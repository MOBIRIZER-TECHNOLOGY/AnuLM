"""Train a tokenizer, AnuLM, and optional omni tasks without a base checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_VERSION = 2
SPLIT_FORMAT = "anulm-document-splits"
ENCODED_FORMAT = "anulm-encoded-splits"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def command_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as e:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another pipeline is already running in {self.path.parent}") from e
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid()}).encode("ascii"))
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def update_status(path: Path, stage: str, status: str, **details) -> None:
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    state["stages"][stage] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    atomic_json(path, state)


def ensure_disk_space(path: Path, reserve_bytes: int, stage: str, required_bytes: int = 0) -> None:
    free = shutil.disk_usage(path).free
    needed = max(reserve_bytes, required_bytes)
    if free < needed:
        raise RuntimeError(
            f"{stage} needs {needed / 2**30:.2f} GiB free at {path}, but only "
            f"{free / 2**30:.2f} GiB is available"
        )
    print(f"disk {stage}: {free / 2**30:.2f} GiB free (required {needed / 2**30:.2f} GiB)", flush=True)


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait()


def run(cmd: list[str], dry_run: bool = False, log_path: Path | None = None,
        timeout: float | None = None) -> None:
    """Run one stage from the repository root, raising on failure or timeout.

    With `log_path`, the output goes to the log and to the console as it is
    written: a pretraining stage runs for hours, and a console that stays
    silent until the stage ends cannot tell a slow run from a hung one."""
    print(f"\n$ {command_text(cmd)}", flush=True)
    if dry_run:
        return
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    log_context = log_path.open("w", encoding="utf-8") if log_path is not None else contextlib.nullcontext()
    with log_context as log:
        if log is None:
            process = subprocess.Popen(cmd, cwd=ROOT, **kwargs)
            pump = None
        else:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", **kwargs)

            def tee():
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)

            pump = threading.Thread(target=tee, daemon=True)
            pump.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            terminate_process_tree(process)
            if pump is not None:
                pump.join(timeout=5)
                # Close our end of the pipe so the handle does not leak. Only
                # once the pump has stopped: closing a buffered stream another
                # thread is blocked reading waits for that read (a grandchild
                # that outlived the kill can hold the pipe open forever).
                if not pump.is_alive():
                    process.stdout.close()
            location = f"; log: {log_path}" if log_path is not None else ""
            raise RuntimeError(f"stage timed out after {timeout:g}s{location}") from e
        if pump is not None:
            pump.join()
    if process.returncode:
        location = f"; log: {log_path}" if log_path is not None else ""
        raise RuntimeError(f"stage command failed with exit code {process.returncode}{location}")


def document_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def prepare_document_splits(corpus: Path, split_dir: Path, seed: int) -> Path:
    from anulm.bpe import DOC_SEP, iter_documents

    split_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    total = 0
    for doc in iter_documents(corpus):
        key = document_hash(doc)
        groups[key] = groups.get(key, 0) + 1
        total += 1
    if not groups:
        raise RuntimeError(f"text corpus has no documents: {corpus}")

    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("ascii")).hexdigest())
    legacy = len(ordered) == 1
    n_test = max(1, round(len(ordered) * 0.1)) if len(ordered) >= 3 else 0
    n_val = max(1, round(len(ordered) * 0.1)) if len(ordered) >= 2 else 0
    assignment = {}
    for i, key in enumerate(ordered):
        assignment[key] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    names = ["train"] if legacy else ["train", "validation"] + (["test"] if n_test else [])
    handles = {}
    temps = {}
    outputs = {}
    counts = {name: 0 for name in names}
    try:
        for name in names:
            outputs[name] = split_dir / f"{name}.txt"
            temps[name] = outputs[name].with_suffix(".txt.tmp")
            handles[name] = temps[name].open("w", encoding="utf-8", newline="\n")
        for doc in iter_documents(corpus):
            name = assignment[document_hash(doc)]
            handles[name].write(doc)
            handles[name].write(DOC_SEP)
            counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for name in names:
        os.replace(temps[name], outputs[name])

    metadata = {
        "format": SPLIT_FORMAT,
        "version": 1,
        "mode": "legacy-single-document" if legacy else "document-grouped",
        "source": {"path": str(corpus), "sha256": sha256(corpus)},
        "seed": seed,
        "documents": {"total": total, "unique": len(groups)},
        "splits": {
            name: {"path": outputs[name].name, "sha256": sha256(outputs[name]), "documents": counts[name]}
            for name in names
        },
    }
    path = split_dir / "splits.json"
    atomic_json(path, metadata)
    if legacy:
        print("WARNING: only one unique document; preserving V2 token-level validation. "
              "Provide at least two unique documents for leak-free holdouts.", flush=True)
    print(f"split: {total} documents / {len(groups)} unique -> "
          + ", ".join(f"{name} {counts[name]}" for name in names), flush=True)
    return path


def validate_document_splits(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != SPLIT_FORMAT:
        raise RuntimeError(f"unsupported split metadata: {path}")
    source = Path(metadata["source"]["path"])
    if sha256(source) != metadata["source"]["sha256"]:
        raise RuntimeError(f"source corpus changed after splitting: {source}")
    seen = {}
    for name, record in metadata["splits"].items():
        split_path = path.parent / record["path"]
        require_file(split_path, f"{name} split")
        if sha256(split_path) != record["sha256"]:
            raise RuntimeError(f"{name} split changed after completion: {split_path}")
        from anulm.bpe import iter_documents
        keys = {document_hash(doc) for doc in iter_documents(split_path)}
        overlap = seen.keys() & keys
        if overlap:
            raise RuntimeError(f"duplicate documents cross data splits in {path}")
        seen.update(dict.fromkeys(keys, name))


def split_path(metadata_path: Path, name: str) -> Path:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata_path.parent / metadata["splits"][name]["path"]


def split_task(spec: str) -> tuple[str, list[str]]:
    name, sep, rest = spec.partition("=")
    if not sep or not name:
        raise ValueError(f"invalid --task {spec!r}; expected name=kind:path[:weight[:batch]]")
    parts = rest.split(":")
    if len(parts) > 2 and len(parts[1]) == 1 and parts[1].isalpha() and parts[2].startswith(("\\", "/")):
        parts[1:3] = [parts[1] + ":" + parts[2]]
    if len(parts) < 2 or parts[0] not in ("text", "asr", "image") or not parts[1]:
        raise ValueError(f"invalid --task {spec!r}; kind must be text, asr, or image")
    return name, parts


def normalize_task(spec: str) -> str:
    name, parts = split_task(spec)
    parts[1] = str(Path(parts[1]).expanduser().resolve())
    return f"{name}=" + ":".join(parts)


def task_manifest(spec: str) -> Path:
    _, parts = split_task(spec)
    path = Path(parts[1])
    return path.with_suffix(".jsonl") if parts[0] == "asr" else path


def task_names(specs: list[str]) -> list[str]:
    return [split_task(spec)[0] for spec in specs]


def validate_task_rows(args, tok) -> None:
    from anulm.training.finetune import _pair_ids

    held = dict(spec.split("=", 1) for spec in args.held)
    for spec in args.task:
        name, parts = split_task(spec)
        path = task_manifest(spec)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if parts[0] == "text":
            rows = [row for row in rows if sum(map(len, _pair_ids(tok, row))) <= args.block_size]
        minimum = 1 if name in held else 2
        if len(rows) < minimum:
            raise RuntimeError(
                f"task {name!r} has {len(rows)} usable rows at block size {args.block_size}; "
                f"at least {minimum} are required for training and automatic holdout"
            )
        if name in held:
            held_rows = [line for line in Path(held[name]).read_text(encoding="utf-8").splitlines()
                         if line.strip()]
            if not held_rows:
                raise RuntimeError(f"held-out task {name!r} has no rows")


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": STATE_VERSION, "stages": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION or not isinstance(state.get("stages"), dict):
        raise RuntimeError(f"unsupported pipeline state in {path}; choose a new --work-dir")
    return state


def save_stage(state_path: Path, state: dict, name: str, identity: dict, output: Path) -> None:
    state["stages"][name] = {
        "identity": identity,
        "output": str(output),
        "sha256": sha256(output),
    }
    atomic_json(state_path, state)


def completed(state: dict, name: str, identity: dict, output: Path, validator) -> bool:
    if not output.exists():
        return False
    record = state["stages"].get(name)
    problem = None
    if not record:
        problem = "it was not recorded as complete"
    elif record.get("identity") != identity or record.get("output") != str(output):
        problem = "it belongs to a different recipe"
    elif record.get("sha256") != sha256(output):
        problem = "its contents changed after completion"
    if problem:
        raise RuntimeError(
            f"refusing to reuse or overwrite {output}: {problem}. Choose a new --work-dir; "
            "use --resume only to continue an interrupted training stage."
        )
    validator(output)
    print(f"skip {name}: verified {output}", flush=True)
    return True


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{description} not found: {path}")


def resume_identity(path: Path, identity: dict) -> None:
    if not path.exists():
        raise RuntimeError(f"resume metadata is missing: {path}; choose a new --work-dir")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != identity:
        raise RuntimeError(
            f"resume options or inputs do not match {path}. Use the original command with "
            f"--resume, or choose a new --work-dir."
        )


def expected_config(args, tokenizer: Path, vocab_size: int):
    from anulm.model import AnuLMConfig

    from anulm.training.train import PRESETS
    builder = PRESETS[args.preset]
    return builder(
        block_size=args.block_size,
        yarn=False,
        yarn_original_context=args.block_size,
        moe_impl=args.moe_impl,
        vocab_size=vocab_size,
        tokenizer_path=str(tokenizer),
        seq_balance_alpha=0.0,
        router_z_alpha=0.0,
        sliding_window=None,
        max_window_layers=0,
    )


def checkpoint_validator(expected, tokenizer: Path, text_vocab: int, grown: bool,
                         names: list[str] | None = None, base: Path | None = None):
    def validate(path: Path) -> None:
        from anulm.model import safe_load

        try:
            ck = safe_load(path, map_location="cpu")
            cfg = ck["cfg"]
            model = ck["model"]
            compatible = asdict(cfg) == asdict(expected)
            compatible = compatible and Path(cfg.tokenizer_path).resolve() == tokenizer
            compatible = compatible and model["embed_tokens.weight"].shape[0] == cfg.vocab_size
            compatible = compatible and model["lm_head.weight"].shape[0] == cfg.vocab_size
            if grown:
                compatible = compatible and ck.get("speech_text_vocab") == text_vocab
            else:
                compatible = compatible and "speech_text_vocab" not in ck
            if names is not None:
                compatible = compatible and ck.get("tasks") == names
            if base is not None:
                compatible = compatible and Path(ck.get("base_ckpt", "")).resolve() == base
        except Exception as e:
            raise RuntimeError(f"incompatible or unreadable checkpoint {path}: {e}") from e
        if not compatible:
            raise RuntimeError(
                f"incompatible checkpoint {path}: architecture, tokenizer, vocabulary, tasks, "
                "or base checkpoint differs; choose a new --work-dir"
            )

    return validate


def validate_tokenizer(path: Path) -> None:
    from anulm.bpe import BPE

    BPE.load(path)


def encoded_validator(tokenizer: Path, vocab_size: int):
    def validate(path: Path) -> None:
        from anulm.model import safe_load

        blob = safe_load(path, map_location="cpu")
        if (blob.get("vocab_size") != vocab_size
                or Path(blob.get("tokenizer", "")).resolve() != tokenizer
                or not len(blob.get("ids", ()))):
            raise RuntimeError(f"encoded corpus is incompatible with {tokenizer}: {path}")

    return validate


def create_encoded_manifest(split_metadata: Path, encoded: dict[str, Path], output: Path) -> None:
    atomic_json(output, {
        "format": ENCODED_FORMAT,
        "version": 1,
        "document_splits_sha256": sha256(split_metadata),
        "splits": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in encoded.items()
        },
    })


def encoded_bundle_validator(tokenizer: Path, vocab_size: int, split_metadata: Path):
    validate_encoded = encoded_validator(tokenizer, vocab_size)

    def validate(path: Path) -> None:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (metadata.get("format") != ENCODED_FORMAT
                or metadata.get("document_splits_sha256") != sha256(split_metadata)):
            raise RuntimeError(f"encoded corpus does not match document splits: {path}")
        for name, record in metadata["splits"].items():
            encoded_path = Path(record["path"])
            if sha256(encoded_path) != record["sha256"]:
                raise RuntimeError(f"encoded {name} split changed after completion: {encoded_path}")
            validate_encoded(encoded_path)

    return validate


def pretrain_command(args, paths: dict[str, Path], resume: bool) -> list[str]:
    cmd = [
        sys.executable, "-m", "anulm.training.train",
        "--preset", args.preset,
        "--data", str(paths["encoded_manifest"]),
        "--out", str(paths["base"]),
        "--steps", str(args.pretrain_steps),
        "--batch-size", str(args.batch_size),
        "--block-size", str(args.block_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--warmup", str(args.warmup),
        "--eval-every", str(args.eval_every),
        "--eval-iters", str(args.eval_iters),
        "--test-iters", str(args.test_iters),
        "--log-every", str(args.log_every),
        "--device", args.device,
        "--moe-impl", args.moe_impl,
        "--seed", str(args.seed),
    ]
    if args.grad_ckpt:
        cmd.append("--grad-ckpt")
    if args.report_gpu_memory:
        cmd.append("--report-gpu-memory")
    if resume:
        cmd.append("--resume")
    return cmd


def omni_command(args, paths: dict[str, Path], resume: bool) -> list[str]:
    cmd = [
        sys.executable, "-m", "anulm.multimodal.omni_train",
        "--ckpt", str(paths["grown"]),
        "--out", str(paths["omni"]),
        "--steps", str(args.omni_steps),
        "--lr", str(args.omni_lr),
        "--proj-lr", str(args.proj_lr),
        "--warmup", str(args.omni_warmup),
        "--eval-every", str(args.omni_eval_every),
        "--eval-n", str(args.omni_eval_n),
        "--log-every", str(args.log_every),
        "--accum", str(args.omni_accum),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    for spec in args.task:
        cmd.extend(["--task", spec])
    for spec in args.held:
        cmd.extend(["--held", spec])
    if args.grad_ckpt:
        cmd.append("--grad-ckpt")
    if resume:
        cmd.append("--resume")
    return cmd


def dry_run(args, paths: dict[str, Path]) -> None:
    print("Dry run: no files are read, created, or modified; every planned stage is shown.")
    print(f"split {paths['corpus']} by document group -> {paths['split_metadata']}")
    run([sys.executable, "-m", "anulm.bpe", "train", "--data", str(paths["train_text"]),
         "--vocab", str(args.vocab), "--out", str(paths["tokenizer"])], True)
    for name in ("train", "validation", "test"):
        run([sys.executable, "-m", "anulm.bpe", "encode", "--tokenizer", str(paths["tokenizer"]),
             "--data", str(paths[f"{name}_text"]), "--out", str(paths[f"{name}_encoded"]),
             "--workers", str(args.workers)], True)
    run(pretrain_command(args, paths, args.resume), True)
    run([sys.executable, "-m", "anulm.multimodal.speech_vocab", "--ckpt", str(paths["base"]),
         "--out", str(paths["grown"]), "--seed", str(args.seed)], True)
    if args.task:
        run(omni_command(args, paths, args.resume), True)
    else:
        print("\nNo --task supplied: omni training is not planned.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus", required=True, help="UTF-8 text used for both BPE training and pretraining")
    p.add_argument("--work-dir", default="runs/from_scratch", help="new directory for all pipeline artifacts")
    p.add_argument("--vocab", type=int, default=16384, help="target BPE size, including EOS; at most 32768")
    p.add_argument("--workers", type=int, default=16, help="BPE encoding worker processes")
    p.add_argument("--preset", choices=("tiny-gqa", "tiny-mla", "350m", "1b"), default="tiny-gqa")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--pretrain-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--test-iters", type=int, default=20, help="final-only test evaluation batches")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--report-gpu-memory", action="store_true", help="report CUDA preflight and peak memory")
    p.add_argument("--stage-timeout", type=float, default=None,
                   help="seconds before a stage is killed; default no limit, since pretraining runs for hours")
    p.add_argument("--min-free-gib", type=float, default=1.0, help="minimum free disk before each large stage")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--moe-impl", choices=("sparse", "dense", "grouped"), default="sparse")
    p.add_argument("--device", help="for example cpu, cuda, or cuda:0; default is cuda when available")
    p.add_argument("--task", action="append", default=[],
                   help="omni task: name=text:manifest.jsonl, name=asr:prefix, or "
                        "name=image:manifest.jsonl; append [:weight[:batch]] as needed")
    p.add_argument("--held", action="append", default=[], help="optional omni held-out manifest: task_name=path")
    p.add_argument("--omni-steps", type=int, default=2000)
    p.add_argument("--omni-accum", type=int, default=1, help="omni batches per optimizer step")
    p.add_argument("--omni-lr", type=float, default=5e-5)
    p.add_argument("--proj-lr", type=float, default=1e-3)
    p.add_argument("--omni-warmup", type=int, default=500)
    p.add_argument("--omni-eval-every", type=int, default=5000)
    p.add_argument("--omni-eval-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true", help="resume interrupted pretrain/omni stages from their .last files")
    p.add_argument("--dry-run", action="store_true", help="print all subprocess commands without reading or writing files")
    args = p.parse_args()

    positive = ("vocab", "workers", "block_size", "pretrain_steps", "batch_size", "grad_accum",
                "eval_every", "eval_iters", "test_iters", "log_every", "omni_steps", "omni_accum",
                "omni_eval_every", "omni_eval_n")
    for name in positive:
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.stage_timeout is not None and args.stage_timeout <= 0:
        p.error("--stage-timeout must be positive")
    if not 257 <= args.vocab <= 32768:
        p.error("--vocab must be between 257 and 32768")
    args.task = [normalize_task(spec) for spec in args.task]
    if len(set(task_names(args.task))) != len(args.task):
        p.error("omni task names must be unique")
    held = []
    for spec in args.held:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            p.error("each --held must have the form task_name=path")
        held.append(f"{name}={Path(path).expanduser().resolve()}")
    args.held = held
    held_names = [spec.split("=", 1)[0] for spec in args.held]
    if any(name not in task_names(args.task) for name in held_names):
        p.error("--held names must refer to supplied --task names")

    if args.min_free_gib < 0:
        p.error("--min-free-gib cannot be negative")

    work = Path(args.work_dir).expanduser().resolve()
    split_dir = work / "splits"
    paths = {
        "corpus": Path(args.corpus).expanduser().resolve(),
        "split_metadata": split_dir / "splits.json",
        "train_text": split_dir / "train.txt",
        "validation_text": split_dir / "validation.txt",
        "test_text": split_dir / "test.txt",
        "tokenizer": work / "bpe.json",
        "train_encoded": work / "corpus.bin",
        "validation_encoded": work / "validation.bin",
        "test_encoded": work / "test.bin",
        "encoded_manifest": work / "corpus.splits.json",
        "base": work / "pretrained.pt",
        "grown": work / "multimodal_init.pt",
        "omni": work / "omni.pt",
    }
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        dry_run(args, paths)
        return

    require_file(paths["corpus"], "text corpus")
    for spec in args.task:
        require_file(task_manifest(spec), "task manifest")
    for spec in args.held:
        require_file(Path(spec.split("=", 1)[1]), "held-out manifest")
    work.mkdir(parents=True, exist_ok=True)
    with RunLock(work / ".pipeline.lock"):
        run_pipeline(args, paths, work)


def run_pipeline(args, paths: dict[str, Path], work: Path) -> None:
    state_path = work / "pipeline.json"
    status_path = work / "status.json"
    state = load_state(state_path)
    reserve = int(args.min_free_gib * 2**30)

    from anulm.bpe import BPE
    from anulm.multimodal.speech_vocab import SpeechVocab

    def invoke(stage: str, cmd: list[str]) -> None:
        log = work / "logs" / f"{stage}.log"
        update_status(status_path, stage, "running", log=str(log), command=cmd)
        try:
            run(cmd, log_path=log, timeout=args.stage_timeout)
        except Exception as e:
            update_status(status_path, stage, "failed", log=str(log), error=str(e))
            raise

    def finish(stage: str, output: Path) -> None:
        update_status(status_path, stage, "completed", output=str(output), sha256=sha256(output))

    def skip(stage: str, output: Path) -> None:
        update_status(status_path, stage, "skipped", output=str(output), sha256=sha256(output))

    corpus_hash = sha256(paths["corpus"])
    split_identity = {"corpus_sha256": corpus_hash, "seed": args.seed}
    if completed(state, "split", split_identity, paths["split_metadata"], validate_document_splits):
        skip("split", paths["split_metadata"])
    else:
        ensure_disk_space(work, reserve, "split", paths["corpus"].stat().st_size * 2)
        update_status(status_path, "split", "running")
        try:
            prepare_document_splits(paths["corpus"], paths["split_metadata"].parent, args.seed)
            validate_document_splits(paths["split_metadata"])
        except Exception as e:
            update_status(status_path, "split", "failed", error=str(e))
            raise
        save_stage(state_path, state, "split", split_identity, paths["split_metadata"])
        finish("split", paths["split_metadata"])

    split_metadata = json.loads(paths["split_metadata"].read_text(encoding="utf-8"))
    split_names = list(split_metadata["splits"])
    split_hash = sha256(paths["split_metadata"])
    tokenizer_identity = {
        "corpus_sha256": corpus_hash,
        "document_splits_sha256": split_hash,
        "train_sha256": split_metadata["splits"]["train"]["sha256"],
        "vocab": args.vocab,
    }
    tokenizer_cmd = [sys.executable, "-m", "anulm.bpe", "train", "--data", str(paths["train_text"]),
                     "--vocab", str(args.vocab), "--out", str(paths["tokenizer"])]
    if completed(state, "tokenizer", tokenizer_identity, paths["tokenizer"], validate_tokenizer):
        skip("tokenizer", paths["tokenizer"])
    else:
        ensure_disk_space(work, reserve, "tokenizer")
        invoke("tokenizer", tokenizer_cmd)
        validate_tokenizer(paths["tokenizer"])
        save_stage(state_path, state, "tokenizer", tokenizer_identity, paths["tokenizer"])
        finish("tokenizer", paths["tokenizer"])

    tok = BPE.load(paths["tokenizer"])
    validate_task_rows(args, tok)
    tokenizer_hash = sha256(paths["tokenizer"])
    encoded_identity = {
        "corpus_sha256": corpus_hash,
        "document_splits_sha256": split_hash,
        "split_text_sha256": {name: split_metadata["splits"][name]["sha256"] for name in split_names},
        "tokenizer_sha256": tokenizer_hash,
    }
    validate_encoded = encoded_bundle_validator(paths["tokenizer"], tok.vocab_size, paths["split_metadata"])
    if completed(state, "encode", encoded_identity, paths["encoded_manifest"], validate_encoded):
        skip("encode", paths["encoded_manifest"])
    else:
        ensure_disk_space(work, reserve, "encode", paths["corpus"].stat().st_size * 4)
        for name in split_names:
            encoded_cmd = [sys.executable, "-m", "anulm.bpe", "encode",
                           "--tokenizer", str(paths["tokenizer"]),
                           "--data", str(paths[f"{name}_text"]),
                           "--out", str(paths[f"{name}_encoded"]),
                           "--workers", str(args.workers)]
            invoke(f"encode_{name}", encoded_cmd)
            finish(f"encode_{name}", paths[f"{name}_encoded"])
        encoded = {name: paths[f"{name}_encoded"] for name in split_names}
        create_encoded_manifest(paths["split_metadata"], encoded, paths["encoded_manifest"])
        validate_encoded(paths["encoded_manifest"])
        save_stage(state_path, state, "encode", encoded_identity, paths["encoded_manifest"])
        finish("encode", paths["encoded_manifest"])

    expected_base = expected_config(args, paths["tokenizer"], tok.vocab_size)
    validate_base = checkpoint_validator(expected_base, paths["tokenizer"], tok.vocab_size, False)
    encoded_metadata = json.loads(paths["encoded_manifest"].read_text(encoding="utf-8"))
    pretrain_identity = {
        "corpus_sha256": corpus_hash,
        "document_splits_sha256": split_hash,
        "encoded_manifest_sha256": sha256(paths["encoded_manifest"]),
        "encoded_sha256": {name: record["sha256"] for name, record in encoded_metadata["splits"].items()},
        "tokenizer_sha256": tokenizer_hash,
        "preset": args.preset,
        "pretrain_steps": args.pretrain_steps,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup": args.warmup,
        "eval_every": args.eval_every,
        "eval_iters": args.eval_iters,
        "test_iters": args.test_iters,
        "grad_ckpt": args.grad_ckpt,
        "moe_impl": args.moe_impl,
        "device": args.device,
        "seed": args.seed,
    }
    pretrain_meta = Path(str(paths["base"]) + ".pipeline.json")
    pretrain_last = Path(str(paths["base"]) + ".last")
    resume_pretrain = args.resume and pretrain_last.exists()
    pretrain_done = False
    if resume_pretrain:
        resume_identity(pretrain_meta, pretrain_identity)
        validate_base(pretrain_last)
    else:
        pretrain_done = completed(state, "pretrain", pretrain_identity, paths["base"], validate_base)
        if not pretrain_done:
            if pretrain_last.exists():
                raise RuntimeError(f"interrupted pretraining found at {pretrain_last}; rerun with --resume")
            if paths["base"].exists():
                raise RuntimeError(
                    f"incomplete pretraining output {paths['base']}; use --resume or a new --work-dir")
            if pretrain_meta.exists():
                resume_identity(pretrain_meta, pretrain_identity)
    if pretrain_done:
        skip("pretrain", paths["base"])
    else:
        ensure_disk_space(work, reserve, "pretrain")
        atomic_json(pretrain_meta, pretrain_identity)
        invoke("pretrain", pretrain_command(args, paths, resume_pretrain))
        validate_base(paths["base"])
        save_stage(state_path, state, "pretrain", pretrain_identity, paths["base"])
        finish("pretrain", paths["base"])

    expected_grown = replace(expected_base, vocab_size=SpeechVocab(tok.vocab_size).total)
    validate_grown = checkpoint_validator(expected_grown, paths["tokenizer"], tok.vocab_size, True,
                                          base=paths["base"])
    grown_identity = {"base_sha256": sha256(paths["base"]), "tokenizer_sha256": tokenizer_hash,
                      "seed": args.seed}
    grown_cmd = [sys.executable, "-m", "anulm.multimodal.speech_vocab", "--ckpt", str(paths["base"]),
                 "--out", str(paths["grown"]), "--seed", str(args.seed)]
    if completed(state, "grow", grown_identity, paths["grown"], validate_grown):
        skip("grow", paths["grown"])
    else:
        ensure_disk_space(work, reserve, "grow")
        invoke("grow", grown_cmd)
        validate_grown(paths["grown"])
        save_stage(state_path, state, "grow", grown_identity, paths["grown"])
        finish("grow", paths["grown"])

    if not args.task:
        update_status(status_path, "pipeline", "completed")
        print(f"\nDone. Fresh pretrained checkpoint: {paths['base']}")
        print(f"Multimodal-ready checkpoint: {paths['grown']}")
        print("Add at least one --task to run omni_train.")
        return

    names = task_names(args.task)
    task_identity = [{"spec": spec, "manifest_sha256": sha256(task_manifest(spec))} for spec in args.task]
    held_identity = [{"spec": spec, "manifest_sha256": sha256(Path(spec.split("=", 1)[1]))}
                     for spec in args.held]
    omni_identity = {
        "grown_sha256": sha256(paths["grown"]),
        "tasks": task_identity,
        "held": held_identity,
        "omni_steps": args.omni_steps,
        "omni_accum": args.omni_accum,
        "omni_lr": args.omni_lr,
        "proj_lr": args.proj_lr,
        "omni_warmup": args.omni_warmup,
        "omni_eval_every": args.omni_eval_every,
        "omni_eval_n": args.omni_eval_n,
        "grad_ckpt": args.grad_ckpt,
        "device": args.device,
        "seed": args.seed,
    }
    validate_omni = checkpoint_validator(expected_grown, paths["tokenizer"], tok.vocab_size, True,
                                         names=names, base=paths["grown"])
    omni_meta = Path(str(paths["omni"]) + ".pipeline.json")
    omni_last = Path(str(paths["omni"]) + ".last")
    resume_omni = args.resume and omni_last.exists()
    omni_done = False
    if resume_omni:
        resume_identity(omni_meta, omni_identity)
        validate_omni(omni_last)
    else:
        omni_done = completed(state, "omni", omni_identity, paths["omni"], validate_omni)
        if not omni_done:
            if omni_last.exists():
                raise RuntimeError(f"interrupted omni training found at {omni_last}; rerun with --resume")
            if paths["omni"].exists():
                raise RuntimeError(f"incomplete omni output {paths['omni']}; use --resume or a new --work-dir")
            if omni_meta.exists():
                resume_identity(omni_meta, omni_identity)
    if omni_done:
        skip("omni", paths["omni"])
    else:
        ensure_disk_space(work, reserve, "omni")
        atomic_json(omni_meta, omni_identity)
        invoke("omni", omni_command(args, paths, resume_omni))
        validate_omni(paths["omni"])
        save_stage(state_path, state, "omni", omni_identity, paths["omni"])
        finish("omni", paths["omni"])
    update_status(status_path, "pipeline", "completed")
    print(f"\nDone. Omni checkpoint: {paths['omni']}")


if __name__ == "__main__":
    main()
