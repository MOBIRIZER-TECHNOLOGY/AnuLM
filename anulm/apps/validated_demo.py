"""
A demo page that shows only recorded, reviewed results -- it runs no model.

    python -m anulm validated_demo                       # http://127.0.0.1:7864
    python -m anulm validated_demo --catalogue examples/approved/catalogue.json
    python -m anulm validated_demo --check               # print what passes and why the rest does not

The live demos (`app_omni`, `app`) answer anything a visitor types, which is
the right tool for exploring a model and the wrong one for showing a company
what it can be relied on to do: one bad answer to an improvised question is
what gets remembered. This page shows a fixed set of cases instead, each one
run on held-out input, judged against a criterion written down before the
output was seen, and approved by a named reviewer.

The checks fail closed. A case is shown only if its report and every file it
names still match the SHA-256 recorded for them, the report says `passed` and
`approved: true`, its dataset is marked held-out with a source and a licence,
and it lists the checkpoint and tokenizer (and, for audio or image, the
encoder) it was produced with. Anything missing, changed or outside the
repository excludes that case, and the page says which case and why.

Hashes prove the files are the ones that were reviewed, not that the review was
right: approval stays a person's decision. docs/DEMO_VALIDATION.md describes
how to record a case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "examples" / "approved" / "catalogue.json"
TASKS = {"text", "image", "audio"}
REPORT_FIELDS = ("reviewed_by", "validated_at", "criteria", "output", "source_revision")


class Excluded(ValueError):
    """Why one catalogue entry cannot be shown."""


class Hasher:
    """SHA-256 of each file, computed once per run however many cases share it
    (a 1.6 GB checkpoint is typically listed by every case). Keyed on path,
    size and modification time, so a file changed mid-run is hashed again."""

    def __init__(self):
        self._seen: dict[tuple[str, int, int], str] = {}

    def __call__(self, path: Path) -> str:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self._seen:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._seen[key] = h.hexdigest()
        return self._seen[key]


def digest(path: Path) -> str:
    """SHA-256 of one file, as the catalogue records it."""
    return Hasher()(Path(path))


def checked_file(root: Path, record, what: str, sha) -> Path:
    """The file a record names, if it is inside `root` and its hash matches."""
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record.get("sha256"):
        raise Excluded(f"{what}: needs a path and a sha256")
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise Excluded(f"{what}: {record['path']} is outside the repository")
    if not path.is_file():
        raise Excluded(f"{what}: {record['path']} is missing")
    if sha(path) != record["sha256"]:
        raise Excluded(f"{what}: {record['path']} has changed since it was reviewed")
    return path


def check_case(entry, root: Path, sha) -> dict:
    """One catalogue entry -> a displayable case, or Excluded with the reason."""
    case_id = entry.get("id") if isinstance(entry, dict) else None
    if not isinstance(case_id, str) or not case_id:
        raise Excluded("needs a non-empty string id")
    report_path = checked_file(root, entry.get("report"), "report", sha)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Excluded(f"report is not readable JSON ({e})") from None
    if not isinstance(report, dict):
        raise Excluded("report is not a JSON object")
    if report.get("case_id") != case_id:
        raise Excluded(f"report is for case {report.get('case_id')!r}")
    if report.get("status") != "passed":
        raise Excluded(f"status is {report.get('status')!r}, not 'passed'")
    if report.get("approved") is not True:
        raise Excluded("not approved by a reviewer")
    task = report.get("task")
    if task not in TASKS:
        raise Excluded(f"task {task!r} is not one of {sorted(TASKS)}")
    empty = [f for f in REPORT_FIELDS if not isinstance(report.get(f), str) or not report[f].strip()]
    if empty:
        raise Excluded(f"report leaves {', '.join(empty)} empty")
    data = report.get("dataset")
    if not isinstance(data, dict) or data.get("held_out") is not True:
        raise Excluded("its input is not marked as held-out data")
    if not data.get("source") or not data.get("license"):
        raise Excluded("its dataset has no source or licence")
    assets = report.get("artifacts")
    if not isinstance(assets, list):
        raise Excluded("report lists no artifacts")
    roles = {a.get("role") for a in assets if isinstance(a, dict)}
    needed = {"checkpoint", "tokenizer"} | ({"encoder"} if task != "text" else set())
    if not needed <= roles:
        raise Excluded(f"artifacts lack {', '.join(sorted(needed - roles))}")
    paths = [checked_file(root, a, f"artifact {a.get('role')}", sha) for a in assets]
    if task == "text":
        if not isinstance(report.get("input"), str) or not report["input"].strip():
            raise Excluded("text input is empty")
        media = None
    else:
        media = checked_file(root, report.get("input"), f"{task} input", sha)
    return {"id": case_id, "report": report, "media": media, "artifacts": paths}


def load_cases(catalogue: Path = CATALOGUE, root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """(cases that pass every check, one reason per excluded entry)."""
    root, catalogue = Path(root), Path(catalogue)
    if not catalogue.exists():
        return [], [f"no catalogue at {catalogue}"]
    try:
        config = json.loads(catalogue.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("version") != 1 or not isinstance(config.get("cases"), list):
            raise ValueError("expected {\"version\": 1, \"cases\": [...]}")
    except (OSError, ValueError) as e:
        return [], [f"catalogue {catalogue.name} is invalid: {e}"]
    sha, cases, excluded, seen = Hasher(), [], [], set()
    for n, entry in enumerate(config["cases"], 1):
        label = entry.get("id") if isinstance(entry, dict) and isinstance(entry.get("id"), str) else f"entry {n}"
        try:
            if label in seen:
                raise Excluded("duplicate id")
            seen.add(label)
            cases.append(check_case(entry, root, sha))
        except Excluded as e:
            excluded.append(f"{label}: {e}")
    return cases, excluded


def build(catalogue: Path = CATALOGUE, root: Path = ROOT):
    """The Gradio page: one tab per passing case, then why the others were left out."""
    import gradio as gr
    cases, excluded = load_cases(catalogue, root)
    with gr.Blocks(title="AnuLM reviewed examples") as demo:
        gr.Markdown("# AnuLM reviewed examples\n"
                    "Recorded results on held-out inputs, each judged against a criterion set before the "
                    "output was seen and approved by a reviewer. This page runs no model. Audio and image "
                    "cases use pretrained encoders (Whisper, SigLIP); each record names every file used.")
        if not cases:
            gr.Markdown("No case has passed the checks in this checkout yet.")
        for case in cases:
            r = case["report"]
            with gr.Tab(case["id"]):
                if r["task"] == "image":
                    gr.Image(value=str(case["media"]), interactive=False, label="Input image")
                elif r["task"] == "audio":
                    gr.Audio(value=str(case["media"]), interactive=False, label="Input recording")
                else:
                    gr.Textbox(value=r["input"], interactive=False, label="Input")
                if r.get("question"):
                    gr.Textbox(value=r["question"], interactive=False, label="Question")
                gr.Textbox(value=r["output"], interactive=False, label="Recorded output")
                gr.Textbox(value=r["criteria"], interactive=False, label="Acceptance criterion and result")
                gr.JSON(value={k: r[k] for k in ("dataset", "reviewed_by", "validated_at", "source_revision")},
                        label="Review record")
        if excluded:
            gr.Markdown("**Not shown** (failed the checks):\n\n" + "\n".join(f"- {x}" for x in excluded))
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--catalogue", type=Path, default=CATALOGUE)
    p.add_argument("--check", action="store_true", help="print the results of the checks and exit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7864)
    args = p.parse_args()
    if args.check:
        cases, excluded = load_cases(args.catalogue)
        print(f"{len(cases)} case(s) pass: {', '.join(c['id'] for c in cases) or 'none'}")
        for reason in excluded:
            print(f"  excluded  {reason}")
        return
    build(args.catalogue).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
