"""Test trained AnuLM checkpoints locally; never trains or changes a model.

    pip install gradio huggingface_hub
    python -m anulm.apps.evaluate --ckpt /path/to/checkpoint.pt

The source may be a local .pt checkpoint, a folder from export_hf.py, or a
Hugging Face Hub id. Nothing loads until Load or an evaluation button is used.
One omni.Omni instance shares its decoder between text, ASR, and vision.

Text batches are UTF-8 JSONL with one {"question": ..., "answer": ...} object
per line. examples/evaluation/text_task.jsonl is only a format example: supply your own
held-out data for meaningful measurements. The server binds to localhost by
default; --host and --share must be chosen explicitly to expose it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path

import gradio as gr

from anulm.evaluation.eval_qa import f1, tokens
from anulm.evaluation.eval_speech import wer

from anulm.paths import PROJECT_ROOT
ROOT = PROJECT_ROOT
MAX_BATCH_ROWS = 50
MAX_NEW_TOKENS = 256
MAX_VISION_TOKENS = 64
MAX_QUESTION_CHARS = 20_000
MODES = ("question", "continue", "translate")
HUB_ID = re.compile(r"[\w.-]+/[\w.-]+")
DEVICE = re.compile(r"cpu|mps|cuda(?::[0-9]+)?")


def uploaded_path(value, description: str = "file") -> Path:
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    elif not isinstance(value, (str, os.PathLike)):
        value = getattr(value, "path", None) or getattr(value, "name", None)
    if not value:
        raise ValueError(f"choose a {description}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{description} is not a readable file: {path}")
    return path


def validate_source(source: str) -> str:
    source = (source or "").strip()
    if not source:
        raise ValueError("enter a local .pt checkpoint, exported folder, or Hub id")
    if "\0" in source:
        raise ValueError("invalid model source")
    path = Path(source).expanduser()
    if path.exists():
        if path.is_file() and path.suffix.lower() != ".pt":
            raise ValueError("a local checkpoint file must have a .pt extension")
        if not path.is_file() and not path.is_dir():
            raise ValueError("model source must be a file or directory")
        return str(path.resolve())
    if not HUB_ID.fullmatch(source):
        raise ValueError(f"local model source does not exist: {source}")
    return source



def validate_device(device: str) -> str:
    import torch

    device = (device or "auto").strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not DEVICE.fullmatch(device):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def question_text(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("enter a non-empty question or prompt")
    value = value.strip()
    if len(value) > MAX_QUESTION_CHARS:
        raise ValueError(f"questions are limited to {MAX_QUESTION_CHARS:,} characters")
    return value


def reference_text(value, required: bool = False) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError("each evaluation row needs a non-empty answer string")
        return None
    if not isinstance(value, str):
        raise ValueError("reference answer must be a string")
    return value.strip()


def read_text_jsonl(value, limit: int = MAX_BATCH_ROWS) -> list[dict]:
    path = uploaded_path(value, "text evaluation JSONL")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if len(rows) >= limit:
                    raise ValueError(f"batch has more than {limit} rows; split it into smaller files")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number}: each row must be a JSON object")
                try:
                    question = question_text(row.get("question"))
                    answer = reference_text(row.get("answer"), required=True)
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}:{line_number}: expected non-empty question and answer strings ({exc})"
                    ) from exc
                item = {"question": question, "answer": answer}
                if row.get('lang') is not None:
                    if not isinstance(row['lang'], str) or not row['lang'].strip():
                        raise ValueError('lang must be a nonempty prompt-template key.')
                    item['lang'] = row['lang']
                rows.append(item)
    except UnicodeDecodeError as exc:
        raise ValueError(f"text evaluation JSONL must be UTF-8: {path}: {exc}") from exc
    if not rows:
        raise ValueError("text evaluation JSONL has no rows")
    return rows


def generation_settings(max_tokens, temperature, top_k, seed, repetition_penalty) -> dict:
    try:
        numeric = {
            "max_tokens": float(max_tokens),
            "temperature": float(temperature),
            "top_k": float(top_k),
            "seed": float(seed),
            "repetition_penalty": float(repetition_penalty),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("generation settings must be numbers") from exc
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("generation settings must be finite")
    for name in ("max_tokens", "top_k", "seed"):
        if not numeric[name].is_integer():
            raise ValueError(f"{name.replace('_', ' ')} must be an integer")
    settings = {
        "max_tokens": int(numeric["max_tokens"]),
        "temperature": numeric["temperature"],
        "top_k": int(numeric["top_k"]),
        "seed": int(numeric["seed"]),
        "repetition_penalty": numeric["repetition_penalty"],
    }
    if not 1 <= settings["max_tokens"] <= MAX_NEW_TOKENS:
        raise ValueError(f"max tokens must be between 1 and {MAX_NEW_TOKENS}")
    if not 0 <= settings["temperature"] <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not 0 <= settings["top_k"] <= 1000:
        raise ValueError("top k must be between 0 and 1000")
    if not 0 <= settings["seed"] <= 2**31 - 1:
        raise ValueError("seed must be between 0 and 2^31 - 1")
    if not 1 <= settings["repetition_penalty"] <= 2:
        raise ValueError("repetition penalty must be between 1 and 2")
    return settings


def vision_token_limit(value) -> int:
    settings = generation_settings(value, 0, 1, 0, 1)
    if settings["max_tokens"] > MAX_VISION_TOKENS:
        raise ValueError(f"image generation is limited to {MAX_VISION_TOKENS} tokens")
    return settings["max_tokens"]


def normalize_answer(text: str) -> str:
    return " ".join(tokens(text.lower()))


def text_scores(prediction: str, reference: str | None) -> dict:
    if reference is None:
        return {"exact_match": None, "f1": None}
    return {
        "exact_match": float(normalize_answer(prediction) == normalize_answer(reference)),
        "f1": f1(prediction.lower(), reference.lower()),
    }


def describe_error(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    lower = message.lower()
    if "projector" in lower or "proj" in lower and "state_dict" in lower:
        return (
            "This checkpoint does not contain the projector needed for that modality. "
            "Use an ASR, caption/VQA, or Omni checkpoint with the matching projector. "
            f"Original error: {message}"
        )
    if isinstance(exc, FileNotFoundError) or "index" in lower or "no such file" in lower:
        return (
            "A model file, tokenizer, or index is missing. Check that the .pt path or exported "
            "folder is complete (including config/tokenizer/weight files), or that the Hub id is correct. "
            f"Original error: {message}"
        )
    if isinstance(exc, ModuleNotFoundError):
        return f"An inference dependency is not installed: {message}"
    return f"{type(exc).__name__}: {message}"


class ModelManager:
    """At most one lazily loaded Omni, with all modalities on its decoder."""

    def __init__(self):
        self._model = None
        self._key = None
        self.lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, source: str, device: str):
        source = validate_source(source)
        device = validate_device(device)
        from anulm.evaluation.evidence import path_record
        identity = path_record(source)
        key = (source, device, json.dumps(identity, sort_keys=True))
        with self.lock:
            if self._model is not None and self._key == key:
                return self._model
            self._model = None
            self._key = None
            try:
                from anulm.multimodal.omni import Omni

                model = Omni(source, device)
            except (Exception, SystemExit) as exc:
                raise RuntimeError(f"Could not load {source}: {describe_error(exc)}") from exc
            if path_record(source) != identity:
                raise RuntimeError("Model artifacts changed while loading; retry with stable files.")
            model.engine.info["ckpt"] = source
            model.engine.info["evaluation_artifacts"] = identity
            cfg = getattr(getattr(model.engine, "model", None), "cfg", None)
            tokenizer = getattr(cfg, "tokenizer_path", None)
            model.engine.info["evaluation_tokenizer"] = path_record(tokenizer) if tokenizer else {"status": "unresolved"}
            model.engine.info["evaluation_encoders"] = {
                key: path_record(value) for key, value in getattr(model, "meta", {}).items()
                if key in ("whisper", "tower")}

            self._model, self._key = model, key
            return model


def model_metadata(model) -> dict:
    info = dict(model.engine.info)
    info["modalities"] = {
        "text": True,
        "asr": getattr(model, "proj", None) is not None,
        "vision": getattr(model, "vision_proj", None) is not None and "tower" in model.meta,
    }
    info["note"] = "Inference only. Use held-out inputs for meaningful evaluation."
    return info


def require_modality(model, modality: str) -> None:
    if modality == "asr" and getattr(model, "proj", None) is None:
        raise ValueError("this checkpoint has no audio projector (proj); choose an ASR or Omni checkpoint")
    if modality == "vision":
        if getattr(model, "vision_proj", None) is None:
            raise ValueError(
                "this checkpoint has no image projector (vision_proj); choose a caption/VQA or Omni checkpoint"
            )
        if "tower" not in model.meta:
            raise ValueError("this checkpoint has no vision tower metadata; choose a caption/VQA or Omni checkpoint")


def _generate(model, question: str, mode: str, settings: dict, lang=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    requested_mode = mode
    if mode == 'question' and lang is not None:
        from anulm.data.make_qa import PROMPTS
        templates = getattr(model.engine, 'qa_templates', None) or PROMPTS
        if lang not in templates:
            raise ValueError(f'Unknown prompt-template language: {lang}')
        question = templates[lang].format(q=question)
        mode = 'continue'  # Already formatted with the declared training template.
    output = model.engine.generate(
        question,
        settings["max_tokens"],
        settings["temperature"],
        settings["top_k"],
        settings["seed"],
        mode=mode,
        repetition_penalty=settings["repetition_penalty"],
    )
    if lang is not None:
        output['mode'] = requested_mode + ':' + lang
    return output


def run_text(manager: ModelManager, source: str, device: str, question: str, reference: str | None,
             mode: str, max_tokens, temperature, top_k, seed, repetition_penalty, lang=None) -> tuple[dict, str]:
    question = question_text(question)
    reference = reference_text(reference)
    settings = generation_settings(max_tokens, temperature, top_k, seed, repetition_penalty)
    with manager.lock:
        model = manager.load(source, device)
        output = _generate(model, question, mode, settings, lang)
    result = {
        "question": question,
        "reference_answer": reference,
        "prediction": output["completion"],
        **text_scores(output["completion"], reference),
        "generation": {key: output.get(key) for key in ("tokens", "seconds", "stopped_at_eos", "mode")},
    }
    payload = {"loaded_artifacts": model_metadata(model), "model": source, "device": validate_device(device), "settings": settings, "results": [result]}
    return result, results_file(payload)


def run_text_batch(manager: ModelManager, source: str, device: str, jsonl_file, mode: str,
                   max_tokens, temperature, top_k, seed, repetition_penalty) -> tuple[dict, str]:
    rows = read_text_jsonl(jsonl_file)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    settings = generation_settings(max_tokens, temperature, top_k, seed, repetition_penalty)
    results = []
    with manager.lock:
        model = manager.load(source, device)
        for index, row in enumerate(rows):
            output = _generate(model, row["question"], mode, settings, row.get('lang'))
            results.append({
                "row": index + 1,
                "question": row["question"],
                "reference_answer": row["answer"],
                "prediction": output["completion"],
                **text_scores(output["completion"], row["answer"]),
                "tokens": output.get("tokens"),
                "seconds": output.get("seconds"),
            })
    payload = {
        "loaded_artifacts": model_metadata(model),
        "model": source,
        "device": validate_device(device),
        "settings": settings,
        "summary": {
            "rows": len(results),
            "exact_match": sum(row["exact_match"] for row in results) / len(results),
            "f1": sum(row["f1"] for row in results) / len(results),
        },
        "results": results,
    }
    return payload, results_file(payload)


def run_asr(manager: ModelManager, source: str, device: str, audio, reference: str | None) -> tuple[dict, str]:
    audio_path = uploaded_path(audio, "audio clip")
    from anulm.evaluation.evidence import file_record
    input_identity = file_record(audio_path)
    reference = reference_text(reference)
    with manager.lock:
        model = manager.load(source, device)
        require_modality(model, "asr")
        try:
            heard = model.ear().hear(str(audio_path))
        except (Exception, SystemExit) as exc:
            raise RuntimeError(f"ASR failed: {describe_error(exc)}") from exc
    if file_record(audio_path) != input_identity:
        raise ValueError("Audio changed during inference.")
    prediction = heard.get("text", "")
    result = {
        "audio": str(audio_path),
        "reference_transcript": reference,
        "transcription": prediction,
        "language": heard.get("language"),
        "seconds": heard.get("seconds"),
        "wer": None,
        "edits": None,
        "reference_words": None,
    }
    if reference is not None:
        edits, words = wer(reference, prediction)
        result.update(wer=edits / max(words, 1), edits=edits, reference_words=words)
    payload = {"loaded_artifacts": model_metadata(model), "model": source, "device": validate_device(device), "results": [result]}
    return result, results_file(payload)


def run_vision(manager: ModelManager, source: str, device: str, image, task: str, question: str | None,
               reference: str | None, max_tokens) -> tuple[dict, str]:
    image_path = uploaded_path(image, "image")
    from anulm.evaluation.evidence import file_record
    input_identity = file_record(image_path)
    if task not in ("caption", "vqa"):
        raise ValueError("image task must be caption or vqa")
    question = question_text(question) if task == "vqa" else None
    reference = reference_text(reference)
    max_tokens = vision_token_limit(max_tokens)
    with manager.lock:
        model = manager.load(source, device)
        require_modality(model, "vision")
        try:
            from anulm.multimodal.vision_encoder import answer, caption

            vl, tower, tok = model.eye()
            if task == "vqa":
                prediction = answer(vl, tower, tok, str(image_path), question,
                                    max_tokens=max_tokens, device=validate_device(device))
            else:
                prediction = caption(vl, tower, tok, str(image_path),
                                     max_tokens=max_tokens, device=validate_device(device))
        except (Exception, SystemExit) as exc:
            raise RuntimeError(f"Image evaluation failed: {describe_error(exc)}") from exc
    if file_record(image_path) != input_identity:
        raise ValueError("Image changed during inference.")
    result = {
        "image": str(image_path),
        "task": task,
        "question": question,
        "reference_answer": reference,
        "prediction": prediction,
        **text_scores(prediction, reference),
        "score_note": "Normalized exact match and token F1 are simple lexical scores, not caption quality judgments.",
    }
    payload = {"loaded_artifacts": model_metadata(model), "model": source, "device": validate_device(device), "results": [result]}
    return result, results_file(payload)


def results_file(payload: dict) -> str:
    from anulm.evaluation.evidence import stamp_result
    payload = stamp_result(payload)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", prefix="anulm-eval-",
                                     delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        return stream.name


def build(manager: ModelManager | None = None, initial_source: str = "", initial_device: str = "auto") -> gr.Blocks:
    manager = manager or ModelManager()
    css = """
    .gradio-container {max-width: 1180px !important; margin: auto}
    #hero h1 {margin-bottom: .2em}
    #hero p {margin-top: 0}
    .note {font-size: .9em; opacity: .8}
    """
    with gr.Blocks(title="AnuLM · research evaluation") as demo:
        gr.Markdown("**Experimental evaluation — outputs are unreviewed and are not approved demo results.**")
        gr.Markdown(
            "# AnuLM · local model testing\n"
            "Evaluate a trained checkpoint on **your held-out data**. Inference only: no training, "
            "fine-tuning, or checkpoint writes. One decoder is shared by text, speech, and image tests.",
            elem_id="hero",
        )
        with gr.Row():
            source = gr.Textbox(value=initial_source, label="checkpoint, exported folder, or Hub id",
                                placeholder="ckpt_omni.pt  or  org/model", scale=4)
            device = gr.Dropdown(("auto", "cpu", "cuda", "cuda:0", "mps"), value=initial_device,
                                 allow_custom_value=True, label="device", scale=1)
            load_button = gr.Button("Load model", variant="secondary", scale=1)
        load_status = gr.Markdown("Nothing loaded. Loading happens on the first load or evaluation request.")
        metadata = gr.JSON(label="model metadata", value={})
        gr.Markdown(
            "Use data the model did not train on. `examples/evaluation/text_task.jsonl` shows the text format only; "
            "its two tiny examples are not a benchmark. Hub models and frozen speech/image encoders may "
            "download on first use.", elem_classes="note",
        )

        with gr.Tab("Text · single & batch"):
            with gr.Row():
                mode = gr.Dropdown(MODES, value="question", label="prompt mode")
                max_tokens = gr.Slider(1, MAX_NEW_TOKENS, value=96, step=1, label="max new tokens")
                temperature = gr.Slider(0, 2, value=0.2, step=0.05, label="temperature")
            with gr.Row():
                top_k = gr.Slider(0, 1000, value=40, step=1, label="top k (0 = all)")
                seed = gr.Number(value=0, precision=0, label="seed")
                repetition = gr.Slider(1, 2, value=1, step=0.05, label="repetition penalty")
            generation_inputs = [mode, max_tokens, temperature, top_k, seed, repetition]
            with gr.Tab("Single question"):
                single_question = gr.Textbox(label="question / prompt", lines=3,
                                             placeholder="What is the capital of India?")
                single_reference = gr.Textbox(label="reference answer (optional)", lines=2)
                single_button = gr.Button("Generate & score", variant="primary")
                single_result = gr.JSON(label="prediction and scores")
                single_download = gr.File(label="download result")
            with gr.Tab("JSONL batch"):
                gr.Markdown(
                    "Each row must be `{\"question\": \"...\", \"answer\": \"...\"}`. "
                    f"UTF-8, at most {MAX_BATCH_ROWS} rows. Exact match is normalized; F1 is token-level."
                )
                batch_file = gr.File(label="held-out question/answer JSONL", type="filepath",
                                     file_types=[".jsonl"])
                batch_button = gr.Button("Run batch evaluation", variant="primary")
                batch_summary = gr.JSON(label="aggregate scores")
                batch_table = gr.Dataframe(
                    headers=["row", "question", "reference_answer", "prediction", "exact_match", "f1"],
                    label="per-row results", interactive=False, wrap=True,
                )
                batch_download = gr.File(label="download full JSON results")

            def do_single(src, dev, question, reference, *settings):
                try:
                    result, path = run_text(manager, src, dev, question, reference, *settings)
                    return result, path
                except (Exception, SystemExit) as exc:
                    return {"error": describe_error(exc)}, None

            def do_batch(src, dev, file, *settings):
                try:
                    payload, path = run_text_batch(manager, src, dev, file, *settings)
                    table = [[row[key] for key in
                              ("row", "question", "reference_answer", "prediction", "exact_match", "f1")]
                             for row in payload["results"]]
                    return payload["summary"], table, path
                except (Exception, SystemExit) as exc:
                    return {"error": describe_error(exc)}, [], None

            single_button.click(do_single, [source, device, single_question, single_reference] + generation_inputs,
                                [single_result, single_download], concurrency_limit=1)
            batch_button.click(do_batch, [source, device, batch_file] + generation_inputs,
                               [batch_summary, batch_table, batch_download], concurrency_limit=1)

        with gr.Tab("Speech · ASR + WER"):
            gr.Markdown("Upload a clip from a held-out ASR set. Add its transcript to compute word error rate "
                        "(lower is better); punctuation and case are normalized by `eval_speech.wer`.")
            audio = gr.Audio(label="audio clip", sources=["upload"], type="filepath")
            transcript = gr.Textbox(label="reference transcript (optional)", lines=3)
            asr_button = gr.Button("Transcribe & score", variant="primary")
            asr_result = gr.JSON(label="transcription and WER")
            asr_download = gr.File(label="download result")

            def do_asr(src, dev, clip, reference):
                try:
                    return run_asr(manager, src, dev, clip, reference)
                except (Exception, SystemExit) as exc:
                    return {"error": describe_error(exc)}, None

            asr_button.click(do_asr, [source, device, audio, transcript], [asr_result, asr_download],
                             concurrency_limit=1)

        with gr.Tab("Images · caption & VQA"):
            gr.Markdown("Use held-out images. Leave the question empty only for captioning. With a reference, "
                        "the app reports normalized exact match and token F1; these are simple lexical scores, "
                        "not a complete measure of caption quality.")
            with gr.Row():
                image = gr.Image(label="image", type="filepath", height=360)
                with gr.Column():
                    image_task = gr.Radio((("Caption", "caption"), ("Visual question answering", "vqa")),
                                          value="caption", label="task")
                    image_question = gr.Textbox(label="question (required for VQA)", lines=2)
                    image_reference = gr.Textbox(label="reference caption / answer (optional)", lines=2)
                    image_tokens = gr.Slider(1, MAX_VISION_TOKENS, value=32, step=1, label="max new tokens")
                    image_button = gr.Button("Run image evaluation", variant="primary")
            image_result = gr.JSON(label="prediction and simple scores")
            image_download = gr.File(label="download result")

            def do_image(src, dev, img, task, question, reference, token_count):
                try:
                    return run_vision(manager, src, dev, img, task, question, reference, token_count)
                except (Exception, SystemExit) as exc:
                    return {"error": describe_error(exc)}, None

            image_button.click(do_image,
                               [source, device, image, image_task, image_question, image_reference, image_tokens],
                               [image_result, image_download], concurrency_limit=1)

        def do_load(src, dev):
            try:
                with manager.lock:
                    info = model_metadata(manager.load(src, dev))
                return f"Loaded **{info['ckpt']}** on `{info['device']}`. Ready for inference.", info
            except (Exception, SystemExit) as exc:
                return f"**Load failed:** {describe_error(exc)}", {}

        load_button.click(do_load, [source, device], [load_status, metadata], concurrency_limit=1)
    return demo


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="", help="local .pt, exported folder, or Hub id (loaded on demand)")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--host", default="127.0.0.1", help="default is localhost only")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true", help="explicitly create a public Gradio link")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    build(initial_source=args.ckpt, initial_device=args.device).launch(
        server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
