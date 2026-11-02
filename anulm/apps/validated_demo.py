"""Evidence-backed demo catalogue. It displays recorded results, not new inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anulm.paths import PROJECT_ROOT

TASKS = {"text", "image", "audio"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_file(root, record):
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Demo artifacts must remain inside the checkout.")
    if not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError("Missing or changed validation artifact.")
    return path


def load_cases(manifest, root=PROJECT_ROOT):
    """Fail closed for absent/unapproved/stale evidence; return cases and reasons.

    Hashes establish consistency, not trust or independent validation. Reports
    must be approved by a project reviewer before being added to the catalogue.
    """
    root, manifest = Path(root), Path(manifest)
    if not manifest.exists():
        return [], ["No validation catalogue is installed."]
    try:
        config = json.loads(manifest.read_text(encoding="utf-8"))
        if config["version"] != 1 or not isinstance(config["cases"], list):
            raise ValueError("Unsupported validation catalogue.")
    except (OSError, ValueError, KeyError, TypeError):
        return [], ["The validation catalogue is invalid."]
    approved, excluded, seen = [], [], set()
    for entry in config["cases"]:
        try:
            case_id = entry["id"]
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ValueError("Case IDs must be unique nonempty strings.")
            seen.add(case_id)
            report = json.loads(checked_file(root, entry["report"]).read_text(encoding="utf-8"))
            if (report["case_id"] != case_id or report["status"] != "passed"
                    or report["approved"] is not True or report["task"] not in TASKS):
                raise ValueError("Case has no approved passing report.")
            for field in ("reviewed_by", "validated_at", "criteria", "output", "source_revision"):
                if not isinstance(report[field], str) or not report[field].strip():
                    raise ValueError("Validation report is incomplete.")
            data = report["dataset"]
            if data["held_out"] is not True or not data["source"] or not data["license"]:
                raise ValueError("Dataset provenance and held-out status are required.")
            assets = report["artifacts"]
            roles = {asset["role"] for asset in assets}
            required = {"checkpoint", "tokenizer"}
            if report["task"] != "text":
                required.add("encoder")
            if not required <= roles:
                raise ValueError("Checkpoint, tokenizer and modality encoder provenance are required.")
            paths = [checked_file(root, asset) for asset in assets]
            media = None
            if report["task"] != "text":
                media = checked_file(root, report["input"])
            elif not isinstance(report["input"], str) or not report["input"].strip():
                raise ValueError("Text input is empty.")
            approved.append({"id": case_id, "report": report, "media": media,
                             "artifacts": paths})
        except (OSError, ValueError, KeyError, TypeError):
            excluded.append("An entry was excluded because its evidence is incomplete or changed.")
    return approved, excluded


def build(manifest, root=PROJECT_ROOT):
    import gradio as gr
    cases, excluded = load_cases(manifest, root)
    with gr.Blocks(title="AnuLM validated examples") as demo:
        gr.Markdown("# AnuLM validated examples\n"
                    "Recorded results for approved test inputs. This page does not run new inference. "
                    "Image and audio paths use pretrained encoders; each result records its model and data provenance.")
        if not cases:
            gr.Markdown("No examples have passed the evidence checks in this checkout. "
                        "Real image/audio evaluation and reviewer approval are still required.")
        for case in cases:
            report = case["report"]
            with gr.Tab(case["id"]):
                if report["task"] == "image":
                    gr.Image(value=str(case["media"]), interactive=False, label="Test image")
                elif report["task"] == "audio":
                    gr.Audio(value=str(case["media"]), interactive=False, label="Test recording")
                else:
                    gr.Textbox(value=report["input"], interactive=False, label="Test input")
                gr.Textbox(value=report.get("question", ""), visible=bool(report.get("question")),
                           interactive=False, label="Test question")
                gr.Textbox(value=report["output"], interactive=False, label="Recorded validated output")
                gr.Textbox(value=report["criteria"], interactive=False, label="Acceptance criteria")
                gr.JSON(value={k: report[k] for k in ("dataset", "reviewed_by", "validated_at", "source_revision")},
                        label="Validation record")
        if excluded:
            gr.Markdown(f"{len(excluded)} catalogue entries were excluded by the evidence checks.")
    return demo
