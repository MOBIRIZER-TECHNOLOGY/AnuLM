"""Run an AnuLM program: python -m anulm COMMAND [options].

    python -m anulm train --preset 30b --steps 50
    python -m anulm sample --ckpt ckpt.pt --prompt "KING RICHARD II:"

Each command is a module of the package, and `python -m anulm.training.train`
is the same program. `python -m anulm` with no command lists them all.
Only modules with an `if __name__ == "__main__":` block are commands; the
library modules (omni, muon, hf_tok, ...) are imported, not run.
tests/test_quality_layout.py keeps this table and the package in step.
"""

import runpy
import sys

COMMANDS = {
    "bidirectional": "anulm.bidirectional",
    "bpe": "anulm.bpe",
    "lora": "anulm.lora",
    "model": "anulm.model",
    "quickstart": "anulm.quickstart",
    "app": "anulm.apps.app",
    "app_omni": "anulm.apps.app_omni",
    "app_vision": "anulm.apps.app_vision",
    "app_voice": "anulm.apps.app_voice",
    "validated_demo": "anulm.apps.validated_demo",
    "convert_parquet": "anulm.data.convert_parquet",
    "convert_translate": "anulm.data.convert_translate",
    "fetch_code": "anulm.data.fetch_code",
    "fetch_hf": "anulm.data.fetch_hf",
    "fetch_hindi": "anulm.data.fetch_hindi",
    "fetch_speech": "anulm.data.fetch_speech",
    "fetch_web": "anulm.data.fetch_web",
    "filter_prose": "anulm.data.filter_prose",
    "make_bench": "anulm.data.make_bench",
    "make_chat": "anulm.data.make_chat",
    "make_golden": "anulm.data.make_golden",
    "make_qa": "anulm.data.make_qa",
    "make_qa_en": "anulm.data.make_qa_en",
    "make_qa_py": "anulm.data.make_qa_py",
    "make_rc": "anulm.data.make_rc",
    "mix_corpus": "anulm.data.mix_corpus",
    "eval_bench": "anulm.eval.eval_bench",
    "eval_code": "anulm.eval.eval_code",
    "eval_context": "anulm.eval.eval_context",
    "eval_golden": "anulm.eval.eval_golden",
    "eval_qa": "anulm.eval.eval_qa",
    "eval_speech": "anulm.eval.eval_speech",
    "eval_translate": "anulm.eval.eval_translate",
    "export_hf": "anulm.hf.export_hf",
    "hf_tokenizer": "anulm.hf.hf_tokenizer",
    "hf_upgrade": "anulm.hf.hf_upgrade",
    "api": "anulm.inference.api",
    "ask": "anulm.inference.ask",
    "rag": "anulm.inference.rag",
    "sample": "anulm.inference.sample",
    "sample_many": "anulm.inference.sample_many",
    "serve": "anulm.inference.serve",
    "voice": "anulm.inference.voice",
    "audio_codec": "anulm.multimodal.audio_codec",
    "omni_data": "anulm.multimodal.omni_data",
    "omni_eval": "anulm.multimodal.omni_eval",
    "omni_images": "anulm.multimodal.omni_images",
    "omni_train": "anulm.multimodal.omni_train",
    "speech_data": "anulm.multimodal.speech_data",
    "speech_encoder": "anulm.multimodal.speech_encoder",
    "speech_vocab": "anulm.multimodal.speech_vocab",
    "vision_encoder": "anulm.multimodal.vision_encoder",
    "decide": "anulm.tasks.decide",
    "rc_pointer": "anulm.tasks.rc_pointer",
    "finetune": "anulm.training.finetune",
    "train": "anulm.training.train",
    "train_from_scratch": "anulm.training.train_from_scratch",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        width = max(map(len, COMMANDS))
        for name, module in COMMANDS.items():
            print(f"  {name:<{width}}  {module}")
        return 0
    name = sys.argv[1]
    if name not in COMMANDS:
        print(f"unknown command: {name} (python -m anulm lists them)", file=sys.stderr)
        return 2
    sys.argv = [f"python -m anulm {name}"] + sys.argv[2:]
    runpy.run_module(COMMANDS[name], run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
