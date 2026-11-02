"""
Chapter 15 checks: shipping.

    python ch15.py                  # your course/build/post.py
    python ch15.py --impl solution  # the reference solution
    python ch15.py --self-test      # proves each check catches the bug it is for

The last bugs are the ones your users find: a release that loses its
projector, router biases rounded to bfloat16, a model card that says "pass"
for a measurement nobody made.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, main, solution, with_replaced  # noqa: E402
from anulm.hf import export_hf  # noqa: E402

spec = importlib.util.spec_from_file_location("omni_one", ROOT / "experiments" / "omni_one.py")
omni_one = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omni_one)

CHECKS = []
B = solution(8)
KW = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)
OMNI = {"chat_heldout_loss": 5.3181, "read_squad_en": 24.6, "intent_banking77": 90.0, "wer_en": 12.2,
        "wer_hi_fleurs": 48.8, "wer_hi_kathbath": 35.5, "wer_hi_mucs": 54.4, "vqa_acc": 17.6}   # logs/omni_one/REPORT.md


def check(fn):
    CHECKS.append(fn)
    return fn


def sample_state():
    torch.manual_seed(0)
    return {"big": torch.randn(400, 300), "edge": torch.randn(100_000), "small": torch.randn(99_999),
            "layers.1.mlp.router.expert_bias": torch.randn(8) * 1e-3,
            "layers.1.mlp.router.load_counts": torch.arange(8), "norm.weight": torch.ones(128)}


@check
def precision_matches_export_hf(M):
    got, want = M.to_bf16(sample_state()), export_hf.to_bf16(sample_state())
    for k in want:
        assert got[k].dtype == want[k].dtype, f"{k}: {got[k].dtype}, export_hf keeps {want[k].dtype}"
        assert torch.equal(got[k], want[k]), f"{k}: values differ from export_hf.to_bf16"


@check
def extras_are_found_like_export_hf(M):
    ck = {"model": {"w": torch.ones(2)}, "cfg": {}, "step": 3, "qa_templates": {"en": "Q: {q}"},
          "proj": {"net.0.weight": torch.ones(2, 2)}, "decide_head": {"weight": torch.ones(1, 4)},
          "lora": {"a.lora_a": torch.ones(2, 2)}, "labels": ["a", "b"], "empty": {}}
    got, want = M.tensor_groups(ck), export_hf.tensor_groups(ck)
    assert sorted(got) == sorted(want), f"groups {sorted(got)}; export_hf finds {sorted(want)}"


def checkpoint():
    torch.manual_seed(0)
    m = B.MoELM(**KW).eval()
    for n, b in m.named_buffers():
        if n.endswith("expert_bias"):
            b.copy_(torch.randn_like(b) * 1e-3)                   # what the balancer leaves behind
    torch.manual_seed(1)
    ck = {"model": m.state_dict(), "cfg": KW, "step": 1500, "val_loss": 1.504,
          "proj": {"net.0.weight": torch.randn(256, 512), "net.0.bias": torch.randn(256)}}
    return m, ck


@check
def a_release_keeps_every_part(M):
    """Model, projector and settings all come back; what was stored in bfloat16 comes back as float32
    within bfloat16's precision; small tensors and integers come back exactly."""
    m, ck = checkpoint()
    tmp = Path(tempfile.mkdtemp())
    try:
        M.export_folder(ck, tmp / "release")
        back = M.load_folder(tmp / "release")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert back["cfg"] == KW and back["step"] == 1500 and back["val_loss"] == 1.504, "settings lost on the way"
    assert "proj" in back and sorted(back["proj"]) == sorted(ck["proj"]), "the projector did not come back"
    w = back["proj"]["net.0.weight"]
    assert w.dtype == torch.float32, f"loaded as {w.dtype}; widen it back to float32"
    torch.testing.assert_close(w, ck["proj"]["net.0.weight"], rtol=1e-2, atol=1e-2, msg="projector weights changed")
    for k, v in ck["model"].items():
        assert torch.equal(back["model"][k].to(v.dtype), v), f"{k} changed; a small tensor must come back exactly"


@check
def the_released_model_answers_the_same(M):
    m, ck = checkpoint()
    tmp = Path(tempfile.mkdtemp())
    try:
        M.export_folder(ck, tmp / "release")
        back = M.load_folder(tmp / "release")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    m2 = B.MoELM(**back["cfg"]).eval()
    m2.load_state_dict(back["model"])
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        torch.testing.assert_close(m2(x)[0], m(x)[0], msg="the released model's logits differ")


@check
def the_report_says_no_when_a_gate_fails(M):
    """The first AnuLM-Omni run, as measured: four gates failed, so the report must say so."""
    res = dict(OMNI)
    g = omni_one.gates(res)                                         # adds wer_hi_mean to res
    text, ok = M.release_report(res, omni_one.GATES, g)
    assert not ok and "NOT all pass" in text, "a failed release was reported as passing"
    row = next(l for l in text.splitlines() if l.startswith("| chat_heldout_loss "))
    assert "**no**" in row and "5.3181" in row, f"chat's row does not show the failure: {row}"
    missing = {k: v for k, v in OMNI.items() if k != "vqa_acc"}
    g = omni_one.gates(missing)
    text, ok = M.release_report(missing, omni_one.GATES, g)
    row = next(l for l in text.splitlines() if l.startswith("| wer_hi_mean "))
    assert "46.2" in row and "yes" in row, f"the Hindi mean's row: {row}"
    row = next(l for l in text.splitlines() if l.startswith("| vqa_acc "))
    assert "**no**" in row and "| - |" in row, f"a measurement nobody made shows as {row}"
    good = {k: (0.0 if op == "<=" else 100.0) for k, (op, _) in omni_one.GATES.items()}
    text, ok = M.release_report(good, omni_one.GATES, omni_one.gates(good))
    assert ok and "all pass" in text and "**no**" not in text, "a release that passes every gate was refused"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("to_bf16", "tensor_groups", "export_folder", "load_folder", "release_report", "CORE_KEYS")


def MUTANTS(sol):
    def all_floats(sd):
        return {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach()) for k, v in sd.items()}

    def everything(sd):
        return {k: v.detach().to(torch.bfloat16) for k, v in sd.items()}

    def strictly_more(sd):
        return {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() and v.numel() > 100_000 else v.detach())
                for k, v in sd.items()}

    def only_projectors(ck):
        return {k: v for k, v in ck.items() if k in ("proj", "vision_proj")}

    def any_dict(ck):
        return {k: v for k, v in ck.items() if k not in sol.CORE_KEYS and isinstance(v, dict)}

    def no_extras(ck, folder):
        return sol.export_folder({k: v for k, v in ck.items() if k in sol.CORE_KEYS}, folder)

    def bf16_model_all(ck, folder):
        folder = sol.export_folder(ck, folder)
        from safetensors.torch import save_file
        save_file(all_floats(ck["model"]), str(Path(folder) / "model.safetensors"))
        return folder

    def not_widened(folder):
        import json
        from safetensors.torch import load_file
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        ck = {"model": load_file(str(folder / "model.safetensors")), "cfg": config["cfg"], "step": config["step"],
              "val_loss": config["val_loss"]}
        if config["extras"]:
            flat = load_file(str(folder / "extras.safetensors"))
            for g in config["extras"]:
                ck[g] = {k[len(g) + 1:]: v for k, v in flat.items() if k.startswith(g + ".")}
        return ck

    def missing_passes(results, rules, gate_results):
        lines = ["| measure | value | gate | pass |", "|---|---|---|---|"]
        for k, (op, thr) in rules.items():
            ok = gate_results.get(k) or k not in results
            lines.append(f"| {k} | {results.get(k, '-')} | {op} {thr} | {'yes' if ok else '**no**'} |")
        good = all(gate_results.get(k) or k not in results for k in rules)
        lines += ["", f"Release gates: **{'all pass' if good else 'NOT all pass -- not published'}**"]
        return "\n".join(lines) + "\n", good

    def always_publishes(results, rules, gate_results):
        text, _ = sol.release_report(results, rules, gate_results)
        return text.replace("NOT all pass -- not published", "all pass"), True

    bugs = [("every float tensor in bfloat16, router biases too", dict(to_bf16=all_floats)),
            ("integer buffers converted as well", dict(to_bf16=everything)),
            ("a tensor of exactly 100,000 kept in float32", dict(to_bf16=strictly_more)),
            ("only projectors treated as extras", dict(tensor_groups=only_projectors)),
            ("any dict treated as tensors", dict(tensor_groups=any_dict)),
            ("extras left out of the release", dict(export_folder=no_extras)),
            ("the backbone rounded whole", dict(export_folder=bf16_model_all)),
            ("bfloat16 not widened on loading", dict(load_folder=not_widened)),
            ("a missing measurement passes", dict(release_report=missing_passes)),
            ("the summary always says publish", dict(release_report=always_publishes))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(15, CHECKS, MUTANTS, __doc__, learner="post.py")
