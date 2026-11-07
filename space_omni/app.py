"""AnuLM-Omni-400M on a Hugging Face Space (ZeroGPU, or any CPU/GPU hardware).

Built into a Space folder by tools/build_space_omni.py, next to a copy of the
`anulm` package and the demo's sample library. The same page as
`python -m anulm app_omni`; three things differ:

- every handler that runs the model is wrapped in `spaces.GPU`, so on ZeroGPU a
  GPU is attached for that call only (a no-op anywhere else);
- every model is loaded once at start-up, before the first visitor, because
  ZeroGPU wants weights placed on CUDA at module level;
- Whisper's encoder stays on the CPU (`ANULM_WHISPER_DEVICE=cpu`): it runs
  through ctranslate2, which needs a real GPU when it loads, and ZeroGPU only
  has one inside a GPU call. It is the frozen part; the decoder still uses the GPU.
"""
import os

os.environ.setdefault("ANULM_WHISPER_DEVICE", "cpu")
os.environ.setdefault("ANULM_OMNI", "toonist/AnuLM-Omni-400M")
os.environ.setdefault("ANULM_OMNI_RESULTS", os.path.join(os.path.dirname(__file__), "results.json"))

try:
    import spaces                          # present on Hugging Face Spaces

    gpu = spaces.GPU(duration=int(os.environ.get("ANULM_GPU_SECONDS", "60")))
except ImportError:                        # local run: no wrapper
    def gpu(f):
        return f

import torch                               # noqa: E402  (after spaces, as ZeroGPU requires)

from anulm.apps import app_omni as A        # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
models = A.Models(device, A.BRAIN)

# Load everything now. The Wikipedia lookup needs a 1.3 GB index that is not in
# the Space, so `reader()` returns None and the Chat tab answers from the model alone.
om = models.omni()
om.ear()
om.eye()
for task in A.OMNI_TASKS:
    models.decider(A.DECIDE_OWN, task)
print(f"AnuLM-Omni loaded on {device}; Whisper encoder on {os.environ['ANULM_WHISPER_DEVICE']}", flush=True)

demo = A.build(models, gpu=gpu)

if __name__ == "__main__":
    demo.launch(theme=A.THEME, css=A.CSS)
