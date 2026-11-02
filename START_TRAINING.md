# Set up AnuLM on your GPU computer


Completed-run update (2 November 2026): text and joint training, native evaluations and both FP32 export checks are finished. Quality is inadequate for production. See [current status](docs/CURRENT_STATUS.md) and the detailed assessment linked there. This guide describes reproducible setup, not an instruction to restart the completed run.

This bundle contains the same AnuLM codebase, the current local improvements,
book, tests and pinned India-focused training recipe. It starts a **new run**
on the destination computer. It does not migrate the running laptop job.
No SSH connection, GitHub access, password or model account is required for the
public sources in this recipe. Internet access is required for installation
and dataset/encoder downloads. Do not start training inside the ZIP viewer.

## 1. Prepare the computer

- NVIDIA GPU with a driver supporting CUDA 13.0 and the supplied PyTorch build.
  A 16 GB GPU can use the same conservative configuration; this does not
  automatically increase model size or batch size.
- Python **3.14**, 64 bit (the source machine used 3.14.5).
- Git installed and available on PATH. No repository clone or login is needed;
  the runner invokes Git to record available source revision information.
- At least **150 GiB free SSD space** to begin; more is preferable. This is a
  planning minimum, not an exact final size. Downloads retain a 40 GiB reserve.
  CPU audio encoding and corpus preparation also require available system RAM.
- Extract into a local writable folder such as `C:\AnuLM`. Avoid a cloud-sync
  folder for the large training data. Linux example: `~/AnuLM`.
- Keep the computer powered and prevent sleep throughout the run.

Check `nvidia-smi` on that computer first. If CUDA installation fails or the
GPU is unavailable, fix the driver/build compatibility before proceeding;
the script does not silently fall back to CPU training. Do not install a new
driver while another GPU job is running.

## 2. Install on Windows

Open PowerShell in the extracted `AnuLM` folder. Python must resolve to 3.14.
The setup creates a separate `.venv`; it does not change the system Python.

```powershell
python --version
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_training.ps1
```

If Python is installed elsewhere:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_training.ps1 -PythonExecutable 'C:\path\to\Python314\python.exe'
```

The execution-policy option applies to that PowerShell process only. Setup
installs the tested torch/torchvision pair from the official CUDA wheel index,
the pinned application dependencies, and Chromium for Hindi text rendering.
It then checks dependency consistency, GPU arithmetic and browser startup.
If any step fails, it stops. Read the error before rerunning.

## 3. Install on Linux

These commands are provided for Linux; the recorded full setup was tested on
Windows. A supported NVIDIA driver and Python 3.14 must already be installed.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r scripts/requirements-india.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/check_training_machine.py
```

If Chromium reports missing operating-system libraries, have the computer's
administrator install those reported libraries (Playwright provides
`python -m playwright install-deps chromium`). Stop on any installation error.

## 4. Check the code, then launch

Windows correctness checks (Linux: substitute `.venv/bin/python`):

```powershell
.\.venv\Scripts\python.exe -m tests.test_model
.\.venv\Scripts\python.exe -m unittest tests.test_layout tests.test_multimodal_contracts tests.test_validated_demo tests.test_dataset_setup tests.test_india_run tests.test_workflow_improvements tests.test_evaluation_app tests.test_from_scratch_pipeline
```

The setup includes Gradio and Transformers for the bundled verification suites.
Inspect any optional dependency skips; API serving and speech-generation tools
are separate extras. The 66/66 core result recorded on the source computer used
its broader environment; the destination's results must be checked separately.

Read `config/india-training.json` **before** launching. Its original not-before
time is 1 November 2026 at 20:00 IST. If that time has passed, training starts
as soon as downloads, preparation and the real-data preflight pass. To choose a
different time, edit `start_not_before_ist` before launch. The initial training
budget is 40 hours, excluding downloads/preparation. Keep the initial settings
for the first run on the 16 GB GPU; tune only after measuring memory/throughput.

Windows background launch:

```powershell
.\scripts\windows\start_india_training.ps1 -PythonExecutable (Resolve-Path '.\.venv\Scripts\python.exe').Path
Get-Content .\logs\india-v1\status.json
Get-Content .\logs\india-v1\orchestrator.log -Tail 15
```

If local PowerShell policy blocks the launcher, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start_india_training.ps1 -PythonExecutable "$PWD\.venv\Scripts\python.exe"
```

Linux foreground launch (keep the terminal open, or use an existing persistent
terminal session):

```bash
.venv/bin/python -u -m anulm.data.india_corpus fetch
# Run the next command only after fetch exits successfully.
.venv/bin/python -u -m anulm.training.india_run
```

The Windows launcher downloads and waits automatically. Both routes perform
preparation, a training-only tokenizer fit, real multimodal preflight, native
decoder training from random initialization, then joint text/image/ASR training.
Whisper and SigLIP are frozen pretrained encoders. Audio here is **speech input
transcription**, not speech generation. This recipe is not broad chat alignment.

## 5. Monitor and recover

The package also includes the [V2 workflow improvements](docs/V2_IMPROVEMENTS.md):
custom-corpus training, joint accumulation, a local evaluation webapp, declared
per-task quality reports and long-audio inference. These are optional tools;
the India launcher keeps accumulation at one unless `joint_accum` is set before
a new run. Its final status includes a generated validation plan for the reporter.

- `logs/india-v1/status.json`: current stage or failure reason.
- `logs/india-v1/download.log` and `download-error.log`: Windows downloads.
- `logs/india-v1/text_training.log` and `joint_training.log`: training losses.
- `runs/india-v1/`: checkpoints, optimizer recovery files and provenance.
- `data/india-v1/`: downloaded corpora, attribution, manifests and token streams.

On a failed stage, read its log first, fix the cause, then rerun the same launch
command. Finished downloads are verified/reused and interrupted downloads resume
when supported. Checkpoint recovery uses `.last` files. Do not edit runtime code
or configuration during an active run. Do not delete locks or start duplicate
jobs while the recorded owner process is still alive. Do not copy prepared
manifests/checkpoints between machines as a shortcut: several contain absolute
paths and must be regenerated or explicitly migrated.

The ZIP intentionally excludes datasets, model weights, live process records,
logs, caches, environments, credentials and Git metadata. Downloading again is
the supported portable path. If you later want to transfer the large raw files
to save bandwidth, transfer their matching `.source.json` metadata too and
validate them before preparation; this bundle does not perform that migration.

Read [the full training guide](docs/INDIA_TRAINING.md) for the 13 languages,
sources, splitting, filters and commercial-use conditions. Dataset terms permit
use subject to their conditions; this is not unconditional commercial clearance.
Training completion still requires held-out quality evaluation. The default
demo remains empty until results are reviewed and approved.

## Bundle integrity and provenance

`BUNDLE_MANIFEST.json` records every included source file's SHA-256 and the
base commit. The bundle includes local uncommitted changes and is not a newly
published Git commit. Verify the ZIP using its external `.sha256` file; the
included packaging script can also verify every member:

```powershell
.\.venv\Scripts\python.exe scripts/package_training_bundle.py --verify C:\path\to\AnuLM-training-setup.zip
```

No training was run on your destination GPU during packaging. Run the machine
check and real preflight there before relying on it.
