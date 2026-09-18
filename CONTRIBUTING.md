# Contributing to AnuLM

Thank you for looking. This is a one-GPU academic project, and the most
useful contributions are the ones that let more people reproduce it, run
it on more machines, or extend it in a measured way. Open items are in
[TODO.md](TODO.md) and as GitHub issues; the ones labelled
`good first issue` need no GPU.

## Set up

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM.git && cd AnuLM
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or the CUDA wheel for your driver
pip install -r requirements.txt
python quickstart.py          # confirms the install; prints the fix for anything missing
python test_model.py          # 50 tests, ~2 min, must pass before and after your change
```

CI runs the tests plus a 50-step training run on Ubuntu and Windows for
every push and pull request, so a change that only works on one of them
will show up before review.

`docs/DEVELOPING.md` explains the layout; `docs/TUTORIAL.md` runs the
whole pipeline. Everything is plain PyTorch with no framework, so a change
usually touches one file.

## What a good change looks like

- **It is measured.** If it touches the model, training or data, report a
  number the way `docs/RESULTS.md` does: the script, the seed, the val
  loss or benchmark before and after, and how long it took on what
  hardware. Val loss carries about ±0.03 of noise at the default
  `--eval-iters`; say so if your difference is inside it.
- **It stays reproducible.** No data checked in; add a fetcher or a
  recipe under `experiments/` instead and a row to `docs/DATASETS.md`
  with the licence. No new dependency for the core paths (model,
  training, evaluation, serving) without a strong reason.
- **It keeps old checkpoints loading.** The config dataclass is pickled
  into every `.pt`; add fields with defaults, never rename or remove one
  without an alias.
- **It runs on Linux and Windows.** Shell scripts are bash and stay LF;
  Windows-only helpers are `.cmd` and stay CRLF (`.gitattributes` does
  this for you).
- **Tests.** Add or extend a test in `test_model.py` for anything
  numerical. The suite has no pytest dependency on purpose.

## Pull requests

One topic per PR. Say what you measured in the description. If a PR
changes a number that appears in `README.md`, a model card or
`docs/RESULTS.md`, update those in the same PR.

## Data and licences

Read `NOTICE` §2 before adding a data source. Contributions that add
non-commercial or share-alike data must say so in `docs/DATASETS.md` and
in any model card of a checkpoint trained on it.

## Reporting problems

Open an issue — the form asks for exactly what is needed, and `python
quickstart.py` prints the environment block it wants in one go. For a reproduction that lands
outside the expected numbers in `docs/TUTORIAL.md` §13, include the
`.last` checkpoint's step and the val-loss curve.

## Conduct

Be kind and specific. Credit sources. Do not paste other people's data or
weights into this repository.
