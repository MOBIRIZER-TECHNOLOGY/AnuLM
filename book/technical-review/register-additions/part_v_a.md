# Register proposals: Part V, first half (Chapters 21–23)

## Proposed: bfloat16 for training (Chapter 23)

| field | value |
|---|---|
| title | A Study of BFLOAT16 for Deep Learning Training |
| first author | Dhiraj Kalamkar |
| year | 2019 (arXiv v1, 29 May 2019) |
| identifier | arXiv:1905.12322 |
| source | arxiv.org/abs/1905.12322 |
| claim to support | BFLOAT16 represents the same range of values as FP32, and keeping that range is why no hyperparameter tuning is needed for convergence, unlike FP16 |
| how verified | the arXiv abstract page, fetched 2026-11-01: title, first author and v1 date as above; the abstract states the range claim ("the same as that of IEEE 754 floating-point format (FP32)") and the tuning claim. The abstract does **not** mention loss scaling, so it must not be cited for "bf16 needs no loss scaler" |

Chapter 23 does not cite it. It states the range from `torch.finfo` (*verified*) and that the
repository has no loss scaler (a repository fact). If the integrator accepts the entry, 23.1
could cite it for the range sentence only.

## Outcome (register worker, 2026-11-03)

**Verified** (arXiv API abstract) and added as [R57], for the range sentence only.
