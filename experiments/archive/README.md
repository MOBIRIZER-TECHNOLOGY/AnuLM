# Archived one-off runs

Machine-operations scripts written for one unattended run on the project's
training machine: they pause and restart scheduled tasks, wait on specific log
lines and checkpoints, and assume the data of the day. They are kept so the
numbers in `docs/RESULTS.md` stay traceable to the code that made them, not as
tools to rerun. Each still resolves the repository root from its own path
(`parents[2]`), so `python experiments/archive/NAME.py` runs from anywhere.

- `demo_branch.py` (2026-09-30): branched base v2 at step 700k, decayed a copy to 800k, then tuned chat and the retrieval reader on it, while the main run waited (RESULTS.md §35).
- `decide_branch.py` (2026-10-01): trained AnuLM-Decide on BANKING77 in the GPU window after the demo branch, then resumed pretraining.
- `decide_general_branch.py` (2026-10-01): trained the general (zero-shot) Decide in a pause right after a base-v2 save and scored it on BANKING77.
- `post_v2.py` (2026-10, after base v2 finished): post-training on the final base v2: chat v3, the lookup reader, the LoRA recipe comparison A-H, the Hindi adapter and Decide general (RESULTS.md §39).
- `prep_indic_extra.py` (2026-10-18): turned the second batch of Indian-language data (Hinglish, indic-instruct, indic-align) into `data/extra_hi/*.jsonl`.
- `chat_v5.py` (2026-10-18): chat v5, the human-written chat data with a small share of clean Indian data, then recipe-F Decide adapters on it.
- `overnight_text.py` (2026-11-03): launched the text-only overnight from-scratch pipeline (`train_from_scratch`, 350m preset) with bounded checkpoint resume.

Still in `experiments/` although they are one-offs too, because package code
cites them by path: `omni_v3.py` and `run_omni_v3.cmd` (cited in
`anulm/apps/app_omni.py`) and `indic_v4.py` (cited in
`anulm/multimodal/omni_data.py`).
