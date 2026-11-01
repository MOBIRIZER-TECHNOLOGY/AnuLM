# Register proposals from Part IV, second half (Chapters 16–19)

None of these is cited in a chapter. Each claim is written in the chapter
without a citation until the integrator decides.

## 1. (Withdrawn after review) Extend R01's claims: the position-wise FFN

The sentence it supported, "the transformer's original FFN was two matrices
with a ReLU, four times as wide", was removed from Ch 16 in the review fixes.
No proposal is needed now. Kept here for the record: R01 §3.3 describes
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ with d_ff = 2048 = 4 × 512.

## 2. Extend R19's claims: why the rotary key is decoupled, and absorption

- **Source:** DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model*, 2024, arXiv:2405.04434 (already R19, CHECKED).
- **Claims to add:**
  (a) at inference the key up-projection W^UK can be absorbed into the query
  projection, and the value up-projection W^UV into the output projection, so
  keys and values need not be computed from the latent;
  (b) RoPE is incompatible with that absorption, because a position-dependent
  rotation would sit between the query and W^UK; recomputing the keys of all
  prefix tokens would then be required, which is why DeepSeek-V2 decouples RoPE
  into additional query slices and a shared key;
  (c) DeepSeek-V2 also applies low-rank compression to the queries, to reduce
  activation memory in training.
- **Used in:** Ch 19, 19.1 (the reason for the decoupled key, stated "without a
  register source"), 19.5 (absorption of both halves, badged `NOT IN THE REPO`).
  19.2 states only the repository fact that `q_proj` is uncompressed.
- **How verified:** from my knowledge of the paper's §2.1.2–2.1.3; not
  re-opened in this session. The chapter's numeric check of the key-side
  algebra (`ch19_mla.py` `absorb`) does not depend on the paper.
