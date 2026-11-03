# Register proposals: Part I and Part II (Chapters 1–7)

Proposed by the Part I–II writer, 2026-11-01. None of these is cited in a
chapter: each claim below is written in the chapter without a citation, and
the integrator decides whether to add the source. "Checked" here means the
writer opened the source named in "How verified"; it is not the register's
CHECKED status.

| Proposed ID | Title | Authors | Year | Identifier | Exact claim it would support | Chapter | How verified |
|---|---|---|---|---|---|---|---|
| (new) | Finding Structure in Time | Jeffrey L. Elman | 1990 | doi:10.1207/s15516709cog1402_1; Cognitive Science 14(2):179–211 | a recurrent network carries a hidden state from one time step to the next (the simple recurrent network) | 2 (2.4) | Crossref metadata (title, author, journal, volume, issue, pages, year) fetched 2026-11-01. The claim itself was **not** checked against the paper's text |
| (new) | Long Short-Term Memory | Sepp Hochreiter, Jürgen Schmidhuber | 1997 | doi:10.1162/neco.1997.9.8.1735; Neural Computation 9(8):1735–1780 | a gated recurrent cell designed to carry information over long time lags | 2 (2.4), optional | Crossref metadata fetched 2026-11-01; claim not checked against the text |
| S01 (claim widening) | BERT | Devlin | 2018 | arXiv:1810.04805 | widen the claims cell to: "pre-trains deep bidirectional representations by jointly conditioning on both left and right context in all layers" (an encoder-only model) | 2 (2.5) | arXiv abstract read 2026-11-01; the sentence is in the abstract |
| (new) | SGDR: Stochastic Gradient Descent with Warm Restarts | Ilya Loshchilov, Frank Hutter | 2016 | arXiv:1608.03983 | the cosine-annealed learning rate schedule (with warm restarts) | 7 (7.3) | arXiv abstract page read 2026-11-01: title, first author, v1 date 13 Aug 2016. The abstract says "warm restart technique" but does not say "cosine"; the cosine form is in the paper body, which the writer did **not** open. Check before use |
| (new) | On the difficulty of training Recurrent Neural Networks | Razvan Pascanu | 2012 | arXiv:1211.5063 | gradient norm clipping as a remedy for exploding gradients | 7 (7.4) | arXiv abstract read 2026-11-01: proposes "a gradient norm clipping strategy to deal with exploding gradients" |

Not proposed: a source for learning-rate warmup. The chapter states warmup's
purpose as reasoning about Adam's early moment estimates and cites nothing;
[R10] (pre-norm without warmup) is the nearest checked source and is not used
for that claim.

## Outcome (register worker, 2026-11-03)

- Elman 1990: **verified**, added as [S14]. Claim from the abstract (Crossref publisher deposit): hidden unit
  patterns fed back to themselves give the network a dynamic memory. The abstract credits the idea to Jordan (1986).
- Hochreiter & Schmidhuber 1997: **verified**, added as [S15] (abstract via Crossref).
- S01 widening: **verified** and added, without "(an encoder-only model)", which the abstract does not say.
- SGDR: **verified**, added as [R55]; the cosine annealing is eq. 5 in the full text (checked in the arXiv PDF).
- Pascanu et al. 2012: **verified**, added as [R56] (abstract).
