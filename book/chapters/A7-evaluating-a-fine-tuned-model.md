---
title: "Appendix G · Evaluating a fine-tuned model"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix G  Evaluating a fine-tuned model

Chapter 24 makes the validation loss honest: a sample mean with an error bar,
paired between checkpoints, comparable across tokenizers through bits per
byte. That is the right instrument for pre-training. After Chapter 30 it stops
being enough. Instruction tuning on this repository taught the format far more
than facts (Chapter 30), and a loss cannot say whether an answer is right, or
whether the model knew that it did not know. The measure becomes passes out
of n prompts, graded by code where code can grade and by a judge where it
cannot. Counts have their own statistics, and judges their own biases. This
appendix covers both, with the repository's own graders where it has them.

What the repository already grades by counts: `anulm/eval/eval_golden.py`
scores exact match and multiple choice on the golden sets (Chapter 24),
`anulm/eval/eval_qa.py` token F1 on held-out questions (Chapter 33),
`anulm/eval/eval_code.py` test runs on generated programs, and `anulm/tasks/decide.py`
accuracy and calibration error (Chapter 32). None of them prints an interval.
Every number below was printed by `book/code-examples/appG_eval_finetuned.py`
(**Book example, not repository code**), which runs on a CPU in seconds; the
worked counts are made up for teaching and say so.

```bash
python book/code-examples/appG_eval_finetuned.py
```

## G.1  What changes after fine-tuning

Before fine-tuning the question is "how surprised is the model by real text?",
and the answer is a mean over tokens. After it the question is "how often does
it do the job?", and the answer is a count. Three rules carry over from
Chapter 24 unchanged: report the error bar, prefer paired comparisons, and
never report the numbers you tuned on. Three things are new: the interval of a
proportion, the bias of a model used as a judge, and the difference between a
question the context can answer and one it cannot.

## G.2  An interval for a pass rate

**By hand** (*illustrative*). 40 passes out of 50 is 80%. With so few prompts
the true rate could be anywhere from the high sixties to the high eighties.
The textbook interval p ± 1.96·√(p(1 − p)/n) misbehaves near 0% and 100%:
at 0 of 20 it has zero width. The Wilson score interval does not, and it is
one function:

```python
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
```

```
   G.2                   40/50  =  80.0%   [ 67.0%,  88.8%]
   G.2                  400/500 =  80.0%   [ 76.3%,  83.3%]
   Ch 33 dense reader     5/20  =  25.0%   [ 11.2%,  46.9%]
   Ch 33 hybrid reader    2/20  =  10.0%   [  2.8%,  30.1%]
```

*verified*: printed by the example. Ten times the prompts make the interval
about three times narrower, not ten: it shrinks with √n. Decide the size of the
gap you need to see before writing the evaluation set, then size the set to
match.

The last two rows are the book's own numbers. Chapter 33 reports that the
hybrid retriever recalled at least as well as the dense one and *answered
worse*, 2 of 20 against 5 of 20 (*measured*, RESULTS §33). On twenty questions
that is a difference of three, and the two intervals overlap almost entirely.
The chapter's reading may well be right, but the count alone cannot carry it;
the per-question pairs would (G.3).

`NOT IN THE REPO`

## G.3  Two models on the same prompts

This is Chapter 24.5's lesson for counts. When two models answer the same
prompts, compare them prompt by prompt. Each prompt gives a difference of +1
(only the candidate passed), −1 (only the baseline passed) or 0. The repository
already has the paired error for losses:

> **In the repo** · `anulm/eval/eval_context.py` · `paired` · lines 66–78 · @2e80b3e

```python
def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    # ...
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return mean, (var / n) ** 0.5
```

It takes any two matched lists, so it takes two lists of 0s and 1s. The
example also resamples the differences, which makes no normal assumption:

```python
def paired_bootstrap(diffs, reps: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    means = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(1)
    return d.mean(), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
```

**By hand** (*illustrative*). On 60 prompts both models pass 40 and both
fail 8; the candidate alone passes 9 and the baseline alone 3.

```
2. 60 prompts: candidate 81.7% (70.1, 89.4), baseline 71.7% (59.2, 81.5)
   paired gap +10.0 points, bootstrap 95% [+0.0, +21.7] (10,000 resamples, seed 0)
   eval_context.paired(): mean +0.1000, standard error 0.0567; +/- 1.96 se = [-1.1, +21.1] points
```

*verified*: printed by the example, with the repository's `paired` unchanged.
Separately the two Wilson intervals overlap heavily; paired, the gap is +10
points with an interval that just touches zero. "Overlapping intervals" was
never the right test. The paired interval is, and here it says: probably
better, collect more prompts before claiming it.

`IN THE REPO` (`paired`) · `NOT IN THE REPO` (the bootstrap)

## G.4  A judge that likes long answers

When no code can grade an answer (is this explanation good?), a stronger model
is asked which of two answers is better. Model judges prefer the longer
answer. If your fine-tune writes shorter answers than the baseline, a raw win
rate understates it; if longer, it flatters it.

The fix is to model the length effect and read off the win rate at equal
length. For each pair let x = ln(candidate length / baseline length) and fit
win ~ σ(a + b·x). σ(a) is the length-controlled win rate and b the judge's
taste for length. The example fits it with Newton's method in four lines and
checks the fit on data whose truth it knows:

```
3. 600 simulated pairs, true equal-length win rate 70%, true length taste b = 1.5; candidate shorter on 91% of pairs
   raw win rate 42.7%; length-controlled sigmoid(a) = 65.4% [59.5%, 73.0%]; fitted b = 1.19
```

*illustrative*: a simulation, seed 0; the arithmetic is *verified* by the
example. The raw number points the wrong way; the controlled one recovers most
of the truth, with an interval from 300 refits on resampled pairs. Whether the
judge you use has this taste, and how much, is *not measured here*: fit b on
your own pairs and report it beside the win rate.

`NOT IN THE REPO`

## G.5  A split that never moves

Chapter 21 holds out by document hash, so a document never drifts between
training and validation. Evaluation sets need the same discipline, plus one
more rule: an item you have looked at to fix a failure can no longer tell you
whether the fix generalises. Split every set once, by a hash of the item id,
into a dev half you may read and a sealed half you only ever score.

```python
def split_of(item_id: str, sealed_share: int = 50) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16)
    return "sealed" if h % 100 < sealed_share else "dev"
```

```
4. split_of over 1,000 ids: sealed 473 (47.3%), identical on a second pass: True; item-0000 is 'sealed'
```

*verified*: printed by the example. The share is near a half, not exactly a
half, because the hash decides item by item; the same id lands on the same
side on every machine and every run, which a random split re-drawn each run
does not give. Release decisions are taken on the sealed half; the dev half is
for finding out why.

`NOT IN THE REPO`

## G.6  Questions the context can answer, and questions it cannot

Chapter 33's readers answer from retrieved passages, and `anulm/eval/eval_qa.py`
grades them by token overlap:

> **In the repo** · `anulm/eval/eval_qa.py` · `f1` · lines 70–80 · @2e80b3e

```python
def f1(pred: str, gold: str) -> float:
    # ...
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
```

One pooled F1 mixes two different skills. Tag every item by whether the
passages it is given contain the answer:

| Item type | Correct behaviour | Grade as |
|---|---|---|
| answer in the context | find it and use it | correct / wrong |
| answer not in the context | say the passages do not settle it; at most give general knowledge, marked as such | honest / fabricated |

The two rates move in opposite directions under training. Teaching a model to
always produce an answer raises the first and quietly raises fabrication on
the second; teaching it to abstain does the reverse and adds over-refusal.
Only the split shows which trade you made. The repository's QA items carry no
such tag, so the second rate is *not measured here*; adding the tag to the
items is the whole change, since `f1` already grades the first.

`IN THE REPO` (`f1`) · `NOT IN THE REPO` (the tag)

## G.7  Export fidelity

Chapter 39 shows that storing the weights in bfloat16 moved the held-out loss
by −0.00004 nats, while rounding the router biases changed about half the
expert choices for 0.003 nats (*measured*). Loss is too blunt to notice a
changed decision. A sharper test of an exported or quantised model is the KL
divergence between the source's and the export's next-token distributions,
position by position, together with the share of positions whose top token
changed.

**By hand** (*illustrative*). Source (0.70, 0.20, 0.10), export (0.66, 0.23,
0.11): KL = Σ p·ln(p/q) = 0.0037 nats, small, and the top token is unchanged.
The example then applies the instrument to `tiny()` against a copy of itself
whose every weight was rounded through bfloat16, the way the export stores
them, with the logits computed in float32 at every position:

```
5. KL by hand: sum p ln(p/q) = 0.0037 nats; top token unchanged: True
   tiny() against its bfloat16-rounded copy over 256 positions: mean KL 2.83e-08 nats, 99th percentile 4.05e-07, top token agrees at 100.0%
```

*verified*: printed by the example. Report the mean KL, its 99th percentile and
the top-token agreement, and pick the thresholds before you measure, as
Chapter 24 asks of every comparison. The same three numbers for the released
model's export are *not measured here*; `book/code-examples/ch39_export_roundtrip.py --measure`
has the two models loaded side by side and is the place to add them.

`NOT IN THE REPO`

## G.8  Common mistakes, and how to catch them

| Mistake | Catch it |
|---|---|
| a pass rate without an interval | `wilson` on every rate; size the set from the gap you need to see (G.2) |
| two models compared by their separate intervals | per-item differences, `paired` or the bootstrap (G.3) |
| a raw judge win rate when answer lengths differ | fit the length term; report σ(a) and b (G.4) |
| fixing failures found on the set you report | hash split once; read dev, score sealed (G.5) |
| one accuracy for a retrieval model | tag items answerable or not; report correct and fabricated separately (G.6) |
| an export judged by loss alone | mean and tail KL plus top-token agreement against the source (G.7) |
| the judge, its prompt or the decoding settings changed mid-project | freeze them with the prompts; any change starts a new baseline |

## G.9  Exercises

1. A fine-tune passes 45 of 60 safety prompts and the base model 39. Without
   the per-item results, what can you say? What do you need to say more?
2. A judge prefers your model 60% of the time, and your model's answers are on
   average twice as long. Is it better?
3. Why is the sealed half chosen by a hash of the id rather than at random on
   each run?

**Answers.** 1. Separately, 75.0% (62.8–84.2%) against 65.0% (52.4–75.8%):
the intervals overlap, so nothing firm. You need the per-item pairs for a
paired interval (G.3). 2. Not necessarily: with a taste for length, an
equal-length win rate below 50% is consistent with 60% raw. Fit G.4's model.
3. So that an item is sealed for ever. A random split re-drawn each run lets
every item be seen in dev eventually, and the sealed score slowly turns into a
tuned one. *verified*: the two intervals of answer 1 are printed by the example.

## Key points

- After fine-tuning the measure is passes out of n. Report a Wilson interval,
  and compare models item by item with a paired interval; the repository's
  `paired` already does the paired part.
- Model judges like length; fit it out and report the equal-length win rate
  with the fitted taste.
- Split once by hash: dev to read, sealed to decide.
- Split retrieval items by whether the context holds the answer; the correct
  and fabricated rates move against each other.
- Judge an export by KL and top-token agreement, not by loss alone.
