# Register proposals: Part III (Chapters 8–10)

None of these is cited in a chapter. Each chapter states the claim without a
citation, as `WRITER_BRIEF.md` requires, and names this file. The integrator
decides whether to add them to `RESEARCH_REFERENCES.md`.

Metadata below is from the writer's knowledge and was **not fetched or checked
at a primary source in this session**. Each needs Stage 3 verification before
it can be CHECKED.

| Proposed for | Title | Author(s) | Year | Identifier | Exact claim it would support | How the claim was checked so far |
|---|---|---|---|---|---|---|
| Ch 8, 8.1 | UTF-8, a transformation format of ISO 10646 (RFC 3629) | F. Yergeau | 2003 | doi:10.17487/RFC3629 | UTF-8 encodes each Unicode code point in one to four bytes; the lead byte's high bits (`0`, `110`, `1110`, `11110`) give the length, and every following byte has the form `10xxxxxx` | `book/code-examples/ch08_bytes.py` implements the bit patterns and checks them against Python's `str.encode("utf-8")` on all 315 characters of `train.FALLBACK` and on a, भ, € and 😀 |
| Ch 10, 10.2 (optional) | Using the Output Embedding to Improve Language Models | O. Press, L. Wolf | 2016 (EACL 2017) | arXiv:1608.05859 | tying a language model's input embedding and output (softmax) weight matrices | not checked; the chapter describes tying only as `tie_word_embeddings` does it and cites `docs/RESULTS.md` §32 for its effect here |
| Ch 9, 9.1 (optional) | A New Algorithm for Data Compression | P. Gage | 1994 | The C Users Journal 12(2) | byte pair encoding as a data-compression method, before its use for subwords [R05] | not checked; the chapter does not make this claim at present |

The register's existing entries cover the rest of Part III: [R05] for BPE
adapted to subword segmentation and [R04] for byte-level BPE with a base
vocabulary of 256, both CHECKED and both cited in Chapter 9 only for those
claims.

## Outcome (register worker, 2026-11-03)

- RFC 3629: **verified** at the RFC Editor's text (§3 table of octet sequences), added as [R54].
- Press & Wolf: **verified** (arXiv abstract; EACL 2017 per the arXiv comment), added as [S17].
- Gage 1994: **not added**. No primary source reachable (secondary pages only, and they disagree on the venue);
  no chapter makes the claim.
