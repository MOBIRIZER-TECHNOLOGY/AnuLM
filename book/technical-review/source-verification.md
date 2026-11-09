# Source verification

Written by `book/tools/verify_source_map.py` against `@2e80b3e`.

Source-map rows checked: 94. Matching their line range (±1): **92**. Not matching: **2**.
Test-map entries: 65. At the `def` line given: **65**. Not: **0**.

Rows whose range deliberately starts with leading comments or constants (accepted): Vocabulary and ranks, EOS above the merges (+3).

## Rows that don't match

| Component | File | Symbols | Map says | Finding |
|---|---|---|---|---|
| QLoRA, 4-bit base | `anulm/lora.py` |  | – | no range: an absent feature or an uncommitted one (documented in the row) |
| DPO, RLHF, GRPO | `anulm/bidirectional.py` |  | – | no range: an absent feature or an uncommitted one (documented in the row) |

## Test map entries that don't match

- none

Test functions defined in `tests/test_model.py` at the pin: 86.

## Drift between the pin and the working tree

- none
