# Custom dataset (B2) — UIT-ViSFD aspect-sentiment

**Task:** a Vietnamese smartphone review → a JSON object listing only the aspects the review
mentions, each mapped to `positive | negative | neutral`.

```
"Pin kém còn lại miễn chê"  ->  {"battery": "negative", "general": "positive"}
```

Aspect keys: `screen, camera, features, battery, performance, storage, design, price,
general, ser_acc` (ser_acc = service & accessories).

Rebuild with `python scripts/make_visfd_data.py`; it is deterministic (seed 42). The
build log, with every filter count and the label distributions, is in `data/visfd_build.json`.

## Source

| | |
|---|---|
| Dataset | **UIT-ViSFD** (Vietnamese Smartphone Feedback Dataset), University of Information Technology, VNU-HCM |
| Copy used | https://huggingface.co/datasets/visolex/ViSFD (one CSV, original split in the `type` column) |
| Size | 11,122 comments (train 7,786 / dev 1,112 / test 2,224) |
| Collection | real customer comments from a major Vietnamese e-commerce site, **human-annotated** with 10 aspects × 3 polarities (plus an `OTHERS` tag without polarity) |
| Licence | the HF card states none; UIT datasets are released for research use. Used here for coursework only, not redistributed beyond the 300 sampled rows. |
| Paper | Luc Phan et al., *SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence*, KSEM 2021 |

## What was taken and how

| Step | Rule | Removed |
|---|---|---|
| Length | keep 20–300 characters (corpus p90 = 279) so a T4 can score 50 items × 3 sweeps in budget | 620 train / 180 test |
| Label | drop `{OTHERS}` (it has no polarity); keep only comments with ≥1 polarity-bearing aspect | 113 train / 36 test |
| Exact dedup | NFC + lowercase + punctuation-stripped text, within each split | 8 train |
| Sample | 50 eval from **test**, then 250 train from **train**, `random.Random(42)` | — |

Result: **250 train** (→ 225/25 after NB1's seed-42 split) and **50 eval**, avg 3.0 / 2.7
aspects per item. Polarity mix (train): positive 437 · negative 217 · neutral 106.

## Decontamination

1. **Split-level:** eval comes only from ViSFD's `test` split and train only from `train`,
   so the annotators' own split is respected. `dev` is unused.
2. **Eval chosen first, then frozen**; the train pool was filtered *against* it, never the
   reverse.
3. **Exact match:** any train comment whose normalized text equals an eval comment is
   dropped (0 found).
4. **Near-duplicate:** any train comment with character-5-gram Jaccard ≥ 0.7 to any eval
   comment is dropped (0 found).
5. **Prompt few-shots:** the two examples inside `VISFD_OPTIMIZED_PROMPT` are invented,
   not taken from any split.
6. `eval_regression.jsonl` (15 general-knowledge probes) is **unchanged** from the
   shipped lab, so the regression gate measures the same thing as on the default corpus.

Frozen eval checksum (sha256, first 16 hex): `eval_target.jsonl = ae103cf33949be65`,
`train_seed.jsonl = 4b9db7f563779ebd`. `data/checksums.json` still holds the shipped
corpus's hashes on purpose, so `make verify` reports this swap as a declared WARN
instead of silently treating the new eval set as the original.

## Why this data is distributionally new (deck §3.3)

* **Informal Vietnamese user text** — teen-code, missing diacritics, run-on sentences,
  store slang ("tgdđ", "bin trâu", "wf"). This is far from the clean web/Wikipedia
  Vietnamese a base model sees most, and unlike the shipped corpus it is not templated.
* **A closed, dataset-specific label space.** What counts as `features` vs `performance`,
  or when a review is `neutral` rather than mixed, is an annotation convention of this
  dataset. A base model cannot know it from pretraining; a prompt can only describe it.
  That is exactly the kind of knowledge fine-tuning should be able to move into weights.
* **Variable-size output.** The number of keys depends on the input (1–8 here), so the
  model must decide *which* aspects are present, not just fill a fixed schema.

## Scoring

* **target** = F1 between predicted and gold `(aspect, polarity)` pairs, averaged over
  items (`evaluate.aspect_f1`). A right aspect with the wrong polarity is both a false
  positive and a false negative. A fixed 10-key schema with `none` was rejected: most
  keys would be `none`, so predicting nothing would already score ~70%.
* **format** = fraction of emitted entries with a valid aspect key and polarity value
  (`evaluate.aspect_format`); non-JSON or `{}` scores 0.
* **regression / latency** — unchanged.

The task is selected by the one-word file `data/TASK` (`visfd`); `$LAB_TASK` overrides it.
Prompts and scorers both read that one switch (`config.TASK`).
