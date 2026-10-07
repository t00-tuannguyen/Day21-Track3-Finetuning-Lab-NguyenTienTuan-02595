#!/usr/bin/env python3
"""Build the B2 custom corpus from UIT-ViSFD — deterministic, decontaminated.

Task: **Vietnamese smartphone review -> aspect-sentiment JSON**, e.g.
    "Pin kém còn lại miễn chê"  ->  {"battery": "negative", "general": "positive"}

Source: https://huggingface.co/datasets/visolex/ViSFD (UIT-ViSFD, 11,122 human-annotated
comments from a Vietnamese e-commerce site, original train/dev/test split kept in the
`type` column). See data/CUSTOM_DATASET.md for why this corpus and how it was cleaned.

Writes (overwrites) data/train_seed.jsonl, data/eval_target.jsonl, data/TASK and
data/visfd_build.json. data/eval_regression.jsonl is NOT touched: the general-capability
probe must stay identical so the regression gate still means the same thing.

Run:  python scripts/make_visfd_data.py            (downloads ~3.3 MB once, then cached)
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
import re
import sys
import unicodedata
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "raw" / "ViSFD.csv"          # gitignored: data/raw/
URL = "https://huggingface.co/datasets/visolex/ViSFD/resolve/main/ViSFD.csv"

SEED = 42
N_TRAIN, N_EVAL = 250, 50
MIN_CHARS, MAX_CHARS = 20, 300               # p90 of the corpus is 279 chars
NEAR_DUP_JACCARD = 0.7

# Source tag -> output key. OTHERS carries no polarity in ViSFD and is dropped.
ASPECT_MAP = {
    "SCREEN": "screen", "CAMERA": "camera", "FEATURES": "features", "BATTERY": "battery",
    "PERFORMANCE": "performance", "STORAGE": "storage", "DESIGN": "design",
    "PRICE": "price", "GENERAL": "general", "SER&ACC": "ser_acc",
}
POLARITY_MAP = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}

# Kept for the `system=None` render only (NB1 shows it side by side); training and eval
# use config.VISFD_NAIVE_PROMPT as the system turn — see data.to_messages / F-31.
INSTRUCTION = ("Phân tích cảm xúc theo khía cạnh cho bình luận điện thoại sau. Trả về JSON, "
               "khóa là khía cạnh được nhắc tới (screen, camera, features, battery, "
               "performance, storage, design, price, general, ser_acc), giá trị là "
               "positive | negative | neutral.")


def parse_label(raw: str) -> dict:
    out = {}
    for tag in re.findall(r"\{([^}]*)\}", raw):
        if "#" not in tag:
            continue                          # {OTHERS}
        aspect, pol = tag.split("#", 1)
        if aspect in ASPECT_MAP and pol in POLARITY_MAP:
            out[ASPECT_MAP[aspect]] = POLARITY_MAP[pol]
    return out


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    t = norm(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def load_rows() -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, CACHE)
    with CACHE.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def candidates(rows: list[dict], split: str, stats: Counter) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["type"] != split:
            continue
        text = re.sub(r"\s+", " ", r["comment"]).strip()
        label = parse_label(r["label"])
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            stats[f"{split}: length out of range"] += 1
            continue
        if not label:
            stats[f"{split}: no polarity-bearing aspect"] += 1
            continue
        key = norm(text)
        if key in seen:
            stats[f"{split}: exact duplicate"] += 1
            continue
        seen.add(key)
        out.append({"text": text, "label": label, "index": int(r["index"]),
                    "n_star": int(r["n_star"]) if r["n_star"].isdigit() else None})
    return out


def to_record(c: dict, split: str) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": c["text"],
        "output": json.dumps(c["label"], ensure_ascii=False),
        "label": c["label"],
        "source": {"dataset": "UIT-ViSFD", "split": split, "index": c["index"],
                   "n_star": c["n_star"]},
    }


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> str:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def main() -> int:
    rows = load_rows()
    stats: Counter = Counter()
    rng = random.Random(SEED)

    # Eval is drawn first, from the TEST split only, and frozen; train is then cleaned
    # against it. Never the other way round.
    eval_pool = candidates(rows, "test", stats)
    eval_pick = rng.sample(eval_pool, N_EVAL)
    eval_sh = [shingles(c["text"]) for c in eval_pick]
    eval_keys = {norm(c["text"]) for c in eval_pick}

    train_pool = []
    for c in candidates(rows, "train", stats):
        if norm(c["text"]) in eval_keys:
            stats["train: exact match with an eval item"] += 1
            continue
        sh = shingles(c["text"])
        if any(jaccard(sh, e) >= NEAR_DUP_JACCARD for e in eval_sh):
            stats["train: near-duplicate of an eval item"] += 1
            continue
        train_pool.append(c)
    train_pick = rng.sample(train_pool, N_TRAIN)

    train = [to_record(c, "train") for c in train_pick]
    evals = [to_record(c, "test") for c in eval_pick]
    sha_train = write_jsonl(DATA / "train_seed.jsonl", train)
    sha_eval = write_jsonl(DATA / "eval_target.jsonl", evals)
    (DATA / "TASK").write_text("visfd\n", encoding="utf-8")

    def dist(rs):
        asp, pol = Counter(), Counter()
        for r in rs:
            for k, v in r["label"].items():
                asp[k] += 1
                pol[v] += 1
        return {"aspects": dict(asp.most_common()), "polarities": dict(pol.most_common()),
                "mean_aspects_per_item": round(sum(len(r["label"]) for r in rs) / len(rs), 2)}

    build = {
        "source": URL, "seed": SEED, "n_train": len(train), "n_eval": len(evals),
        "pool_sizes": {"train": len(train_pool), "test": len(eval_pool)},
        "filters": dict(sorted(stats.items())),
        "sha256_16": {"train_seed.jsonl": sha_train, "eval_target.jsonl": sha_eval},
        "train": dist(train), "eval": dist(evals),
    }
    (DATA / "visfd_build.json").write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(build, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
