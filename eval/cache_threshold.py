"""Pick a safe SEMANTIC_CACHE_THRESHOLD from labelled leads.

The semantic cache may reuse an answer only for a lead that *needs the same answer*. Its guards
already require identical numbers and the same classifier label, so the dangerous pairs are the
ones that pass both guards and still have **different known priorities**. This script embeds every
labelled lead with the same model the cache uses, finds those dangerous pairs, and reports the
highest similarity any of them reaches. The threshold must sit above it.

    python -m eval.cache_threshold          # dry run: how many leads, estimated cost
    python -m eval.cache_threshold --yes    # embeds them (~$0.0001 with text-embedding-3-small)

Uses OPENAI_API_KEY from .env. Reads finetune/data/test.jsonl and val.jsonl by default.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

import semantic_cache
from finetune.generate_dataset import DATA_DIR, read_jsonl

MARGIN = 0.02


def unsafe_pairs(rows: list[dict], vectors: np.ndarray, labels: list[str | None]) -> list[tuple[float, int, int]]:
    """(similarity, i, j) for pairs the guards would let through but whose known labels differ."""
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    nums = [semantic_cache.numbers(r["text"]) for r in rows]
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if nums[i] == nums[j] and labels[i] == labels[j] and rows[i]["label"] != rows[j]["label"]:
            out.append((float(vectors[i] @ vectors[j]), i, j))
    return sorted(out, reverse=True)


def recommend(pairs: list[tuple[float, int, int]], floor: float = 0.85) -> float:
    highest = pairs[0][0] if pairs else 0.0
    return round(max(floor, highest + MARGIN), 2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--file", type=Path, action="append", help="labelled JSONL (repeatable)")
    p.add_argument("--yes", action="store_true", help="call the OpenAI embeddings API")
    args = p.parse_args(argv)
    files = args.file or [DATA_DIR / "test.jsonl", DATA_DIR / "val.jsonl"]
    rows = [r for f in files for r in read_jsonl(f)]
    print(f"{len(rows)} labelled leads from {', '.join(f.name for f in files)}; model {semantic_cache.EMBED_MODEL}")
    if not args.yes:
        print("dry run: nothing sent. Add --yes (costs about $0.0001).")
        return 0

    from openai import OpenAI

    from config import get_settings

    client = OpenAI(api_key=get_settings().openai_api_key)
    vectors = []
    for start in range(0, len(rows), 100):
        batch = [r["text"] for r in rows[start : start + 100]]
        resp = client.embeddings.create(
            model=semantic_cache.EMBED_MODEL, input=batch, dimensions=semantic_cache.EMBED_DIMENSIONS
        )
        vectors += [d.embedding for d in resp.data]
    labels = [semantic_cache._label(r["text"]) for r in rows]
    pairs = unsafe_pairs(rows, np.asarray(vectors, dtype=np.float32), labels)

    print(f"\n{len(pairs)} pairs pass both guards but have different known priorities. Most similar:")
    for sim, i, j in pairs[:5]:
        print(f"  {sim:.3f}  [{rows[i]['label']}] {rows[i]['text'][:60]!r}")
        print(f"         [{rows[j]['label']}] {rows[j]['text'][:60]!r}")
    print(f"\nRecommended SEMANTIC_CACHE_THRESHOLD: {recommend(pairs)} (highest unsafe pair + {MARGIN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
