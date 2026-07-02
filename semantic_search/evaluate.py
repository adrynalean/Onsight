"""Retrieval benchmark: sentence-probe self-retrieval.

For N randomly sampled passages, take one distinctive sentence as the query and
check whether the retriever brings back its source passage (top-1 / top-5) or at
least its source episode (top-5). Seeded and reproducible.

Usage:  python -m semantic_search.evaluate [n_queries]
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .embeddings import get_embedder  # noqa: E402
from .index import VectorIndex  # noqa: E402


def _probe_sentence(text: str) -> str | None:
    """Pick the longest sentence (most distinctive) with >= 6 words."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    sents = [s for s in sents if len(s.split()) >= 6]
    return max(sents, key=len) if sents else None


def evaluate(n_queries: int = 120, k: int = 5, seed: int = 42) -> dict:
    index = VectorIndex()
    if not index.load():
        raise RuntimeError("no index — run: python -m semantic_search.ingest")
    embedder = get_embedder()

    rng = random.Random(seed)
    candidates = [m for m in index.meta if _probe_sentence(m["text"])]
    sample = rng.sample(candidates, min(n_queries, len(candidates)))

    top1 = top5 = ep5 = 0
    latencies: list[float] = []
    for m in sample:
        query = _probe_sentence(m["text"])
        t0 = time.perf_counter()
        hits = index.query(embedder.embed([query])[0], k)
        latencies.append(time.perf_counter() - t0)
        keys = [(h["episode"], h["passage"]) for h in hits]
        target = (m["episode"], m["passage"])
        top1 += keys[:1] == [target]
        top5 += target in keys
        ep5 += m["episode"] in [h["episode"] for h in hits]

    n = len(sample)
    lat = sorted(latencies)
    results = {
        "queries": n,
        "passages": len(index),
        "embedder": embedder.backend,
        "index": index.backend,
        "top1_passage": round(top1 / n, 3),
        "top5_passage": round(top5 / n, 3),
        "top5_episode": round(ep5 / n, 3),
        "p95_latency_ms": round(lat[int(0.95 * n) - 1] * 1000, 1),
    }
    print(f"[eval] {n} queries over {len(index)} passages "
          f"({results['embedder']} + {results['index']})")
    print(f"[eval] passage recall  top-1: {results['top1_passage']:.1%}   "
          f"top-5: {results['top5_passage']:.1%}")
    print(f"[eval] episode recall  top-5: {results['top5_episode']:.1%}")
    print(f"[eval] retrieval p95: {results['p95_latency_ms']}ms")
    return results


if __name__ == "__main__":
    evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
