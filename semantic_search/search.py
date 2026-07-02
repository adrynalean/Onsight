"""Query the passage index.

Usage:  python -m semantic_search.search "who is the pirate hunter?"
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .embeddings import get_embedder  # noqa: E402
from .index import VectorIndex  # noqa: E402


@lru_cache
def _loaded() -> tuple:
    index = VectorIndex()
    if not index.load():
        raise RuntimeError("no index found — run: python -m semantic_search.ingest")
    return index, get_embedder()


def search(query: str, k: int = 5) -> list[dict]:
    """Return top-k passages: [{episode, passage, text, score}]."""
    index, embedder = _loaded()
    return index.query(embedder.embed([query])[0], k)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "who is the pirate hunter?"
    for hit in search(q):
        print(f"ep{hit['episode']:>3}  {hit['score']:.3f}  {hit['text'][:90]}…")
