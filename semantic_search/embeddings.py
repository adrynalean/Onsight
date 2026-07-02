"""Embeddings: sentence-transformers when available, hashed n-grams otherwise."""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_DIM = 512


class HashingEmbedder:
    """Deterministic character-n-gram hashing embedder (no downloads, no keys)."""

    backend = "hashing"
    dim = _DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = re.findall(r"\w+", text.lower())
            grams = toks + [t[i:i + 4] for t in toks for i in range(max(len(t) - 3, 1))]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                out[row, h % _DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SbertEmbedder:
    backend = "sentence-transformers"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(EMBED_MODEL)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def get_embedder():
    try:
        return SbertEmbedder()
    except ModuleNotFoundError:
        return HashingEmbedder()
