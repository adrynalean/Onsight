"""Vector index: FAISS when installed, numpy cosine otherwise. Persisted to disk."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

INDEX_DIR = Path(os.getenv("SEMSEARCH_DIR", Path(__file__).resolve().parent / "index"))


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


class VectorIndex:
    """Cosine-similarity index over passage vectors, with passage metadata."""

    def __init__(self) -> None:
        self.meta: list[dict] = []
        self._vecs: np.ndarray | None = None
        self._faiss = None

    # ── build / persist ──────────────────────────────────────────────────
    def build(self, vectors: np.ndarray, meta: list[dict]) -> None:
        self.meta = meta
        self._vecs = vectors.astype(np.float32)
        if _has_faiss():
            import faiss
            self._faiss = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss.add(self._vecs)

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "vectors.npy", self._vecs)
        (INDEX_DIR / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")

    def load(self) -> bool:
        vec_path, meta_path = INDEX_DIR / "vectors.npy", INDEX_DIR / "meta.json"
        if not (vec_path.exists() and meta_path.exists()):
            return False
        self.build(np.load(vec_path), json.loads(meta_path.read_text(encoding="utf-8")))
        return True

    # ── query ────────────────────────────────────────────────────────────
    def query(self, vector: np.ndarray, k: int = 5) -> list[dict]:
        if self._vecs is None or not len(self.meta):
            return []
        q = vector.astype(np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, min(k, len(self.meta)))
            pairs = zip(idxs[0], scores[0])
        else:
            sims = (self._vecs @ q.T).ravel()
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [{**self.meta[i], "score": round(float(s), 4)} for i, s in pairs]

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return len(self.meta)
