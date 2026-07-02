"""Index episode dialogue into overlapping passages for semantic search.

Usage:  python -m semantic_search.ingest [subtitles_dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_loader import load_subtitles_dataset  # noqa: E402

from .embeddings import get_embedder  # noqa: E402
from .index import VectorIndex  # noqa: E402

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "One_Piece_Anime_S1_English_Text"


def passages(script: str, sentences_per_passage: int = 8, overlap: int = 2) -> list[str]:
    """Split an episode script into overlapping sentence windows."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if len(s.strip()) > 2]
    step = max(sentences_per_passage - overlap, 1)
    return [" ".join(sents[i:i + sentences_per_passage])
            for i in range(0, max(len(sents), 1), step)]


def ingest(subtitles_dir: str | Path = DEFAULT_DATA) -> VectorIndex:
    df = load_subtitles_dataset(str(subtitles_dir))
    texts, meta = [], []
    for _, row in df.iterrows():
        for j, p in enumerate(passages(row["script"])):
            texts.append(p)
            meta.append({"episode": int(row["episode"]), "passage": j, "text": p})

    embedder = get_embedder()
    index = VectorIndex()
    index.build(embedder.embed(texts), meta)
    index.save()
    print(f"[ingest] indexed {len(texts)} passages from {df.shape[0]} episodes "
          f"(embedder: {embedder.backend}, index: {index.backend})")
    return index


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
