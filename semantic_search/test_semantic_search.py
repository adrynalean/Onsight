"""Tests for the semantic-search module (keyless fallbacks, temp index dir)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["SEMSEARCH_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from semantic_search.embeddings import get_embedder  # noqa: E402
from semantic_search.index import VectorIndex  # noqa: E402
from semantic_search.ingest import passages  # noqa: E402


def test_embeddings_are_normalized():
    vecs = get_embedder().embed(["hello world", "pirate king"])
    assert vecs.shape[0] == 2
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_passage_windows_overlap():
    script = " ".join(f"Sentence number {i} is here." for i in range(20))
    ps = passages(script, sentences_per_passage=8, overlap=2)
    assert len(ps) >= 3
    assert "number 6" in ps[0] and "number 6" in ps[1]   # overlap region shared


def test_index_roundtrip_and_query():
    emb = get_embedder()
    texts = ["Zoro is the pirate hunter.", "Nami draws sea charts.", "Luffy stretches."]
    idx = VectorIndex()
    idx.build(emb.embed(texts), [{"episode": i, "passage": 0, "text": t}
                                 for i, t in enumerate(texts)])
    idx.save()

    idx2 = VectorIndex()
    assert idx2.load() and len(idx2) == 3
    hits = idx2.query(emb.embed(["who hunts pirates? Zoro"])[0], k=1)
    assert hits[0]["text"].startswith("Zoro")
