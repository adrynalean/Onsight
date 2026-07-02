# Semantic Search — Benchmark Results

Sentence-probe **self-retrieval** benchmark: for 120 randomly sampled passages
(seed 42), the longest sentence of the passage is used as the query, and we check
whether the retriever brings back the source passage / episode.

## Measured (keyless run — hashed n-gram embeddings + numpy cosine index)

| Metric | Value |
|--------|-------|
| Indexed passages | 2,036 (30 episodes, 8-sentence windows, 2-sentence overlap) |
| Passage recall @1 | 67.5% |
| Passage recall @5 | **95.0%** |
| Episode recall @5 | **97.5%** |
| Retrieval p95 latency | 0.6 ms |

Reproduce:

```bash
python -m semantic_search.ingest      # build the index
python -m semantic_search.evaluate    # run the 120-query benchmark
```

## Notes — how to read this honestly

- **Self-retrieval is a soft benchmark**: the query is a sentence *from* the target
  passage, so lexical methods score well. It measures indexing/retrieval correctness
  and ranking quality, not paraphrase-level semantic matching.
- For paraphrase robustness, install the full stack (`pip install faiss-cpu
  sentence-transformers`) — the module upgrades automatically to dense MiniLM
  embeddings + FAISS, and the same benchmark then measures true semantic recall.
