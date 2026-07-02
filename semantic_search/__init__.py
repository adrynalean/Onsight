"""Semantic search over episode dialogue.

FAISS + sentence-transformers when installed; falls back to a numpy cosine index
with hashed n-gram embeddings so it runs (and evaluates) with no extra installs.

Public API:  from semantic_search.search import search
"""
