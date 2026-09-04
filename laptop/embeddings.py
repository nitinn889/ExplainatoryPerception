"""
Caption -> vector embeddings (Phase 6), via sentence-transformers.

Model is loaded once and reused; embed()/embed_batch() are the only entry
points memory_store.py should need.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(text: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    return _get_model(model_name).encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    vectors = _get_model(model_name).encode(texts, normalize_embeddings=True)
    return vectors.tolist()
