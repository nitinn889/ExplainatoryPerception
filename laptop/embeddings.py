"""
Caption -> vector embeddings (Phase 6).

Primary backend is sentence-transformers (`all-MiniLM-L6-v2`), which gives
genuine semantic similarity ("blue flask" ~ "water bottle"). That model has
to be downloaded from HuggingFace the first time it's used, so on a machine
with no network access (or behind a proxy that blocks huggingface.co) it
cannot load.

Rather than crash mid-demo, this module falls back to a deterministic
**lexical** embedder: word unigrams plus character 4-grams hashed into the
same 384-dimensional space and L2-normalised, so cosine similarity
approximates word/substring overlap. It needs no model download and no
network.

The fallback is a genuinely weaker embedder — it will match "where is the
bottle?" to "A bottle is on the table." but *not* "blue flask" to "water
bottle". `active_backend()` reports which one is live so the demo UI and the
evaluation write-up can state it honestly rather than implying full semantic
retrieval was used.

Force a backend with the EPISODIC_EMBEDDINGS env var: "sentence-transformers"
or "lexical".
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger("laptop.embeddings")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

BACKEND_TRANSFORMER = "sentence-transformers"
BACKEND_LEXICAL = "lexical"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Without this, function words sink the lexical fallback: every caption contains
# "a", "is", "on" and "the", so after L2 normalisation those shared tokens
# outweigh the one word the question was actually about.
_STOPWORDS = frozenset(
    """a an and are as at be been by did do does for from had has have how i in
    into is it its my of on onto or our that the their there these this those to
    was were what when where which who why with you your""".split()
)


# --------------------------------------------------------------------------
# Lexical fallback
# --------------------------------------------------------------------------

def _bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % EMBED_DIM


def _lexical_embed(text: str) -> list[float]:
    """Hashed bag of word unigrams + character 4-grams, L2-normalised."""
    vector = [0.0] * EMBED_DIM
    lowered = text.lower()
    words = [w for w in _WORD_RE.findall(lowered) if w not in _STOPWORDS]

    for word in words:
        vector[_bucket(f"w:{word}")] += 1.0

    squashed = " ".join(words)
    for i in range(max(0, len(squashed) - 3)):
        vector[_bucket(f"c:{squashed[i:i + 4]}")] += 0.5

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_transformer(model_name: str) -> Any | None:
    """Load the sentence-transformers model, or return None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("sentence-transformers not installed (%s)", exc)
        return None

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning(
            "Could not load '%s' (%s). Falling back to the lexical embedder — "
            "retrieval will be keyword-based, not semantic.",
            model_name,
            exc,
        )
        return None


@lru_cache(maxsize=1)
def active_backend(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Which embedding backend is actually in use: 'sentence-transformers' or 'lexical'."""
    requested = os.environ.get("EPISODIC_EMBEDDINGS", "").strip().lower()
    if requested == BACKEND_LEXICAL:
        logger.info("Lexical embedder forced via EPISODIC_EMBEDDINGS.")
        return BACKEND_LEXICAL

    if _load_transformer(model_name) is not None:
        return BACKEND_TRANSFORMER

    if requested == BACKEND_TRANSFORMER:
        raise RuntimeError(
            "EPISODIC_EMBEDDINGS=sentence-transformers was requested but the model "
            f"'{model_name}' could not be loaded. Install sentence-transformers and "
            "make sure huggingface.co is reachable, or unset the variable."
        )
    return BACKEND_LEXICAL


def embedding_dim(model_name: str = DEFAULT_MODEL_NAME) -> int:
    if active_backend(model_name) == BACKEND_TRANSFORMER:
        model = _load_transformer(model_name)
        return int(model.get_sentence_embedding_dimension())
    return EMBED_DIM


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def embed(text: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    if active_backend(model_name) == BACKEND_TRANSFORMER:
        model = _load_transformer(model_name)
        return model.encode(text, normalize_embeddings=True).tolist()
    return _lexical_embed(text)


def embed_batch(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    if active_backend(model_name) == BACKEND_TRANSFORMER:
        model = _load_transformer(model_name)
        return model.encode(texts, normalize_embeddings=True).tolist()
    return [_lexical_embed(t) for t in texts]
