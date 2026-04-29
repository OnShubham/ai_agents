"""
embeddings.py
─────────────
Singleton wrapper around SentenceTransformer.
Provides both single-text and batch embedding helpers.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.constant.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


# ── Singleton ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
    return model


# ── Public helpers ─────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """
    Embed a single string (e.g. a user query).

    Returns
    -------
    list[float]
        Dense vector of length 384 (all-MiniLM-L6-v2 output dim).
    """
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings in one batched call (efficient for ingestion).

    Returns
    -------
    list[list[float]]
        One dense vector per input text.
    """
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]
