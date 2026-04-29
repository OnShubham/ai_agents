"""
retrival.py
───────────
Full retrieval pipeline:

  1. semantic_search()  — embed query, pull top-k × MULTIPLIER candidates from ChromaDB
  2. rerank()           — score candidates with a CrossEncoder, keep best top-k
  3. retrieve()         — one-call convenience that wires 1 + 2 together
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from app.constant.config import (
    RERANKER_MODEL_NAME,
    TOP_K,
    TOP_K_CANDIDATES_MULTIPLIER,
)
from app.rag.embeddings import embed_text
from app.rag.vector_db import similarity_search

logger = logging.getLogger(__name__)


# ── Singleton CrossEncoder ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
    model = CrossEncoder(RERANKER_MODEL_NAME)
    logger.info("Reranker loaded successfully.")
    return model


# ── Stage 1: Semantic search ───────────────────────────────────────────────────

def semantic_search(
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Embed *query* and return *top_k* most similar chunks from ChromaDB.

    Parameters
    ----------
    query  : the raw user question
    top_k  : how many candidates to pull

    Returns
    -------
    list of candidate dicts {"text", "metadata", "score", "id"}
    """
    query_embedding = embed_text(query)
    candidates = similarity_search(query_embedding, top_k=top_k)
    logger.debug("semantic_search returned %d candidates.", len(candidates))
    return candidates


# ── Stage 2: Cross-encoder reranking ──────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Re-score *candidates* with a CrossEncoder and return the best *top_k*.

    The CrossEncoder reads both the query and each candidate passage jointly,
    giving a much more accurate relevance score than cosine similarity alone.

    Parameters
    ----------
    query      : original user question
    candidates : output of semantic_search()
    top_k      : number of chunks to keep

    Returns
    -------
    Reranked list (highest relevance first), truncated to *top_k*.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    # Build (query, passage) pairs for the CrossEncoder
    pairs = [(query, c["text"]) for c in candidates]
    scores: List[float] = reranker.predict(pairs).tolist()

    # Attach reranker scores and sort descending
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    top = ranked[:top_k]

    logger.debug(
        "rerank: kept %d/%d candidates. Top score=%.4f",
        len(top),
        len(candidates),
        top[0]["rerank_score"] if top else 0.0,
    )
    return top


# ── Stage 3: Full pipeline ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline: semantic_search → rerank.

    Fetches `top_k × TOP_K_CANDIDATES_MULTIPLIER` candidates from the vector
    DB (broadening the recall window), then reranks them down to *top_k*.

    Parameters
    ----------
    query  : user question
    top_k  : number of final chunks to return

    Returns
    -------
    list of top-k dicts, sorted by relevance (best first).
    """
    candidate_count = top_k * TOP_K_CANDIDATES_MULTIPLIER
    logger.info(
        "retrieve: query='%s', fetching %d candidates then reranking to %d.",
        query[:80],
        candidate_count,
        top_k,
    )

    candidates = semantic_search(query, top_k=candidate_count)
    final = rerank(query, candidates, top_k=top_k)
    return final
