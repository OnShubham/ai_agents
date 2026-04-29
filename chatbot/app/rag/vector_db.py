"""
vector_db.py
────────────
ChromaDB persistent vector store wrapper.

Responsibilities:
  - Persist embeddings to disk (VECTOR_DB_PATH).
  - Provide similarity_search() returning scored candidate chunks.
  - Guard against double-ingestion via collection_count().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.constant.config import COLLECTION_NAME, VECTOR_DB_PATH

logger = logging.getLogger(__name__)


# ── Singleton client & collection ──────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None


def _get_collection() -> Any:
    """
    Lazily initialise the ChromaDB persistent client and return the collection.
    Calling this multiple times is safe — it reuses the cached objects.
    """
    global _client, _collection
    if _collection is None:
        logger.info("Initialising ChromaDB at: %s", VECTOR_DB_PATH)
        _client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d docs).",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ── Public API ─────────────────────────────────────────────────────────────────

def collection_count() -> int:
    """Return how many documents are currently stored."""
    return _get_collection().count()


def add_documents(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    Upsert chunks into the collection.

    Parameters
    ----------
    ids        : unique string id per chunk  (e.g. "chunk_0", "chunk_1", …)
    embeddings : pre-computed dense vectors
    documents  : raw text of each chunk
    metadatas  : dict of metadata per chunk  (e.g. {"source": "foo.pdf", "page": 3})
    """
    collection = _get_collection()
    # Use upsert so re-running ingestion is idempotent
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into ChromaDB.", len(ids))


def similarity_search(
    query_embedding: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar documents to *query_embedding*.

    Returns
    -------
    list of dicts, each containing:
      - "text"      : str  — chunk text
      - "metadata"  : dict — stored metadata
      - "score"     : float — cosine distance (lower = more similar)
      - "id"        : str  — chunk id
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("similarity_search called on an empty collection.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates: List[Dict[str, Any]] = []
    for doc, meta, dist, cid in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        candidates.append(
            {
                "text": doc,
                "metadata": meta,
                "score": float(dist),   # cosine distance (0 = identical)
                "id": cid,
            }
        )

    return candidates
