"""
main.py
───────
FastAPI application entry point.

On startup:
  - Auto-ingests the default PDF (app/data/Learning_Python.pdf) into ChromaDB
    if the collection is empty.

Endpoints (via chat_router):
  GET  /health   — liveness check
  POST /chat     — conversational RAG endpoint
  POST /ingest   — on-demand document ingestion
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI

from app.chat_router import router

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Auto-ingest the default PDF on first startup.
    Skipped if the ChromaDB collection already has documents.
    """
    from app.rag.pipeline import RAGPipeline
    from app.rag.vector_db import collection_count

    DEFAULT_PDF = os.path.join(
        os.path.dirname(__file__),
        "app", "data", "Learning_Python.pdf",
    )

    if collection_count() == 0:
        logger.info("Knowledge base is empty — starting auto-ingestion of default PDF...")
        pipeline = RAGPipeline()
        try:
            result = pipeline.ingest(DEFAULT_PDF)
            logger.info(
                "Auto-ingestion complete: %d chunks stored from '%s'.",
                result["chunks_added"],
                result["source"],
            )
        except FileNotFoundError:
            logger.warning(
                "Default PDF not found at '%s'. "
                "Start the server and use POST /ingest to add documents.",
                DEFAULT_PDF,
            )
    else:
        logger.info(
            "Knowledge base already has %d documents. Skipping auto-ingestion.",
            collection_count(),
        )

    yield  # application runs here
    logger.info("Shutting down RAG Chatbot API.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A Retrieval-Augmented Generation chatbot that answers questions from "
        "ingested PDF documents using semantic search, top-k retrieval, and "
        "cross-encoder reranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# ── Dev entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
