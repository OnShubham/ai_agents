"""
chat_router.py
──────────────
FastAPI router providing three endpoints:

  POST /chat    — stateful conversational RAG endpoint
  POST /ingest  — trigger PDF ingestion on demand
  GET  /health  — liveness check
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app.rag.pipeline import RAGPipeline
from app.rag.vector_db import collection_count
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Singletons ─────────────────────────────────────────────────────────────────

_pipeline = RAGPipeline()

# In-memory session store: {session_id: [{"role": ..., "content": ...}, ...]}
_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and receive a RAG-powered answer",
    description=(
        "Submit a question with a session_id. The endpoint maintains per-session "
        "conversation history so follow-up questions are answered in context."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational RAG endpoint.

    Flow:
      1. Look up (or create) conversation history for *session_id*.
      2. Call RAGPipeline.query() → retrieve + rerank + generate.
      3. Append the user message and assistant reply to the session history.
      4. Return the structured ChatResponse.
    """
    logger.info("Chat | session=%s | message='%s'", request.session_id, request.message[:80])

    try:
        history = _sessions[request.session_id]

        result = _pipeline.query(
            question=request.message,
            history=history,
            top_k=request.top_k or 5,
        )
    except Exception as exc:
        logger.exception("Chat endpoint error for session=%s", request.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Persist this turn in history
    _sessions[request.session_id].extend(
        [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": result["answer"]},
        ]
    )

    sources = [SourceChunk(**s) for s in result["sources"]]

    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        sources=sources,
        history_length=len(_sessions[request.session_id]),
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a PDF into the vector store",
    description=(
        "Load, chunk, embed, and store a PDF document. "
        "Set `force=true` to re-ingest an already-processed file."
    ),
)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Trigger document ingestion on demand."""
    logger.info("Ingest | pdf_path=%s | force=%s", request.pdf_path, request.force)

    try:
        result = _pipeline.ingest(pdf_path=request.pdf_path, force=request.force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest endpoint error.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(**result)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> HealthResponse:
    """Return service status and the number of documents currently indexed."""
    return HealthResponse(
        status="ok",
        documents_in_store=collection_count(),
    )
