"""
schemas.py
──────────
Pydantic v2 request / response models for the /chat and /ingest endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique identifier for the conversation session.",
        examples=["user_abc_123"],
    )
    message: str = Field(
        ...,
        description="The user's question or message.",
        min_length=1,
        examples=["What is Python used for?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (after reranking).",
    )


class SourceChunk(BaseModel):
    text: str = Field(..., description="Preview of the retrieved chunk text.")
    source: str = Field(..., description="Source file name.")
    page: str = Field(..., description="Page number within the source document.")
    rerank_score: float = Field(..., description="CrossEncoder relevance score.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk]
    history_length: int = Field(
        ..., description="Total number of messages in the session history after this turn."
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    pdf_path: str = Field(
        ...,
        description="Absolute or relative path to the PDF file to ingest.",
        examples=["app/data/Learning_Python.pdf"],
    )
    force: bool = Field(
        default=False,
        description="Re-ingest even if the collection already has documents.",
    )


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    source: str


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    documents_in_store: int
