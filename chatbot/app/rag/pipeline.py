"""
pipeline.py
───────────
RAGPipeline: high-level orchestrator that wires together:
  - Document ingestion  (PDF → chunk → embed → store)
  - Query answering     (retrieve → rerank → generate)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

from app.constant.config import TOP_K
from app.rag.chunking import chunk_document, chunking_strategy
from app.rag.embeddings import embed_documents
from app.rag.llm import generate
from app.rag.retrival import retrieve
from app.rag.vector_db import add_documents, collection_count

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation lifecycle.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.ingest("path/to/document.pdf")
    result   = pipeline.query("What is Python?", history=[])
    print(result["answer"])
    """

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Load a PDF, chunk it, embed the chunks, and upsert into ChromaDB.

        Parameters
        ----------
        pdf_path : absolute or relative path to a PDF file
        force    : if True, re-ingest even when documents already exist

        Returns
        -------
        dict with {"status", "chunks_added", "source"}
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        existing = collection_count()
        if existing > 0 and not force:
            logger.info(
                "Collection already has %d docs. Skipping ingestion (use force=True to override).",
                existing,
            )
            return {"status": "skipped", "chunks_added": 0, "source": pdf_path}

        logger.info("Ingesting PDF: %s", pdf_path)

        # 1. Load raw pages from PDF
        raw_pages = chunk_document(pdf_path)

        # 2. Split into smaller chunks
        splitter = chunking_strategy()
        chunks = splitter.split_documents(raw_pages)
        logger.info("Created %d chunks from PDF.", len(chunks))

        if not chunks:
            return {"status": "empty", "chunks_added": 0, "source": pdf_path}

        # 3. Embed all chunk texts in one batch call
        texts = [c.page_content for c in chunks]
        embeddings = embed_documents(texts)

        # 4. Build metadata and stable IDs
        source_name = os.path.basename(pdf_path)
        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source_name,
                "page": str(c.metadata.get("page", i)),
                "chunk_index": str(i),
            }
            for i, c in enumerate(chunks)
        ]

        # 5. Upsert into ChromaDB
        add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info("Ingestion complete. %d chunks stored.", len(chunks))
        return {"status": "success", "chunks_added": len(chunks), "source": pdf_path}

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int = TOP_K,
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Parameters
        ----------
        question : user's natural language question
        history  : list of prior {"role": ..., "content": ...} messages
        top_k    : number of chunks to retrieve after reranking

        Returns
        -------
        dict with:
          - "answer"  : str
          - "sources" : list of {"text", "source", "page", "rerank_score"}
        """
        if history is None:
            history = []

        if collection_count() == 0:
            return {
                "answer": (
                    "The knowledge base is empty. "
                    "Please ingest a document first via POST /ingest."
                ),
                "sources": [],
            }

        # 1. Retrieve + rerank
        chunks = retrieve(question, top_k=top_k)

        # 2. Generate answer with LLM
        answer = generate(question, chunks, history)

        # 3. Build source citations for the response
        sources = [
            {
                "text": c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
                "source": c["metadata"].get("source", "unknown"),
                "page": c["metadata"].get("page", "?"),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
            }
            for c in chunks
        ]

        return {"answer": answer, "sources": sources}
