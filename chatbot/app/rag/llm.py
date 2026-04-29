"""
llm.py
──────
Thin wrapper around the OpenAI chat completions API.

Builds a structured prompt that injects:
  - Retrieved context chunks (with source citations)
  - Conversation history (trimmed to MAX_HISTORY_TURNS)
  - The current user question
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import OpenAI

from app.constant.config import (
    LLM_MODEL,
    MAX_HISTORY_TURNS,
    MAX_TOKENS,
    OPEN_API_KEY,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

# ── Singleton client ───────────────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPEN_API_KEY)
    return _client


# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful, precise AI assistant that answers questions based strictly on \
the provided context passages. 

Rules:
- Only use information present in the context below.
- If the answer is not in the context, say: "I don't have enough information to answer that."
- Cite which passage(s) you used by referring to [Source X] tags.
- Be concise and direct.
"""


def _build_messages(
    query: str,
    context_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Construct the messages array for the Chat Completions API.

    Structure:
      [system] → [trimmed history] → [user message with injected context]
    """
    # Build context block from retrieved chunks
    context_parts: List[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        page = chunk.get("metadata", {}).get("page", "?")
        context_parts.append(
            f"[Source {i}] (file: {source}, page: {page})\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Trim history to the last N turns (each turn = 2 messages: user + assistant)
    max_messages = MAX_HISTORY_TURNS * 2
    trimmed_history = history[-max_messages:] if len(history) > max_messages else history

    # Final user message includes the context
    user_content = (
        f"Context:\n\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_content})

    return messages


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(
    query: str,
    context_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> str:
    """
    Generate an answer using the LLM given retrieved context.

    Parameters
    ----------
    query         : user's question
    context_chunks: list of chunk dicts from retrieve()
    history       : list of {"role": "user"|"assistant", "content": "..."} dicts

    Returns
    -------
    str — the model's answer text
    """
    if history is None:
        history = []

    messages = _build_messages(query, context_chunks, history)
    client = _get_client()

    logger.info(
        "Calling LLM model=%s, chunks=%d, history_turns=%d",
        LLM_MODEL,
        len(context_chunks),
        len(history) // 2,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,       # type: ignore[arg-type]
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    answer: str = response.choices[0].message.content or ""
    logger.info("LLM response received (%d chars).", len(answer))
    return answer.strip()
