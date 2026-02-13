"""
Streaming helpers for SSE responses.
"""

from __future__ import annotations

import json
import time
from typing import Generator, Iterable


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text_tokens(text: str, chunk_words: int = 3) -> Iterable[str]:
    words = (text or "").split()
    if not words:
        return []
    chunk_size = max(1, int(chunk_words))
    chunks: list[str] = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current) + " ")
            current = []
    if current:
        chunks.append(" ".join(current) + " ")
    return chunks


def stream_text_sse(
    text: str,
    request_id: str = "",
    delay_s: float = 0.015,
    start_payload: dict | None = None,
) -> Generator[str, None, None]:
    start = {"request_id": request_id}
    if start_payload:
        start.update(start_payload)
    yield sse_event("start", start)
    for token in chunk_text_tokens(text):
        yield sse_event("token", {"text": token})
        if delay_s > 0:
            time.sleep(delay_s)
    yield sse_event("done", {"request_id": request_id})
