"""
Streaming event handler with SSE event schema.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from backend.app.config.runtime import get_runtime_config
from backend.app.services.streaming import chunk_text_tokens


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class OrchestratorStreamHandler:
    def __init__(self):
        self.cfg = get_runtime_config()

    async def stream(
        self,
        text: str,
        request_id: str,
        chat_id: str,
        usage: dict,
        tool_events: list[dict],
        is_disconnected=None,
    ) -> AsyncGenerator[str, None]:
        yield sse("start", {"request_id": request_id, "chat_id": chat_id, "usage": usage})
        for evt in tool_events:
            yield sse("tool_call", evt)

        delay = max(self.cfg.stream_delay_ms, 0) / 1000.0
        for chunk in chunk_text_tokens(text, chunk_words=self.cfg.stream_chunk_words):
            if is_disconnected is not None and await is_disconnected():
                return
            yield sse("token", {"text": chunk})
            if delay:
                await asyncio.sleep(delay)
        yield sse("done", {"request_id": request_id, "chat_id": chat_id})

    async def stream_error(self, message: str, request_id: str = "") -> AsyncGenerator[str, None]:
        yield sse("error", {"request_id": request_id, "message": message})
        yield sse("done", {"request_id": request_id})
