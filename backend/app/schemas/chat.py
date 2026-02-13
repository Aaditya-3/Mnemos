"""
Schemas for orchestrated chat endpoints.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class OrchestratedChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    use_tools: bool = False
    scope: Optional[str] = None


class TokenUsageSchema(BaseModel):
    input_tokens_est: int
    output_tokens_est: int
    cost_est_usd: float
    llm_latency_ms: float


class OrchestratedChatResponse(BaseModel):
    reply: str
    chat_id: str
    usage: TokenUsageSchema
    semantic_memories: list[dict[str, Any]]
    tool_events: list[dict[str, Any]] = []

