"""
Factory for orchestrator dependency wiring.
"""

from __future__ import annotations

from typing import Callable

from backend.app.llm.client import get_llm_client
from backend.app.orchestrator.pipeline import ChatOrchestrator
from backend.app.orchestrator.types import OrchestratorDependencies


def build_chat_orchestrator(
    deterministic_memory_fn: Callable[[str, str], str],
    semantic_retrieve_fn: Callable[[str, str], tuple[list[dict], str]],
    recency_buffer_fn: Callable[[object], str],
    realtime_fn: Callable[[str], str],
    should_realtime_fn: Callable[[str], bool],
    tool_agent_fn: Callable[[str], dict],
    sanitize_reply_fn: Callable[[str, str], str],
) -> ChatOrchestrator:
    llm = get_llm_client()
    deps = OrchestratorDependencies(
        deterministic_memory_fn=deterministic_memory_fn,
        semantic_retrieve_fn=semantic_retrieve_fn,
        recency_buffer_fn=recency_buffer_fn,
        llm_complete_fn=llm.complete,
        realtime_fn=realtime_fn,
        should_realtime_fn=should_realtime_fn,
        tool_agent_fn=tool_agent_fn,
        sanitize_reply_fn=sanitize_reply_fn,
    )
    return ChatOrchestrator(deps=deps)

