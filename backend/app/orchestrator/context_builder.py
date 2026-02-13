"""
Context builder layer for orchestrator pipeline.
"""

from __future__ import annotations

import time
from typing import Any

from backend.app.config.runtime import get_runtime_config
from backend.app.observability.logging import log_event
from backend.app.observability.metrics import metrics
from backend.app.orchestrator.types import ContextBundle, OrchestratorDependencies, OrchestratorInput
from backend.app.tools.registry import tool_registry


class ContextBuilder:
    def __init__(self, deps: OrchestratorDependencies):
        self.deps = deps
        self.cfg = get_runtime_config()

    def build(self, payload: OrchestratorInput, chat_session: Any, deterministic_hints: list[str] | None = None) -> ContextBundle:
        bundle = ContextBundle()
        bundle.deterministic_hints = list(deterministic_hints or [])

        # Deterministic memory retrieval.
        try:
            bundle.deterministic_memory_context = self.deps.deterministic_memory_fn(payload.continuity_message, payload.user_id) or ""
        except Exception as exc:
            log_event("deterministic_memory_failed", user_id=payload.user_id, error=str(exc))
            bundle.deterministic_memory_context = ""

        # Semantic memory retrieval.
        try:
            t0 = time.perf_counter()
            rows, semantic_context = self.deps.semantic_retrieve_fn(payload.user_id, payload.continuity_message)
            metrics.observe("memory_retrieval_time_seconds", time.perf_counter() - t0)
            bundle.semantic_rows = rows
            bundle.semantic_memory_context = semantic_context
        except Exception as exc:
            log_event("semantic_memory_failed", user_id=payload.user_id, error=str(exc))
            bundle.semantic_rows = []
            bundle.semantic_memory_context = ""

        # Recency buffer.
        try:
            bundle.recency_buffer = self.deps.recency_buffer_fn(chat_session) or ""
        except Exception as exc:
            log_event("recency_buffer_failed", user_id=payload.user_id, error=str(exc))
            bundle.recency_buffer = ""

        # Tool hints.
        try:
            tools = tool_registry.list_tools()
            if tools:
                hints = []
                for tool in tools:
                    hints.append(f"- {tool.get('name')}: {tool.get('description')}")
                bundle.tool_hints = "\n".join(hints)
            else:
                bundle.tool_hints = ""
        except Exception:
            bundle.tool_hints = ""

        # Optional realtime context.
        try:
            if self.deps.should_realtime_fn(payload.continuity_message):
                bundle.realtime_context = self.deps.realtime_fn(payload.continuity_message) or ""
        except Exception as exc:
            log_event("realtime_context_failed", user_id=payload.user_id, error=str(exc))
            bundle.realtime_context = ""

        return bundle

