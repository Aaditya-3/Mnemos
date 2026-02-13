"""
Layered chat orchestrator pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from backend.app.config.runtime import get_runtime_config
from backend.app.observability.logging import log_event
from backend.app.observability.metrics import metrics
from backend.app.observability.tracing import trace_span
from backend.app.orchestrator.context_builder import ContextBuilder
from backend.app.orchestrator.context_ranker import ContextRanker
from backend.app.orchestrator.prompt_assembler import PromptAssembler
from backend.app.orchestrator.types import (
    ContextBundle,
    OrchestratorDependencies,
    OrchestratorInput,
    PipelineResult,
)
from backend.app.services.token_usage import estimate_cost_usd, estimate_tokens


@dataclass
class PersistenceHooks:
    before_persist: Callable[[OrchestratorInput, PipelineResult], None] | None = None
    after_persist: Callable[[OrchestratorInput, PipelineResult], None] | None = None


class ChatOrchestrator:
    def __init__(self, deps: OrchestratorDependencies, persistence_hooks: PersistenceHooks | None = None):
        self.deps = deps
        self.cfg = get_runtime_config()
        self.context_builder = ContextBuilder(deps=deps)
        self.context_ranker = ContextRanker()
        self.prompt_assembler = PromptAssembler()
        self.persistence_hooks = persistence_hooks or PersistenceHooks()

    def _invoke_llm(self, prompt: str) -> tuple[str, float]:
        started = time.perf_counter()
        with trace_span("llm_invoke"):
            reply = self.deps.llm_complete_fn(prompt, self.cfg.llm_timeout_seconds)
        latency = time.perf_counter() - started
        metrics.observe("llm_latency_seconds", latency)
        return reply, latency

    def run(
        self,
        payload: OrchestratorInput,
        chat_session: Any,
        deterministic_hints: list[str] | None = None,
    ) -> PipelineResult:
        context = self.context_builder.build(payload=payload, chat_session=chat_session, deterministic_hints=deterministic_hints)
        context = self.context_ranker.rank(context)

        tool_events: list[dict[str, Any]] = []
        if payload.use_tools:
            try:
                agent = self.deps.tool_agent_fn(payload.continuity_message) or {}
                candidate = str(agent.get("reply") or "").strip()
                tool_events = list(agent.get("tool_events") or [])
                if candidate:
                    reply = self.deps.sanitize_reply_fn(payload.continuity_message, candidate)
                    input_tokens = estimate_tokens(payload.continuity_message)
                    output_tokens = estimate_tokens(reply)
                    usage = {
                        "input_tokens_est": input_tokens,
                        "output_tokens_est": output_tokens,
                        "cost_est_usd": estimate_cost_usd(input_tokens, output_tokens),
                        "llm_latency_ms": 0.0,
                    }
                    metrics.inc("llm_tokens_input_total", input_tokens)
                    metrics.inc("llm_tokens_output_total", output_tokens)
                    metrics.inc("llm_cost_usd_total", usage["cost_est_usd"])
                    result = PipelineResult(
                        reply=reply,
                        usage=usage,
                        semantic_rows=context.semantic_rows,
                        tool_events=tool_events,
                        prompt_used="tool_agent",
                    )
                    return result
            except Exception as exc:
                log_event("tool_agent_failed", user_id=payload.user_id, error=str(exc))

        prompt = self.prompt_assembler.build(payload=payload, context=context)
        reply, llm_latency = self._invoke_llm(prompt)
        reply = self.deps.sanitize_reply_fn(payload.continuity_message, reply)
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(reply)
        usage = {
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
            "cost_est_usd": estimate_cost_usd(input_tokens, output_tokens),
            "llm_latency_ms": round(llm_latency * 1000, 2),
        }
        metrics.inc("llm_tokens_input_total", input_tokens)
        metrics.inc("llm_tokens_output_total", output_tokens)
        metrics.inc("llm_cost_usd_total", usage["cost_est_usd"])
        result = PipelineResult(
            reply=reply,
            usage=usage,
            semantic_rows=context.semantic_rows,
            tool_events=tool_events,
            prompt_used=prompt,
        )
        return result
