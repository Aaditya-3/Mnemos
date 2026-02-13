"""
LLM client adapter with timeout and retry policy.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from backend.app.config.runtime import get_runtime_config
from backend.app.core.llm.groq_client import generate_response
from backend.app.observability.logging import log_event
from backend.app.utils.interfaces import LLMClient


@dataclass
class RetryPolicy:
    retries: int
    backoff_seconds: float


class GroqLLMClient(LLMClient):
    def __init__(self, policy: RetryPolicy | None = None):
        cfg = get_runtime_config()
        self.policy = policy or RetryPolicy(
            retries=cfg.llm_retry_count,
            backoff_seconds=cfg.llm_retry_backoff_seconds,
        )

    def _call_once(self, prompt: str, timeout_seconds: float) -> str:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_response, prompt)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"LLM call timed out after {timeout_seconds}s") from exc

    def complete(self, prompt: str, timeout_seconds: float) -> str:
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.policy.retries:
            started = time.perf_counter()
            try:
                response = self._call_once(prompt, timeout_seconds=timeout_seconds)
                log_event(
                    "llm_call_success",
                    attempt=attempt + 1,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                return response
            except Exception as exc:
                last_error = exc
                log_event("llm_call_failure", attempt=attempt + 1, error=str(exc))
                if attempt >= self.policy.retries:
                    break
                sleep_for = self.policy.backoff_seconds * (attempt + 1)
                time.sleep(sleep_for)
                attempt += 1
        raise RuntimeError(f"LLM invocation failed: {last_error}")


_llm_client: GroqLLMClient | None = None


def get_llm_client() -> GroqLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = GroqLLMClient()
    return _llm_client

