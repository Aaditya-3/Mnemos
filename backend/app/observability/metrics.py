"""
Metrics collection with optional Prometheus client.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict


try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore

    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class _LocalMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, amount: float = 1.0):
        with self._lock:
            self.counters[name] += amount

    def observe(self, name: str, value: float):
        with self._lock:
            self.histograms[name].append(float(value))

    def exposition(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, val in self.counters.items():
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {val}")
            for key, vals in self.histograms.items():
                lines.append(f"# TYPE {key}_count counter")
                lines.append(f"{key}_count {len(vals)}")
                if vals:
                    lines.append(f"# TYPE {key}_avg gauge")
                    lines.append(f"{key}_avg {sum(vals)/len(vals)}")
                    lines.append(f"# TYPE {key}_max gauge")
                    lines.append(f"{key}_max {max(vals)}")
        return "\n".join(lines) + "\n"


class AppMetrics:
    def __init__(self):
        self.local = _LocalMetrics()
        if _PROM_AVAILABLE:
            self.http_requests_total = Counter("http_requests_total", "Total HTTP requests")
            self.http_latency_seconds = Histogram("http_request_latency_seconds", "Request latency in seconds")
            self.memory_retrieval_seconds = Histogram("memory_retrieval_latency_seconds", "Semantic memory retrieval latency")
            self.embedding_seconds = Histogram("embedding_latency_seconds", "Embedding generation latency")
            self.tokens_input_total = Counter("llm_tokens_input_total", "Estimated LLM input tokens")
            self.tokens_output_total = Counter("llm_tokens_output_total", "Estimated LLM output tokens")
        else:
            self.http_requests_total = None
            self.http_latency_seconds = None
            self.memory_retrieval_seconds = None
            self.embedding_seconds = None
            self.tokens_input_total = None
            self.tokens_output_total = None

    def inc(self, name: str, amount: float = 1.0):
        self.local.inc(name, amount)
        if not _PROM_AVAILABLE:
            return
        if name == "http_requests_total":
            self.http_requests_total.inc(amount)  # type: ignore[union-attr]
        if name == "llm_tokens_input_total":
            self.tokens_input_total.inc(amount)  # type: ignore[union-attr]
        if name == "llm_tokens_output_total":
            self.tokens_output_total.inc(amount)  # type: ignore[union-attr]

    def observe(self, name: str, value: float):
        self.local.observe(name, value)
        if not _PROM_AVAILABLE:
            return
        if name == "http_request_latency_seconds":
            self.http_latency_seconds.observe(value)  # type: ignore[union-attr]
        if name == "memory_retrieval_latency_seconds":
            self.memory_retrieval_seconds.observe(value)  # type: ignore[union-attr]
        if name == "embedding_latency_seconds":
            self.embedding_seconds.observe(value)  # type: ignore[union-attr]

    def export(self) -> tuple[str, str]:
        if _PROM_AVAILABLE:
            return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
        return self.local.exposition(), CONTENT_TYPE_LATEST


metrics = AppMetrics()

