"""
Metrics collection with optional Prometheus client.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict


try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore

    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class _LocalMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, list[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = defaultdict(float)

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
            for key, val in self.gauges.items():
                lines.append(f"# TYPE {key} gauge")
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
        self._active_user_lock = threading.Lock()
        self._active_users_seen: dict[str, datetime] = {}
        if _PROM_AVAILABLE:
            self.total_requests = Counter("total_requests", "Total HTTP requests")
            self.http_latency_seconds = Histogram("http_request_latency_seconds", "Request latency in seconds")
            self.llm_latency_seconds = Histogram("llm_latency_seconds", "LLM latency in seconds")
            self.memory_retrieval_seconds = Histogram("memory_retrieval_time_seconds", "Semantic memory retrieval latency")
            self.embedding_seconds = Histogram("embedding_time_seconds", "Embedding generation latency")
            self.tool_invocations_total = Counter("tool_invocations_total", "Total tool invocations")
            self.memory_decay_events_total = Counter("memory_decay_events_total", "Total memory decay archive/delete events")
            self.active_users = Gauge("active_users", "Active users in recent window")
            self.tokens_input_total = Counter("llm_tokens_input_total", "Estimated LLM input tokens")
            self.tokens_output_total = Counter("llm_tokens_output_total", "Estimated LLM output tokens")
            self.llm_cost_usd_total = Counter("llm_cost_usd_total", "Estimated accumulated LLM cost in USD")
        else:
            self.total_requests = None
            self.http_latency_seconds = None
            self.llm_latency_seconds = None
            self.memory_retrieval_seconds = None
            self.embedding_seconds = None
            self.tool_invocations_total = None
            self.memory_decay_events_total = None
            self.active_users = None
            self.tokens_input_total = None
            self.tokens_output_total = None
            self.llm_cost_usd_total = None

    def inc(self, name: str, amount: float = 1.0):
        self.local.inc(name, amount)
        if not _PROM_AVAILABLE:
            return
        if name == "total_requests":
            self.total_requests.inc(amount)  # type: ignore[union-attr]
        if name == "tool_invocations_total":
            self.tool_invocations_total.inc(amount)  # type: ignore[union-attr]
        if name == "memory_decay_events_total":
            self.memory_decay_events_total.inc(amount)  # type: ignore[union-attr]
        if name == "llm_tokens_input_total":
            self.tokens_input_total.inc(amount)  # type: ignore[union-attr]
        if name == "llm_tokens_output_total":
            self.tokens_output_total.inc(amount)  # type: ignore[union-attr]
        if name == "llm_cost_usd_total":
            self.llm_cost_usd_total.inc(amount)  # type: ignore[union-attr]

    def observe(self, name: str, value: float):
        self.local.observe(name, value)
        if not _PROM_AVAILABLE:
            return
        if name == "http_request_latency_seconds":
            self.http_latency_seconds.observe(value)  # type: ignore[union-attr]
        if name == "llm_latency_seconds":
            self.llm_latency_seconds.observe(value)  # type: ignore[union-attr]
        if name == "memory_retrieval_time_seconds":
            self.memory_retrieval_seconds.observe(value)  # type: ignore[union-attr]
        if name == "embedding_time_seconds":
            self.embedding_seconds.observe(value)  # type: ignore[union-attr]

    def mark_user_active(self, user_id: str, ttl_minutes: int = 10):
        uid = (user_id or "").strip()
        if not uid or uid == "anonymous":
            return
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=max(ttl_minutes, 1))
        with self._active_user_lock:
            self._active_users_seen[uid] = now
            expired = [k for k, ts in self._active_users_seen.items() if ts < cutoff]
            for k in expired:
                self._active_users_seen.pop(k, None)
            active = float(len(self._active_users_seen))
            self.local.gauges["active_users"] = active
            if _PROM_AVAILABLE and self.active_users is not None:
                self.active_users.set(active)

    def export(self) -> tuple[str, str]:
        if _PROM_AVAILABLE:
            return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
        return self.local.exposition(), CONTENT_TYPE_LATEST


metrics = AppMetrics()
