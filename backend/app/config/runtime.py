"""
Runtime configuration for orchestrator and memory ranking behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RankingWeights:
    similarity: float
    importance: float
    recency: float


@dataclass(frozen=True)
class OrchestratorRuntimeConfig:
    ranking_weights: RankingWeights
    llm_timeout_seconds: float
    llm_retry_count: int
    llm_retry_backoff_seconds: float
    semantic_top_k: int
    memory_archive_threshold: float
    memory_delete_threshold: float
    compression_cluster_min_size: int
    stream_chunk_words: int
    stream_delay_ms: int
    tool_timeout_seconds: float
    max_tool_calls: int
    enable_tool_sandbox: bool
    memory_scope_whitelist: tuple[str, ...]


def get_runtime_config() -> OrchestratorRuntimeConfig:
    sim = _float("MEM_RANK_WEIGHT_SIMILARITY", 0.60)
    imp = _float("MEM_RANK_WEIGHT_IMPORTANCE", 0.25)
    rec = _float("MEM_RANK_WEIGHT_RECENCY", 0.15)
    total = sim + imp + rec
    if total <= 0:
        sim, imp, rec = 0.60, 0.25, 0.15
    else:
        sim, imp, rec = sim / total, imp / total, rec / total

    return OrchestratorRuntimeConfig(
        ranking_weights=RankingWeights(similarity=sim, importance=imp, recency=rec),
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 30.0),
        llm_retry_count=_int("LLM_RETRY_COUNT", 2),
        llm_retry_backoff_seconds=_float("LLM_RETRY_BACKOFF_SECONDS", 0.8),
        semantic_top_k=_int("SEMANTIC_TOP_K", 12),
        memory_archive_threshold=_float("MEMORY_ARCHIVE_THRESHOLD", 0.18),
        memory_delete_threshold=_float("MEMORY_DELETE_THRESHOLD", 0.10),
        compression_cluster_min_size=_int("MEMORY_COMPRESSION_CLUSTER_MIN", 4),
        stream_chunk_words=_int("STREAM_CHUNK_WORDS", 3),
        stream_delay_ms=_int("STREAM_DELAY_MS", 12),
        tool_timeout_seconds=_float("TOOL_TIMEOUT_SECONDS", 6.0),
        max_tool_calls=_int("MAX_TOOL_CALLS", 3),
        enable_tool_sandbox=_bool("ENABLE_TOOL_SANDBOX", True),
        memory_scope_whitelist=("global", "user", "conversation", "project"),
    )

