"""
Semantic memory ingestion, retrieval, decay, and compression.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.embeddings.provider import get_embedding_provider
from backend.app.memory.models import SemanticMemory
from backend.app.observability.logging import log_event
from backend.app.observability.metrics import metrics
from backend.app.vector_store.repository import VectorHit, get_vector_store, iter_memory_tokens


EMOTIONAL_TOKENS = {"love", "hate", "angry", "excited", "anxious", "important", "critical", "never", "always"}
GOAL_TOKENS = {"goal", "plan", "roadmap", "target", "build", "launch", "ship", "deadline"}
PROJECT_TOKENS = {"project", "repo", "feature", "api", "frontend", "backend"}
PREFERENCE_PATTERNS = [
    r"\bi prefer\b",
    r"\bmy favorite\b",
    r"\bi like\b",
    r"\bi usually\b",
]
FACT_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bi work\b",
    r"\bi study\b",
]
TEMPORARY_TOKENS = {"today", "tomorrow", "this week", "right now", "currently"}


class SemanticMemoryService:
    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedding_provider()
        self.store = get_vector_store()

    def classify_memory_type(self, message: str) -> str:
        text = (message or "").lower()
        if any(re.search(p, text) for p in PREFERENCE_PATTERNS):
            return "preference"
        if any(re.search(p, text) for p in FACT_PATTERNS):
            return "factual"
        if any(t in text for t in GOAL_TOKENS):
            return "long_term_goal"
        if any(t in text for t in PROJECT_TOKENS):
            return "project_specific"
        if any(t in text for t in TEMPORARY_TOKENS):
            return "temporary_context"
        if any(t in text for t in EMOTIONAL_TOKENS):
            return "emotional"
        return "factual"

    def detect_scope(self, message: str) -> str:
        text = (message or "").lower()
        if "this project" in text or "in this repo" in text:
            return "project"
        if "in this conversation" in text:
            return "conversation"
        return "user"

    def score_importance(
        self,
        message: str,
        memory_type: str,
        previous_similar_count: int = 0,
    ) -> float:
        base = {
            "factual": 0.62,
            "preference": 0.68,
            "emotional": 0.55,
            "long_term_goal": 0.78,
            "project_specific": 0.66,
            "temporary_context": 0.42,
        }.get(memory_type, 0.5)
        text = (message or "").lower()
        emotional_boost = 0.12 if any(t in text for t in EMOTIONAL_TOKENS) else 0.0
        repeat_boost = min(0.2, previous_similar_count * 0.04)
        explicit_boost = 0.08 if any(k in text for k in ["remember", "important", "don't forget", "must"]) else 0.0
        score = base + emotional_boost + repeat_boost + explicit_boost
        return max(0.0, min(1.0, score))

    def normalize_memory_text(self, message: str) -> str:
        text = re.sub(r"\s+", " ", (message or "").strip())
        text = text.strip(".!?")
        return text[:500]

    def extract_tags(self, message: str, limit: int = 8) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        for token in iter_memory_tokens(message):
            counts[token] += 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [k for k, _ in ranked[:limit]]

    def _existing_similarity_count(self, user_id: str, text: str) -> int:
        try:
            query = self.embedder.embed(text)
            hits = self.store.search(query.vector, user_id=user_id, top_k=12, scopes=None)
            return len([h for h in hits if h.similarity >= 0.84])
        except Exception:
            return 0

    def ingest_message(
        self,
        user_id: str,
        message: str,
        source_message_id: str = "",
        scope: Optional[str] = None,
    ) -> Optional[SemanticMemory]:
        if not self.settings.enable_semantic_memory:
            return None
        normalized = self.normalize_memory_text(message)
        if len(normalized) < 6:
            return None

        memory_type = self.classify_memory_type(normalized)
        resolved_scope = scope or self.detect_scope(normalized)
        similar_count = self._existing_similarity_count(user_id=user_id, text=normalized)
        importance = self.score_importance(normalized, memory_type, previous_similar_count=similar_count)
        tags = self.extract_tags(normalized)
        decay_factor = self.settings.importance_decay_per_day
        memory = SemanticMemory.create(
            user_id=user_id,
            content=normalized,
            memory_type=memory_type,
            scope=resolved_scope,
            importance_score=importance,
            decay_factor=decay_factor,
            tags=tags,
            source_message_id=source_message_id,
        )

        t0 = time.perf_counter()
        embedding = self.embedder.embed(normalized)
        metrics.observe("embedding_latency_seconds", time.perf_counter() - t0)
        memory.embedding = embedding.vector
        memory.embedding_model = embedding.model
        memory.embedding_provider = embedding.provider
        memory.metadata = {
            "similar_count": similar_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.upsert(memory)
        log_event(
            "semantic_memory_ingested",
            user_id=user_id,
            memory_id=memory.id,
            memory_type=memory.memory_type,
            scope=memory.scope,
            importance=memory.importance_score,
            embedding_provider=memory.embedding_provider,
        )
        return memory

    def _rank_score(self, similarity: float, importance: float, created_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
        recency_weight = math.exp(-0.03 * age_days)
        return (similarity * 0.65) + (importance * 0.25) + (recency_weight * 0.10)

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        scopes: Optional[list[str]] = None,
    ) -> tuple[list[dict], str]:
        if not self.settings.enable_semantic_memory:
            return [], ""

        top_k = top_k or self.settings.semantic_top_k
        t0 = time.perf_counter()
        query_emb = self.embedder.embed(query)
        metrics.observe("embedding_latency_seconds", time.perf_counter() - t0)

        t1 = time.perf_counter()
        hits = self.store.search(query_emb.vector, user_id=user_id, top_k=top_k * 3, scopes=scopes)
        metrics.observe("memory_retrieval_latency_seconds", time.perf_counter() - t1)

        ranked = sorted(
            hits,
            key=lambda h: self._rank_score(
                similarity=h.similarity,
                importance=h.memory.importance_score,
                created_at=h.memory.created_at,
            ),
            reverse=True,
        )

        selected: list[VectorHit] = []
        token_budget = self.settings.semantic_token_budget
        used = 0
        for hit in ranked:
            if len(selected) >= top_k:
                break
            if hit.memory.importance_score < self.settings.importance_drop_threshold:
                continue
            est = max(1, len(hit.memory.content) // 4)
            if selected and (used + est) > token_budget:
                break
            selected.append(hit)
            used += est

        rows: list[dict] = []
        lines: list[str] = []
        for idx, hit in enumerate(selected, start=1):
            row = {
                "rank": idx,
                "memory_id": hit.memory.id,
                "content": hit.memory.content,
                "memory_type": hit.memory.memory_type,
                "scope": hit.memory.scope,
                "importance_score": hit.memory.importance_score,
                "similarity_score": hit.similarity,
                "created_at": hit.memory.created_at.isoformat(),
                "tags": hit.memory.tags,
            }
            rows.append(row)
            lines.append(
                f"- ({idx}) {hit.memory.content} "
                f"[type={hit.memory.memory_type}; scope={hit.memory.scope}; importance={hit.memory.importance_score:.2f}; sim={hit.similarity:.2f}]"
            )

        context = "\n".join(lines)
        return rows, context

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        return self.store.list_user_memories(user_id)

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        return self.store.delete_memory(user_id=user_id, memory_id=memory_id)

    def apply_decay(self, user_id: Optional[str] = None) -> dict:
        rows = self.store.list_user_memories(user_id) if user_id else []
        if not user_id:
            # Local fallback: iterate users if underlying store does not support global list.
            # For now we decay via per-user endpoint unless user_id is provided.
            return {"updated": 0, "deactivated": 0}

        updated = 0
        deactivated = 0
        now = datetime.now(timezone.utc)
        for memory in rows:
            age_days = max((now - memory.updated_at).total_seconds() / 86400.0, 0.0)
            decayed = memory.importance_score - (memory.decay_factor * age_days)
            next_score = max(0.0, min(1.0, decayed))
            if abs(next_score - memory.importance_score) < 1e-5:
                continue
            memory.importance_score = next_score
            memory.updated_at = now
            if memory.importance_score < self.settings.importance_drop_threshold:
                memory.is_active = False
                deactivated += 1
            self.store.upsert(memory)
            updated += 1
        return {"updated": updated, "deactivated": deactivated}

    def compress_user_memories(self, user_id: str) -> dict:
        rows = [m for m in self.store.list_user_memories(user_id) if m.is_active]
        if not rows:
            return {"compressed": 0}

        now = datetime.now(timezone.utc)
        buckets: dict[tuple[str, str], list[SemanticMemory]] = defaultdict(list)
        for m in rows:
            age_days = (now - m.created_at).total_seconds() / 86400.0
            if age_days < self.settings.semantic_compression_age_days:
                continue
            key = (m.memory_type, m.scope)
            buckets[key].append(m)

        compressed_count = 0
        for (memory_type, scope), bucket in buckets.items():
            if len(bucket) < self.settings.semantic_compression_min_cluster:
                continue

            # Keep top 2 and compress the rest into one summary memory.
            bucket.sort(key=lambda x: (x.importance_score, x.updated_at.timestamp()), reverse=True)
            keep = bucket[:2]
            compress_items = bucket[2:]
            if not compress_items:
                continue
            summary_parts = [m.content for m in compress_items[:10]]
            summary = " | ".join(summary_parts)
            summary_text = f"Summary of older {memory_type} memories: {summary}"
            summary_mem = SemanticMemory.create(
                user_id=user_id,
                content=summary_text[:800],
                memory_type=f"{memory_type}_summary",
                scope=scope,
                importance_score=max(0.35, sum(m.importance_score for m in keep) / max(len(keep), 1)),
                decay_factor=self.settings.importance_decay_per_day * 0.5,
                tags=["summary", memory_type, scope],
                source_message_id="compression",
            )
            emb = self.embedder.embed(summary_mem.content)
            summary_mem.embedding = emb.vector
            summary_mem.embedding_model = emb.model
            summary_mem.embedding_provider = emb.provider
            summary_mem.metadata = {
                "compressed_from": [m.id for m in compress_items],
                "compressed_at": now.isoformat(),
            }
            self.store.upsert(summary_mem)
            compressed_count += 1

            for m in compress_items:
                m.is_active = False
                m.updated_at = now
                self.store.upsert(m)

        return {"compressed": compressed_count}


_semantic_service: SemanticMemoryService | None = None


def get_semantic_memory_service() -> SemanticMemoryService:
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticMemoryService()
    return _semantic_service

