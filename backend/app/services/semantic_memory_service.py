"""
Semantic memory ingestion, ranking, decay, compression, and re-embedding.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from backend.app.config.runtime import get_runtime_config
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
TRANSIENT_TOKENS = {"today", "tomorrow", "this week", "right now", "currently"}


class SemanticMemoryService:
    def __init__(self):
        self.settings = get_settings()
        self.runtime = get_runtime_config()
        self.embedder = get_embedding_provider()
        self.store = get_vector_store()

    def classify_memory_type(self, message: str) -> str:
        text = (message or "").lower()
        if any(re.search(p, text) for p in PREFERENCE_PATTERNS):
            return "preference"
        if any(re.search(p, text) for p in FACT_PATTERNS):
            return "fact"
        if any(t in text for t in GOAL_TOKENS):
            return "goal"
        if any(t in text for t in PROJECT_TOKENS):
            return "project"
        if any(t in text for t in TRANSIENT_TOKENS):
            return "transient"
        if any(t in text for t in EMOTIONAL_TOKENS):
            return "emotional"
        return "fact"

    def detect_scope(self, message: str) -> str:
        text = (message or "").lower()
        if "this project" in text or "in this repo" in text:
            return "project"
        if "in this conversation" in text:
            return "conversation"
        if "global rule" in text or "for everyone" in text:
            return "global"
        return "user"

    def score_importance(
        self,
        message: str,
        memory_type: str,
        previous_similar_count: int = 0,
    ) -> float:
        base = {
            "fact": 0.62,
            "preference": 0.68,
            "emotional": 0.55,
            "goal": 0.78,
            "project": 0.66,
            "transient": 0.42,
        }.get(memory_type, 0.5)
        text = (message or "").lower()
        emotional_signal = 0.12 if any(t in text for t in EMOTIONAL_TOKENS) else 0.0
        reinforcement_boost = min(0.2, previous_similar_count * 0.04)
        explicit_boost = 0.08 if any(k in text for k in ["remember", "important", "don't forget", "must"]) else 0.0
        score = base + emotional_signal + reinforcement_boost + explicit_boost
        return max(0.0, min(0.95, score))

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
            hits = self.store.search(query.vector, user_id=user_id, top_k=16, scopes=None)
            return len([h for h in hits if h.similarity >= 0.84 and h.memory.is_active and not h.memory.is_archived])
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
        resolved_scope = (scope or self.detect_scope(normalized)).strip().lower()
        if resolved_scope not in self.runtime.memory_scope_whitelist:
            resolved_scope = "user"

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
        memory.reinforcement_count = max(0, similar_count)

        t0 = time.perf_counter()
        embedding = self.embedder.embed(normalized)
        metrics.observe("embedding_time_seconds", time.perf_counter() - t0)
        memory.embedding = embedding.vector
        memory.embedding_model = embedding.model
        memory.embedding_provider = embedding.provider
        memory.metadata = {
            "similar_count": similar_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "importance_model": "base+reinforcement+emotion",
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

    def _recency_weight(self, created_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
        return math.exp(-0.03 * age_days)

    def _rank_score(self, similarity: float, importance: float, created_at: datetime) -> float:
        w = self.runtime.ranking_weights
        recency = self._recency_weight(created_at)
        return (similarity * w.similarity) + (importance * w.importance) + (recency * w.recency)

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        scopes: Optional[list[str]] = None,
    ) -> tuple[list[dict], str]:
        if not self.settings.enable_semantic_memory:
            return [], ""

        top_k = top_k or self.runtime.semantic_top_k
        t0 = time.perf_counter()
        query_emb = self.embedder.embed(query)
        metrics.observe("embedding_time_seconds", time.perf_counter() - t0)

        t1 = time.perf_counter()
        hits = self.store.search(query_emb.vector, user_id=user_id, top_k=top_k * 4, scopes=scopes)
        metrics.observe("memory_retrieval_time_seconds", time.perf_counter() - t1)

        ranked = sorted(
            [h for h in hits if h.memory.is_active and not h.memory.is_archived],
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
            if hit.memory.importance_score < self.runtime.memory_archive_threshold:
                continue
            est = max(1, len(hit.memory.content) // 4)
            if selected and (used + est) > token_budget:
                break
            selected.append(hit)
            used += est

        rows: list[dict] = []
        lines: list[str] = []
        now = datetime.now(timezone.utc)
        for idx, hit in enumerate(selected, start=1):
            rank_score = self._rank_score(hit.similarity, hit.memory.importance_score, hit.memory.created_at)
            row = {
                "rank": idx,
                "memory_id": hit.memory.id,
                "content": hit.memory.content,
                "memory_type": hit.memory.memory_type,
                "scope": hit.memory.scope,
                "importance_score": hit.memory.importance_score,
                "similarity_score": hit.similarity,
                "recency_weight": self._recency_weight(hit.memory.created_at),
                "final_score": rank_score,
                "reinforcement_count": hit.memory.reinforcement_count,
                "created_at": hit.memory.created_at.isoformat(),
                "tags": hit.memory.tags,
            }
            rows.append(row)
            lines.append(
                f"- ({idx}) {hit.memory.content} "
                f"[type={hit.memory.memory_type}; scope={hit.memory.scope}; importance={hit.memory.importance_score:.2f}; "
                f"sim={hit.similarity:.2f}; final={rank_score:.2f}]"
            )

            # Reinforcement logic: if retrieved into context, reinforce.
            hit.memory.reinforcement_count += 1
            hit.memory.last_accessed = now
            hit.memory.importance_score = min(0.99, hit.memory.importance_score + 0.01)
            hit.memory.updated_at = now
            self.store.upsert(hit.memory)

        context = "\n".join(lines)
        return rows, context

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        return self.store.list_user_memories(user_id)

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        return self.store.delete_memory(user_id=user_id, memory_id=memory_id)

    def apply_decay(self, user_id: Optional[str] = None) -> dict:
        rows = self.store.list_user_memories(user_id) if user_id else []
        if not user_id:
            return {"updated": 0, "archived": 0, "deleted": 0}

        updated = 0
        archived = 0
        deleted = 0
        now = datetime.now(timezone.utc)
        for memory in rows:
            if memory.is_archived:
                continue
            # Periodic multiplicative decay.
            prev = memory.importance_score
            decay_factor = max(0.01, min(1.0, float(memory.decay_factor)))
            memory.importance_score = max(0.0, min(1.0, memory.importance_score * decay_factor))
            memory.updated_at = now
            if abs(prev - memory.importance_score) > 1e-6:
                updated += 1

            if memory.importance_score <= self.runtime.memory_delete_threshold:
                if self.store.delete_memory(memory.user_id, memory.id):
                    deleted += 1
                continue
            if memory.importance_score <= self.runtime.memory_archive_threshold:
                memory.is_active = False
                memory.is_archived = True
                memory.archived_at = now
                archived += 1

            self.store.upsert(memory)

        metrics.inc("memory_decay_events_total", archived + deleted)
        return {"updated": updated, "archived": archived, "deleted": deleted}

    def compress_user_memories(self, user_id: str) -> dict:
        rows = [m for m in self.store.list_user_memories(user_id) if m.is_active and not m.is_archived]
        if not rows:
            return {"compressed": 0}

        now = datetime.now(timezone.utc)
        buckets: dict[tuple[str, str], list[SemanticMemory]] = defaultdict(list)
        for m in rows:
            if m.importance_score > self.runtime.memory_archive_threshold:
                continue
            key = (m.memory_type, m.scope)
            buckets[key].append(m)

        compressed_count = 0
        for (memory_type, scope), bucket in buckets.items():
            if len(bucket) < self.runtime.compression_cluster_min_size:
                continue

            bucket.sort(key=lambda x: (x.importance_score, x.updated_at.timestamp()))
            cluster = bucket[: min(12, len(bucket))]
            summary_parts = [m.content for m in cluster]
            summary_text = " | ".join(summary_parts)
            try:
                from backend.app.llm.client import get_llm_client

                llm = get_llm_client()
                prompt = (
                    "Summarize these related memories into one concise durable memory node.\n"
                    "Preserve core facts and preferences. Avoid fluff.\n"
                    f"Memory type: {memory_type}\n"
                    f"Scope: {scope}\n"
                    f"Items: {summary_text}"
                )
                summary_text = llm.complete(prompt, timeout_seconds=self.runtime.llm_timeout_seconds)
            except Exception:
                summary_text = f"Compressed {memory_type} memories: {summary_text}"

            summary_mem = SemanticMemory.create(
                user_id=user_id,
                content=summary_text[:1000],
                memory_type=f"{memory_type}_summary",
                scope=scope,
                importance_score=max(0.3, sum(m.importance_score for m in cluster) / len(cluster)),
                decay_factor=max(0.02, self.settings.importance_decay_per_day * 0.7),
                tags=["summary", memory_type, scope],
                source_message_id="compression",
            )
            emb = self.embedder.embed(summary_mem.content)
            summary_mem.embedding = emb.vector
            summary_mem.embedding_model = emb.model
            summary_mem.embedding_provider = emb.provider
            summary_mem.metadata = {
                "reference_graph": {
                    "cluster_type": memory_type,
                    "scope": scope,
                    "sources": [m.id for m in cluster],
                    "created_by": "compression_engine",
                },
                "compressed_at": now.isoformat(),
            }
            self.store.upsert(summary_mem)
            compressed_count += 1

            for m in cluster:
                m.is_active = False
                m.is_archived = True
                m.archived_at = now
                m.updated_at = now
                self.store.upsert(m)

        return {"compressed": compressed_count}

    def reembed_user_memories(self, user_id: str, reason: str = "model_update") -> dict:
        rows = self.store.list_user_memories(user_id)
        reembedded = 0
        now = datetime.now(timezone.utc)
        for memory in rows:
            try:
                emb = self.embedder.embed(memory.content)
                memory.embedding = emb.vector
                memory.embedding_model = emb.model
                memory.embedding_provider = emb.provider
                memory.updated_at = now
                metadata = dict(memory.metadata or {})
                metadata["reembedded_at"] = now.isoformat()
                metadata["reembed_reason"] = reason
                memory.metadata = metadata
                self.store.upsert(memory)
                reembedded += 1
            except Exception as exc:
                log_event("memory_reembed_failed", user_id=user_id, memory_id=memory.id, error=str(exc))
        return {"reembedded": reembedded}


_semantic_service: SemanticMemoryService | None = None


def get_semantic_memory_service() -> SemanticMemoryService:
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticMemoryService()
    return _semantic_service
