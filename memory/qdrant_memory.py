"""
Qdrant-backed semantic memory service.

This module is the single semantic memory backend used by Mnemos.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.app.core.config import get_settings
from backend.app.memory.models import SemanticMemory
from backend.app.observability.logging import log_event

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )
except Exception:
    QdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    MatchAny = None
    MatchValue = None
    PayloadSchemaType = None
    PointStruct = None
    VectorParams = None


def _importance_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


class QdrantMemoryService:
    def __init__(self):
        self.settings = get_settings()
        self.collection = self.settings.qdrant_collection
        self.vector_size = max(32, int(self.settings.embedding_dims or 1536))
        self.client: Any | None = self._build_client()
        if self.client is not None:
            self.ensure_collection()

    def _build_client(self) -> Any | None:
        if QdrantClient is None:
            return None
        try:
            client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
            client.get_collections()
            return client
        except Exception as exc:
            log_event("qdrant_client_init_failed", error=str(exc))
            return None

    def ensure_collection(self):
        if self.client is None or VectorParams is None or Distance is None:
            return
        try:
            existing = self.client.get_collections().collections
            names = {c.name for c in existing}
            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                log_event(
                    "qdrant_collection_created",
                    collection=self.collection,
                    vector_size=self.vector_size,
                    distance="cosine",
                )
            # Ensure payload indexes required by filtered searches are present.
            self._ensure_payload_indexes()
        except Exception as exc:
            log_event(
                "qdrant_collection_init_failed",
                collection=self.collection,
                error=str(exc),
            )
            raise

    def _ensure_payload_indexes(self):
        if self.client is None:
            return
        # Search filters rely on these payload keys in _search_filter().
        wanted = [
            ("user_id", "KEYWORD"),
            ("scope", "KEYWORD"),
            ("importance", "KEYWORD"),
            ("is_active", "BOOL"),
        ]
        for field_name, schema_name in wanted:
            try:
                schema = getattr(PayloadSchemaType, schema_name) if PayloadSchemaType is not None else schema_name.lower()
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
                log_event(
                    "qdrant_payload_index_ensured",
                    collection=self.collection,
                    field=field_name,
                    schema=schema_name.lower(),
                )
            except Exception as exc:
                # Benign when index already exists; still log for observability.
                message = str(exc)
                if "already exists" in message.lower():
                    log_event(
                        "qdrant_payload_index_exists",
                        collection=self.collection,
                        field=field_name,
                    )
                    continue
                log_event(
                    "qdrant_payload_index_ensure_failed",
                    collection=self.collection,
                    field=field_name,
                    error=message,
                )

    def _to_payload(self, memory: SemanticMemory) -> dict[str, Any]:
        importance = _importance_label(float(memory.importance_score))
        metadata = dict(memory.metadata or {})
        metadata.setdefault("importance", importance)
        return {
            "memory_id": memory.id,
            "user_id": memory.user_id,
            "message": memory.content,
            "content": memory.content,
            "timestamp": memory.created_at.isoformat(),
            "type": "conversation_memory",
            "memory_type": memory.memory_type,
            "scope": memory.scope,
            "importance": importance,
            "importance_score": float(memory.importance_score),
            "reinforcement_count": int(memory.reinforcement_count),
            "decay_factor": float(memory.decay_factor),
            "tags": list(memory.tags or []),
            "source_message_id": memory.source_message_id,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "embedding_model": memory.embedding_model,
            "embedding_provider": memory.embedding_provider,
            "is_active": bool(memory.is_active),
            "is_archived": bool(memory.is_archived),
            "archived_at": memory.archived_at.isoformat() if memory.archived_at else None,
            "metadata": metadata,
        }

    def _point_to_memory(self, point: Any) -> SemanticMemory:
        payload = dict(getattr(point, "payload", {}) or {})
        vector = list(getattr(point, "vector", []) or [])
        return SemanticMemory.from_dict(
            {
                "id": payload.get("memory_id") or str(getattr(point, "id", "")),
                "user_id": payload.get("user_id"),
                "content": payload.get("content") or payload.get("message") or "",
                "memory_type": payload.get("memory_type") or payload.get("type") or "fact",
                "scope": payload.get("scope") or "user",
                "importance_score": payload.get("importance_score", 0.5),
                "reinforcement_count": payload.get("reinforcement_count", 0),
                "decay_factor": payload.get("decay_factor", 0.985),
                "tags": payload.get("tags", []),
                "source_message_id": payload.get("source_message_id", ""),
                "embedding": vector,
                "embedding_model": payload.get("embedding_model", ""),
                "embedding_provider": payload.get("embedding_provider", ""),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "last_accessed": payload.get("last_accessed"),
                "is_active": payload.get("is_active", True),
                "is_archived": payload.get("is_archived", False),
                "archived_at": payload.get("archived_at"),
                "metadata": payload.get("metadata", {}),
            }
        )

    def _search_filter(
        self,
        user_id: str,
        scopes: Optional[list[str]] = None,
        importance_levels: Optional[list[str]] = None,
    ):
        if Filter is None or FieldCondition is None or MatchValue is None:
            return None
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
        if scopes and MatchAny is not None:
            must_conditions.append(FieldCondition(key="scope", match=MatchAny(any=scopes)))
        if importance_levels and MatchAny is not None:
            must_conditions.append(
                FieldCondition(
                    key="importance",
                    match=MatchAny(any=[str(x).strip().lower() for x in importance_levels if str(x).strip()]),
                )
            )
        return Filter(must=must_conditions)

    def upsert(self, memory: SemanticMemory):
        if self.client is None:
            raise RuntimeError("Qdrant client unavailable")
        if PointStruct is None:
            raise RuntimeError("Qdrant PointStruct unavailable")
        if not memory.embedding:
            raise ValueError("Memory embedding is required for Qdrant upsert")
        point = PointStruct(
            id=memory.id,
            vector=list(memory.embedding),
            payload=self._to_payload(memory),
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[point],
            wait=False,
        )

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        limit: int = 5,
        scopes: Optional[list[str]] = None,
        importance_levels: Optional[list[str]] = None,
        with_vectors: bool = True,
    ) -> list[tuple[SemanticMemory, float]]:
        if self.client is None:
            return []
        if not query_vector:
            return []
        query_filter = self._search_filter(user_id=user_id, scopes=scopes, importance_levels=importance_levels)
        points = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=max(1, int(limit or 5)),
            with_payload=True,
            with_vectors=with_vectors,
        )
        out: list[tuple[SemanticMemory, float]] = []
        for point in points:
            memory = self._point_to_memory(point)
            out.append((memory, float(getattr(point, "score", 0.0))))
        return out

    def has_duplicate(
        self,
        query_vector: list[float],
        user_id: str,
        threshold: float = 0.95,
    ) -> tuple[bool, float]:
        hits = self.search(
            query_vector=query_vector,
            user_id=user_id,
            limit=3,
            scopes=None,
            with_vectors=False,
        )
        if not hits:
            return False, 0.0
        top_score = float(hits[0][1])
        return top_score >= float(threshold), top_score

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        if self.client is None:
            return []
        if Filter is None or FieldCondition is None or MatchValue is None:
            return []
        all_rows: list[SemanticMemory] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=200,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            if not points:
                break
            for point in points:
                all_rows.append(self._point_to_memory(point))
            if offset is None:
                break
        all_rows.sort(key=lambda m: m.updated_at, reverse=True)
        return all_rows

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        if self.client is None:
            return False
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return False
            payload = dict(getattr(points[0], "payload", {}) or {})
            if str(payload.get("user_id") or "").strip() != str(user_id).strip():
                log_event("qdrant_delete_blocked_user_mismatch", memory_id=memory_id, requested_user=user_id)
                return False
            self.client.delete(collection_name=self.collection, points_selector=[memory_id], wait=True)
            return True
        except Exception as exc:
            log_event("qdrant_delete_failed", memory_id=memory_id, error=str(exc))
            return False


_qdrant_memory: QdrantMemoryService | None = None


def get_qdrant_memory_service() -> QdrantMemoryService:
    global _qdrant_memory
    if _qdrant_memory is None:
        _qdrant_memory = QdrantMemoryService()
    return _qdrant_memory
