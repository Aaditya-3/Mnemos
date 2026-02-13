"""
Semantic memory domain models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SemanticMemory:
    id: str
    user_id: str
    content: str
    memory_type: str
    scope: str
    importance_score: float
    reinforcement_count: int
    decay_factor: float
    tags: list[str] = field(default_factory=list)
    source_message_id: str = ""
    embedding: list[float] = field(default_factory=list)
    embedding_model: str = ""
    embedding_provider: str = ""
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    last_accessed: datetime | None = None
    is_active: bool = True
    is_archived: bool = False
    archived_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        user_id: str,
        content: str,
        memory_type: str,
        scope: str,
        importance_score: float,
        decay_factor: float,
        tags: list[str] | None = None,
        source_message_id: str = "",
    ) -> "SemanticMemory":
        now = utcnow()
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content.strip(),
            memory_type=memory_type,
            scope=scope,
            importance_score=max(0.0, min(1.0, importance_score)),
            reinforcement_count=0,
            decay_factor=max(0.01, min(1.0, decay_factor)),
            tags=tags or [],
            source_message_id=source_message_id,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "scope": self.scope,
            "importance_score": self.importance_score,
            "reinforcement_count": self.reinforcement_count,
            "decay_factor": self.decay_factor,
            "tags": self.tags,
            "source_message_id": self.source_message_id,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticMemory":
        def _dt(key: str, fallback: datetime | None = None) -> datetime | None:
            raw = data.get(key)
            if not raw:
                return fallback
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return fallback

        created = _dt("created_at", utcnow()) or utcnow()
        updated = _dt("updated_at", created) or created
        accessed = _dt("last_accessed", _dt("last_accessed_at", None))
        archived_at = _dt("archived_at", None)
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())),
            user_id=str(data.get("user_id") or "guest"),
            content=str(data.get("content") or "").strip(),
            memory_type=str(data.get("memory_type") or "fact"),
            scope=str(data.get("scope") or "user"),
            importance_score=float(data.get("importance_score", 0.5)),
            reinforcement_count=int(data.get("reinforcement_count", 0) or 0),
            decay_factor=max(0.01, min(1.0, float(data.get("decay_factor", 0.985)))),
            tags=[str(x) for x in (data.get("tags") or []) if str(x).strip()],
            source_message_id=str(data.get("source_message_id") or ""),
            embedding=[float(x) for x in (data.get("embedding") or [])],
            embedding_model=str(data.get("embedding_model") or ""),
            embedding_provider=str(data.get("embedding_provider") or ""),
            created_at=created,
            updated_at=updated,
            last_accessed=accessed,
            is_active=bool(data.get("is_active", True)),
            is_archived=bool(data.get("is_archived", False)),
            archived_at=archived_at,
            metadata=dict(data.get("metadata") or {}),
        )
