"""
Memory Schema Definition

Defines the structure for long-term memory storage.
Memories are persistent, reusable user information.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import uuid


@dataclass
class Memory:
    """Represents a single memory entry."""
    
    id: str
    user_id: str
    type: Literal["preference", "constraint", "fact"]
    key: str
    value: str
    confidence: float
    metadata: dict | None
    created_at: datetime
    last_updated: datetime
    memory_type: str = ""
    embedding: list[float] | None = None
    importance_score: float = 0.5
    
    def __post_init__(self):
        """Validate memory data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.type not in ["preference", "constraint", "fact"]:
            raise ValueError("Type must be 'preference', 'constraint', or 'fact'")

        try:
            self.importance_score = float(self.importance_score)
        except Exception:
            self.importance_score = 0.5
        self.importance_score = max(0.0, min(1.0, self.importance_score))
        if self.embedding is None:
            self.embedding = []
    
    @classmethod
    def create(
        cls,
        user_id: str,
        type: Literal["preference", "constraint", "fact"],
        key: str,
        value: str,
        confidence: float = 0.7,
        metadata: dict | None = None,
        memory_type: str = "",
        embedding: list[float] | None = None,
        importance_score: float = 0.5,
    ) -> "Memory":
        """Create a new memory with auto-generated ID and timestamps."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=type,
            key=key,
            value=value,
            confidence=confidence,
            metadata=metadata,
            created_at=now,
            last_updated=now,
            memory_type=memory_type,
            embedding=list(embedding or []),
            importance_score=importance_score,
        )
    
    def to_dict(self) -> dict:
        """Convert memory to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "memory_type": self.memory_type,
            "embedding": list(self.embedding or []),
            "importance_score": self.importance_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create memory from dictionary."""
        created_at_raw = data.get("created_at")
        updated_at_raw = data.get("last_updated")
        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except Exception:
            created_at = datetime.now()
        try:
            last_updated = datetime.fromisoformat(str(updated_at_raw))
        except Exception:
            last_updated = created_at

        embedding = data.get("embedding") or []
        safe_embedding = []
        if isinstance(embedding, list):
            for val in embedding:
                try:
                    safe_embedding.append(float(val))
                except Exception:
                    continue

        return cls(
            id=data["id"],
            user_id=data.get("user_id", "guest"),
            type=data["type"],
            key=data["key"],
            value=data["value"],
            confidence=data["confidence"],
            metadata=data.get("metadata"),
            created_at=created_at,
            last_updated=last_updated,
            memory_type=str(data.get("memory_type") or ""),
            embedding=safe_embedding,
            importance_score=float(data.get("importance_score", data.get("confidence", 0.5))),
        )
