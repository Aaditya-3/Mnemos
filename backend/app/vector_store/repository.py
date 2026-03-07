"""
Vector storage abstraction backed exclusively by Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from backend.app.memory.models import SemanticMemory
from backend.app.observability.logging import log_event
from memory.qdrant_memory import get_qdrant_memory_service


@dataclass
class VectorHit:
    memory: SemanticMemory
    similarity: float


class VectorStoreRepository:
    def upsert(self, memory: SemanticMemory):
        raise NotImplementedError

    def search(
        self,
        vector: list[float],
        user_id: str,
        top_k: int,
        scopes: Optional[list[str]] = None,
        importance_levels: Optional[list[str]] = None,
    ) -> list[VectorHit]:
        raise NotImplementedError

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        raise NotImplementedError

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        raise NotImplementedError

    def has_duplicate(self, query_vector: list[float], user_id: str, threshold: float = 0.95) -> tuple[bool, float]:
        raise NotImplementedError


class QdrantVectorStoreRepository(VectorStoreRepository):
    def __init__(self):
        self.backend = get_qdrant_memory_service()
        self.client = self.backend.client
        self.collection = self.backend.collection

    def upsert(self, memory: SemanticMemory):
        self.backend.upsert(memory)

    def search(
        self,
        vector: list[float],
        user_id: str,
        top_k: int,
        scopes: Optional[list[str]] = None,
        importance_levels: Optional[list[str]] = None,
    ) -> list[VectorHit]:
        rows = self.backend.search(
            query_vector=vector,
            user_id=user_id,
            limit=top_k,
            scopes=scopes,
            importance_levels=importance_levels,
            with_vectors=True,
        )
        return [VectorHit(memory=memory, similarity=score) for memory, score in rows]

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        return self.backend.list_user_memories(user_id)

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        return self.backend.delete_memory(user_id=user_id, memory_id=memory_id)

    def has_duplicate(self, query_vector: list[float], user_id: str, threshold: float = 0.95) -> tuple[bool, float]:
        return self.backend.has_duplicate(query_vector=query_vector, user_id=user_id, threshold=threshold)


def iter_memory_tokens(text: str) -> Iterable[str]:
    for token in (text or "").lower().replace("\n", " ").split():
        t = token.strip(".,!?;:\"'()[]{}")
        if len(t) >= 3:
            yield t


_repo_cache: VectorStoreRepository | None = None


def get_vector_store() -> VectorStoreRepository:
    global _repo_cache
    if _repo_cache is not None:
        return _repo_cache

    repo = QdrantVectorStoreRepository()
    if repo.client is None:
        raise RuntimeError("Qdrant is unavailable. Set QDRANT_URL and QDRANT_API_KEY.")
    log_event("vector_store_ready", backend="qdrant", collection=repo.collection)
    _repo_cache = repo
    return _repo_cache
