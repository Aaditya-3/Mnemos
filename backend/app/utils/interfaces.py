"""
Shared interface contracts for dependency injection.
"""

from __future__ import annotations

from typing import Protocol

from backend.app.memory.models import SemanticMemory


class LLMClient(Protocol):
    def complete(self, prompt: str, timeout_seconds: float) -> str:
        ...


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class VectorStoreClient(Protocol):
    def upsert(self, memory: SemanticMemory):
        ...

    def search(self, vector: list[float], user_id: str, top_k: int, scopes: list[str] | None = None):
        ...

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        ...

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        ...

