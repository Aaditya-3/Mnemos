"""
Embedding client adapter for DI.
"""

from __future__ import annotations

from backend.app.embeddings.provider import get_embedding_provider
from backend.app.utils.interfaces import EmbeddingClient


class DefaultEmbeddingClient(EmbeddingClient):
    def __init__(self):
        self.provider = get_embedding_provider()

    def embed(self, text: str) -> list[float]:
        return self.provider.embed(text).vector


_client: DefaultEmbeddingClient | None = None


def get_embedding_client() -> DefaultEmbeddingClient:
    global _client
    if _client is None:
        _client = DefaultEmbeddingClient()
    return _client

