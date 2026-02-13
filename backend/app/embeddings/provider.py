"""
Embedding provider abstraction with safe fallbacks.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from backend.app.core.config import get_settings


@dataclass
class EmbeddingVector:
    vector: list[float]
    model: str
    provider: str


class EmbeddingProvider:
    def embed(self, text: str) -> EmbeddingVector:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """
    Dependency-free fallback embedder.
    Deterministic and fast for local ranking when no external model is configured.
    """

    def __init__(self, dims: int, model: str = "local-hash-v1"):
        self.dims = max(32, dims)
        self.model = model

    def embed(self, text: str) -> EmbeddingVector:
        clean = (text or "").strip().lower()
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dims)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        return EmbeddingVector(vector=vec, model=self.model, provider="local")


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> EmbeddingVector:
        vector = self.model.encode((text or "").strip(), normalize_embeddings=True).tolist()
        return EmbeddingVector(vector=[float(x) for x in vector], model=self.model_name, provider="sentence_transformers")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from openai import OpenAI  # type: ignore
        import os

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> EmbeddingVector:
        response = self.client.embeddings.create(model=self.model_name, input=(text or "").strip())
        data = response.data[0].embedding
        return EmbeddingVector(vector=[float(x) for x in data], model=self.model_name, provider="openai")


_provider_cache: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider_cache
    if _provider_cache is not None:
        return _provider_cache

    settings = get_settings()
    provider = settings.embedding_provider
    model = settings.embedding_model
    dims = settings.embedding_dims

    try:
        if provider == "openai":
            _provider_cache = OpenAIEmbeddingProvider(model)
            return _provider_cache
        if provider in {"sentence_transformers", "bge", "local_model"}:
            _provider_cache = SentenceTransformerEmbeddingProvider(model)
            return _provider_cache
    except Exception:
        # Fall back to safe deterministic embeddings when optional deps are unavailable.
        pass

    _provider_cache = LocalHashEmbeddingProvider(dims=dims)
    return _provider_cache


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))

