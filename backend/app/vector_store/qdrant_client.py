"""
Qdrant integration helpers.
"""

from __future__ import annotations

from typing import Any

from backend.app.core.config import get_settings


def get_qdrant_client() -> Any | None:
    settings = get_settings()
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        return None

    kwargs = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    try:
        return QdrantClient(**kwargs)
    except Exception:
        return None

