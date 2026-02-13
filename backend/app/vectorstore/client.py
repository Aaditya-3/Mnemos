"""
Vector store adapter exposing protocol-compatible client.
"""

from __future__ import annotations

from backend.app.vector_store.repository import get_vector_store


def get_vectorstore_client():
    return get_vector_store()

