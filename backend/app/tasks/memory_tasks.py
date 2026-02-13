"""
Background task entrypoints for semantic memory operations.
"""

from __future__ import annotations

from backend.app.observability.logging import log_event
from backend.app.services.semantic_memory_service import get_semantic_memory_service


def enqueue_ingest_message(user_id: str, message: str, source_message_id: str = "", scope: str | None = None):
    service = get_semantic_memory_service()
    try:
        memory = service.ingest_message(user_id=user_id, message=message, source_message_id=source_message_id, scope=scope)
        if memory:
            log_event("task_ingest_done", user_id=user_id, memory_id=memory.id)
    except Exception as exc:
        log_event("task_ingest_failed", user_id=user_id, error=str(exc))


def run_decay(user_id: str):
    service = get_semantic_memory_service()
    try:
        result = service.apply_decay(user_id=user_id)
        log_event("task_decay_done", user_id=user_id, **result)
        return result
    except Exception as exc:
        log_event("task_decay_failed", user_id=user_id, error=str(exc))
        return {"updated": 0, "deactivated": 0, "error": str(exc)}


def run_compression(user_id: str):
    service = get_semantic_memory_service()
    try:
        result = service.compress_user_memories(user_id=user_id)
        log_event("task_compression_done", user_id=user_id, **result)
        return result
    except Exception as exc:
        log_event("task_compression_failed", user_id=user_id, error=str(exc))
        return {"compressed": 0, "error": str(exc)}

