"""
Celery worker wiring (optional in production).

If Celery/Redis are not configured, app still works with FastAPI BackgroundTasks.
"""

from __future__ import annotations

import os

from backend.app.tasks.memory_tasks import enqueue_ingest_message, run_compression, run_decay, run_reembedding


def _create_celery():
    try:
        from celery import Celery  # type: ignore
    except Exception:
        return None

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    app = Celery("mnemos", broker=redis_url, backend=redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = _create_celery()

if celery_app is not None:
    @celery_app.task(name="semantic.ingest_message")
    def ingest_message_task(user_id: str, message: str, source_message_id: str = "", scope: str | None = None):
        enqueue_ingest_message(user_id=user_id, message=message, source_message_id=source_message_id, scope=scope)

    @celery_app.task(name="semantic.decay_user")
    def decay_user_task(user_id: str):
        return run_decay(user_id=user_id)

    @celery_app.task(name="semantic.compress_user")
    def compress_user_task(user_id: str):
        return run_compression(user_id=user_id)

    @celery_app.task(name="semantic.reembed_user")
    def reembed_user_task(user_id: str, reason: str = "model_update"):
        return run_reembedding(user_id=user_id, reason=reason)
