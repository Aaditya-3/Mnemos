"""
Celery beat schedule helpers.
"""

from __future__ import annotations

from backend.app.tasks.worker import celery_app


def configure_periodic_tasks():
    if celery_app is None:
        return

    celery_app.conf.beat_schedule = {
        "semantic-decay-nightly": {
            "task": "semantic.decay_user",
            "schedule": 24 * 60 * 60,
            "args": ("system",),
        },
        "semantic-compress-nightly": {
            "task": "semantic.compress_user",
            "schedule": 24 * 60 * 60,
            "args": ("system",),
        },
        "semantic-reembed-weekly": {
            "task": "semantic.reembed_user",
            "schedule": 7 * 24 * 60 * 60,
            "args": ("system", "scheduled_reembed"),
        },
    }
