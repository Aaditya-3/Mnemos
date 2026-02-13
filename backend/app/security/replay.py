"""
Replay protection helper based on nonce cache.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import Lock


class ReplayProtector:
    def __init__(self, ttl_seconds: int = 120):
        self.ttl_seconds = max(10, ttl_seconds)
        self._lock = Lock()
        self._seen: dict[str, dict[str, datetime]] = defaultdict(dict)

    def accept(self, user_id: str, nonce: str) -> bool:
        uid = (user_id or "").strip() or "anonymous"
        token = (nonce or "").strip()
        if not token:
            return True
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.ttl_seconds)
        with self._lock:
            user_cache = self._seen[uid]
            expired = [k for k, ts in user_cache.items() if ts < cutoff]
            for k in expired:
                user_cache.pop(k, None)
            if token in user_cache:
                return False
            user_cache[token] = now
            return True


replay_protector = ReplayProtector()

