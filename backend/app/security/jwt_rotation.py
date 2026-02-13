"""
JWT refresh token rotation utilities.

This module keeps an in-memory store by default. In production this should
be backed by Redis or database for multi-instance consistency.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock

from jose import jwt


JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TTL_MIN = int(os.getenv("JWT_ACCESS_TTL_MIN", "30"))
REFRESH_TTL_DAYS = int(os.getenv("JWT_REFRESH_TTL_DAYS", "7"))


def _hash_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest


class RefreshTokenStore:
    def __init__(self):
        self._lock = Lock()
        self._items: dict[str, dict] = {}

    def put(self, user_id: str, refresh_token: str, expires_at: datetime):
        with self._lock:
            self._items[_hash_token(refresh_token)] = {
                "user_id": user_id,
                "expires_at": expires_at,
            }

    def pop_valid(self, user_id: str, refresh_token: str) -> bool:
        key = _hash_token(refresh_token)
        now = datetime.now(timezone.utc)
        with self._lock:
            item = self._items.get(key)
            if not item:
                return False
            if item["user_id"] != user_id:
                return False
            if item["expires_at"] < now:
                self._items.pop(key, None)
                return False
            self._items.pop(key, None)
            return True


refresh_store = RefreshTokenStore()


def issue_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TTL_MIN)).timestamp()),
        "jti": str(uuid.uuid4()),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def issue_refresh_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=REFRESH_TTL_DAYS)).timestamp()),
        "jti": str(uuid.uuid4()),
        "type": "refresh",
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    refresh_store.put(user_id=user_id, refresh_token=token, expires_at=now + timedelta(days=REFRESH_TTL_DAYS))
    return token


def rotate_tokens(user_id: str, refresh_token: str) -> tuple[str, str]:
    if not refresh_store.pop_valid(user_id, refresh_token):
        raise ValueError("Invalid refresh token")
    access = issue_access_token(user_id)
    refresh = issue_refresh_token(user_id)
    return access, refresh

