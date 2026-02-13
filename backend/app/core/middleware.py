"""
Middleware and request guards for production behavior.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.core.config import get_settings
from backend.app.observability.logging import log_event
from backend.app.observability.metrics import metrics
from backend.app.security.replay import replay_protector


_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"reveal\s+(system|developer)\s+prompt",
    r"bypass\s+safety",
    r"act\s+as\s+system",
    r"show\s+me\s+your\s+hidden\s+instructions",
    r"print\s+the\s+prompt",
    r"jailbreak",
    r"developer\s+mode",
    r"BEGIN\s+SYSTEM\s+PROMPT",
]


@dataclass
class _RateWindow:
    hits: deque[float]
    lock: Lock


class InMemoryRateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self._store: dict[str, _RateWindow] = defaultdict(lambda: _RateWindow(deque(), Lock()))

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = self._store[key]
        with bucket.lock:
            while bucket.hits and (now - bucket.hits[0]) > 60:
                bucket.hits.popleft()
            if len(bucket.hits) >= self.max_requests_per_minute:
                return False
            bucket.hits.append(now)
            return True


_settings = get_settings()
_user_limiter = InMemoryRateLimiter(max_requests_per_minute=_settings.max_requests_per_minute)
_ip_limiter = InMemoryRateLimiter(max_requests_per_minute=max(30, _settings.max_requests_per_minute * 2))


def validate_message_payload(message: str):
    settings = get_settings()
    if not isinstance(message, str):
        raise HTTPException(status_code=400, detail="Message must be a string")
    text = message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(text) > settings.max_prompt_chars:
        raise HTTPException(status_code=413, detail=f"Message too long (max {settings.max_prompt_chars} chars)")
    if settings.enable_prompt_injection_guard:
        lowered = text.lower()
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                raise HTTPException(status_code=400, detail="Potential prompt injection pattern detected")


def setup_middleware(app: FastAPI):
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = start

        user_id = request.headers.get("X-User-ID", "anonymous").strip() or "anonymous"
        ip = request.client.host if request.client else "local"
        user_rate_key = f"user:{user_id}"
        ip_rate_key = f"ip:{ip}"
        if not _user_limiter.allow(user_rate_key) or not _ip_limiter.allow(ip_rate_key):
            log_event("rate_limit_triggered", request_id=request_id, user_id=user_id, ip=ip, path=request.url.path)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        nonce = (request.headers.get("X-Request-Nonce") or "").strip()
        if nonce and not replay_protector.accept(user_id=user_id, nonce=nonce):
            log_event("replay_rejected", request_id=request_id, user_id=user_id, ip=ip)
            return JSONResponse(status_code=409, content={"detail": "Replay detected"})

        response = await call_next(request)
        elapsed = time.perf_counter() - start

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = response.headers.get(
            "Cache-Control",
            "no-store, no-cache, must-revalidate, max-age=0",
        )

        metrics.inc("total_requests", 1)
        metrics.observe("http_request_latency_seconds", elapsed)
        metrics.mark_user_active(user_id)
        log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            user_id=user_id,
            latency_ms=round(elapsed * 1000, 2),
            status_code=response.status_code,
        )
        return response
