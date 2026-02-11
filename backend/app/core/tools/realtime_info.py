"""
Realtime Info Helpers

Provides lightweight live web access for dynamic queries (exchange rates and
general web snippets) without hardcoding personal facts.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import httpx


_CURRENCY_ALIASES = {
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
    "cad": "CAD",
    "aud": "AUD",
    "sgd": "SGD",
}


def should_fetch_realtime(user_message: str) -> bool:
    msg = (user_message or "").lower().strip()
    if not msg:
        return False

    markers = (
        "today",
        "current",
        "latest",
        "right now",
        "exchange rate",
        "convert",
        "price of",
        "stock price",
        "weather",
        "news",
        "usd",
        "inr",
        "eur",
        "gbp",
        "jpy",
        "btc",
        "eth",
    )
    return any(m in msg for m in markers)


def get_realtime_context(user_message: str) -> Optional[str]:
    msg = (user_message or "").strip()
    if not msg:
        return None

    currency = _try_currency_conversion(msg)
    if currency:
        return currency

    snippet = _try_duckduckgo_summary(msg)
    if snippet:
        return snippet

    return None


def _normalize_currency(token: str) -> Optional[str]:
    t = (token or "").strip().lower()
    if not t:
        return None
    if t in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[t]
    if re.fullmatch(r"[a-zA-Z]{3}", t):
        return t.upper()
    return None


def _try_currency_conversion(user_message: str) -> Optional[str]:
    msg = user_message.lower().strip()

    patterns = [
        r"^\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<base>[a-zA-Z]{3}|[a-zA-Z]+)\s*(?:to|in)\s*(?P<target>[a-zA-Z]{3}|[a-zA-Z]+)\s*$",
        r"^\s*convert\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<base>[a-zA-Z]{3}|[a-zA-Z]+)\s*(?:to|in)\s*(?P<target>[a-zA-Z]{3}|[a-zA-Z]+)\s*$",
        r"^\s*(?P<base>[a-zA-Z]{3}|[a-zA-Z]+)\s*(?:to|in)\s*(?P<target>[a-zA-Z]{3}|[a-zA-Z]+)\s*$",
    ]

    matched = None
    for p in patterns:
        m = re.match(p, msg)
        if m:
            matched = m
            break
    if not matched:
        return None

    amount_raw = matched.groupdict().get("amount")
    amount = float(amount_raw) if amount_raw is not None else 1.0
    base = _normalize_currency(matched.group("base"))
    target = _normalize_currency(matched.group("target"))
    if not base or not target:
        return None

    try:
        url = "https://api.frankfurter.app/latest"
        params = {"from": base, "to": target}
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        rates = data.get("rates", {})
        rate = rates.get(target)
        if not isinstance(rate, (int, float)):
            return None
        converted = amount * float(rate)
        rate_date = data.get("date") or datetime.now(timezone.utc).date().isoformat()
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"Live exchange data: {amount:g} {base} = {converted:.2f} {target} "
            f"(1 {base} = {float(rate):.4f} {target}, rate date {rate_date}, fetched {fetched_at}, "
            "source: frankfurter.app)."
        )
    except Exception:
        return None


def _try_duckduckgo_summary(query: str) -> Optional[str]:
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    answer = (data.get("Answer") or "").strip()
    abstract = (data.get("AbstractText") or "").strip()
    heading = (data.get("Heading") or "").strip()

    text = answer or abstract
    if not text:
        related = data.get("RelatedTopics") or []
        for item in related:
            if isinstance(item, dict) and item.get("Text"):
                text = str(item["Text"]).strip()
                break
    if not text:
        return None

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"{heading}: " if heading else ""
    return f"Live web snippet: {title}{text} (fetched {fetched_at}, source: DuckDuckGo Instant Answer API)."
