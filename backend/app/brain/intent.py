"""
Rule-based intent classification for pre-LLM orchestration.
"""

from __future__ import annotations

import re

from backend.app.brain.models import Intent


_GREETING_RE = re.compile(r"^(hi|hello|hey|yo|good\s+morning|good\s+afternoon|good\s+evening)\b[!.,\s]*$", re.I)
_MEMORY_QUERY_PATTERNS = [
    r"\bwhat\s+is\s+my\b",
    r"\bwhat\s+are\s+my\b",
    r"\bwhen\s+(?:is|was)\s+my\b",
    r"\bwho\s+am\s+i\b",
    r"\bwhere\s+do\s+i\s+study\b",
    r"\b(?:what|which)\s+year\s+am\s+i\b",
    r"\b(?:say|tell|show|remind|recall|remember|repeat)\s+(?:me\s+)?my\s+(?:name|college|university|year|graduation\s+year|year\s+of\s+graduation)\b",
    r"\b(?:about|regarding)\s+my\s+(?:name|college|university|year|graduation\s+year|year\s+of\s+graduation)\b",
    r"\bdo\s+you\s+remember\b",
    r"\bwhat\s+do\s+i\s+like\b",
    r"\bwhat\s+is\s+my\s+favorite\b",
]
_MEMORY_UPDATE_PATTERNS = [
    r"\bi\s+like\b",
    r"\bi\s+love\b",
    r"\bi\s+prefer\b",
    r"\bi\s+dislike\b",
    r"\bi\s+hate\b",
    r"\bmy\s+\w+\s+(?:is|was|will\s+be|on)\b",
    r"\bi\s+have\b",
]
_GUESS_PATTERNS = [
    r"\bguess\b",
    r"\bwhat\s+would\s+i\s+like\b",
    r"\bguess\s+my\b",
]

_TRAIT_STYLE_RE = re.compile(r"\bi(?:\s+am|'m)\s+(.+)$", re.I)
_TRANSIENT_LEAD_RE = re.compile(r"^[a-z]+ing\b", re.I)


def _looks_like_trait_update(text: str) -> bool:
    """
    Accept stable self-descriptions while avoiding transient clarification phrases
    such as "I'm asking about ...".
    """
    m = _TRAIT_STYLE_RE.search((text or "").strip())
    if not m:
        return False
    phrase = (m.group(1) or "").strip(" .?!,")
    if not phrase:
        return False
    tokens = re.findall(r"[a-z0-9']+", phrase.lower())
    if not tokens or len(tokens) > 8:
        return False
    if _TRANSIENT_LEAD_RE.match(tokens[0] or ""):
        return False
    if {"asking", "question", "clarifying"}.intersection(tokens):
        return False
    return True


def classify_intent(message: str) -> Intent:
    text = (message or "").strip()
    if not text:
        return Intent.OTHER
    if _GREETING_RE.match(text):
        return Intent.GREETING

    lowered = text.lower()
    if any(re.search(p, lowered) for p in _GUESS_PATTERNS):
        return Intent.GUESS_REQUEST
    if any(re.search(p, lowered) for p in _MEMORY_QUERY_PATTERNS):
        return Intent.MEMORY_QUERY
    if any(re.search(p, lowered) for p in _MEMORY_UPDATE_PATTERNS):
        return Intent.MEMORY_UPDATE
    if _looks_like_trait_update(lowered):
        return Intent.MEMORY_UPDATE
    if "?" in lowered:
        return Intent.FACTUAL_QUESTION
    return Intent.OTHER
