"""
Generic preference reasoning helpers.

Works across domains (anime, food, drink, sports, etc.) without hardcoding
entity/domain lists.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from backend.app.core.llm.groq_client import generate_response
from backend.app.core.tools.realtime_info import get_realtime_context


def _canonicalize_subject(subject_label: str) -> tuple[str, str]:
    subject_label = re.sub(r"\s+", " ", (subject_label or "").strip(" .?!,").lower())
    if not subject_label:
        return "", ""

    if subject_label in {"char", "character", "fav char", "favorite character", "favourite character"}:
        return "character", "character"
    if subject_label in {"mc", "main character", "main char"}:
        return "character", "character"
    if subject_label in {"soft drink", "beverage"}:
        return "drink", "drink"
    return re.sub(r"[^a-z0-9]+", "_", subject_label).strip("_"), subject_label


def parse_preference_query(user_message: str) -> tuple[str, str, str]:
    """
    Returns (subject_key, subject_label, entity_key).
    """
    msg = (user_message or "").strip().lower()
    if not msg:
        return "", "", ""

    patterns = [
        r"(?:which|what)\s+(?:is|s)\s+my\s+(?:fav(?:ou)?rite|fav)\s+([a-z0-9' \-]+?)\s+(?:in|from|of)\s+([a-z0-9' \-]+)",
        r"my\s+(?:fav(?:ou)?rite|fav)\s+([a-z0-9' \-]+?)\s+(?:in|from|of)\s+([a-z0-9' \-]+)",
        r"which\s+([a-z0-9' \-]+?)\s+do\s+i\s+like\s+(?:in|from|of)\s+([a-z0-9' \-]+)",
        r"what\s+([a-z0-9' \-]+?)\s+do\s+i\s+like\s+(?:in|from|of)\s+([a-z0-9' \-]+)",
        r"(?:which|what)\s+(?:is|s)\s+my\s+(?:fav(?:ou)?rite|fav)\s+([a-z0-9' \-]+)",
        r"my\s+(?:fav(?:ou)?rite|fav)\s+([a-z0-9' \-]+)",
        r"which\s+([a-z0-9' \-]+?)\s+do\s+i\s+like",
        r"what\s+([a-z0-9' \-]+?)\s+do\s+i\s+like",
    ]

    for pattern in patterns:
        m = re.search(pattern, msg)
        if not m:
            continue

        subject_label = re.sub(r"\s+", " ", m.group(1)).strip(" .?!,")
        entity_label = ""
        if len(m.groups()) > 1 and m.group(2):
            entity_label = re.sub(r"\s+", " ", m.group(2)).strip(" .?!,")

        subject_key, normalized_label = _canonicalize_subject(subject_label)
        entity_key = re.sub(r"[^a-z0-9]+", "_", entity_label).strip("_") if entity_label else ""
        if subject_key:
            return subject_key, normalized_label, entity_key

    return "", "", ""


def classify_memory_for_query(memory, subject_key: str, entity_key: str) -> tuple[bool, bool]:
    """
    Returns (is_specific, is_global_rule) for this query context.
    """
    md = memory.metadata or {}
    if memory.confidence < 0.5:
        return False, False
    if not md.get("domain") or not md.get("category") or not md.get("scope"):
        return False, False

    key_l = (memory.key or "").lower()
    value_l = (memory.value or "").lower()
    rule_type = str(md.get("rule_type", "")).lower()
    md_subject = str(md.get("subject", "")).lower()
    md_category = str(md.get("category", "")).lower()
    scope = str(md.get("scope", "")).upper()

    subject_tokens = set(re.findall(r"[a-z0-9]+", subject_key))
    if "character" in subject_tokens:
        subject_tokens.update({"char", "mc", "main"})

    mem_tokens = set(re.findall(r"[a-z0-9]+", key_l + " " + value_l))
    category_tokens = set(re.findall(r"[a-z0-9]+", md_category))

    subject_related = (
        (not subject_tokens or bool(subject_tokens.intersection(mem_tokens)))
        and (not subject_tokens or bool(subject_tokens.intersection(category_tokens.union(mem_tokens))))
    )
    if md_category and subject_key and subject_key in md_category:
        subject_related = True

    is_global_rule = False
    if scope == "GLOBAL_SCOPE" and rule_type in {"universal", "generic"}:
        is_global_rule = subject_related
    elif scope == "GLOBAL_SCOPE" and "global" in key_l and subject_related:
        is_global_rule = True

    is_specific = False
    if entity_key:
        if scope == "LOCAL_SCOPE" and md_subject and md_subject == entity_key and subject_related:
            is_specific = True
        elif scope == "LOCAL_SCOPE" and subject_related and (key_l.endswith(f".{entity_key}") or f"_{entity_key}" in key_l):
            is_specific = True
        elif scope == "LOCAL_SCOPE" and subject_related and entity_key in value_l:
            is_specific = True
    else:
        if subject_related and memory.type in {"preference", "fact"} and scope in {"GLOBAL_SCOPE", "PROFILE"}:
            is_specific = True

    return is_specific, is_global_rule


def external_reasoning_hint(subject_label: str, entity_key: str) -> Optional[str]:
    if not subject_label:
        return None
    query = subject_label
    if entity_key:
        query = f"{subject_label} in {entity_key.replace('_', ' ')}"
    return get_realtime_context(query)


def _extract_candidate_value(text: str) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None

    raw = re.sub(r"^\s*live web snippet:\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*\(fetched[^)]*\)\s*$", "", raw, flags=re.IGNORECASE)

    patterns = [
        r"\b(?:is|are|was|were)\s+([^.;:()]{2,100})",
        r":\s*([^.;()]{2,100})",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;")
            if 1 < len(candidate) <= 80 and len(candidate.split()) <= 8:
                return candidate

    m = re.search(r"\b([A-Za-z][A-Za-z0-9' -]{1,80})\b", raw)
    if m:
        candidate = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;")
        if 1 < len(candidate) <= 80 and len(candidate.split()) <= 8:
            return candidate
    return None


def _is_concrete_guess(value: str, entity_label: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return False
    blocked_phrases = {
        "mc",
        "main character",
        "protagonist",
        "lead",
        "character",
        "anime",
        "series",
        "show",
    }
    if v in blocked_phrases:
        return False
    if v == (entity_label or "").strip().lower():
        return False
    if any(p in v for p in {"main character", "protagonist", "anime television", "animated series"}):
        return False
    return True


def _hint_mentions_entity(hint_text: str, entity_label: str) -> bool:
    text_l = (hint_text or "").lower()
    entity_tokens = [t for t in re.findall(r"[a-z0-9]+", (entity_label or "").lower()) if len(t) >= 3]
    if not entity_tokens:
        return False
    if any(t in text_l for t in entity_tokens):
        return True

    for token in entity_tokens:
        if not re.fullmatch(r"[a-z]{2,6}", token):
            continue
        words = re.findall(r"[a-z]+", text_l)
        for i in range(len(words)):
            for n in range(2, min(6, len(words) - i) + 1):
                window = words[i : i + n]
                initials = "".join(w[0] for w in window if w)
                if initials == token:
                    return True
    return False


def resolve_external_guess(subject_label: str, entity_key: str, global_values: list[str]) -> Optional[str]:
    if not entity_key:
        return None

    entity_label = entity_key.replace("_", " ")
    queries = [f"{subject_label} in {entity_label}", f"{entity_label} {subject_label}"]
    for rule in global_values[:3]:
        rule_text = re.sub(r"\s+", " ", (rule or "").strip()).lower()
        if not rule_text:
            continue
        queries.append(f"{subject_label} in {entity_label} with preference {rule_text}")
        queries.append(f"{entity_label} {subject_label} {rule_text}")

    for query in queries:
        hint = get_realtime_context(query)
        if not hint:
            continue
        value = _extract_candidate_value(hint)
        if value and _is_concrete_guess(value, entity_label) and _hint_mentions_entity(hint, entity_label):
            return value
    return None


def resolve_rule_with_model(subject_label: str, entity_key: str, global_values: list[str]) -> Optional[str]:
    """
    Generic rule execution with model knowledge (no domain hardcoding).
    Returns a concrete value only when model confidence is sufficiently high.
    """
    if not entity_key or not subject_label or not global_values:
        return None

    entity = entity_key.replace("_", " ").strip()
    rule_lines = "\n".join(f"- {r}" for r in global_values[:3])
    prompt = (
        "You are a strict resolver.\n"
        "Given user preference rules and a target entity, output the most likely specific value.\n"
        "If unsure, return null.\n"
        "Return ONLY compact JSON: {\"value\": \"...\", \"confidence\": 0.0-1.0} or null.\n"
        f"Subject: {subject_label}\n"
        f"Entity: {entity}\n"
        f"Rules:\n{rule_lines}\n"
    )
    try:
        raw = (generate_response(prompt) or "").strip()
    except Exception:
        return None
    if not raw:
        return None

    cleaned = raw
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.lower() == "null":
        return None

    value = None
    confidence = 0.0
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            value = str(data.get("value", "")).strip() or None
            conf_raw = data.get("confidence", 0.0)
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.0
    except Exception:
        first_line = cleaned.splitlines()[0].strip(" .,:;\"'")
        if first_line:
            value = first_line
            confidence = 0.6

    if not value:
        return None

    v = value.lower()
    blocked = {
        "unknown",
        "not sure",
        "cannot determine",
        "mc",
        "main character",
        "protagonist",
        "character",
    }
    if v in blocked:
        return None
    if any(token in v for token in ["not sure", "unknown", "depends"]):
        return None
    if len(value) > 80 or len(value.split()) > 8:
        return None
    if confidence < 0.72:
        return None

    return value
