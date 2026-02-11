"""
Memory Extractor

Strict extractor that stores only explicit, stable user facts/preferences.
"""

import json
import re
from typing import Optional

from backend.app.core.llm.groq_client import generate_response
from .memory_schema import Memory
from .memory_store import get_memory_store

SCHEMA_PREFIXES = {
    "core_identity",
    "lifestyle",
    "entertainment",
    "technical_stack",
}

NEVER_EXTRACT_VALUES = {
    "aria",
    "i understand",
    "i get it",
    "as an ai",
}

NEVER_EXTRACT_KEYS = {
    "assistant_name",
    "ai_name",
}

INVALID_NAME_WORDS = {
    "pursuing",
    "working",
    "studying",
    "trying",
    "doing",
    "going",
    "learning",
}

CRITICAL_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "my",
    "i",
    "me",
    "it",
    "this",
    "that",
    "be",
    "with",
    "as",
    "at",
    "from",
    "by",
    "or",
    "if",
    "then",
    "must",
    "should",
}

DAY_WORDS = {
    "today",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def _normalize_for_extraction(user_message: str) -> str:
    """
    Normalize short confirmation wrappers into declarative text so extractor
    can retain facts across follow-up turns like "yeah".
    """
    raw = re.sub(r"\s+", " ", (user_message or "").strip())
    if not raw:
        return raw

    m = re.match(r"i\s+(confirm|reject)\s+this:\s*(.+)$", raw, flags=re.IGNORECASE)
    if not m:
        return raw

    stance = (m.group(1) or "").strip().lower()
    statement = re.sub(r"\s+", " ", (m.group(2) or "").strip()).rstrip(".!? ")
    if not statement:
        return raw
    if stance == "reject":
        # Keep explicit rejection untouched so we do not store false positives.
        return raw

    stmt = statement.lower()
    stmt = stmt.replace("?", " ")
    stmt = re.sub(r"^(is|are|am|was|were|do|does|did|can|could|will|would|should|have|has|had)\s+", "", stmt)
    stmt = re.sub(r"\byour\b", "my", stmt)
    stmt = re.sub(r"\bfor you\b", "for me", stmt)
    stmt = re.sub(r"\s+", " ", stmt).strip()
    if not stmt:
        return raw

    # Convert "tomorrow a class day for me" -> "i have class day tomorrow"
    day_pattern = r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    dm = re.match(rf"(?P<day>{day_pattern})\s+(?P<event>.+)$", stmt)
    if dm:
        day = dm.group("day").strip().lower()
        event = re.sub(r"^(?:a|an|the)\s+", "", dm.group("event").strip().lower())
        event = event.replace("?", " ")
        event = re.sub(r"\s+for\s+me$", "", event).strip()
        if event:
            return f"i have {event} {day}"

    return f"i confirm {stmt}"


def _is_general_statement(user_message: str) -> bool:
    msg_l = (user_message or "").lower()
    if not msg_l:
        return False
    markers = {"always", "every", "all", "any", "usually", "generally", "in general"}
    return any(m in msg_l for m in markers)


def _apply_general_confidence(memory: Memory, user_message: str) -> Memory:
    """
    For generalized statements, keep confidence at 0.5 so they are treated as
    broad/soft rules rather than hard facts.
    Critical instructions are exempt and should stay high-confidence.
    """
    md = memory.metadata or {}
    if str(md.get("importance", "")).lower() == "critical":
        return memory
    if not _is_general_statement(user_message):
        return memory
    scope = str(md.get("scope", "")).upper()
    if scope == "GLOBAL_SCOPE":
        memory.confidence = min(memory.confidence, 0.5)
    return memory


def _is_memory_supported_by_message(key: str, value: str, user_message: str) -> bool:
    msg = (user_message or "").lower()
    key_l = (key or "").strip().lower()
    value_l = (value or "").strip().lower()
    if not msg or not value_l:
        return False
    if key_l in NEVER_EXTRACT_KEYS:
        return False
    if value_l in NEVER_EXTRACT_VALUES:
        return False

    if value_l in msg:
        return True

    trusted_keys = {
        "name",
        "role",
        "occupation",
        "location",
        "anime_preference",
        "language_preference",
        "tool_preference",
        "preference",
    }
    if key_l in trusted_keys:
        tokens = [t for t in value_l.split() if len(t) > 2]
        if tokens and any(t in msg for t in tokens):
            return True

    return False


def _looks_like_question(msg_l: str) -> bool:
    if not msg_l:
        return False
    if "?" in msg_l:
        return True
    starters = (
        "what ",
        "which ",
        "who ",
        "when ",
        "where ",
        "why ",
        "how ",
        "can ",
        "could ",
        "do ",
        "does ",
        "did ",
        "is ",
        "are ",
        "am ",
        "will ",
        "would ",
        "should ",
    )
    return msg_l.startswith(starters)


def _build_critical_key(raw_text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (raw_text or "").lower())
    filtered = [t for t in tokens if len(t) > 2 and t not in CRITICAL_STOPWORDS]
    slug = "_".join(filtered[:5]) if filtered else "general"
    return f"custom.critical.{slug}"


def _extract_critical_memories(user_message: str, user_id: str) -> list[Memory]:
    msg = (user_message or "").strip()
    msg_l = msg.lower()
    if not msg:
        return []

    critical_patterns = [
        r"^\s*(?:remember(?:\s+this)?|don't forget(?:\s+this)?|note(?:\s+this)?|important)\s*[:,-]?\s*(.+)$",
        r"^\s*(?:from now on|for future reference)\s*[:,-]?\s*(.+)$",
        r"\bit is (?:very\s+)?important that\s+(.+)$",
        r"\bplease remember(?: that)?\s+(.+)$",
    ]

    extracted: list[Memory] = []
    for pattern in critical_patterns:
        m = re.search(pattern, msg_l)
        if not m:
            continue
        value = re.sub(r"\s+", " ", (m.group(1) or "").strip(" .,!?:;"))
        if len(value) < 6 or len(value) > 220:
            continue
        key = _build_critical_key(value)
        category = key.replace("custom.critical.", "") or "general"
        extracted.append(
            Memory.create(
                user_id=user_id,
                type="constraint",
                key=key,
                value=value,
                confidence=0.9,
                metadata={
                    "domain": "critical_context",
                    "category": category,
                    "scope": "GLOBAL_SCOPE",
                    "importance": "critical",
                    "source": "explicit_user_instruction",
                },
            )
        )
        break

    # Also capture explicit high-priority declarative facts.
    if not extracted and not _looks_like_question(msg_l):
        m = re.search(r"^\s*(?:this is critical|this is important)\s*[:,-]?\s*(.+)$", msg_l)
        if m:
            value = re.sub(r"\s+", " ", (m.group(1) or "").strip(" .,!?:;"))
            if 6 <= len(value) <= 220:
                key = _build_critical_key(value)
                category = key.replace("custom.critical.", "") or "general"
                extracted.append(
                    Memory.create(
                        user_id=user_id,
                        type="fact",
                        key=key,
                        value=value,
                        confidence=0.9,
                        metadata={
                            "domain": "critical_context",
                            "category": category,
                            "scope": "GLOBAL_SCOPE",
                            "importance": "critical",
                            "source": "explicit_user_instruction",
                        },
                    )
                )

    return extracted


def _extract_schedule_memories(user_message: str, user_id: str) -> list[Memory]:
    """
    Extract upcoming event constraints from explicit user statements like:
    - "I have an important test on Friday"
    - "I have a lab tomorrow"
    """
    msg = (user_message or "").strip()
    msg_l = msg.lower()
    if not msg or _looks_like_question(msg_l):
        return []

    day_pattern = r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    clause_pattern = re.compile(
        rf"\b(?:i\s+have|i've\s+got|there\s+is|there's|i\s+got)\b"
        rf"(?P<event>[^.?!]{{1,90}}?)"
        rf"\b(?:on\s+|this\s+|upcoming\s+)?(?P<day>{day_pattern})\b",
        flags=re.IGNORECASE,
    )

    memories: list[Memory] = []
    clauses = re.split(r"\b(?:and|also)\b|[,;]", msg_l)
    for clause in clauses:
        match = clause_pattern.search(clause)
        if not match:
            continue
        event_raw = re.sub(r"\s+", " ", (match.group("event") or "").strip()).lower()
        day_raw = (match.group("day") or "").strip().lower()
        if day_raw not in DAY_WORDS:
            continue

        # Normalize event phrase to a compact slug without hardcoding domain nouns.
        event_clean = re.sub(r"\b(on|this|upcoming)\b", " ", event_raw)
        event_clean = re.sub(r"\b(a|an|the|my|very)\b", " ", event_clean)
        event_clean = re.sub(r"\bimportant\b", " ", event_clean)
        event_clean = re.sub(r"\s+", " ", event_clean).strip()
        if len(event_clean) < 2:
            continue
        event_slug = re.sub(r"[^a-z0-9]+", "_", event_clean).strip("_")
        if not event_slug:
            continue

        important = "important" in clause or "important" in msg_l
        key = f"lifestyle.upcoming_event.{event_slug}"
        confidence = 0.9 if important else 0.8
        memories.append(
            Memory.create(
                user_id=user_id,
                type="constraint",
                key=key,
                value=day_raw,
                confidence=confidence,
                metadata={
                    "domain": "lifestyle",
                    "category": "upcoming_event",
                    "scope": "GLOBAL_SCOPE",
                    "subject": event_slug,
                    "event": event_clean,
                    "importance": "critical" if important else "normal",
                    "source": "explicit_user_schedule",
                },
            )
        )

    return memories


def _extract_patterns(user_message: str, user_id: str) -> list[Memory]:
    memories: list[Memory] = []
    message_lower = user_message.lower()

    memories.extend(_extract_critical_memories(user_message, user_id))
    memories.extend(_extract_schedule_memories(user_message, user_id))

    # Keep "I'm" preamble flexible, but keep captured name strict.
    name_patterns = [
        r"i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"my\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"i\s+am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"call\s+me\s+([A-Z][a-z]+)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_message)
        if match:
            name = match.group(1).strip()
            if 1 < len(name) < 50 and name.lower() not in INVALID_NAME_WORDS:
                memories.append(Memory.create(user_id=user_id, type="fact", key="name", value=name, confidence=0.8))
                break

    # Food-specific preference capture.
    favorite_food_patterns = [
        r"my\s+favo(?:u)?rite\s+food\s+is\s+([^,.!?]+)",
        r"i\s+love\s+eating\s+([^,.!?]+)",
        r"i\s+like\s+eating\s+([^,.!?]+)",
    ]
    for pattern in favorite_food_patterns:
        match = re.search(pattern, message_lower)
        if match:
            food = match.group(1).strip()
            if 1 < len(food) < 80:
                memories.append(
                    Memory.create(
                        user_id=user_id,
                        type="preference",
                        key="favorite_food",
                        value=food,
                        confidence=0.8,
                    )
                )
                break

    # Generic favorite character preference capture from user phrasing.
    favorite_character_patterns = [
        r"my\s+favo(?:u)?rite\s+char(?:acter)?\s+(?:is|are)\s+([^,.!?]+)",
        r"i\s+usually\s+like\s+([^,.!?]+)\s+as\s+my\s+favo(?:u)?rite\s+char(?:acter)?",
        r"i\s+prefer\s+([^,.!?]+)\s+as\s+my\s+favo(?:u)?rite\s+char(?:acter)?",
    ]
    for pattern in favorite_character_patterns:
        match = re.search(pattern, message_lower)
        if match:
            char_pref = match.group(1).strip()
            if 2 < len(char_pref) < 100:
                char_pref = re.sub(r"\s+", " ", char_pref).strip()
                char_pref = char_pref.replace(" main character ", " mc ")
                char_pref = re.sub(r"\bmain character\b", "mc", char_pref)
                memories.append(
                    Memory.create(
                        user_id=user_id,
                        type="preference",
                        key="entertainment.character_preference",
                        value=char_pref,
                        confidence=0.8,
                        metadata={"rule_type": "generic", "scope": "LOCAL_SCOPE", "subject": "anime_character_preference"},
                    )
                )
            break

    # Global rule extraction for character preference.
    global_rule_patterns = [
        r"my\s+favo(?:u)?rite\s+char(?:acter)?\s+is\s+always\s+([^,.!?]+)",
        r"i\s+always\s+like\s+([^,.!?]+)\s+as\s+my\s+favo(?:u)?rite\s+char(?:acter)?",
        r"i\s+prefer\s+([^,.!?]+)\s+in\s+every\s+anime",
        r"i\s+like\s+the\s+mc\s+of\s+(?:any|every|all)\s+anime",
        r"i\s+like\s+the\s+main\s+character\s+of\s+(?:any|every|all)\s+anime",
        r"i\s+prefer\s+the\s+mc\s+of\s+(?:any|every|all)\s+anime",
        r"i\s+prefer\s+the\s+main\s+character\s+of\s+(?:any|every|all)\s+anime",
    ]
    for pattern in global_rule_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        if match.lastindex and match.group(1):
            rule_value = re.sub(r"\s+", " ", match.group(1).strip())
        else:
            rule_value = "mc"
        if 1 < len(rule_value) < 120:
            memories.append(
                Memory.create(
                    user_id=user_id,
                    type="preference",
                    key="entertainment.global_character_rule",
                    value=rule_value,
                    confidence=0.5,
                    metadata={"rule_type": "universal", "scope": "GLOBAL_SCOPE", "subject": "anime_character_preference"},
                )
            )
            break

    # Exception extraction, e.g. "except in JJK, my favorite is Gojo".
    exception_patterns = [
        r"except\s+in\s+([a-z0-9\s]+?)[, ]+\s*(?:my\s+favo(?:u)?rite(?:\s+char(?:acter)?)?\s+is|i\s+like)\s+([a-z0-9\s]+)",
        r"except\s+in\s+([a-z0-9\s]+?)\s*[,\.]",
    ]
    for pattern in exception_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        anime = re.sub(r"\s+", " ", match.group(1).strip())
        anime_key = re.sub(r"[^a-z0-9]+", "_", anime).strip("_")
        if not anime_key:
            continue

        exception_value = "override"
        if len(match.groups()) > 1 and match.group(2):
            candidate = re.sub(r"\s+", " ", match.group(2).strip())
            if 1 < len(candidate) < 120:
                exception_value = candidate
        memories.append(
            Memory.create(
                user_id=user_id,
                type="preference",
                key=f"entertainment.character_exception.{anime_key}",
                value=exception_value,
                confidence=0.8,
                metadata={"rule_type": "exception", "scope": "LOCAL_SCOPE", "subject": anime_key},
            )
        )
        break

    # Anime-specific favorite character capture.
    anime_char_patterns = [
        r"my\s+favo(?:u)?rite\s+char(?:acter)?\s+in\s+([a-z0-9\s]+?)\s+is\s+([a-z0-9\s]+)",
        r"in\s+([a-z0-9\s]+?)\s*,?\s*my\s+favo(?:u)?rite\s+char(?:acter)?\s+is\s+([a-z0-9\s]+)",
    ]
    for pattern in anime_char_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        anime = re.sub(r"\s+", " ", match.group(1).strip())
        character = re.sub(r"\s+", " ", match.group(2).strip())
        # avoid placeholder/echo values
        if len(anime) < 2 or len(character) < 2:
            continue
        if character in {"fav char", "favorite character", "favourite character"}:
            continue
        anime_key = re.sub(r"[^a-z0-9]+", "_", anime).strip("_")
        if not anime_key:
            continue
        memories.append(
            Memory.create(
                user_id=user_id,
                type="fact",
                key=f"entertainment.favorite_character_in_{anime_key}",
                value=character,
                confidence=0.8,
                metadata={"rule_type": "specific", "scope": "LOCAL_SCOPE", "subject": anime_key},
            )
        )
        break

    generic_universal_patterns = [
        r"my\s+favo(?:u)?rite\s+([a-z0-9\s_\-]{2,40})\s+is\s+always\s+([^,.!?]+)",
        r"i\s+always\s+(?:like|prefer|choose)\s+([^,.!?]+)\s+as\s+my\s+([a-z0-9\s_\-]{2,40})",
    ]
    for pattern in generic_universal_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        if len(match.groups()) != 2:
            continue
        if pattern.startswith("my\\s+favo"):
            subject_raw = match.group(1).strip().lower()
            rule_value = match.group(2).strip()
        else:
            rule_value = match.group(1).strip()
            subject_raw = match.group(2).strip().lower()

        subject_key = re.sub(r"[^a-z0-9]+", "_", subject_raw).strip("_")
        rule_value = re.sub(r"\s+", " ", rule_value).strip()
        if not subject_key or len(rule_value) < 2:
            continue

        memories.append(
            Memory.create(
                user_id=user_id,
                type="preference",
                key=f"custom.preference_rule.{subject_key}",
                value=rule_value,
                confidence=0.8,
                metadata={
                    "domain": "generic",
                    "category": subject_key,
                    "scope": "GLOBAL_SCOPE",
                    "rule_type": "universal",
                    "subject": subject_key,
                },
            )
        )
        break

    preference_patterns = [
        r"i\s+prefer\s+([^,.!?]+)",
        r"i\s+like\s+([^,.!?]+)",
        r"my\s+favorite\s+([^,.!?]+)\s+is\s+([^,.!?]+)",
    ]
    for pattern in preference_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        if len(match.groups()) == 2:
            key = match.group(1).strip()
            value = match.group(2).strip()
        else:
            key = "preference"
            value = match.group(1).strip()
        if key in {"entertainment.global_character_rule", "entertainment.character_exception"}:
            continue
        if "character" in key and (" in " in key or key in {"character", "favorite character", "favourite character"}):
            # handled above with stronger structure; skip weak capture.
            continue
        if value.startswith("eating "):
            # Already handled in favorite_food_patterns to avoid duplicate generic preference.
            continue
        if key in {"food", "favorite food", "favourite food"}:
            key = "lifestyle.food"
        if 2 < len(value) < 100:
            memories.append(Memory.create(user_id=user_id, type="preference", key=key, value=value, confidence=0.8))

    return memories


def _map_key_to_schema(key: str, value: str) -> Optional[str]:
    k = (key or "").strip().lower()
    v = (value or "").strip().lower()
    if not k or not v:
        return None
    if any(k.startswith(f"{prefix}.") for prefix in SCHEMA_PREFIXES):
        return k
    if k.startswith("entertainment."):
        return k
    if k.startswith("custom.preference_rule."):
        return k
    if k.startswith("custom.critical."):
        return k

    # CORE_IDENTITY
    if k in {"name"}:
        return "core_identity.name"
    if k in {"college", "university", "location"} or any(t in v for t in {"college", "university", "iiit"}):
        return "core_identity.college"
    if k in {"year", "year of graduation", "graduation year"}:
        return "core_identity.year"

    # LIFESTYLE
    if any(t in k for t in {"food", "drink"}):
        return "lifestyle.food" if "food" in k else "lifestyle.drinks"
    if "gym" in k or "gym" in v or "workout" in v:
        return "lifestyle.gym"
    if k in {"favorite_food"}:
        return "lifestyle.food"

    # ENTERTAINMENT
    if "anime" in k or "anime" in v:
        return "entertainment.anime"
    if "character" in k:
        return "entertainment.character_preference"
    if k in {"anime_preference"}:
        return "entertainment.anime"

    # TECHNICAL_STACK
    if "c++" in k or "c++" in v:
        return "technical_stack.cpp"
    if "mongodb" in k or "mongodb" in v:
        return "technical_stack.mongodb"
    if "groq" in k or "groq" in v:
        return "technical_stack.groq"

    return None


def _apply_schema_guard(memory: Memory) -> Optional[Memory]:
    mapped_key = _map_key_to_schema(memory.key, memory.value)
    if not mapped_key:
        return None
    memory.key = mapped_key
    if mapped_key.startswith("technical_stack.") and len(memory.value) > 120:
        memory.value = memory.value[:120]
    return memory


def _entity_from_message(user_message: str) -> str:
    msg = (user_message or "").lower()
    m = re.search(r"\b(?:in|from|of)\s+([a-z0-9 ]{2,40})", msg)
    if not m:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", m.group(1)).strip("_")


def _context_metadata_for_memory(memory: Memory, user_message: str) -> Optional[dict]:
    key = (memory.key or "").lower()
    value = (memory.value or "").lower()
    md = dict(memory.metadata or {})

    if key.startswith("core_identity."):
        md.update({"domain": "identity", "category": key.split(".", 1)[1], "scope": "PROFILE"})
        return md
    if key.startswith("lifestyle."):
        md.update({"domain": "lifestyle", "category": key.split(".", 1)[1], "scope": "GLOBAL_SCOPE"})
        return md
    if key.startswith("technical_stack."):
        md.update({"domain": "technical_stack", "category": key.split(".", 1)[1], "scope": "GLOBAL_SCOPE"})
        return md
    if key.startswith("entertainment.favorite_character_in_"):
        subject = key.replace("entertainment.favorite_character_in_", "").strip()
        if not subject:
            return None
        md.update({"domain": "entertainment", "category": "favorite_character", "scope": "LOCAL_SCOPE", "subject": subject})
        return md
    if key.startswith("entertainment.character_exception."):
        subject = key.replace("entertainment.character_exception.", "").strip()
        if not subject:
            return None
        md.update({"domain": "entertainment", "category": "favorite_character", "scope": "LOCAL_SCOPE", "subject": subject})
        return md
    if key == "entertainment.global_character_rule":
        md.update({"domain": "entertainment", "category": "favorite_character", "scope": "GLOBAL_SCOPE", "subject": "anime"})
        return md
    if key == "entertainment.character_preference":
        # Require explicit scope context.
        entity = _entity_from_message(user_message)
        msg_l = (user_message or "").lower()
        if entity:
            md.update({"domain": "entertainment", "category": "favorite_character", "scope": "LOCAL_SCOPE", "subject": entity})
            return md
        if any(t in msg_l for t in ["always", "every", "all anime"]):
            md.update({"domain": "entertainment", "category": "favorite_character", "scope": "GLOBAL_SCOPE", "subject": "anime"})
            return md
        return None
    if key == "entertainment.anime":
        md.update({"domain": "entertainment", "category": "anime", "scope": "GLOBAL_SCOPE"})
        return md
    if key.startswith("custom.critical."):
        category = key.replace("custom.critical.", "") or "general"
        md.setdefault("domain", "critical_context")
        md.setdefault("category", category)
        md.setdefault("scope", "GLOBAL_SCOPE")
        md.setdefault("importance", "critical")
        return md

    # Unknown/custom keys must carry explicit context to be stored.
    if md.get("domain") and md.get("category") and md.get("scope"):
        return md
    return None


def extract_memory(user_message: str, user_id: str) -> Optional[Memory]:
    normalized_message = _normalize_for_extraction(user_message)
    pattern_memories = _extract_patterns(normalized_message, user_id)
    stored_any = None
    if pattern_memories:
        for memory in pattern_memories:
            memory = _apply_schema_guard(memory)
            if memory is None:
                continue
            memory.metadata = _context_metadata_for_memory(memory, normalized_message)
            if not memory.metadata:
                continue
            memory = _apply_general_confidence(memory, normalized_message)
            stored = get_memory_store().add_or_update_memory(memory)
            if stored:
                stored_any = stored
                print(f"Memory stored (pattern): {stored.key} = {stored.value}")
        # Pattern path is strict and explicit; avoid LLM extraction here.
        return stored_any

    prompt = """You are a skeptical auditor.

Source of Truth:
- ONLY extract facts explicitly stated by the USER.
- If the ASSISTANT suggested something (e.g., "Do you like coding?") and the user didn't say "Yes," you MUST NOT store it.
- If the user's message is ambiguous, do not guess. Return null.

Never extract:
- AI names (like "Aria")
- Polite filler ("I understand")
- Hypothetical examples used in the conversation

Output format:
- Return either null, or a JSON array.
- Each object must be: {"type":"fact|preference|constraint","key":"...","value":"..."}.
- Do not include anything else.

User message: """ + normalized_message

    try:
        response = generate_response(prompt)
        if not response:
            return None

        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        if response.lower() == "null" or not response:
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        payload = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
        if payload is None:
            return None

        stored_any = None
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not {"type", "key", "value"}.issubset(item.keys()):
                continue

            m_type = item.get("type")
            m_key = item.get("key")
            m_value = item.get("value")
            if m_type not in ["preference", "constraint", "fact"]:
                continue
            if not isinstance(m_key, str) or not isinstance(m_value, str):
                continue
            if m_key.strip().lower() in NEVER_EXTRACT_KEYS:
                continue
            if m_value.strip().lower() in NEVER_EXTRACT_VALUES:
                continue
            if not _is_memory_supported_by_message(m_key, m_value, normalized_message):
                continue

            memory = Memory.create(
                user_id=user_id,
                type=m_type,
                key=m_key.strip(),
                value=m_value.strip(),
                confidence=0.8,
            )
            memory = _apply_schema_guard(memory)
            if memory is None:
                continue
            memory.metadata = _context_metadata_for_memory(memory, normalized_message)
            if not memory.metadata:
                continue
            memory = _apply_general_confidence(memory, normalized_message)
            stored = get_memory_store().add_or_update_memory(memory)
            if stored:
                stored_any = stored
                print(f"Memory stored: {stored.key} = {stored.value} (confidence: {stored.confidence:.2f})")

        return stored_any

    except RuntimeError as e:
        print(f"Memory extraction API error (non-fatal): {e}")
        return None
    except Exception as e:
        print(f"Memory extraction error (non-fatal): {type(e).__name__}: {e}")
        return None
