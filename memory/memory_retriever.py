"""
Memory Retriever

Retrieves relevant memories for the current conversation context.
"""

import heapq
import os
import re
from .memory_store import get_memory_store

ESSENTIAL_KEYS = {
    "core_identity.name",
    "core_identity.college",
    "core_identity.year",
    "lifestyle.food",
    "lifestyle.gym",
    "lifestyle.drinks",
    "entertainment.anime",
    "entertainment.character_preference",
    "entertainment.global_character_rule",
    "technical_stack.cpp",
    "technical_stack.mongodb",
    "technical_stack.groq",
}

CRITICAL_MARKERS = {
    "important",
    "critical",
    "remember",
    "forget",
    "must",
    "never",
    "always",
    "rule",
    "rules",
    "instruction",
    "instructions",
    "constraint",
}

MAX_CANDIDATES = int(os.getenv("MEMORY_RETRIEVER_MAX_CANDIDATES", "80"))
MAX_SELECTED = int(os.getenv("MEMORY_RETRIEVER_MAX_SELECTED", "12"))
GENERIC_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "about",
    "what", "which", "who", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should",
    "my", "me", "i", "your", "you", "it", "that", "this", "there",
    "please", "tell", "show", "list", "give", "have",
}


def _is_critical_memory(memory) -> bool:
    md = memory.metadata or {}
    key_l = (memory.key or "").lower()
    domain = str(md.get("domain", "")).lower()
    return (
        domain == "critical_context"
        or key_l.startswith("custom.critical.")
    )


def _semantic_tokens(tokens: set[str]) -> set[str]:
    return {t for t in tokens if len(t) > 2 and t not in GENERIC_QUERY_STOPWORDS}


def _memory_tokens(memory) -> set[str]:
    md = memory.metadata or {}
    raw = " ".join(
        [
            str(memory.key or ""),
            str(memory.value or ""),
            str(md.get("domain", "") or ""),
            str(md.get("category", "") or ""),
            str(md.get("subject", "") or ""),
        ]
    ).lower()
    return set(re.findall(r"[a-z0-9]+", raw))


def _is_broad_rule(memory) -> bool:
    md = memory.metadata or {}
    scope = str(md.get("scope", "")).upper()
    rule_type = str(md.get("rule_type", "")).lower()
    key_l = (memory.key or "").lower()
    return (
        (scope == "GLOBAL_SCOPE" and rule_type in {"universal", "generic"})
        or key_l == "entertainment.global_character_rule"
        or key_l.startswith("custom.preference_rule.")
    )


def retrieve_memories(user_message: str, user_id: str) -> str:
    """
    Retrieve relevant memories for the current user message.
    
    Returns a short summary string suitable for system prompt.
    Uses scoped memories with confidence >= 0.5 for balanced recall.
    """
    store = get_memory_store()
    # Use the configured medium-confidence threshold.
    memories = store.get_memories_by_confidence(user_id=user_id, min_confidence=0.5)
    
    if not memories:
        return ""
    
    msg_l = (user_message or "").lower()
    msg_tokens = set(re.findall(r"[a-z0-9]+", msg_l))
    semantic_tokens = _semantic_tokens(msg_tokens)
    entity = _extract_query_entity(msg_l)
    query_domain, query_category = _infer_query_context(msg_tokens)
    query_is_critical = bool(CRITICAL_MARKERS.intersection(msg_tokens))

    # Keep candidate set bounded for predictable latency.
    scoped = []
    for m in memories:
        md = m.metadata or {}
        domain = str(md.get("domain", "")).lower()
        category = str(md.get("category", "")).lower()
        scope = str(md.get("scope", "")).upper()
        subject = str(md.get("subject", "")).lower().strip()

        is_critical = _is_critical_memory(m)

        # Must have explicit scope metadata to be used.
        if not domain or not category or not scope:
            continue
        if query_domain and domain != query_domain and not is_critical:
            continue
        if query_category and query_category not in category and not is_critical:
            continue
        if entity and scope == "LOCAL_SCOPE" and subject and subject != entity and not is_critical:
            continue

        if not query_domain and semantic_tokens and not is_critical:
            mem_tokens = _memory_tokens(m)
            overlap_semantic = len(semantic_tokens.intersection(mem_tokens))
            if overlap_semantic == 0:
                continue
            # Global broad rules must be strongly tied to current query terms.
            if _is_broad_rule(m) and overlap_semantic < 2:
                continue

        scoped.append(m)

    if not scoped:
        return ""

    # Keep top candidates without sorting the full set for better performance as memory grows.
    candidate_count = min(MAX_CANDIDATES, max(24, len(scoped) // 2))
    candidates = heapq.nlargest(
        candidate_count,
        scoped,
        key=lambda m: (m.confidence, m.last_updated.timestamp()),
    )

    def score(memory):
        key_tokens = set(re.findall(r"[a-z0-9]+", memory.key.lower()))
        value_tokens = set(re.findall(r"[a-z0-9]+", memory.value.lower()))
        overlap = len(msg_tokens.intersection(key_tokens.union(value_tokens)))
        identity_bonus = 1 if memory.key.lower() == "core_identity.name" else 0
        food_bonus = 0
        if {"food", "favourite", "favorite"}.intersection(msg_tokens):
            if memory.key.lower() == "lifestyle.food":
                food_bonus = 3
            elif memory.key.lower().startswith("entertainment."):
                food_bonus = -1
        character_bonus = 0
        if {"character", "char", "mc", "main", "anime"}.intersection(msg_tokens):
            if memory.key.lower() in {"entertainment.character_preference", "entertainment.global_character_rule"}:
                character_bonus = 3
        schedule_bonus = 0
        if {"today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "schedule", "important", "when"}.intersection(msg_tokens):
            if memory.key.lower().startswith("lifestyle.upcoming_event."):
                schedule_bonus = 4

        entity_bonus = 0
        entity_penalty = 0
        if entity:
            key_l = memory.key.lower()
            md = memory.metadata or {}
            subject = str(md.get("subject", "")).lower().strip()
            if subject and subject == entity:
                entity_bonus = 8
            if key_l.endswith(f".{entity}") or f"_{entity}" in key_l:
                entity_bonus = max(entity_bonus, 7)
            # If query names an entity, suppress exceptions for other entities.
            if "character_exception." in key_l and not key_l.endswith(f".{entity}"):
                entity_penalty = 9

        rule_bonus = 0
        md = memory.metadata or {}
        rule_type = str(md.get("rule_type", "")).lower()
        if {"anime", "character", "char", "fav", "favorite"}.intersection(msg_tokens):
            if rule_type == "exception":
                rule_bonus = 4
            elif rule_type == "universal":
                rule_bonus = 2
            elif rule_type == "specific":
                rule_bonus = 3
        critical_bonus = 0
        if _is_critical_memory(memory):
            critical_bonus = 6 if query_is_critical else 2
        constraint_bonus = 2 if query_is_critical and memory.type == "constraint" else 0

        return (
            overlap + identity_bonus + food_bonus + character_bonus + schedule_bonus + entity_bonus + rule_bonus + critical_bonus + constraint_bonus - entity_penalty,
            memory.confidence,
            memory.last_updated.timestamp(),
        )

    memories_sorted = sorted(candidates, key=score, reverse=True)
    selected = []
    seen_ids = set()
    for m in memories_sorted:
        if len(selected) >= MAX_SELECTED:
            break
        if m.id in seen_ids:
            continue
        selected.append(m)
        seen_ids.add(m.id)

    # Ensure critical memories are not lost even if lexical overlap is weak.
    critical_candidates = heapq.nlargest(
        3,
        [m for m in candidates if _is_critical_memory(m)],
        key=lambda m: (m.confidence, m.last_updated.timestamp()),
    )
    required_critical = 2 if query_is_critical else 1
    for cm in critical_candidates:
        if required_critical <= 0:
            break
        if cm.id in seen_ids:
            continue
        if len(selected) < MAX_SELECTED:
            selected.append(cm)
        else:
            selected[-1] = cm
        seen_ids.add(cm.id)
        required_critical -= 1

    # Build concise summary string
    parts = []
    for memory in selected:
        md = memory.metadata or {}
        rule_type = md.get("rule_type")
        scope = md.get("scope")
        subject = md.get("subject")
        importance = md.get("importance")
        if importance == "critical":
            parts.append(
                f"- {memory.key}: {memory.value} [importance=critical; scope={scope}; subject={subject}]"
            )
        elif rule_type:
            parts.append(
                f"- {memory.key}: {memory.value} [rule_type={rule_type}; scope={scope}; subject={subject}]"
            )
        else:
            parts.append(f"- {memory.key}: {memory.value}")
    
    context = "\n".join(parts)
    return context


def _extract_query_entity(message_l: str) -> str:
    # Generic entity extraction for patterns like:
    # "fav char from/in <entity>" or "about <entity>".
    patterns = [
        r"\b(?:from|in)\s+([a-z0-9 ]{2,40})\b",
        r"\babout\s+([a-z0-9 ]{2,40})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, message_l)
        if not m:
            continue
        entity = re.sub(r"[^a-z0-9]+", "_", m.group(1)).strip("_")
        if entity:
            return entity
    return ""


def _infer_query_context(msg_tokens: set[str]) -> tuple[str, str]:
    # Returns (domain, category)
    if {"name", "college", "year", "identity"}.intersection(msg_tokens):
        return "identity", ""
    if {"who", "am", "i"}.issubset(msg_tokens):
        return "identity", ""
    if {"food", "drink", "drinks", "gym", "lifestyle", "eat"}.intersection(msg_tokens):
        if {"food", "eat"}.intersection(msg_tokens):
            return "lifestyle", "food"
        if {"drink", "drinks"}.intersection(msg_tokens):
            return "lifestyle", "drinks"
        if {"gym"}.intersection(msg_tokens):
            return "lifestyle", "gym"
        return "lifestyle", ""
    if {"today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "schedule", "important", "when"}.intersection(msg_tokens):
        return "lifestyle", "upcoming"
    if {"anime", "character", "char", "mc"}.intersection(msg_tokens):
        return "entertainment", "character"
    if {"c++", "mongodb", "groq", "tech", "stack"}.intersection(msg_tokens):
        return "technical_stack", ""
    return "", ""
