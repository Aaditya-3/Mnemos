"""
Memory Retriever

Topic-aware semantic retrieval pipeline for deterministic memory injection.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from backend.app.embeddings.provider import cosine_similarity, get_embedding_provider
from backend.app.observability.logging import log_event
from .memory_store import get_memory_store


TOP_K = int(os.getenv("MEMORY_RETRIEVER_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_RETRIEVER_SIMILARITY_THRESHOLD", "0.75"))
RAW_SIMILARITY_FLOOR = float(os.getenv("MEMORY_RETRIEVER_RAW_SIMILARITY_FLOOR", "0.05"))
MAX_INJECT = int(os.getenv("MEMORY_RETRIEVER_MAX_INJECT", "3"))

SIMILARITY_WEIGHT = float(os.getenv("MEMORY_RETRIEVER_WEIGHT_SIMILARITY", "0.7"))
RECENCY_WEIGHT = float(os.getenv("MEMORY_RETRIEVER_WEIGHT_RECENCY", "0.2"))
IMPORTANCE_WEIGHT = float(os.getenv("MEMORY_RETRIEVER_WEIGHT_IMPORTANCE", "0.1"))

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "favorite_anime": ("anime", "show", "series", "character", "mc", "manga"),
    "drink_preference": ("drink", "drinks", "beverage", "soda", "cola", "juice"),
    "food_preference": ("food", "eat", "meal", "cuisine", "dish"),
    "hobby": ("hobby", "hobbies", "sport", "sports", "gym", "music", "game"),
    "relationship": ("relationship", "partner", "girlfriend", "boyfriend", "wife", "husband"),
    "work": ("work", "job", "office", "project", "company", "tech", "stack", "backend", "frontend"),
    "personal_fact": ("name", "age", "college", "university", "city", "where am i from"),
}


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _is_memory_intent(query: str) -> bool:
    text = (query or "").lower()
    tokens = _word_tokens(text)
    markers = {
        "my",
        "mine",
        "favorite",
        "favourite",
        "fav",
        "prefer",
        "preference",
        "preferences",
        "like",
        "likes",
        "remember",
        "recall",
    }
    if bool(tokens.intersection(markers)):
        return True
    patterns = [
        r"\bwhat(?:'s| is)\s+my\b",
        r"\bwhat\s+do\s+i\s+like\b",
        r"\bdo\s+you\s+remember\b",
        r"\bwhen\s+(?:is|was)\s+my\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _memory_text(memory: Any) -> str:
    key = re.sub(r"\s+", " ", str(getattr(memory, "key", "") or "").strip())
    value = re.sub(r"\s+", " ", str(getattr(memory, "value", "") or "").strip())
    if key and value:
        return f"{key}: {value}"
    return value or key


def _infer_memory_type(memory: Any) -> str:
    current = str(getattr(memory, "memory_type", "") or "").strip().lower()
    if current:
        return current

    key_l = str(getattr(memory, "key", "") or "").lower()
    value_l = str(getattr(memory, "value", "") or "").lower()
    mem_type = str(getattr(memory, "type", "") or "").lower()
    joined = " ".join([key_l, value_l])

    if key_l.startswith("entertainment.") or "anime" in joined or "character" in joined:
        return "favorite_anime"
    if key_l == "lifestyle.drinks" or any(t in joined for t in ("drink", "beverage", "cola", "soda")):
        return "drink_preference"
    if key_l == "lifestyle.food" or any(t in joined for t in ("food", "eat", "meal", "dish")):
        return "food_preference"
    if key_l.startswith("technical_stack.") or any(t in joined for t in ("work", "job", "project", "company")):
        return "work"
    if any(t in joined for t in ("girlfriend", "boyfriend", "wife", "husband", "relationship")):
        return "relationship"
    if "gym" in joined or "hobby" in joined or mem_type == "preference":
        return "hobby"
    return "personal_fact"


def _detect_query_topic(query: str) -> str:
    text = (query or "").lower()
    if not text:
        return ""

    # Prioritize stronger topic identifiers first.
    priority = (
        "favorite_anime",
        "drink_preference",
        "food_preference",
        "relationship",
        "work",
        "hobby",
        "personal_fact",
    )
    for topic in priority:
        keywords = _TOPIC_KEYWORDS.get(topic, ())
        if any(k in text for k in keywords):
            return topic
    return ""


def _topic_match(memory: Any, topic: str) -> bool:
    if not topic:
        return True

    memory_type = _infer_memory_type(memory)
    if memory_type == topic:
        return True

    # Fallback lexical guard for legacy memories with weak typing.
    text = _memory_text(memory).lower()
    keywords = _TOPIC_KEYWORDS.get(topic, ())
    return any(k in text for k in keywords)


def _importance_score(memory: Any) -> float:
    raw = getattr(memory, "importance_score", None)
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))
    except Exception:
        pass
    try:
        conf = float(getattr(memory, "confidence", 0.5) or 0.5)
    except Exception:
        conf = 0.5
    return max(0.0, min(1.0, conf))


def _recency_score(memory: Any) -> float:
    now = datetime.now(timezone.utc)
    raw_dt = getattr(memory, "last_updated", None) or getattr(memory, "created_at", None)
    if not isinstance(raw_dt, datetime):
        return 0.5
    dt = raw_dt if raw_dt.tzinfo else raw_dt.replace(tzinfo=timezone.utc)
    age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
    return math.exp(-0.03 * age_days)


def _serialize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": hit["memory_id"],
        "memory_text": hit["memory_text"],
        "memory_type": hit["memory_type"],
        "similarity": round(float(hit["similarity"]), 6),
        "similarity_normalized": round(float(hit["similarity_normalized"]), 6),
        "recency": round(float(hit["recency"]), 6),
        "importance": round(float(hit["importance"]), 6),
        "rank_score": round(float(hit["rank_score"]), 6),
    }


def retrieve_memories(user_message: str, user_id: str) -> str:
    """
    Retrieval pipeline:
    query -> topic detection -> type filter -> vector search -> similarity threshold
    -> weighted rank -> top-3 injection.
    """
    query = (user_message or "").strip()
    if not query:
        return ""

    store = get_memory_store()
    memories = store.get_memories_by_confidence(user_id=user_id, min_confidence=0.5)
    if not memories:
        log_event(
            "memory_retrieval_debug",
            user_id=user_id,
            user_query=query,
            query_topic="",
            query_embedding=[],
            retrieved_memories=[],
            filtered_memories=[],
            final_injected_memories=[],
        )
        return ""

    topic = _detect_query_topic(query)
    memory_intent = _is_memory_intent(query)
    type_filtered = [m for m in memories if _topic_match(m, topic)]
    if topic and not type_filtered:
        # Known topic but no matching typed memory means "no relevant memory found".
        type_filtered = []
    elif not topic:
        type_filtered = list(memories)

    if not type_filtered:
        embedder = get_embedding_provider()
        query_embedding = embedder.embed(query).vector
        log_event(
            "memory_retrieval_debug",
            user_id=user_id,
            user_query=query,
            query_topic=topic,
            query_embedding=[round(float(x), 6) for x in query_embedding],
            retrieved_memories=[],
            filtered_memories=[],
            final_injected_memories=[],
        )
        return ""

    embedder = get_embedding_provider()
    query_vector = list(embedder.embed(query).vector or [])
    vector_dims = len(query_vector)

    retrieved_hits: list[dict[str, Any]] = []
    dirty = False

    for memory in type_filtered:
        memory_type = _infer_memory_type(memory)
        if str(getattr(memory, "memory_type", "") or "").strip().lower() != memory_type:
            memory.memory_type = memory_type
            dirty = True

        memory_text = _memory_text(memory)
        if not memory_text:
            continue

        embedding = list(getattr(memory, "embedding", []) or [])
        if not embedding or (vector_dims and len(embedding) != vector_dims):
            try:
                embedding = list(embedder.embed(memory_text).vector or [])
                memory.embedding = embedding
                dirty = True
            except Exception:
                continue
        if not embedding:
            continue

        similarity = cosine_similarity(query_vector, embedding)
        similarity = max(0.0, min(1.0, float(similarity)))
        recency = _recency_score(memory)
        importance = _importance_score(memory)
        rank_score = (
            (similarity * SIMILARITY_WEIGHT)
            + (recency * RECENCY_WEIGHT)
            + (importance * IMPORTANCE_WEIGHT)
        )

        retrieved_hits.append(
            {
                "memory_id": str(getattr(memory, "id", "")),
                "memory_text": memory_text,
                "memory_type": memory_type,
                "similarity": similarity,
                "similarity_normalized": similarity,
                "recency": recency,
                "importance": importance,
                "rank_score": rank_score,
            }
        )

    if dirty:
        try:
            store.save()
        except Exception:
            pass

    retrieved_hits.sort(key=lambda x: x["similarity"], reverse=True)
    top_hits = retrieved_hits[: max(1, TOP_K)]

    if top_hits:
        min_sim = min(h["similarity"] for h in top_hits)
        max_sim = max(h["similarity"] for h in top_hits)
        span = max(1e-9, max_sim - min_sim)
        for hit in top_hits:
            if max_sim - min_sim <= 1e-9:
                hit["similarity_normalized"] = 1.0
            else:
                hit["similarity_normalized"] = (hit["similarity"] - min_sim) / span
            hit["rank_score"] = (
                (hit["similarity_normalized"] * SIMILARITY_WEIGHT)
                + (hit["recency"] * RECENCY_WEIGHT)
                + (hit["importance"] * IMPORTANCE_WEIGHT)
            )

    effective_threshold = SIMILARITY_THRESHOLD
    effective_raw_floor = RAW_SIMILARITY_FLOOR
    if not topic and not memory_intent:
        effective_threshold = max(0.9, SIMILARITY_THRESHOLD)
        effective_raw_floor = max(0.12, RAW_SIMILARITY_FLOOR)

    filtered_hits = [
        h
        for h in top_hits
        if h["similarity_normalized"] >= effective_threshold and h["similarity"] >= effective_raw_floor
    ]
    filtered_hits.sort(key=lambda x: x["rank_score"], reverse=True)
    final_hits = filtered_hits[: max(1, MAX_INJECT)]

    log_event(
        "memory_retrieval_debug",
        user_id=user_id,
        user_query=query,
        query_topic=topic,
        query_embedding=[round(float(x), 6) for x in query_vector],
        retrieved_memories=[_serialize_hit(h) for h in top_hits],
        filtered_memories=[_serialize_hit(h) for h in filtered_hits],
        final_injected_memories=[_serialize_hit(h) for h in final_hits],
        top_k=TOP_K,
        similarity_threshold=effective_threshold,
        raw_similarity_floor=effective_raw_floor,
        memory_intent=memory_intent,
        max_inject=MAX_INJECT,
    )

    if not final_hits:
        return ""

    lines = [f"- {h['memory_text']}" for h in final_hits]
    return "\n".join(lines)
