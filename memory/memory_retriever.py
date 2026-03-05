"""
Memory Retriever

Logical semantic retrieval pipeline for deterministic memory injection.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
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

TOPIC_SCORE_THRESHOLD = float(os.getenv("MEMORY_RETRIEVER_TOPIC_SCORE_THRESHOLD", "0.23"))
TOPIC_SEMANTIC_WEIGHT = float(os.getenv("MEMORY_RETRIEVER_TOPIC_SEMANTIC_WEIGHT", "0.68"))
TOPIC_MIN_SEMANTIC = float(os.getenv("MEMORY_RETRIEVER_TOPIC_MIN_SEMANTIC", "0.10"))
NON_INTENT_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_RETRIEVER_NON_INTENT_SIM_THRESHOLD", "0.90"))
NON_INTENT_RAW_FLOOR = float(os.getenv("MEMORY_RETRIEVER_NON_INTENT_RAW_FLOOR", "0.12"))

CANONICAL_TYPES = (
    "preference_anime",
    "preference_food",
    "preference_drink",
    "personal_fact",
    "hobby",
    "relationship",
    "career",
)


def _word_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(t) >= 2}


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


def _memory_intent_kind(query: str) -> str:
    tokens = _word_tokens(query)
    preference_markers = {"favorite", "favourite", "fav", "like", "likes", "prefer", "preference", "preferences"}
    fact_markers = {"name", "age", "college", "university", "where", "who"}
    if tokens.intersection(preference_markers):
        return "preference"
    if tokens.intersection(fact_markers):
        return "fact"
    return ""


def _memory_text(memory: Any) -> str:
    key = re.sub(r"\s+", " ", str(getattr(memory, "key", "") or "").strip())
    value = re.sub(r"\s+", " ", str(getattr(memory, "value", "") or "").strip())
    if key and value:
        return f"{key}: {value}"
    return value or key


def _metadata(memory: Any) -> dict[str, Any]:
    raw = getattr(memory, "metadata", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _normalize_memory_type(memory: Any) -> str:
    current = str(getattr(memory, "memory_type", "") or "").strip().lower()
    key_l = str(getattr(memory, "key", "") or "").strip().lower()
    value_l = str(getattr(memory, "value", "") or "").strip().lower()
    mem_kind = str(getattr(memory, "type", "") or "").strip().lower()
    md = _metadata(memory)
    domain = str(md.get("domain") or "").strip().lower()
    category = str(md.get("category") or "").strip().lower()
    subject = str(md.get("subject") or "").strip().lower()
    joined = " ".join([current, key_l, value_l, domain, category, subject])

    # Keep canonical type if already valid.
    if current in CANONICAL_TYPES:
        return current

    # Compatibility normalization for older labels.
    if current in {"favorite_anime", "anime_preference"}:
        return "preference_anime"
    if current in {"food_preference", "preference_food"}:
        return "preference_food"
    if current in {"drink_preference", "preference_drink"}:
        return "preference_drink"
    if current in {"work", "career"}:
        return "career"
    if current in {"personal_fact"}:
        return "personal_fact"

    if domain == "entertainment" or key_l.startswith("entertainment.") or "anime" in joined or "character" in joined:
        return "preference_anime"
    if domain == "lifestyle" and ("drink" in category or key_l.startswith("lifestyle.drink") or "beverage" in joined):
        return "preference_drink"
    if domain == "lifestyle" and ("food" in category or key_l.startswith("lifestyle.food") or "eat" in joined):
        return "preference_food"
    if domain in {"technical_stack", "career", "work"} or key_l.startswith("technical_stack."):
        return "career"
    if any(x in joined for x in ("girlfriend", "boyfriend", "wife", "husband", "relationship", "partner")):
        return "relationship"
    if domain == "identity" or key_l.startswith("core_identity.") or mem_kind == "fact":
        return "personal_fact"
    if mem_kind == "preference":
        return "hobby"

    return "personal_fact"


def _memory_feature_text(memory: Any, normalized_type: str) -> str:
    md = _metadata(memory)
    return " ".join(
        [
            str(getattr(memory, "key", "") or ""),
            str(getattr(memory, "value", "") or ""),
            normalized_type,
            str(md.get("domain") or ""),
            str(md.get("category") or ""),
            str(md.get("subject") or ""),
        ]
    ).strip()


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


def _vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    if dims == 0:
        return []
    out = [0.0] * dims
    count = 0
    for vec in vectors:
        if len(vec) != dims:
            continue
        for i in range(dims):
            out[i] += float(vec[i])
        count += 1
    if count <= 0:
        return []
    out = [x / count for x in out]
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 0:
        return []
    return [v / norm for v in out]


def _detect_query_topic(
    query: str,
    query_vector: list[float],
    grouped_entries: dict[str, list[dict[str, Any]]],
) -> tuple[str, float, list[dict[str, Any]]]:
    query_tokens = _word_tokens(query)
    if not grouped_entries:
        return "", 0.0, []

    results: list[dict[str, Any]] = []
    lexical_weight = max(0.0, 1.0 - TOPIC_SEMANTIC_WEIGHT)
    for memory_type, rows in grouped_entries.items():
        type_tokens: set[str] = set()
        vectors: list[list[float]] = []
        for row in rows:
            type_tokens.update(_word_tokens(str(row.get("feature_text") or "")))
            emb = list(row.get("embedding") or [])
            if emb:
                vectors.append(emb)

        centroid = _vector_mean(vectors)
        semantic_score = cosine_similarity(query_vector, centroid) if query_vector and centroid else 0.0
        if query_tokens:
            overlap = len(query_tokens.intersection(type_tokens))
            lexical_score = overlap / max(1, len(query_tokens))
        else:
            lexical_score = 0.0
        topic_score = (semantic_score * TOPIC_SEMANTIC_WEIGHT) + (lexical_score * lexical_weight)
        results.append(
            {
                "memory_type": memory_type,
                "semantic_score": max(0.0, min(1.0, float(semantic_score))),
                "lexical_score": max(0.0, min(1.0, float(lexical_score))),
                "topic_score": max(0.0, min(1.0, float(topic_score))),
                "memory_count": len(rows),
            }
        )

    results.sort(key=lambda x: x["topic_score"], reverse=True)
    if not results:
        return "", 0.0, []
    best = results[0]
    if best["topic_score"] < TOPIC_SCORE_THRESHOLD or best["semantic_score"] < TOPIC_MIN_SEMANTIC:
        return "", float(best["topic_score"]), results[:6]
    return str(best["memory_type"]), float(best["topic_score"]), results[:6]


def retrieve_memories(user_message: str, user_id: str) -> str:
    """
    Retrieval pipeline:
    query -> embedding -> topic inference -> type filter -> vector search
    -> threshold filter -> weighted rank -> top-3 injection.
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
            topic_confidence=0.0,
            topic_candidates=[],
            query_embedding=[],
            retrieved_memories=[],
            filtered_memories=[],
            final_injected_memories=[],
        )
        return ""

    embedder = get_embedding_provider()
    query_vector = list(embedder.embed(query).vector or [])
    vector_dims = len(query_vector)
    memory_intent = _is_memory_intent(query)
    intent_kind = _memory_intent_kind(query)

    entries: list[dict[str, Any]] = []
    grouped_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dirty = False

    for memory in memories:
        memory_type = _normalize_memory_type(memory)
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

        feature_text = _memory_feature_text(memory, memory_type)
        row = {
            "memory": memory,
            "memory_id": str(getattr(memory, "id", "")),
            "memory_text": memory_text,
            "memory_type": memory_type,
            "embedding": embedding,
            "feature_text": feature_text,
        }
        entries.append(row)
        grouped_entries[memory_type].append(row)

    if dirty:
        try:
            store.save()
        except Exception:
            pass

    if not entries:
        log_event(
            "memory_retrieval_debug",
            user_id=user_id,
            user_query=query,
            query_topic="",
            topic_confidence=0.0,
            topic_candidates=[],
            query_embedding=[round(float(x), 6) for x in query_vector],
            retrieved_memories=[],
            filtered_memories=[],
            final_injected_memories=[],
        )
        return ""

    topic, topic_confidence, topic_candidates = _detect_query_topic(query, query_vector, grouped_entries)
    type_filtered = grouped_entries.get(topic, []) if topic else entries
    if not type_filtered:
        type_filtered = []
    if not topic and intent_kind == "preference":
        preference_rows = [
            row
            for row in entries
            if (
                str(getattr(row.get("memory"), "type", "") or "").strip().lower() == "preference"
                or row.get("memory_type") in {"preference_anime", "preference_food", "preference_drink", "hobby"}
            )
        ]
        if preference_rows:
            type_filtered = preference_rows
    elif not topic and intent_kind == "fact":
        fact_rows = [
            row
            for row in entries
            if (
                str(getattr(row.get("memory"), "type", "") or "").strip().lower() == "fact"
                or row.get("memory_type") == "personal_fact"
            )
        ]
        if fact_rows:
            type_filtered = fact_rows

    retrieved_hits: list[dict[str, Any]] = []
    for row in type_filtered:
        memory = row["memory"]
        similarity = cosine_similarity(query_vector, list(row["embedding"]))
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
                "memory_id": row["memory_id"],
                "memory_text": row["memory_text"],
                "memory_type": row["memory_type"],
                "similarity": similarity,
                "similarity_normalized": similarity,
                "recency": recency,
                "importance": importance,
                "rank_score": rank_score,
            }
        )

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
    if not memory_intent:
        effective_threshold = max(effective_threshold, NON_INTENT_SIMILARITY_THRESHOLD)
        effective_raw_floor = max(effective_raw_floor, NON_INTENT_RAW_FLOOR)
    elif not topic:
        effective_threshold = max(effective_threshold, 0.80)
        effective_raw_floor = max(effective_raw_floor, 0.08)

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
        topic_confidence=round(float(topic_confidence), 6),
        topic_candidates=topic_candidates,
        query_embedding=[round(float(x), 6) for x in query_vector],
        retrieved_memories=[_serialize_hit(h) for h in top_hits],
        filtered_memories=[_serialize_hit(h) for h in filtered_hits],
        final_injected_memories=[_serialize_hit(h) for h in final_hits],
        top_k=TOP_K,
        similarity_threshold=effective_threshold,
        raw_similarity_floor=effective_raw_floor,
        memory_intent=memory_intent,
        memory_intent_kind=intent_kind,
        max_inject=MAX_INJECT,
    )

    if not final_hits:
        return ""

    lines = [f"- {h['memory_text']}" for h in final_hits]
    return "\n".join(lines)
