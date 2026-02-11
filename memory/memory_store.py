"""
Memory Store

Persistent multi-user memory storage with confidence management.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .memory_schema import Memory
from backend.app.core.db.mongo import get_db


class MemoryStore:
    """Persistent memory store (multi-user via user_id)."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}
        self._key_index: dict[tuple[str, str], str] = {}
        self._storage_path = Path(__file__).resolve().parent / "memories.json"
        mongo_enabled = os.getenv("ENABLE_MONGO", "false").strip().lower() in {"1", "true", "yes", "on"}
        if mongo_enabled:
            try:
                self._db = get_db()
                self._collection = self._db["memories"]
            except Exception:
                self._collection = None
        else:
            self._collection = None
        self._load()

    def _load(self):
        self._memories = {}
        self._key_index = {}
        dirty = False
        if self._collection is not None:
            try:
                data = list(self._collection.find({}, {"_id": 0}))
                if data:
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        try:
                            memory = Memory.from_dict(item)
                            if memory.confidence < 0.5:
                                memory.confidence = 0.5
                                dirty = True
                            if self._backfill_metadata(memory):
                                dirty = True
                            if self._is_broad_rule(memory):
                                capped = min(max(memory.confidence, 0.5), 0.55)
                                if capped != memory.confidence:
                                    memory.confidence = capped
                                    dirty = True
                            self._memories[memory.id] = memory
                        except Exception:
                            continue
                    self._rebuild_index()
                    if dirty:
                        self._save()
                    return
            except Exception:
                pass
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    memory = Memory.from_dict(item)
                    if memory.confidence < 0.5:
                        memory.confidence = 0.5
                        dirty = True
                    if self._backfill_metadata(memory):
                        dirty = True
                    if self._is_broad_rule(memory):
                        capped = min(max(memory.confidence, 0.5), 0.55)
                        if capped != memory.confidence:
                            memory.confidence = capped
                            dirty = True
                    # Load existing memories as-is to avoid accidental data loss
                    # during schema migrations or normalization rule updates.
                    self._memories[memory.id] = memory
                except Exception:
                    continue
            self._rebuild_index()
            if dirty:
                self._save()
        except Exception:
            pass

    def _save(self):
        if self._collection is not None:
            try:
                existing_ids = set(self._memories.keys())
                for memory in self._memories.values():
                    self._collection.replace_one({"id": memory.id}, memory.to_dict(), upsert=True)
                self._collection.delete_many({"id": {"$nin": list(existing_ids)}})
                return
            except Exception:
                pass
        try:
            payload = [m.to_dict() for m in self._memories.values()]
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def add_or_update_memory(self, memory: Memory) -> Optional[Memory]:
        """
        Add new memory or update existing one.

        Rules:
        - New memory starts at 0.7
        - Same key+value confirms (+0.1)
        - Same key + different value overwrites and resets to 0.7
        """
        memory = self._normalize_memory(memory)
        if memory is None:
            return None

        existing = self._find_by_key(memory.key, memory.user_id)
        if existing:
            if existing.value.lower() == memory.value.lower():
                existing.confidence = min(1.0, existing.confidence + 0.1)
            else:
                existing.value = memory.value
                existing.confidence = 0.7
            self._apply_confidence_policy(existing)
            existing.last_updated = datetime.now()
            self._save()
            return existing

        self._apply_confidence_policy(memory)
        self._memories[memory.id] = memory
        self._key_index[(memory.user_id, memory.key.lower())] = memory.id
        self._save()
        return memory

    def get_all_memories(self) -> List[Memory]:
        return list(self._memories.values())

    def get_user_memories(self, user_id: str) -> List[Memory]:
        return [m for m in self._memories.values() if m.user_id == user_id]

    def get_memories_by_confidence(self, user_id: str, min_confidence: float = 0.5) -> List[Memory]:
        return [
            m for m in self._memories.values()
            if m.user_id == user_id and m.confidence >= min_confidence
        ]

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            self._key_index.pop((memory.user_id, memory.key.lower()), None)
            del self._memories[memory_id]
            self._save()
            return True
        return False

    def apply_strong_feedback(self, user_id: str, user_message: str) -> int:
        """
        Apply strong sentiment updates to matching memories.
        - Strong criticism: confidence -= 0.7
        - Strong appreciation: confidence += 0.7
        Returns number of affected memories.
        """
        msg = (user_message or "").lower()
        if not msg:
            return 0

        negative_targets = self._extract_feedback_targets(
            msg,
            patterns=[
                r"\bi\s+(?:really\s+)?hate\s+([^,.!?]+)",
                r"\bi\s+strongly\s+dislike\s+([^,.!?]+)",
                r"\bi\s+absolutely\s+dislike\s+([^,.!?]+)",
                r"\bi\s+can't\s+stand\s+([^,.!?]+)",
            ],
        )
        positive_targets = self._extract_feedback_targets(
            msg,
            patterns=[
                r"\bi\s+(?:really\s+)?love\s+([^,.!?]+)",
                r"\bi\s+absolutely\s+love\s+([^,.!?]+)",
                r"\bi\s+strongly\s+like\s+([^,.!?]+)",
                r"\bi\s+adore\s+([^,.!?]+)",
            ],
        )

        adjusted = 0
        user_memories = self.get_user_memories(user_id)
        for memory in user_memories:
            mem_text = f"{memory.key} {memory.value}".lower()
            if any(t and (t in mem_text or mem_text in t) for t in negative_targets):
                memory.confidence = max(0.5, memory.confidence - 0.7)
                memory.last_updated = datetime.now()
                adjusted += 1
            elif any(t and (t in mem_text or mem_text in t) for t in positive_targets):
                memory.confidence = min(1.0, memory.confidence + 0.7)
                memory.last_updated = datetime.now()
                adjusted += 1

        if adjusted:
            self._save()
        return adjusted

    def _extract_feedback_targets(self, msg: str, patterns: List[str]) -> List[str]:
        targets: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, msg):
                target = (match.group(1) or "").strip().lower()
                target = re.sub(r"^(?:eating|watching|using)\s+", "", target).strip()
                if len(target) >= 2:
                    targets.append(target)
        return targets

    def _find_by_key(self, key: str, user_id: str) -> Optional[Memory]:
        key_l = (key or "").strip().lower()
        if not key_l:
            return None
        indexed_id = self._key_index.get((user_id, key_l))
        if indexed_id:
            memory = self._memories.get(indexed_id)
            if memory and memory.user_id == user_id and memory.key.lower() == key_l:
                return memory
            self._key_index.pop((user_id, key_l), None)
        for memory in self._memories.values():
            if memory.user_id == user_id and memory.key.lower() == key_l:
                self._key_index[(user_id, key_l)] = memory.id
                return memory
        return None

    def _rebuild_index(self):
        self._key_index = {}
        for memory in self._memories.values():
            key_l = (memory.key or "").strip().lower()
            if not key_l:
                continue
            self._key_index[(memory.user_id, key_l)] = memory.id

    def _is_broad_rule(self, memory: Memory) -> bool:
        md = memory.metadata or {}
        scope = str(md.get("scope", "")).upper()
        rule_type = str(md.get("rule_type", "")).lower()
        key_l = (memory.key or "").lower()
        return (
            (scope == "GLOBAL_SCOPE" and rule_type in {"universal", "generic"})
            or key_l == "entertainment.global_character_rule"
            or key_l.startswith("custom.preference_rule.")
        )

    def _apply_confidence_policy(self, memory: Memory):
        memory.confidence = min(1.0, max(0.5, memory.confidence))
        if self._is_broad_rule(memory):
            memory.confidence = min(memory.confidence, 0.55)

    def _normalize_memory(self, memory: Memory) -> Optional[Memory]:
        key = (memory.key or "").strip().lower()
        value = (memory.value or "").strip()
        value_l = value.lower()

        drop_keys = {
            "contextual_conversation_topic",
            "has_not_shared_favourite_food",
            "has_not_shared_favorite_food",
            "study_location",
        }
        drop_values = {
            "somewhere to be determined",
            "in year",
            "the city where my college is",
            "true",
            "false",
            "unknown",
            "n/a",
        }
        invalid_name_values = {
            "pursuing",
            "working",
            "studying",
            "trying",
            "doing",
            "going",
            "learning",
        }

        if not key or not value:
            return None
        # Reject self-referential/placeholder values that only restate the key.
        key_tokens = set(re.findall(r"[a-z0-9]+", key))
        value_tokens = set(re.findall(r"[a-z0-9]+", value_l))
        if value_tokens and len(value_tokens) <= 4 and value_tokens.issubset(key_tokens.union({"fav", "favorite", "favourite", "char", "character"})):
            return None
        if value_l in {
            "fav char in naruto",
            "favorite character in naruto",
            "in hells paradise",
            "favorite character",
        }:
            return None
        if key in drop_keys:
            return None
        if value_l in drop_values:
            return None
        if key == "name" and value_l in invalid_name_values:
            return None

        # Migrate generic food preferences into schema key.
        food_terms = {"pizza", "burger", "biryani", "pasta", "noodles", "rice", "sushi"}
        if key == "preference":
            if re.search(r"\beat(ing)?\b", value_l) or any(t in value_l for t in food_terms):
                key = "lifestyle.food"
                value = re.sub(r"^\s*eating\s+", "", value, flags=re.IGNORECASE).strip()
                value_l = value.lower()
                if not value:
                    return None

        # Normalize identity/lifestyle/entertainment aliases into schema.
        if key in {"name"}:
            key = "core_identity.name"
        if key in {"location", "college", "university"}:
            key = "core_identity.college"
        if key in {"year", "year of graduation", "graduation year"}:
            key = "core_identity.year"
        if key in {"favorite_food"}:
            key = "lifestyle.food"
        if key in {"favourite soft drink", "favorite soft drink", "soft drink"}:
            key = "lifestyle.drinks"
        if key in {"interest", "interests"} and "gym" in value_l:
            key = "lifestyle.gym"

        # Normalize entertainment aliases.
        if key in {"interest", "interests"} and "aot" in value_l:
            key = "entertainment.anime"
        if key in {"anime_preference"}:
            key = "entertainment.anime"

        # Normalize favorite-character preference keys without hardcoding values.
        if key in {
            "favorite_character",
            "favourite_character",
            "character_preference",
            "fav_character",
            "favorite_character_rule",
            "favorite anime character",
            "favourite anime character",
        }:
            key = "entertainment.character_preference"
            value = re.sub(r"\s+", " ", value).strip()
            value_l = value.lower()

        # Normalize anime-specific favorite character key format.
        if key.startswith("favorite character in "):
            anime = key.replace("favorite character in ", "").strip()
            anime = re.sub(r"[^a-z0-9]+", "_", anime).strip("_")
            if not anime:
                return None
            key = f"entertainment.favorite_character_in_{anime}"
        if key.startswith("favourite character in "):
            anime = key.replace("favourite character in ", "").strip()
            anime = re.sub(r"[^a-z0-9]+", "_", anime).strip("_")
            if not anime:
                return None
            key = f"entertainment.favorite_character_in_{anime}"

        # Keep schema-only keys plus approved exception/rule keys.
        schema_allowed = (
            key.startswith("core_identity.")
            or key.startswith("lifestyle.")
            or key.startswith("entertainment.")
            or key.startswith("technical_stack.")
            or key.startswith("custom.")
        )
        if not schema_allowed:
            # Preserve unknown keys instead of dropping user data.
            safe_key = re.sub(r"[^a-z0-9._-]+", "_", key).strip("_.-")
            if not safe_key:
                return None
            key = f"custom.{safe_key}"

        memory.key = key
        memory.value = value
        self._backfill_metadata(memory)
        return memory

    def _backfill_metadata(self, memory: Memory) -> bool:
        """
        Backfill missing metadata for legacy memories so strict scoped retrieval
        can still use previously stored user info.
        Returns True if metadata was updated.
        """
        md = dict(memory.metadata or {})
        if md.get("domain") and md.get("category") and md.get("scope"):
            return False

        key = (memory.key or "").lower()
        updated = False

        if key.startswith("core_identity."):
            md.setdefault("domain", "identity")
            md.setdefault("category", key.split(".", 1)[1])
            md.setdefault("scope", "PROFILE")
            updated = True
        elif key.startswith("lifestyle."):
            md.setdefault("domain", "lifestyle")
            md.setdefault("category", key.split(".", 1)[1])
            md.setdefault("scope", "GLOBAL_SCOPE")
            updated = True
        elif key.startswith("technical_stack."):
            md.setdefault("domain", "technical_stack")
            md.setdefault("category", key.split(".", 1)[1])
            md.setdefault("scope", "GLOBAL_SCOPE")
            updated = True
        elif key.startswith("entertainment.character_exception."):
            subject = key.replace("entertainment.character_exception.", "").strip()
            if subject:
                md.setdefault("domain", "entertainment")
                md.setdefault("category", "favorite_character")
                md.setdefault("scope", "LOCAL_SCOPE")
                md.setdefault("subject", subject)
                md.setdefault("rule_type", "exception")
                updated = True
        elif key.startswith("entertainment.favorite_character_in_"):
            subject = key.replace("entertainment.favorite_character_in_", "").strip()
            if subject:
                md.setdefault("domain", "entertainment")
                md.setdefault("category", "favorite_character")
                md.setdefault("scope", "LOCAL_SCOPE")
                md.setdefault("subject", subject)
                md.setdefault("rule_type", "specific")
                updated = True
        elif key == "entertainment.global_character_rule":
            md.setdefault("domain", "entertainment")
            md.setdefault("category", "favorite_character")
            md.setdefault("scope", "GLOBAL_SCOPE")
            md.setdefault("subject", "anime")
            md.setdefault("rule_type", "universal")
            updated = True
        elif key == "entertainment.anime":
            md.setdefault("domain", "entertainment")
            md.setdefault("category", "anime")
            md.setdefault("scope", "GLOBAL_SCOPE")
            updated = True
        elif key.startswith("custom.preference_rule."):
            subject = key.replace("custom.preference_rule.", "").strip()
            if subject:
                md.setdefault("domain", "generic")
                md.setdefault("category", subject)
                md.setdefault("scope", "GLOBAL_SCOPE")
                md.setdefault("subject", subject)
                md.setdefault("rule_type", "universal")
                updated = True
        elif key.startswith("custom.critical."):
            subject = key.replace("custom.critical.", "").strip() or "general"
            md.setdefault("domain", "critical_context")
            md.setdefault("category", subject)
            md.setdefault("scope", "GLOBAL_SCOPE")
            md.setdefault("subject", subject)
            md.setdefault("importance", "critical")
            updated = True

        if updated:
            memory.metadata = md
        return updated

    def ensure_bootstrap_memories(self, user_id: str):
        # Kept as a no-op for backward compatibility with existing callers.
        return

    def clear(self):
        self._memories.clear()
        self._key_index.clear()
        self._save()


_memory_store = MemoryStore()


def get_memory_store() -> MemoryStore:
    return _memory_store

