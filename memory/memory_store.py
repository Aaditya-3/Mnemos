"""
Memory Store

Persistent storage for user memories.
Handles CRUD operations, confidence management, and disk persistence.
"""

from typing import Optional, List
from datetime import datetime
from pathlib import Path
import json

from .memory_schema import Memory


class MemoryStore:
    """
    Persistent storage for memories (single user).

    Backed by an in-memory dict with JSON file persistence so memories
    survive server restarts and are shared across all chat sessions.
    """
    
    def __init__(self):
        """Initialize memory store and load any existing memories from disk."""
        self._memories: dict[str, Memory] = {}
        # Store memories in a JSON file next to this module
        self._storage_path: Path = Path(__file__).resolve().parent / "memories.json"
        self._load_from_disk()
    
    def add_or_update_memory(self, memory: Memory) -> Memory:
        """
        Add new memory or update existing one.
        
        Confidence rules:
        - Initial: 0.7
        - Confirmation (same key/value): +0.1
        - Contradiction (same key, different value): -0.3
        - Delete if confidence < 0.3
        """
        # Check if memory with same key exists
        existing = self._find_by_key(memory.key)
        
        if existing:
            # Same value = confirmation
            if existing.value.lower() == memory.value.lower():
                existing.confidence = min(1.0, existing.confidence + 0.1)
            else:
                # Different value = contradiction
                existing.confidence = max(0.0, existing.confidence - 0.3)
                existing.value = memory.value
            
            existing.last_updated = datetime.now()
            
            # Delete if confidence too low
            if existing.confidence < 0.3:
                del self._memories[existing.id]
                return None
            
            return existing
        else:
            # New memory: ignore obviously bad entries
            # - empty/whitespace key or value
            # - generic/unknown keys that are not helpful
            bad_keys = {"unknown", "n/a", "none", "null", "undefined"}
            if (not memory.key or not memory.key.strip() or
                not memory.value or not memory.value.strip() or
                memory.key.strip().lower() in bad_keys or
                memory.value.strip().lower() in bad_keys):
                # Do NOT store this memory at all
                print(f"Ignoring low-quality memory: key={memory.key!r}, value={memory.value!r}")
                return None

            self._memories[memory.id] = memory
        
        # Persist changes
        self._save_to_disk()
        return memory
    
    def get_all_memories(self) -> List[Memory]:
        """Get all stored memories."""
        return list(self._memories.values())
    
    def get_memories_by_confidence(self, min_confidence: float = 0.3) -> List[Memory]:
        """
        Get memories above confidence threshold.
        
        Default threshold is 0.3 so that any memory that has not been
        "forgotten" (deleted) will still be available to the model.
        """
        return [m for m in self._memories.values() if m.confidence >= min_confidence]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID. Returns True if deleted, False if not found."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save_to_disk()
            return True
        return False
    
    def _find_by_key(self, key: str) -> Optional[Memory]:
        """Find memory by key (case-insensitive)."""
        for memory in self._memories.values():
            if memory.key.lower() == key.lower():
                return memory
        return None
    
    def clear(self):
        """Clear all memories (for testing)."""
        self._memories.clear()
        self._save_to_disk()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load memories from JSON file if it exists."""
        if not self._storage_path.exists():
            return
        try:
            raw = self._storage_path.read_text(encoding="utf-8")
            if not raw.strip():
                return
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    try:
                        mem = Memory.from_dict(item)
                        self._memories[mem.id] = mem
                    except Exception as e:
                        # Corrupt entry – skip but don't crash
                        print(f"Skipping invalid memory entry in storage: {e}")
        except Exception as e:
            # Any disk/JSON issue should not crash the backend
            print(f"Failed to load memories from disk: {e}")

    def _save_to_disk(self) -> None:
        """Persist all memories to JSON file."""
        try:
            payload = [m.to_dict() for m in self._memories.values()]
            self._storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            # Persistence errors should not crash the app, but log them
            print(f"Failed to save memories to disk: {e}")


# Global singleton instance
_memory_store = MemoryStore()


def get_memory_store() -> MemoryStore:
    """Get the global memory store instance."""
    return _memory_store
