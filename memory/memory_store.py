"""
Memory Store

In-memory storage for user memories.
Handles CRUD operations and confidence management.
"""

from typing import Optional, List
from datetime import datetime
from .memory_schema import Memory


class MemoryStore:
    """In-memory storage for memories (single user)."""
    
    def __init__(self):
        """Initialize empty memory store."""
        self._memories: dict[str, Memory] = {}
    
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
            # New memory
            self._memories[memory.id] = memory
            return memory
    
    def get_all_memories(self) -> List[Memory]:
        """Get all stored memories."""
        return list(self._memories.values())
    
    def get_memories_by_confidence(self, min_confidence: float = 0.5) -> List[Memory]:
        """Get memories above confidence threshold."""
        return [m for m in self._memories.values() if m.confidence >= min_confidence]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID. Returns True if deleted, False if not found."""
        if memory_id in self._memories:
            del self._memories[memory_id]
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


# Global singleton instance
_memory_store = MemoryStore()


def get_memory_store() -> MemoryStore:
    """Get the global memory store instance."""
    return _memory_store
