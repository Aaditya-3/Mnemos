"""
Memory Retriever

Retrieves relevant memories for the current conversation context.
"""

from typing import List
from .memory_store import get_memory_store
from .memory_schema import Memory


def retrieve_memories(user_message: str) -> str:
    """
    Retrieve relevant memories for the current user message.
    
    Returns a short summary string suitable for system prompt.
    Uses all memories with confidence >= 0.5 (new memories start at 0.7).
    """
    store = get_memory_store()
    # Get all memories with confidence >= 0.5 (includes new memories at 0.7)
    memories = store.get_memories_by_confidence(min_confidence=0.5)
    
    if not memories:
        return ""
    
    # Sort by confidence (highest first) and recency
    memories_sorted = sorted(memories, key=lambda m: (m.confidence, m.last_updated), reverse=True)
    
    # Build detailed summary string with better formatting
    parts = []
    for memory in memories_sorted:
        parts.append(f"- {memory.key}: {memory.value}")
    
    context = "\n".join(parts)
    return context
