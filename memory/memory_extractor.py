"""
Memory Extractor

Uses LLM to extract long-term memories from user messages.
Only extracts persistent, reusable information.
"""

import json
import re
from typing import Optional
from .memory_schema import Memory
from .memory_store import get_memory_store
from backend.app.core.llm.groq_client import generate_response


def _extract_patterns(user_message: str) -> Optional[Memory]:
    """
    Pattern-based memory extraction as fallback.
    Extracts common patterns without calling LLM.
    """
    message_lower = user_message.lower()
    
    # Pattern: "I'm [name]" or "My name is [name]"
    name_patterns = [
        r"i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"my\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"i\s+am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"call\s+me\s+([A-Z][a-z]+)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if len(name) > 1 and len(name) < 50:  # Reasonable name length
                return Memory.create(type="fact", key="name", value=name, confidence=0.7)
    
    # Pattern: "I prefer [X]" or "I like [X]"
    preference_patterns = [
        r"i\s+prefer\s+([^,.!?]+)",
        r"i\s+like\s+([^,.!?]+)",
        r"my\s+favorite\s+([^,.!?]+)\s+is\s+([^,.!?]+)"
    ]
    for pattern in preference_patterns:
        match = re.search(pattern, message_lower)
        if match:
            if len(match.groups()) == 2:
                key = match.group(1).strip()
                value = match.group(2).strip()
            else:
                value = match.group(1).strip()
                key = "preference"
            if len(value) > 2 and len(value) < 100:
                return Memory.create(type="preference", key=key, value=value, confidence=0.7)
    
    return None


def extract_memory(user_message: str) -> Optional[Memory]:
    """
    Extract long-term memory from user message if present.
    
    Returns Memory object if found, None otherwise.
    Uses pattern matching first, then LLM extraction.
    """
    # Try pattern-based extraction first (fast, no API call)
    pattern_memory = _extract_patterns(user_message)
    if pattern_memory:
        stored = get_memory_store().add_or_update_memory(pattern_memory)
        if stored:
            print(f"Memory stored (pattern): {stored.key} = {stored.value}")
        return stored
    
    # Fallback to LLM-based extraction
    prompt = """You are a memory extraction system. Your job is to identify and extract ANY personal information, preferences, facts, or details about the user that should be remembered for future conversations.

EXTRACT these types of information:
- Personal facts: name, age, location, occupation, role, company, education, skills, hobbies
- Preferences: programming languages, tools, frameworks, styles, formats, colors, themes
- Constraints: deadlines, budgets, time zones, availability, limitations
- Context: current projects, goals, interests, relationships, background
- Any stable information about the user that would be useful to remember

IMPORTANT: Be proactive! Extract information even if it's mentioned casually or indirectly.

Examples of what to extract:
- "I'm John" → fact: name = John
- "I prefer Python" → preference: programming_language = Python
- "I work at Google" → fact: company = Google
- "I'm a software engineer" → fact: occupation = software engineer
- "I'm in New York" → fact: location = New York
- "I like dark themes" → preference: theme = dark
- "My deadline is next week" → constraint: deadline = next week

DO NOT extract:
- Simple greetings without information (hello, hi, thanks)
- Pure emotions without context (happy, sad)
- One-off questions that don't reveal user info

Return JSON format:
{
  "type": "preference" | "constraint" | "fact",
  "key": "descriptive_key_name",
  "value": "the actual information"
}

If you find ANY extractable information, return the JSON. Only return null if there is absolutely no personal information.

User message: """ + user_message
    
    try:
        # Call LLM to extract memory
        response = generate_response(prompt)
        
        if not response:
            print("Memory extraction: Empty response from LLM")
            return None
        
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # Handle null response
        if response.lower() == "null" or not response:
            return None
        
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Invalid JSON, no memory extracted
            return None
        
        # Validate and create memory
        if isinstance(data, dict) and "type" in data and "key" in data and "value" in data:
            # Validate type
            if data["type"] not in ["preference", "constraint", "fact"]:
                return None
            
            # Validate key and value are strings
            if not isinstance(data["key"], str) or not isinstance(data["value"], str):
                return None
            
            try:
                memory = Memory.create(
                    type=data["type"],
                    key=data["key"],
                    value=data["value"],
                    confidence=0.7
                )
                
                # Store in memory store
                stored = get_memory_store().add_or_update_memory(memory)
                if stored:
                    print(f"Memory stored: {stored.key} = {stored.value} (confidence: {stored.confidence:.2f})")
                return stored
            except (ValueError, TypeError) as e:
                # Invalid memory data
                print(f"Invalid memory data: {e}")
                return None
        
        return None
        
    except RuntimeError as e:
        # API key missing or Groq API/model error - log but don't crash
        print(f"Memory extraction API error (non-fatal): {e}")
        return None
    except json.JSONDecodeError:
        # Invalid JSON, no memory extracted
        return None
    except Exception as e:
        # Log error but don't crash (e.g. network, rate limit, model error)
        print(f"Memory extraction error (non-fatal): {type(e).__name__}: {e}")
        return None
