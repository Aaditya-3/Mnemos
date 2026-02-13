"""
Pydantic schemas for semantic memory APIs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


MemoryType = Literal["fact", "preference", "emotional", "goal", "project", "transient", "factual", "long_term_goal", "project_specific", "temporary_context"]
MemoryScope = Literal["global", "user", "conversation", "project"]


class MemoryNodeSchema(BaseModel):
    memory_id: str
    user_id: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    type: str
    importance_score: float
    reinforcement_count: int
    created_at: datetime
    last_accessed: Optional[datetime] = None
    decay_factor: float
    scope: str
    source_message_id: str = ""
    metadata: dict = Field(default_factory=dict)
    is_active: bool = True
    is_archived: bool = False
    archived_at: Optional[datetime] = None


class MemoryMaintenanceResult(BaseModel):
    updated: int = 0
    archived: int = 0
    deleted: int = 0
    compressed: int = 0

