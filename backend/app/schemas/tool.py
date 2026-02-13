"""
Tool call schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCallSchema(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    arguments: dict = Field(default_factory=dict)


class ToolPlannerResponseSchema(BaseModel):
    tool_call: ToolCallSchema | None = None
    reason: str = ""

