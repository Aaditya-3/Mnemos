"""
Web lookup tool using existing realtime context helpers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.core.tools.realtime_info import get_realtime_context
from backend.app.tools.registry import ToolSpec, tool_registry


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300)


def run_web_search(payload: WebSearchInput) -> dict:
    result = get_realtime_context(payload.query) or "No result found."
    return {"query": payload.query, "result": result}


tool_registry.register(
    ToolSpec(
        name="web_search",
        description="Fetch a concise web snippet for a query.",
        input_model=WebSearchInput,
        execute=run_web_search,
    )
)

