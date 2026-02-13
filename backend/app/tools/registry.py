"""
Tool registry and execution framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from backend.app.observability.logging import log_event


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[[BaseModel], dict[str, Any]]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def list_tools(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for spec in self._tools.values():
            items.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_model.model_json_schema(),
                }
            )
        return items

    def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(tool_name)
        if not spec:
            raise ValueError(f"Unknown tool: {tool_name}")
        try:
            parsed = spec.input_model.model_validate(payload or {})
        except ValidationError as exc:
            raise ValueError(f"Invalid input for {tool_name}: {exc}") from exc
        result = spec.execute(parsed)
        log_event("tool_executed", tool_name=tool_name)
        return result


tool_registry = ToolRegistry()

