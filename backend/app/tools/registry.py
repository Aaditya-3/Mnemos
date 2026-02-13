"""
Tool registry and execution framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from backend.app.observability.logging import log_event

_BLOCKED_TOOL_ARG_PATTERNS = ("__", "import", "exec", "eval", "subprocess", "os.")


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
        self._sandbox_validate_payload(tool_name=tool_name, payload=payload or {})
        try:
            parsed = spec.input_model.model_validate(payload or {})
        except ValidationError as exc:
            raise ValueError(f"Invalid input for {tool_name}: {exc}") from exc
        result = spec.execute(parsed)
        log_event("tool_executed", tool_name=tool_name)
        return result

    def _sandbox_validate_payload(self, tool_name: str, payload: dict[str, Any]):
        def _walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    _walk(str(k))
                    _walk(v)
                return
            if isinstance(value, list):
                for item in value:
                    _walk(item)
                return
            text = str(value).lower()
            for pattern in _BLOCKED_TOOL_ARG_PATTERNS:
                if pattern in text:
                    raise ValueError(f"Blocked tool payload for `{tool_name}` due to sandbox pattern `{pattern}`")

        _walk(payload)


tool_registry = ToolRegistry()
