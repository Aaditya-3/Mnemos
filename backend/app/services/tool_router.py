"""
Agent-style tool routing and execution.
"""

from __future__ import annotations

import json
from typing import Any

from backend.app.core.llm.groq_client import generate_response
from backend.app.observability.logging import log_event
from backend.app.tools import calculator  # noqa: F401
from backend.app.tools import currency  # noqa: F401
from backend.app.tools import web_search  # noqa: F401
from backend.app.tools.registry import tool_registry


def _strip_json_block(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def decide_tool_call(user_message: str) -> dict[str, Any]:
    tools = tool_registry.list_tools()
    planner_prompt = (
        "You are a tool planner.\n"
        "Return ONLY JSON object with shape:\n"
        '{"call_tool": "tool_name_or_null", "tool_input": {...}, "reason": "short"}\n'
        "If no tool is needed, set call_tool to null.\n"
        f"Available tools: {json.dumps(tools)}\n"
        f"User message: {user_message}"
    )
    raw = generate_response(planner_prompt)
    parsed_raw = _strip_json_block(raw)
    try:
        data = json.loads(parsed_raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"call_tool": None, "tool_input": {}, "reason": "planner_parse_failed"}


def run_agent_turn(user_message: str, max_loops: int = 2) -> dict[str, Any]:
    tool_events: list[dict[str, Any]] = []
    context_note = ""

    for _ in range(max_loops):
        planner_input = user_message if not context_note else f"{user_message}\n\nTool context:\n{context_note}"
        decision = decide_tool_call(planner_input)
        tool_name = decision.get("call_tool")
        tool_input = decision.get("tool_input") or {}
        if not tool_name:
            break
        try:
            tool_result = tool_registry.execute(str(tool_name), dict(tool_input))
            tool_events.append({"tool": tool_name, "input": tool_input, "result": tool_result})
            context_note = json.dumps(tool_result)
        except Exception as exc:
            tool_events.append({"tool": tool_name, "input": tool_input, "error": str(exc)})
            context_note = f"Tool {tool_name} error: {exc}"
            break

    final_prompt = (
        "You are an assistant that may receive tool results.\n"
        "Give a concise and direct answer.\n"
        f"User message: {user_message}\n"
        f"Tool events: {json.dumps(tool_events)}"
    )
    reply = generate_response(final_prompt)
    log_event("agent_turn_completed", tool_calls=len(tool_events))
    return {"reply": reply, "tool_events": tool_events}

