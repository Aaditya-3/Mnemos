"""
Token/cost estimation utilities.
"""

from __future__ import annotations

from backend.app.core.config import get_settings


def estimate_tokens(text: str) -> int:
    return max(1, len((text or "").strip()) // 4)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    settings = get_settings()
    in_cost = (input_tokens / 1000.0) * settings.llm_cost_input_per_1k
    out_cost = (output_tokens / 1000.0) * settings.llm_cost_output_per_1k
    return round(in_cost + out_cost, 8)

