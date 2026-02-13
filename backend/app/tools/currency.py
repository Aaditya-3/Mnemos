"""
Currency conversion tool via existing realtime helper.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.core.tools.realtime_info import get_realtime_context
from backend.app.tools.registry import ToolSpec, tool_registry


class CurrencyInput(BaseModel):
    amount: float = Field(default=1.0, gt=0)
    base: str = Field(min_length=3, max_length=10)
    target: str = Field(min_length=3, max_length=10)


def run_currency(payload: CurrencyInput) -> dict:
    q = f"{payload.amount:g} {payload.base} to {payload.target}"
    context = get_realtime_context(q) or "No live exchange data available."
    return {"query": q, "result": context}


tool_registry.register(
    ToolSpec(
        name="currency_convert",
        description="Convert currency amounts using live exchange rates.",
        input_model=CurrencyInput,
        execute=run_currency,
    )
)

