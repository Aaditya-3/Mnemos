"""
Safe calculator tool.
"""

from __future__ import annotations

import ast
import operator
from pydantic import BaseModel

from backend.app.tools.registry import ToolSpec, tool_registry


class CalculatorInput(BaseModel):
    expression: str


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _eval_node(node.operand)
        return _UNARY_OPS[type(node.op)](value)
    raise ValueError("Unsupported expression")


def run_calculator(payload: CalculatorInput) -> dict:
    expr = (payload.expression or "").strip()
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree.body)
    return {"expression": expr, "result": result}


tool_registry.register(
    ToolSpec(
        name="calculator",
        description="Evaluate arithmetic expressions",
        input_model=CalculatorInput,
        execute=run_calculator,
    )
)

