"""
Context ranking layer.
"""

from __future__ import annotations

from backend.app.core.config import get_settings
from backend.app.orchestrator.types import ContextBundle


class ContextRanker:
    def __init__(self):
        self.settings = get_settings()

    def rank(self, bundle: ContextBundle) -> ContextBundle:
        if not bundle.semantic_rows:
            return bundle

        rows = sorted(
            bundle.semantic_rows,
            key=lambda r: (
                float(r.get("final_score", 0.0)),
                float(r.get("similarity_score", 0.0)),
                float(r.get("importance_score", 0.0)),
            ),
            reverse=True,
        )
        selected = []
        used = 0
        for row in rows:
            content = str(row.get("content", ""))
            est = max(1, len(content) // 4)
            if selected and (used + est) > self.settings.semantic_token_budget:
                break
            selected.append(row)
            used += est

        bundle.semantic_rows = selected
        lines = []
        for idx, row in enumerate(selected, start=1):
            lines.append(
                f"- ({idx}) {row.get('content', '')} "
                f"[type={row.get('memory_type', '')}; scope={row.get('scope', '')}; "
                f"importance={float(row.get('importance_score', 0.0)):.2f}; "
                f"final={float(row.get('final_score', 0.0)):.2f}]"
            )
        bundle.semantic_memory_context = "\n".join(lines)
        return bundle

