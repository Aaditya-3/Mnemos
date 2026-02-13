"""
Optional OpenTelemetry tracing hooks.
"""

from __future__ import annotations

from contextlib import contextmanager


try:
    from opentelemetry import trace  # type: ignore

    _tracer = trace.get_tracer("mnemos")
except Exception:
    _tracer = None


@contextmanager
def trace_span(name: str, **attrs):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        yield span

