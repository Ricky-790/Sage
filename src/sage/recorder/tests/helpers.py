"""Shared helpers for recorder tests: parse JSONL traces and verify span trees."""

import json
from pathlib import Path

from sage.recorder.models import Span


def read_jsonl_spans(path) -> list[Span]:
    """Read a traces JSONL file and validate every line against the Span schema."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, f"no spans written to {path}"
    for line in lines:
        # extra="forbid" makes this reject any field outside the schema.
        Span.model_validate(line)
    return [Span.model_validate(line) for line in lines]


def single_trace(spans: list[Span]) -> str:
    trace_ids = {span.trace_id for span in spans}
    assert len(trace_ids) == 1, f"expected one trace, got {trace_ids}"
    return trace_ids.pop()


def build_children(spans: list[Span]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {}
    for span in spans:
        if span.parent_span_id:
            children.setdefault(span.parent_span_id, set()).add(span.span_id)
    return children


def assert_time_containment(spans: list[Span]) -> None:
    """Every child span must start and end inside its parent's interval.

    Parents can dangle: LangChain wrapper runs reference themselves as parents
    without emitting their own events, and those are skipped.
    """
    by_id = {span.span_id: span for span in spans}
    for span in spans:
        if not span.parent_span_id:
            continue
        parent = by_id.get(span.parent_span_id)
        if parent is None:
            continue
        assert span.started_at >= parent.started_at, (
            f"span {span.name!r} starts before parent {parent.name!r}"
        )
        assert span.ended_at is not None
        assert parent.ended_at is not None
        assert span.ended_at <= parent.ended_at, (
            f"span {span.name!r} ends after parent {parent.name!r}"
        )


def find_by_name(spans: list[Span], name: str) -> Span:
    matches = [span for span in spans if span.name == name]
    # The recorder names spans by __qualname__; locally-defined functions record
    # as "factory.<locals>.name" and must still match their plain name.
    matches += [
        span
        for span in spans
        if span.name not in matches and span.name.endswith("<locals>." + name)
    ]
    assert matches, f"no span named {name!r} (got {[s.name for s in spans]})"
    return matches[0]
