"""Unit tests for the @observe decorator and the Recorder's JSONL contract.

Tests that read the JSONL file carry an xfail marker with the contract they
encode the recorder must satisfy; they stop being xfail when the recorder
writes pydantic Span schema lines.
"""

import threading

import pytest

from sage.recorder.models import Span, SpanKind, SpanStatus
from sage.recorder.tests.helpers import (
    assert_time_containment,
    build_children,
    find_by_name,
    read_jsonl_spans,
    single_trace,
)

CONTRACT = "Recorder must write pydantic Span schema JSONL; span() emits legacy dict fields today"


def make_agent_pair(recorder):
    @recorder.observe(kind="tool")
    async def search_tool(query: str) -> dict:
        return {"matches": [query]}

    @recorder.observe(kind="agent")
    async def run_agent(query: str) -> str:
        return f"Search result: {await search_tool(query)}"

    return run_agent


@pytest.mark.asyncio
@pytest.mark.xfail(reason=CONTRACT)
async def test_observe_nested_lineage(recorder, trace_path):
    run_agent = make_agent_pair(recorder)

    result = await run_agent("observability")
    assert result == "Search result: {'matches': ['observability']}"

    spans = read_jsonl_spans(trace_path)
    trace_id = single_trace(spans)

    agent_span = find_by_name(spans, "run_agent")
    tool_span = find_by_name(spans, "search_tool")

    assert agent_span.trace_id == trace_id
    assert agent_span.kind == SpanKind.AGENT
    assert agent_span.status == SpanStatus.OK
    assert agent_span.input["args"] == ["observability"]
    assert agent_span.output == "Search result: {'matches': ['observability']}"

    assert tool_span.trace_id == trace_id
    assert tool_span.parent_span_id == agent_span.span_id
    assert tool_span.kind == SpanKind.TOOL
    assert tool_span.status == SpanStatus.OK
    assert tool_span.output == {"matches": ["observability"]}

    assert build_children(spans) == {agent_span.span_id: {tool_span.span_id}}
    assert_time_containment(spans)


@pytest.mark.asyncio
@pytest.mark.xfail(reason=CONTRACT)
async def test_observe_three_level_nesting(recorder, trace_path):
    @recorder.observe(kind="span")
    async def inner(value: str) -> str:
        return value.upper()

    @recorder.observe(kind="tool")
    async def wrap(value: str) -> str:
        return await inner(value)

    @recorder.observe(kind="agent")
    async def root(value: str) -> str:
        return await wrap(value)

    assert await root("seed") == "SEED"

    spans = read_jsonl_spans(trace_path)
    root_span = find_by_name(spans, "root")
    mid_span = find_by_name(spans, "wrap")
    inner_span = find_by_name(spans, "inner")

    assert single_trace(spans) == root_span.trace_id
    assert mid_span.parent_span_id == root_span.span_id
    assert inner_span.parent_span_id == mid_span.span_id
    assert inner_span.output == "SEED"
    assert_time_containment(spans)


@pytest.mark.xfail(reason=CONTRACT)
def test_observe_sync_function(recorder, trace_path):
    @recorder.observe(kind="tool")
    def double(n: int) -> int:
        return n * 2

    assert double(21) == 42

    spans = read_jsonl_spans(trace_path)
    assert find_by_name(spans, "double").output == 42
    assert_time_containment(spans)


@pytest.mark.asyncio
async def test_observe_custom_name_and_kwargs(recorder, trace_path):
    @recorder.observe(name="custom.span", kind="agent")
    async def work(x: int, *, power: int = 2) -> int:
        return x**power

    assert await work(3, power=3) == 27

    spans = read_jsonl_spans(trace_path)
    span = find_by_name(spans, "custom.span")
    assert span.kind == SpanKind.AGENT
    assert span.input == {"args": [3], "kwargs": {"power": 3}}


@pytest.mark.asyncio
@pytest.mark.xfail(reason=CONTRACT)
async def test_observe_unserializable_output(recorder, trace_path):
    marker = object()

    @recorder.observe(kind="tool")
    async def keep() -> object:
        return marker

    result = await keep()
    assert result is marker

    spans = read_jsonl_spans(trace_path)
    assert isinstance(find_by_name(spans, "keep").output, str)


@pytest.mark.asyncio
@pytest.mark.xfail(reason=CONTRACT)
async def test_observe_two_sibling_tool_calls(recorder, trace_path):
    @recorder.observe(kind="tool")
    async def tool_a() -> str:
        return "a"

    @recorder.observe(kind="tool")
    async def tool_b() -> str:
        return "b"

    @recorder.observe(kind="agent")
    async def root() -> list:
        return [await tool_a(), await tool_b()]

    assert await root() == ["a", "b"]

    spans = read_jsonl_spans(trace_path)
    root_span = find_by_name(spans, "root")
    assert build_children(spans) == {
        root_span.span_id: {
            find_by_name(spans, "tool_a").span_id,
            find_by_name(spans, "tool_b").span_id,
        },
    }
    assert_time_containment(spans)


@pytest.mark.asyncio
@pytest.mark.xfail(reason=CONTRACT)
async def test_observe_errors_bubble_to_every_enclosing_span(recorder, trace_path):
    @recorder.observe(kind="tool")
    async def failing() -> None:
        raise ValueError("tool blew up")

    @recorder.observe(kind="agent")
    async def run() -> str:
        return str(await failing())

    with pytest.raises(ValueError):
        await run()

    spans = read_jsonl_spans(trace_path)
    for name in ("failing", "run"):
        span = find_by_name(spans, name)
        assert span.status == SpanStatus.ERROR
        assert span.error.type == "ValueError"
        assert span.error.message == "tool blew up"
    assert_time_containment(spans)


def test_observe_threads_get_isolated_traces(recorder, trace_path):
    @recorder.observe(kind="span")
    def work(n: int) -> int:
        return n * 2

    def run(collection, i):
        collection.append(work(i))

    collected = []
    threads = [threading.Thread(target=run, args=(collected, i)) for i in (2, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert collected == [4, 6]
    spans = read_jsonl_spans(trace_path)
    trace_ids = {span.trace_id for span in spans}
    assert len(spans) == 2, "each thread call records exactly one span"
    assert len(trace_ids) == 2, "each thread call opens its own trace"
    assert all(span.parent_span_id is None for span in spans)


def test_record_span_contract(recorder, trace_path):
    span = Span(
        trace_id="t1",
        span_id="s1",
        name="manual",
        kind=SpanKind.TOOL,
        status=SpanStatus.OK,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration=1.0,
        input={"x": 1},
        output="ok",
    )
    recorder.record_span(span)

    spans = read_jsonl_spans(trace_path)
    assert len(spans) == 1
    recorded = spans[0]
    assert recorded.trace_id == "t1"
    assert recorded.span_id == "s1"
    assert recorded.name == "manual"
    assert recorded.duration == 1.0
