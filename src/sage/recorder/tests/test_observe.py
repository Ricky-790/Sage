import pytest

from sage.recorder import Recorder
from sage.recorder.models import SpanKind, SpanStatus
from sage.recorder.tests.helpers import (
    assert_time_containment,
    build_children,
    find_by_name,
    read_jsonl_spans,
    single_trace,
)

TRACE_PATH = "src/sage/recorder/tests/outputs/observe/traces.jsonl"

recorder = Recorder(TRACE_PATH)

@pytest.fixture(scope="module", autouse=True)
def _trace_dir():
    from pathlib import Path

    Path(TRACE_PATH).parent.mkdir(parents=True, exist_ok=True)


@recorder.observe(kind="agent")
async def run_agent(query: str) -> str:
    result = await search_tool(query)
    return f"Search result: {result}"


@recorder.observe(kind="tool")
async def search_tool(query: str) -> dict:
    return {"matches": [query]}


@recorder.observe(kind="tool")
async def failing_tool(query: str) -> dict:
    raise ValueError("tool blew up")


@recorder.observe(kind="agent")
async def run_failing_agent(query: str) -> str:
    return str(await failing_tool(query))


@pytest.mark.asyncio
async def test_observe_nested_lineage(tmp_path):
    result = await run_agent("observability")
    assert result == "Search result: {'matches': ['observability']}"

    spans = read_jsonl_spans(TRACE_PATH)
    trace_id = single_trace(spans)

    agent_span = find_by_name(spans, "run_agent")
    tool_span = find_by_name(spans, "search_tool")

    assert agent_span.trace_id == trace_id
    assert agent_span.kind == SpanKind.AGENT
    assert agent_span.status == SpanStatus.OK
    assert agent_span.input["args"] == ["observability"]
    assert agent_span.output == "Search result: {'matches': ['observability']}"

    # The tool call inside the agent becomes a child span under the agent span.
    assert tool_span.trace_id == trace_id
    assert tool_span.parent_span_id == agent_span.span_id
    assert tool_span.kind == SpanKind.TOOL
    assert tool_span.status == SpanStatus.OK
    assert tool_span.output == {"matches": ["observability"]}

    assert build_children(spans) == {agent_span.span_id: {tool_span.span_id}}
    assert_time_containment(spans)


@pytest.mark.asyncio
async def test_observe_captures_error_span():
    spans_before = len(read_jsonl_spans(TRACE_PATH))
    with pytest.raises(ValueError):
        await run_failing_agent("boom")

    spans = read_jsonl_spans(TRACE_PATH)[spans_before:]
    error_span = find_by_name(spans, "failing_tool")
    assert error_span.status == SpanStatus.ERROR
    assert error_span.error.type == "ValueError"
    assert error_span.error.message == "tool blew up"
    assert_time_containment(spans)
