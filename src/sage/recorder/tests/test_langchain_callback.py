"""Offline tests of the LangChain callback path through a stubbed chat model.

No API keys are needed; the handler is driven directly with the same
callbacks LangChain emits during an agent run.
"""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from sage.recorder.integrations.langchain import LangChainRecorderCallback
from sage.recorder.models import SpanKind, SpanStatus, SpanUsage
from sage.recorder.tests.helpers import (
    assert_time_containment,
    build_children,
    find_by_name,
    read_jsonl_spans,
    single_trace,
)

CONTRACT = "Recorder must expose record_span(Span) and write pydantic Span JSONL"


def drive_happy_path(handler):
    root, llm, tool = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({"name": "AgentExecutor"}, {"input": "weather?"}, run_id=root)
    handler.on_chat_model_start(
        {"name": "fake-chat-model"}, [[{"role": "user", "content": "weather?"}]],
        run_id=llm, parent_run_id=root,
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[
                ChatGeneration(
                    message=AIMessage(
                        content="calling get_location",
                        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                    )
                )
            ]],
            llm_output={},
        ),
        run_id=llm,
        parent_run_id=root,
    )
    handler.on_tool_start(
        {"name": "get_location"}, '{"user_id": "demo"}',
        run_id=tool, parent_run_id=llm, inputs={"user_id": "demo"},
    )
    handler.on_tool_end({"city": "Amsterdam"}, run_id=tool, parent_run_id=llm)
    return root, llm, tool


@pytest.mark.xfail(reason=CONTRACT)
def test_callback_records_tree_with_usage(recorder, trace_path):
    handler = LangChainRecorderCallback(recorder)
    root, llm, tool = drive_happy_path(handler)

    spans = read_jsonl_spans(trace_path)
    root_span = find_by_name(spans, "AgentExecutor")
    llm_span = find_by_name(spans, "fake-chat-model")
    tool_span = find_by_name(spans, "get_location")

    assert single_trace(spans) == root_span.trace_id
    assert root_span.kind == SpanKind.CHAIN
    assert root_span.status == SpanStatus.OK
    assert llm_span.kind == SpanKind.LLM
    assert llm_span.parent_span_id == root_span.span_id
    assert llm_span.usage == SpanUsage(input_tokens=5, output_tokens=3, total_tokens=8)
    assert tool_span.kind == SpanKind.TOOL
    assert tool_span.parent_span_id == llm_span.span_id
    assert tool_span.output == {"city": "Amsterdam"}

    assert build_children(spans) == {
        root_span.span_id: {llm_span.span_id},
        llm_span.span_id: {tool_span.span_id},
    }
    assert_time_containment(spans)


@pytest.mark.xfail(reason="on_chain_start must tolerate serialized=None from LangChain")
def test_callback_tolerates_none_serialized(recorder, trace_path):
    handler = LangChainRecorderCallback(recorder)
    handler.on_chain_start(None, {"input": "x"}, run_id=uuid4())

    spans = read_jsonl_spans(trace_path)
    assert spans[0].name == "chain"
    assert spans[0].kind == SpanKind.CHAIN


@pytest.mark.xfail(reason=CONTRACT)
def test_callback_records_llm_and_tool_errors(recorder, trace_path):
    handler = LangChainRecorderCallback(recorder)
    llm, tool = uuid4(), uuid4()

    handler.on_chat_model_start({"name": "fake-chat-model"}, [[{"role": "user", "content": "hi"}]], run_id=llm)
    handler.on_llm_error(RuntimeError("provider down"), run_id=llm)

    handler.on_tool_start({"name": "get_weather"}, '{"city": "Amsterdam"}', run_id=tool)
    handler.on_tool_error(ValueError("bad city"), run_id=tool)

    spans = read_jsonl_spans(trace_path)
    llm_span = find_by_name(spans, "fake-chat-model")
    tool_span = find_by_name(spans, "get_weather")

    for span, message in ((llm_span, "provider down"), (tool_span, "bad city")):
        assert span.status == SpanStatus.ERROR
        assert span.error.type in ("RuntimeError", "ValueError")
        assert span.error.message == message
    assert all(span.parent_span_id is None for span in spans)


def test_callback_end_without_start_is_ignored(recorder, trace_path):
    handler = LangChainRecorderCallback(recorder)
    handler.on_tool_end({"city": "Amsterdam"}, run_id=uuid4())
    handler.on_chain_end({"output": "x"}, run_id=uuid4())

    spans = read_jsonl_spans(trace_path)
    assert spans == [], "end events without a start must record nothing"
