"""Live weather agent test: full trace recorded via the LangChain callback handler."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from sage.recorder import Recorder
from sage.recorder.integrations.langchain import LangChainRecorderCallback
from sage.recorder.tests.helpers import (
    assert_time_containment,
    build_children,
    find_by_name,
    read_jsonl_spans,
    single_trace,
)

load_dotenv()
MODEL_NAME = "google/gemma-4-31b-it:free"
TRACE_PATH = "src/sage/recorder/tests/outputs/weather_agent/traces.jsonl"
chat_model = None


def get_chat_model():
    """Google first, openrouter fallback. Same model either way."""
    global chat_model
    if chat_model is not None:
        return chat_model

    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat_model = ChatGoogleGenerativeAI(
            model="gemma-4-26b-a4b-it", api_key=google_key
        )
        return chat_model

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openrouter import ChatOpenRouter

        chat_model = ChatOpenRouter(model=MODEL_NAME, api_key=openrouter_key)
        return chat_model

    pytest.skip("No API key: set GOOGLE_API_KEY or OPENROUTER_API_KEY")


@tool
def get_location(user_id: str = "demo-user") -> dict:
    """Get the user's current city for the given user id."""
    return {"user_id": user_id, "city": "Amsterdam"}


@tool
def get_weather(city: str) -> dict:
    """Check the current weather for a city."""
    return {"city": city, "condition": "sunny", "temperature_c": 21}


def make_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[get_location, get_weather],
        system_prompt=(
            "You are a weather assistant.\n"
            "To answer any weather question you must, in order:\n"
            "1. Call get_location to find the user's city.\n"
            "2. Call get_weather with that city.\n"
            "3. Report the weather in one sentence."
        ),
    )


def test_weather_agent_full_trace():
    Path(TRACE_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Append-mode writes accumulate across runs; one run must see its own trace only.
    Path(TRACE_PATH).unlink(missing_ok=True)
    recorder = Recorder(TRACE_PATH)
    agent = make_agent()
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What's the weather like where I am?"}
            ]
        },
        config={
            "callbacks": [LangChainRecorderCallback(recorder)],
            "recursion_limit": 12,
        },
    )

    assert result["messages"], "no messages returned"
    spans = read_jsonl_spans(TRACE_PATH)
    assert len(spans) >= 3, (
        f"expected agent, llm and tool spans, got {[s.name for s in spans]}"
    )

    trace_id = single_trace(spans)
    assert_time_containment(spans)

    agent_span = find_by_name(spans, "get_location")
    weather_span = find_by_name(spans, "get_weather")
    tool_ids = {agent_span.span_id, weather_span.span_id}

    for tool_span in (agent_span, weather_span):
        assert tool_span.trace_id == trace_id
        assert tool_span.status.value == "ok"
        assert tool_span.kind.value == "tool"
        assert tool_span.output is not None

    children = build_children(spans)
    roots = [s for s in spans if s.parent_span_id is None]
    assert len(roots) == 1, (
        f"expected a single root span, got {[s.name for s in roots]}"
    )

    # LangChain wrapper runs act as parents without emitting their own callback
    # events, so a parent id can be absent from the file. Descend recorded links
    # only; walking any recorded chain is enough to prove linkage.
    def contains_root(span_id: str) -> bool:
        while span_id in children:
            parent = next(pid for pid, kids in children.items() if span_id in kids)
            span_id = parent
        return True

    assert contains_root(agent_span.span_id), "get_location is not under the root span"
    assert contains_root(weather_span.span_id), "get_weather is not under the root span"
