from __future__ import annotations

import time
import traceback
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from sage.recorder import Recorder
from sage.recorder.models import (
    Span,
    SpanError,
    SpanKind,
    SpanStatus,
    SpanUsage,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _jsonable(value: Any) -> Any:
    """Convert common LangChain objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))

    if hasattr(value, "dict"):
        return _jsonable(value.dict())

    if hasattr(value, "content"):
        return {
            "type": type(value).__name__,
            "content": _jsonable(value.content),
        }

    return str(value)


def _error_info(error: BaseException) -> SpanError:
    return SpanError(
        type=type(error).__name__,
        message=str(error),
        traceback="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )


def _token_usage(response: LLMResult) -> SpanUsage | None:
    """Extract token usage from LangChain AIMessage metadata."""
    try:
        generation = response.generations[0][0]
    except IndexError:
        return None

    if not isinstance(generation, ChatGeneration):
        return None

    message = generation.message
    if not isinstance(message, AIMessage):
        return None

    usage = message.usage_metadata
    if not usage:
        return None

    return SpanUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
    )


class LangChainRecorderCallback(BaseCallbackHandler):
    def __init__(self, recorder: Any) -> None:
        super().__init__()
        self.recorder = recorder
        self._spans: dict[UUID, Span] = {}  # Holds each operation's in-progress Span.
        self._start_times: dict[
            UUID, float
        ] = {}  # Holds monotonic start times for measuring duration.
        self._run_context: dict[
            UUID, tuple[str, str | None]
        ] = {}  # { current_span_id : ( trace_id, parent_span_id) }

    def _start(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        kind: SpanKind,
        inputs: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        parent_context = (
            self._run_context.get(parent_run_id) if parent_run_id is not None else None
        )

        trace_id = parent_context[0] if parent_context else str(run_id)

        span = Span(
            trace_id=trace_id,
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            name=name,
            kind=kind,
            status=SpanStatus.RUNNING,
            started_at=_now(),
            input=_jsonable(inputs),
            output=None,
            metadata=_jsonable(metadata or {}),
        )

        self._spans[run_id] = span
        self._start_times[run_id] = time.perf_counter()
        self._run_context[run_id] = (
            trace_id,
            str(parent_run_id) if parent_run_id else None,
        )

    def _finish(
        self,
        run_id: UUID,
        *,
        output: Any = None,
        error: BaseException | None = None,
        usage: SpanUsage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        span = self._spans.pop(run_id, None)
        started = self._start_times.pop(run_id, None)

        if span is None:
            return

        span.ended_at = _now()
        if started is not None:
            span.duration = max(0.0, time.perf_counter() - started)

        span.status = SpanStatus.ERROR if error else SpanStatus.OK
        span.output = _jsonable(output)

        if error is not None:
            span.error = _error_info(error)

        if usage is not None:
            span.usage = usage

        if metadata:
            span.metadata.update(_jsonable(metadata))

        # Assumes Recorder exposes record_span(Span).
        self.recorder.record_span(span)

    # ---- LLMs and chat models ----

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name", "llm"),
            kind=SpanKind.LLM,
            inputs={"prompts": prompts},
            metadata={
                "serialized": serialized,
                "tags": tags or [],
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name", "chat_model"),
            kind=SpanKind.LLM,
            inputs={"messages": messages},
            metadata={
                "serialized": serialized,
                "tags": tags or [],
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(
            run_id,
            output=response,
            usage=_token_usage(response),
            metadata={"llm_output": response.llm_output or {}},
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, error=error)

    # ---- Tools ----

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name", "tool"),
            kind=SpanKind.TOOL,
            inputs=inputs if inputs is not None else input_str,
            metadata={
                "serialized": serialized,
                "tags": tags or [],
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, error=error)

    # ---- Chains ----

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name", "chain"),
            kind=SpanKind.CHAIN,
            inputs=inputs,
            metadata={
                "serialized": serialized,
                "tags": tags or [],
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, error=error)

    # ---- Retrievers ----

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name", "retriever"),
            kind=SpanKind.RETRIEVER,
            inputs={"query": query},
            metadata={
                "serialized": serialized,
                "tags": tags or [],
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_retriever_end(
        self,
        documents: "Sequence[Document]",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, output=documents)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, error=error)
