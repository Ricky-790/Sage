from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .adapters import to_jsonable
from .models import Span, SpanError, SpanKind, SpanStatus


def new_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> datetime:
    return datetime.now(UTC)


def _error_info(error: BaseException) -> SpanError:
    return SpanError(
        type=type(error).__name__,
        message=str(error),
        traceback="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )


class Recorder:
    """Write spans to a JSONL transcript.

    Two capture modes share the same writer:

    1. Decorate functions with ``observe``. Nested calls become child spans.

    .. code-block:: python

        recorder = Recorder("transcript.jsonl")

        @recorder.observe(kind="tool")
        def lookup_order(order_id):
            ...

    2. Pass ``callback_handler()`` to a LangChain invocation. Nested model,
       tool, and chain runs are linked by LangChain's run tree.

    .. code-block:: python

        agent.invoke(input, config={"callbacks": [recorder.callback_handler()]})

    The modes can be combined. A callback root started inside an observed span
    attaches to that span when it can see the active context.

    Context propagation uses ``contextvars``. Asyncio tasks inherit it, but new
    threads do not. Wrap thread targets with ``contextvars.copy_context().run``
    when a worker thread needs to continue the active trace.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "trace_id", default=None
        )
        self._span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "span_id", default=None
        )

    def record_span(self, span: Span) -> None:
        """Apply the span to the transcript. This is the only writer."""
        span.input = to_jsonable(span.input)
        span.output = to_jsonable(span.output)
        span.metadata = to_jsonable(span.metadata)
        line = span.model_dump_json()

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @contextmanager
    def span(self, name, kind=SpanKind.SPAN, input=None, metadata=None):
        """Record a span around a block of work."""
        active_trace_id = self._trace_id.get()
        active_span_id = self._span_id.get()
        is_root = active_trace_id is None or active_span_id is None
        trace_id = new_id() if is_root else active_trace_id

        span = Span(
            trace_id=trace_id,
            span_id=new_id(),
            parent_span_id=None if is_root else active_span_id,
            name=name,
            kind=kind,
            status=SpanStatus.RUNNING,
            started_at=utc_now(),
            input=input,
            metadata=metadata or {},
        )
        started = time.perf_counter()

        trace_token = self._trace_id.set(trace_id) if is_root else None
        span_token = self._span_id.set(span.span_id)

        try:
            yield span
        except asyncio.CancelledError:
            span.status = SpanStatus.CANCELLED
            raise
        except BaseException as exc:
            span.status = SpanStatus.ERROR
            span.error = _error_info(exc)
            raise
        finally:
            span.ended_at = utc_now()
            span.duration = max(0.0, time.perf_counter() - started)
            if span.status == SpanStatus.RUNNING:
                span.status = SpanStatus.OK
            self._span_id.reset(span_token)
            if trace_token is not None:
                self._trace_id.reset(trace_token)
            self.record_span(span)

    def current_ids(self) -> tuple[str | None, str | None]:
        """Return the active trace and span IDs for the current context."""
        return self._trace_id.get(), self._span_id.get()

    def observe(self, name=None, kind=SpanKind.SPAN, metadata=None):
        """Decorate a sync or async function so each call is recorded."""

        def decorator(fn):
            span_name = name or fn.__qualname__

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with self.span(
                        span_name,
                        kind=kind,
                        input={"args": args, "kwargs": kwargs},
                        metadata=metadata,
                    ) as span:
                        result = await fn(*args, **kwargs)
                        span.output = to_jsonable(result)
                        return result

                return async_wrapper

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with self.span(
                    span_name,
                    kind=kind,
                    input={"args": args, "kwargs": kwargs},
                    metadata=metadata,
                ) as span:
                    result = fn(*args, **kwargs)
                    span.output = to_jsonable(result)
                    return result

            return sync_wrapper

        return decorator

    def callback_handler(self):
        """Return a LangChain callback handler bound to this recorder."""
        from .integrations.langchain import LangChainRecorderCallback

        return LangChainRecorderCallback(self)
