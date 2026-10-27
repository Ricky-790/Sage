import contextvars
import functools
import inspect
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def new_id():
    return uuid.uuid4().hex


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def serialize(value):
    """Best-effort JSON serialization for SDK response objects."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return repr(value)


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._trace_id = contextvars.ContextVar("trace_id", default=None)
        self._span_id = contextvars.ContextVar("span_id", default=None)

    @contextmanager
    def span(self, name, kind="span", input=None, metadata=None):
        # Reuse the active trace, or create a new root trace.
        trace_id = self._trace_id.get()
        is_root = trace_id is None

        if is_root:
            trace_id = new_id()

        parent_id = self._span_id.get()
        span_id = new_id()

        span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_id,
            "name": name,
            "kind": kind,
            "status": "ok",
            "started_at": utc_now(),
            "start_ns": time.perf_counter_ns(),
            "input": serialize(input),
            "metadata": serialize(metadata or {}),
        }

        trace_token = None
        if is_root:
            trace_token = self._trace_id.set(trace_id)

        span_token = self._span_id.set(span_id)

        try:
            print("yielded span")
            yield span
        except BaseException as exc:
            span["status"] = "error"
            span["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            span["ended_at"] = utc_now()
            span["duration_ms"] = (
                time.perf_counter_ns() - span["start_ns"]
            ) / 1_000_000

            span.pop("start_ns", None)

            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(serialize(span)) + "\n")

            self._span_id.reset(span_token)

            if is_root:
                self._trace_id.reset(trace_token)

    def observe(self, name=None, kind="span"):
        """Decorate either a synchronous or asynchronous function."""

        def decorator(fn):
            span_name = name or fn.__qualname__

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with self.span(
                        span_name,
                        kind=kind,
                        input={"args": args, "kwargs": kwargs},
                    ) as span:
                        result = await fn(*args, **kwargs)
                        span["output"] = serialize(result)
                        return result

                return async_wrapper

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with self.span(
                    span_name,
                    kind=kind,
                    input={"args": args, "kwargs": kwargs},
                ) as span:
                    result = fn(*args, **kwargs)
                    span["output"] = serialize(result)
                    return result

            return sync_wrapper

        return decorator
