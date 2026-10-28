from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpanKind(StrEnum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    SPAN = "span"


class SpanStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class SpanError(BaseModel):
    type: str = Field(..., description="Exception class name.")
    message: str = Field(..., description="Exception message.")
    traceback: str | None = Field(
        None, description="Formatted traceback, if available."
    )


class SpanUsage(BaseModel):
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    total_tokens: int | None = Field(None, ge=0)


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)

    trace_id: str = Field(..., description="ID of the trace this span belongs to.")
    span_id: str = Field(..., description="Unique ID of this span.")
    parent_span_id: str | None = Field(
        None, description="ID of the parent span, if any."
    )

    name: str = Field(..., description="Name of the operation.")
    kind: SpanKind = Field(..., description="Type of operation being recorded.")
    status: SpanStatus = Field(
        default=SpanStatus.RUNNING, description="Execution status of the span."
    )

    started_at: datetime = Field(..., description="Timestamp when the span started.")
    ended_at: datetime | None = Field(
        default=None, description="Timestamp when the span ended."
    )
    duration: float | None = Field(
        default=None, ge=0, description="Duration in seconds."
    )

    input: Any = Field(None, description="Input captured for the operation.")
    output: Any = Field(None, description="Output captured for the operation.")

    error: SpanError | None = Field(
        default=None, description="Exception details if the operation failed."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional operation-specific metadata.",
    )
    usage: SpanUsage | None = Field(
        default=None, description="Token usage for LLM operations, when available."
    )
