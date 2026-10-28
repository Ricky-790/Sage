"""Normalize framework-specific data."""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID


def to_jsonable(value: Any) -> Any:
    """Convert common SDK objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return to_jsonable(value.value)

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json"))
        except Exception:
            try:
                return to_jsonable(value.model_dump())
            except Exception:
                pass

    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict())
        except Exception:
            pass

    if hasattr(value, "content"):
        return {
            "type": type(value).__name__,
            "content": to_jsonable(value.content),
        }

    return str(value)
