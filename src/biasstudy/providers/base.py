"""Provider-neutral request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from biasstudy.config import ModelSpec


@dataclass
class Request:
    prompt: str
    search: bool
    temperature: float | None  # None = vendor default
    max_output_tokens: int
    location: str = "United States"  # only used by Google search products


@dataclass
class Response:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    model_id_reported: str | None = None
    finish_reason: str | None = None
    shown: bool = True  # False when the product showed no AI answer (e.g. no AI Overview box)
    served_by: str | None = None  # upstream provider that actually served the request
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    spec: ModelSpec

    async def complete(self, req: Request) -> Response: ...
