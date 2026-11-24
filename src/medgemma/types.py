"""Shared value types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
    """Result returned by :meth:`medgemma.MedGemma.ask`."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0
