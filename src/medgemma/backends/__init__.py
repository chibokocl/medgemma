"""Inference backends.

``mlx``           – Apple Silicon (macOS), 4-bit quantised via mlx-vlm.
``transformers``  – Linux/CUDA (or CPU) via Hugging Face transformers.
"""

import os
import platform
import sys
from typing import Iterator, Protocol

from ..types import Response


class Backend(Protocol):
    name: str

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(
        self, prompt: str, image: str | None, max_tokens: int, temperature: float
    ) -> Response: ...

    def stream(
        self, prompt: str, image: str | None, max_tokens: int, temperature: float
    ) -> Iterator[str]: ...


def resolve_backend_name(name: str | None = None) -> str:
    """Resolve ``auto`` / ``None`` to a concrete backend name."""
    name = (name or os.environ.get("MEDGEMMA_BACKEND", "auto")).lower()
    if name == "auto":
        on_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        return "mlx" if on_apple_silicon else "transformers"
    if name not in {"mlx", "transformers"}:
        raise ValueError(f"Unknown backend {name!r}; expected 'mlx' or 'transformers'.")
    return name


def create_backend(name: str | None = None, model_path: str | None = None) -> Backend:
    resolved = resolve_backend_name(name)
    if resolved == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(model_path)
    from .transformers import TransformersBackend

    return TransformersBackend(model_path)
