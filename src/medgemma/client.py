"""High-level MedGemma client – the main public API."""

from pathlib import Path
from typing import Iterator

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from .model import get_backend, model_info, unload_model
from .types import Response

__all__ = ["MedGemma", "Response"]


class MedGemma:
    """Friendly wrapper around the MedGemma model.

    The inference backend is chosen automatically (MLX on Apple Silicon,
    transformers elsewhere) and can be forced with ``MEDGEMMA_BACKEND``.

    Parameters
    ----------
    model_path:
        Local model directory (MLX backend) or Hugging Face repo id / path
        (transformers backend). Defaults to the standard cache / repo.
    max_tokens, temperature:
        Default generation settings.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

    def ask(
        self,
        prompt: str,
        *,
        image: str | Path | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Send a prompt (and optional image) and return the full response."""
        return self._backend().generate(*self._args(prompt, image, max_tokens, temperature))

    def stream(
        self,
        prompt: str,
        *,
        image: str | Path | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream generated text chunk by chunk."""
        return self._backend().stream(*self._args(prompt, image, max_tokens, temperature))

    def load(self) -> None:
        """Load the model now instead of on first request."""
        self._backend()

    def unload(self) -> None:
        """Release the model from memory."""
        unload_model()

    @staticmethod
    def info():
        """Return model info without loading."""
        return model_info()

    def _backend(self):
        return get_backend(str(self._model_path) if self._model_path else None)

    def _args(self, prompt, image, max_tokens, temperature):
        return (
            prompt,
            str(image) if image is not None else None,
            max_tokens if max_tokens is not None else self.max_tokens,
            temperature if temperature is not None else self.temperature,
        )
