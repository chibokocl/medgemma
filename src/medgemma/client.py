"""High-level MedGemma client – the main public API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, SYSTEM_PROMPT
from .model import get_model, model_info, unload_model


@dataclass(frozen=True)
class Response:
    """Result returned by :meth:`MedGemma.ask`."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0


class MedGemma:
    """Friendly wrapper around the MLX MedGemma model.

    Parameters
    ----------
    model_path:
        Path to a local converted MLX model directory. When ``None`` the
        default cache at ``~/.medgemma/model`` is used (auto-downloaded on
        first call).
    max_tokens:
        Default maximum tokens for generation.
    temperature:
        Default sampling temperature.
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

    # -- public API --------------------------------------------------------

    def ask(
        self,
        prompt: str,
        *,
        image: str | Path | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Send a prompt (and optional image) to the model.

        Returns a :class:`Response` with ``.text`` and generation stats.
        """
        model, processor = self._ensure_loaded()
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        formatted_prompt = self._apply_template(processor, prompt, image)
        image_arg = self._resolve_image(image)

        from mlx_vlm import generate

        result = generate(
            model,
            processor,
            formatted_prompt,
            image=image_arg,
            max_tokens=max_tok,
            temperature=temp,
            verbose=False,
        )

        # generate returns a GenerationResult dataclass
        text = result.text if hasattr(result, "text") else str(result)
        prompt_tokens = getattr(result, "prompt_tokens", 0)
        gen_tokens = getattr(result, "generation_tokens", 0)
        gen_tps = getattr(result, "generation_tps", 0.0)
        # Compute elapsed from tokens / tps
        elapsed = gen_tokens / gen_tps if gen_tps > 0 else 0.0

        return Response(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            tokens_per_second=round(gen_tps, 1),
            elapsed_seconds=round(elapsed, 2),
        )

    def stream(
        self,
        prompt: str,
        *,
        image: str | Path | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Stream generated text chunk by chunk."""
        model, processor = self._ensure_loaded()
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        formatted_prompt = self._apply_template(processor, prompt, image)
        image_arg = self._resolve_image(image)

        from mlx_vlm import stream_generate

        for chunk in stream_generate(
            model,
            processor,
            formatted_prompt,
            image=image_arg,
            max_tokens=max_tok,
            temperature=temp,
        ):
            if isinstance(chunk, str):
                yield chunk
            elif hasattr(chunk, "text"):
                yield chunk.text
            else:
                yield str(chunk)

    def unload(self) -> None:
        """Release the model from memory."""
        unload_model()

    @staticmethod
    def info():
        """Return model info without loading."""
        return model_info()

    # -- internals ---------------------------------------------------------

    def _ensure_loaded(self):
        return get_model(self._model_path)

    @staticmethod
    def _apply_template(processor, prompt: str, image=None) -> str:
        """Build chat messages and apply the processor's chat template."""
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _resolve_image(image):
        if image is None:
            return None
        p = Path(image).expanduser()
        if p.is_file():
            return [str(p)]
        # Might be a URL – let mlx_vlm handle it
        return [str(image)]
