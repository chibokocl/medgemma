"""Apple Silicon backend (mlx-vlm)."""

import threading
from pathlib import Path
from typing import Any, Iterator

from ..config import CACHE_DIR
from ..convert import is_model_ready, setup_model
from ..types import Response


def _resolve(image: str) -> str:
    p = Path(image).expanduser()
    return str(p) if p.is_file() else image  # otherwise assume a URL


class MLXBackend:
    name = "mlx"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.path = Path(model_path).expanduser() if model_path else CACHE_DIR
        self._model: Any = None
        self._processor: Any = None
        self._gen_lock = threading.Lock()  # Metal is not safe for concurrent generation

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is required on Apple Silicon: pip install 'medgemma[mlx]'"
            ) from exc
        if not is_model_ready(self.path):
            setup_model(cache_dir=self.path)
        self._model, self._processor = load(str(self.path))

    def unload(self) -> None:
        self._model = self._processor = None

    def _prepare(self, prompt: str, image: str | None):
        from ..config import SYSTEM_PROMPT

        content: list[dict] = [{"type": "image"}] if image else []
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text, ([_resolve(image)] if image else None)

    def generate(self, prompt, image, max_tokens, temperature) -> Response:
        from mlx_vlm import generate

        text, images = self._prepare(prompt, image)
        with self._gen_lock:
            result = generate(
                self._model, self._processor, text, image=images,
                max_tokens=max_tokens, temperature=temperature, verbose=False,
            )
        gen_tokens = getattr(result, "generation_tokens", 0)
        tps = getattr(result, "generation_tps", 0.0)
        return Response(
            text=(result.text if hasattr(result, "text") else str(result)).strip(),
            prompt_tokens=getattr(result, "prompt_tokens", 0),
            completion_tokens=gen_tokens,
            tokens_per_second=round(tps, 1),
            elapsed_seconds=round(gen_tokens / tps, 2) if tps > 0 else 0.0,
        )

    def stream(self, prompt, image, max_tokens, temperature) -> Iterator[str]:
        from mlx_vlm import stream_generate

        text, images = self._prepare(prompt, image)
        with self._gen_lock:
            for chunk in stream_generate(
                self._model, self._processor, text, image=images,
                max_tokens=max_tokens, temperature=temperature,
            ):
                yield chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
