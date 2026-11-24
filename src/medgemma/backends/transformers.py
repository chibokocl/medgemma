"""Linux backend: Hugging Face transformers (CUDA GPU, falls back to CPU)."""

import threading
import time
from typing import Any, Iterator

from ..config import HF_REPO_ID, SYSTEM_PROMPT
from ..types import Response


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = str(model_id) if model_id else HF_REPO_ID
        self._model: Any = None
        self._processor: Any = None
        self._gen_lock = threading.Lock()  # one generation at a time per GPU

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "torch/transformers are required on Linux: pip install 'medgemma[cuda]'"
            ) from exc

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto"
        ).eval()

    def unload(self) -> None:
        self._model = self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _inputs(self, prompt: str, image: str | None):
        import torch

        user: list[dict] = []
        if image:
            from transformers.image_utils import load_image

            user.append({"type": "image", "image": load_image(image).convert("RGB")})
        user.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user},
        ]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        floats = {k: v.to(self._model.dtype) for k, v in inputs.items() if v.is_floating_point()}
        inputs.update(floats)
        return inputs, torch

    @staticmethod
    def _gen_kwargs(max_tokens: int, temperature: float) -> dict:
        kwargs: dict = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        return kwargs

    def generate(self, prompt, image, max_tokens, temperature) -> Response:
        inputs, torch = self._inputs(prompt, image)
        n_prompt = inputs["input_ids"].shape[-1]
        with self._gen_lock, torch.inference_mode():
            start = time.perf_counter()
            out = self._model.generate(**inputs, **self._gen_kwargs(max_tokens, temperature))
            elapsed = time.perf_counter() - start
        new = out[0][n_prompt:]
        text = self._processor.decode(new, skip_special_tokens=True)
        return Response(
            text=text.strip(),
            prompt_tokens=n_prompt,
            completion_tokens=len(new),
            tokens_per_second=round(len(new) / elapsed, 1) if elapsed > 0 else 0.0,
            elapsed_seconds=round(elapsed, 2),
        )

    def stream(self, prompt, image, max_tokens, temperature) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs, torch = self._inputs(prompt, image)
        streamer = TextIteratorStreamer(
            self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def run() -> None:
            with torch.inference_mode():
                self._model.generate(
                    **inputs, streamer=streamer, **self._gen_kwargs(max_tokens, temperature)
                )

        with self._gen_lock:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                yield from streamer
            finally:
                thread.join()
