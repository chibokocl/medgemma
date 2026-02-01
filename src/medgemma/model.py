"""Thread-safe lazy-loading model singleton."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CACHE_DIR
from .convert import is_model_ready, setup_model

_lock = threading.Lock()
_model: Any | None = None
_processor: Any | None = None
_model_path: Path | None = None


@dataclass(frozen=True)
class ModelInfo:
    loaded: bool
    path: str | None
    cache_dir: str
    cache_ready: bool


def get_model(model_path: str | Path | None = None) -> tuple[Any, Any]:
    """Return ``(model, processor)``, loading on first call.

    Thread-safe: only one thread will trigger the load.
    """
    global _model, _processor, _model_path

    if _model is not None and _processor is not None:
        return _model, _processor

    with _lock:
        # Double-checked locking
        if _model is not None and _processor is not None:
            return _model, _processor

        path = Path(model_path).expanduser() if model_path else CACHE_DIR

        if not is_model_ready(path):
            setup_model(cache_dir=path)

        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is required. Install with: pip install mlx-vlm>=0.3.10"
            ) from exc

        _model, _processor = load(str(path))
        _model_path = path

    return _model, _processor


def unload_model() -> None:
    """Release the loaded model from memory."""
    global _model, _processor, _model_path
    with _lock:
        _model = None
        _processor = None
        _model_path = None


def model_info() -> ModelInfo:
    """Return current model state without triggering a load."""
    return ModelInfo(
        loaded=_model is not None,
        path=str(_model_path) if _model_path else None,
        cache_dir=str(CACHE_DIR),
        cache_ready=is_model_ready(),
    )
