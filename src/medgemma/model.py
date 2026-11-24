"""Thread-safe lazy-loading backend singleton."""

import threading
from dataclasses import dataclass

from .backends import Backend, create_backend, resolve_backend_name
from .config import CACHE_DIR
from .convert import is_model_ready

_lock = threading.Lock()
_backend: Backend | None = None


@dataclass(frozen=True)
class ModelInfo:
    loaded: bool
    backend: str
    path: str | None
    cache_dir: str
    cache_ready: bool


def get_backend(model_path: str | None = None) -> Backend:
    """Return the loaded backend, loading it on first call (thread-safe)."""
    global _backend
    if _backend is not None and _backend.loaded:
        return _backend
    with _lock:
        if _backend is None or not _backend.loaded:
            backend = create_backend(model_path=model_path)
            backend.load()
            _backend = backend
    return _backend


def unload_model() -> None:
    """Release the loaded model from memory."""
    global _backend
    with _lock:
        if _backend is not None:
            _backend.unload()
        _backend = None


def model_info() -> ModelInfo:
    """Return current model state without triggering a load."""
    return ModelInfo(
        loaded=_backend is not None and _backend.loaded,
        backend=resolve_backend_name(),
        path=getattr(_backend, "path", None) and str(_backend.path),
        cache_dir=str(CACHE_DIR),
        cache_ready=is_model_ready(),
    )
