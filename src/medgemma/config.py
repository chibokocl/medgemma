"""Constants and environment-driven configuration for medgemma.

Every runtime setting can be overridden with a ``MEDGEMMA_*`` environment
variable, which makes the package easy to configure under Docker/systemd.
"""

import os
from pathlib import Path

# Hugging Face repo for the original model
HF_REPO_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")

# Default cache location for the converted MLX model (macOS backend)
CACHE_DIR = Path(os.environ.get("MEDGEMMA_CACHE_DIR", Path.home() / ".medgemma" / "model"))

# MLX-VLM quantisation defaults
DEFAULT_QUANT_BITS = 4
DEFAULT_QUANT_GROUP_SIZE = 64

# Generation defaults
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant. Provide accurate, "
    "evidence-based medical information. Always recommend consulting "
    "a healthcare professional for personal medical advice."
)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(int(default))).lower() in {"1", "true", "yes", "on"}


class ServerSettings:
    """Server settings, read from the environment when instantiated."""

    def __init__(self) -> None:
        self.host: str = os.environ.get("MEDGEMMA_HOST", "127.0.0.1")
        self.port: int = _int("MEDGEMMA_PORT", 8000)
        self.api_key: str | None = os.environ.get("MEDGEMMA_API_KEY") or None
        self.preload: bool = _bool("MEDGEMMA_PRELOAD", True)
        self.max_tokens_limit: int = _int("MEDGEMMA_MAX_TOKENS_LIMIT", 2048)
        self.max_image_mb: int = _int("MEDGEMMA_MAX_IMAGE_MB", 10)
