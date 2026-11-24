"""MedGemma – local medical AI (MLX on Apple Silicon, CUDA on Linux)."""

from ._version import __version__
from .client import MedGemma, Response

__all__ = ["MedGemma", "Response", "__version__"]
