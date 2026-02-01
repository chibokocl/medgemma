"""MedGemma – Medical AI on Apple Silicon via MLX."""

from ._compat import check_platform
from ._version import __version__
from .client import MedGemma, Response

check_platform()

__all__ = ["MedGemma", "Response", "__version__"]
