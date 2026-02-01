"""Platform compatibility checks for Apple Silicon."""

import platform
import sys


def check_platform() -> None:
    """Fail fast if not running on macOS with Apple Silicon."""
    if sys.platform != "darwin":
        raise RuntimeError(
            "medgemma requires macOS with Apple Silicon (M1/M2/M3/M4). "
            f"Detected platform: {sys.platform}"
        )
    machine = platform.machine()
    if machine != "arm64":
        raise RuntimeError(
            "medgemma requires Apple Silicon (arm64). "
            f"Detected architecture: {machine}"
        )


def is_apple_silicon() -> bool:
    """Return True if running on macOS arm64."""
    return sys.platform == "darwin" and platform.machine() == "arm64"
