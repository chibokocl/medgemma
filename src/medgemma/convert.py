"""Model download and MLX conversion utilities."""

import shutil
from pathlib import Path

from .config import CACHE_DIR, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, HF_REPO_ID


def is_model_ready(model_path: str | Path | None = None) -> bool:
    """Check whether a converted MLX model exists at the given path."""
    path = Path(model_path) if model_path else CACHE_DIR
    return (path / "config.json").is_file()


def setup_model(
    *,
    force: bool = False,
    hf_repo: str = HF_REPO_ID,
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download / convert the model into the cache directory.

    Parameters
    ----------
    force:
        Re-download even if a model already exists in cache.
    hf_repo:
        Hugging Face repo ID to download from.
    local_path:
        Path to an already-converted local model directory.  When provided the
        files are copied (or symlinked) into the cache instead of downloading.
    cache_dir:
        Override the default cache directory (~/.medgemma/model).

    Returns
    -------
    Path to the ready-to-load model directory.
    """
    dest = Path(cache_dir) if cache_dir else CACHE_DIR

    if local_path is not None:
        return _copy_local(Path(local_path).expanduser(), dest, force=force)

    if not force and is_model_ready(dest):
        return dest

    return _convert_from_hf(hf_repo, dest)


def _copy_local(src: Path, dest: Path, *, force: bool = False) -> Path:
    """Copy a local converted model into the cache directory."""
    if not src.is_dir():
        raise FileNotFoundError(f"Local model path does not exist: {src}")
    if not (src / "config.json").is_file():
        raise FileNotFoundError(
            f"No config.json in {src} – is this a converted MLX model?"
        )

    if src.resolve() == dest.resolve():
        return dest

    if force and dest.exists():
        shutil.rmtree(dest)

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)

    return dest


def _convert_from_hf(hf_repo: str, dest: Path) -> Path:
    """Download from Hugging Face and convert to MLX format."""
    try:
        from mlx_vlm import convert as mlx_convert
    except ImportError as exc:
        raise ImportError(
            "mlx-vlm is required for model conversion. "
            "Install it with: pip install mlx-vlm>=0.3.10"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)

    mlx_convert(
        hf_repo,
        mlx_path=str(dest),
        q_bits=DEFAULT_QUANT_BITS,
        q_group_size=DEFAULT_QUANT_GROUP_SIZE,
    )

    if not is_model_ready(dest):
        raise RuntimeError(
            f"Conversion finished but {dest / 'config.json'} not found. "
            "The model may not have been converted correctly."
        )

    return dest
