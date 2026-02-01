"""Constants and default configuration for medgemma."""

from pathlib import Path

# Default cache location for the converted MLX model
CACHE_DIR = Path.home() / ".medgemma" / "model"

# Hugging Face repo for the original model
HF_REPO_ID = "google/medgemma-4b-it"

# MLX-VLM quantisation defaults
DEFAULT_QUANT_BITS = 4
DEFAULT_QUANT_GROUP_SIZE = 64

# Generation defaults
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

# Chat template for MedGemma
SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant. Provide accurate, "
    "evidence-based medical information. Always recommend consulting "
    "a healthcare professional for personal medical advice."
)
