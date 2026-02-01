"""Tests for medgemma.config."""

from pathlib import Path

from medgemma.config import CACHE_DIR, DEFAULT_MAX_TOKENS, HF_REPO_ID


def test_cache_dir_is_in_home():
    assert str(CACHE_DIR).startswith(str(Path.home()))


def test_hf_repo_id_set():
    assert "medgemma" in HF_REPO_ID.lower()


def test_defaults():
    assert DEFAULT_MAX_TOKENS > 0
