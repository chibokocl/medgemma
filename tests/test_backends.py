import pytest

from medgemma.backends import resolve_backend_name


def test_explicit_backend(monkeypatch):
    monkeypatch.delenv("MEDGEMMA_BACKEND", raising=False)
    assert resolve_backend_name("transformers") == "transformers"


def test_env_backend(monkeypatch):
    monkeypatch.setenv("MEDGEMMA_BACKEND", "mlx")
    assert resolve_backend_name() == "mlx"


def test_auto_on_linux(monkeypatch):
    monkeypatch.delenv("MEDGEMMA_BACKEND", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    assert resolve_backend_name() == "transformers"


def test_unknown_backend():
    with pytest.raises(ValueError):
        resolve_backend_name("nope")
