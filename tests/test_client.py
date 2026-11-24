"""Tests for medgemma.client (fake backend – no real model load)."""

from unittest.mock import MagicMock, patch

from medgemma.client import MedGemma, Response


def test_defaults():
    mg = MedGemma()
    assert mg.max_tokens == 512
    assert mg.temperature == 0.1


def test_info_without_load():
    assert MedGemma.info().loaded is False


@patch("medgemma.client.get_backend")
def test_ask_delegates_to_backend(get_backend):
    backend = MagicMock()
    backend.generate.return_value = Response(text="hi")
    get_backend.return_value = backend

    resp = MedGemma(max_tokens=10).ask("q", image="/x.png", temperature=0.5)

    backend.generate.assert_called_once_with("q", "/x.png", 10, 0.5)
    assert resp.text == "hi"


@patch("medgemma.client.get_backend")
def test_stream_delegates_to_backend(get_backend):
    backend = MagicMock()
    backend.stream.return_value = iter(["a", "b"])
    get_backend.return_value = backend
    assert list(MedGemma().stream("q")) == ["a", "b"]
