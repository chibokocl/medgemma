"""Tests for medgemma.client (mocked – no real model load)."""

from unittest.mock import MagicMock, patch

from medgemma.client import MedGemma, Response


def test_response_dataclass():
    r = Response(text="hello", tokens_per_second=90.0, elapsed_seconds=1.0)
    assert r.text == "hello"
    assert r.tokens_per_second == 90.0


def test_medgemma_init_defaults():
    mg = MedGemma()
    assert mg.max_tokens == 512
    assert mg.temperature == 0.1
    assert mg._model_path is None


def test_medgemma_custom_path():
    mg = MedGemma(model_path="/tmp/test-model")
    assert mg._model_path == "/tmp/test-model"


def test_info_without_load():
    info = MedGemma.info()
    assert info.loaded is False


@patch("medgemma.client.get_model")
@patch("mlx_vlm.generate")
def test_ask_mocked(mock_generate, mock_get_model):
    mock_model = MagicMock()
    mock_proc = MagicMock()
    mock_proc.apply_chat_template.return_value = "<formatted prompt>"
    mock_get_model.return_value = (mock_model, mock_proc)

    mock_result = MagicMock()
    mock_result.text = "Test response text"
    mock_result.prompt_tokens = 10
    mock_result.generation_tokens = 5
    mock_result.generation_tps = 90.0
    mock_generate.return_value = mock_result

    mg = MedGemma()
    resp = mg.ask("test prompt")

    assert resp.text == "Test response text"
    assert resp.tokens_per_second == 90.0
    assert resp.completion_tokens == 5
    mock_get_model.assert_called_once()
    mock_generate.assert_called_once()
