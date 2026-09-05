import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_llm_client_imports_without_api_key(monkeypatch):
    # Prevent .env from restoring the API key during the test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Force a fresh import of the module.
    sys.modules.pop("src.generation.llm_client", None)

    module = importlib.import_module("src.generation.llm_client")

    assert module is not None
    assert module.api_key is None
    assert module.client is None


def test_generate_answer_returns_text(monkeypatch):
    from src.generation import llm_client

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = "mocked answer"

    mock_client.models.generate_content.return_value = mock_response

    # Reset the module-level client so _get_client() creates it.
    monkeypatch.setattr(llm_client, "client", None)
    monkeypatch.setattr(llm_client, "api_key", "fake-api-key")

    with patch.object(
        llm_client.genai,
        "Client",
        return_value=mock_client,
    ) as mock_client_cls:

        result = llm_client.generate_answer(
            "What does x do?",
            "def x():\n    return 42",
        )

    assert result == "mocked answer"

    mock_client_cls.assert_called_once_with(api_key="fake-api-key")

    mock_client.models.generate_content.assert_called_once()

    call_kwargs = mock_client.models.generate_content.call_args.kwargs

    assert call_kwargs["model"] == "gemini-3.1-flash-lite"
    assert "What does x do?" in call_kwargs["contents"]
    assert "def x():" in call_kwargs["contents"]


def test_generate_answer_propagates_api_error(monkeypatch):
    from src.generation import llm_client

    mock_client = MagicMock()

    mock_client.models.generate_content.side_effect = RuntimeError(
        "Gemini API failed"
    )

    # Reset the cached client so _get_client() creates our mocked client.
    monkeypatch.setattr(llm_client, "client", None)
    monkeypatch.setattr(llm_client, "api_key", "fake-api-key")

    with patch.object(
        llm_client.genai,
        "Client",
        return_value=mock_client,
    ):

        with pytest.raises(RuntimeError, match="Gemini API failed"):
            llm_client.generate_answer(
                "What does x do?",
                "def x():\n    return 42",
            )

def test_generate_answer_rejects_empty_response(monkeypatch):
    from src.generation import llm_client

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = ""

    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr(llm_client, "client", None)
    monkeypatch.setattr(llm_client, "api_key", "fake-api-key")

    with patch.object(
        llm_client.genai,
        "Client",
        return_value=mock_client,
    ):
        with pytest.raises(
            RuntimeError,
            match="Gemini returned an empty response",
        ):
            llm_client.generate_answer(
                "What does x do?",
                "def x():\n    return 42",
            )

def test_generate_answer_rejects_none_response_text(monkeypatch):
    from src.generation import llm_client

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = None

    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr(llm_client, "client", None)
    monkeypatch.setattr(llm_client, "api_key", "fake-api-key")

    with patch.object(
        llm_client.genai,
        "Client",
        return_value=mock_client,
    ):
        with pytest.raises(
            RuntimeError,
            match="Gemini returned an empty response",
        ):
            llm_client.generate_answer(
                "What does x do?",
                "def x():\n    return 42",
            )