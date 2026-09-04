import importlib
import sys


def test_llm_client_imports_without_api_key(monkeypatch):
    # Prevent .env from restoring the API key during the test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Force a fresh import of the module.
    sys.modules.pop("src.generation.llm_client", None)

    module = importlib.import_module("src.generation.llm_client")

    assert module is not None