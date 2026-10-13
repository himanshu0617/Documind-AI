import pytest
from src.config import Settings, ConfigError

def test_settings_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY environment variable is not set"):
        settings.validate()

def test_settings_placeholder_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    settings = Settings()
    with pytest.raises(ConfigError, match="using the placeholder value"):
        settings.validate()

def test_settings_valid_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_123")
    settings = Settings()
    settings.validate()

def test_settings_invalid_chunk_overlap(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_123")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    settings = Settings()
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP must be strictly less than CHUNK_SIZE"):
        settings.validate()

def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("MAX_FILE_SIZE_MB", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)
    settings = Settings()
    assert settings.max_file_size_mb == 25
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_top_k == 4
    assert settings.max_upload_files == 5
    assert settings.max_question_length == 1000
    assert settings.gemini_model_name == "gemini-3.8-flash"

def test_settings_rejects_unbounded_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_123")
    monkeypatch.setenv("API_MAX_RETRIES", "6")
    with pytest.raises(ConfigError, match="API_MAX_RETRIES must not exceed 5"):
        Settings().validate()

def test_runtime_settings_reject_wildcard_cors(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ConfigError, match="explicit origins"):
        Settings().validate_runtime()
