import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

class Settings:
    """Application settings and configuration management."""
    def __init__(self):
        self.app_title: str = os.getenv("APP_TITLE", "DocuMind AI — Document Intelligence Assistant")
        self.debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        
        # RAG and Document processing configurations
        self.max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
        self.max_upload_files: int = int(os.getenv("MAX_UPLOAD_FILES", "5"))
        self.max_question_length: int = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.embedding_model_name: str = os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash")
        self.retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

        # Resilience & API configurations
        self.api_timeout_seconds: float = float(os.getenv("API_TIMEOUT_SECONDS", "30.0"))
        self.api_max_retries: int = int(os.getenv("API_MAX_RETRIES", "3"))
        self.cors_origins: str = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501"
        )

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Read the API credential from the environment only."""
        return os.getenv("GEMINI_API_KEY")

    def validate(self) -> None:
        """Validate credentials and runtime settings before generation."""
        key = self.gemini_api_key
        if not key or key.strip() == "":
            raise ConfigError("GEMINI_API_KEY environment variable is not set or empty.")
        if key == "your_gemini_api_key_here":
            raise ConfigError("GEMINI_API_KEY is using the placeholder value. Please provide a real API key.")

        self.validate_runtime()

    def validate_runtime(self) -> None:
        """Validate non-secret application limits and service settings."""
        if self.max_file_size_mb <= 0:
            raise ConfigError("MAX_FILE_SIZE_MB must be greater than 0.")
        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ConfigError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be strictly less than CHUNK_SIZE.")
        if self.retrieval_top_k <= 0:
            raise ConfigError("RETRIEVAL_TOP_K must be greater than 0.")
        if self.api_timeout_seconds <= 0:
            raise ConfigError("API_TIMEOUT_SECONDS must be greater than 0.")
        if self.api_timeout_seconds > 120:
            raise ConfigError("API_TIMEOUT_SECONDS must not exceed 120.")
        if self.api_max_retries < 0:
            raise ConfigError("API_MAX_RETRIES cannot be negative.")
        if self.api_max_retries > 5:
            raise ConfigError("API_MAX_RETRIES must not exceed 5.")
        if not 1 <= self.max_upload_files <= 10:
            raise ConfigError("MAX_UPLOAD_FILES must be between 1 and 10.")
        if self.max_question_length <= 0:
            raise ConfigError("MAX_QUESTION_LENGTH must be greater than 0.")
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        if not origins or "*" in origins:
            raise ConfigError("CORS_ORIGINS must contain explicit origins, not '*'.")

# Global settings instance
settings = Settings()
