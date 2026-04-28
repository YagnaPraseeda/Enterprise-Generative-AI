"""
Centralized configuration management using Pydantic Settings.
Loads settings from environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Groq ---
    groq_api_key: str = ""

    # --- ChromaDB ---
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")

    # --- Document Storage ---
    document_upload_dir: str = str(PROJECT_ROOT / "data" / "uploads")

    # --- Application ---
    app_name: str = "Enterprise GenAI Knowledge Platform"
    app_env: str = "development"
    log_level: str = "INFO"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Frontend ---
    streamlit_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    # --- RAG Settings ---
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    default_model: str = "llama-3.3-70b-versatile"

    # --- Local Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Governance ---
    enable_toxicity_filter: bool = True
    enable_hallucination_detection: bool = True
    enable_prompt_guard: bool = True
    toxicity_threshold: float = 0.7
    hallucination_threshold: float = 0.3


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
