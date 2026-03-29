from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # App
    # -------------------------------------------------------------------------
    APP_ENV: str = "development"
    APP_PORT: int = 8003

    # -------------------------------------------------------------------------
    # Internal Security
    # Shared secret dengan peri-bugi-api — wajib diisi
    # -------------------------------------------------------------------------
    INTERNAL_SECRET: str

    # -------------------------------------------------------------------------
    # LLM Provider
    # -------------------------------------------------------------------------
    LLM_PROVIDER: str = "ollama"        # ollama | gemini | openai

    # Ollama
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "gemma2:2b"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-pro"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # LLM params
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: int = 60

    # -------------------------------------------------------------------------
    # Qdrant
    # -------------------------------------------------------------------------
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "peri_bugi_dental"

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = "local"   # local | gemini | openai
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # -------------------------------------------------------------------------
    # External APIs
    # -------------------------------------------------------------------------
    AI_CV_URL: str = ""                 # kosong = skip image inference

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def llm_model_name(self) -> str:
        """Nama model aktif berdasarkan provider."""
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_MODEL
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_MODEL
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_MODEL
        return "unknown"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
