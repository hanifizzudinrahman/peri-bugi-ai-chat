from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_PORT: int = 8003

    # Internal security (shared dengan peri-bugi-api)
    INTERNAL_SECRET: str = ""

    # LLM Provider
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "gemma2:2b"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-pro"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: int = 60

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "peri_bugi_dental"
    QDRANT_FAQ_COLLECTION: str = "peri_bugi_faq"      # NEW: untuk app_faq agent

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding model (untuk RAG retrieval — kb_dental, app_faq agents)
    # Diakses oleh app/agents/sub_agents/__init__.py (_get_embeddings) +
    # app/agents/tools/retrieve.py + admin info endpoint di main.py.
    # Sebelumnya field ini tidak declared di Settings — pakai .env value via
    # extra="ignore" tidak cukup karena pydantic-settings butuh field declaration
    # supaya bisa diakses via settings.EMBEDDING_PROVIDER.
    #   Provider: "local" (HuggingFace, default) | "gemini" | "openai"
    #   Model:    nama model (untuk local: HF model name)
    #   Device:   "cpu" | "cuda" | "auto" (auto = detect via torch)
    # ─────────────────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "auto"

    # Redis (shared dengan peri-bugi-api — connect via host.docker.internal)
    # Kosong = rate limiting RnD endpoint dinonaktifkan
    REDIS_URL: str = ""

    # External
    AI_CV_URL: str = ""
    PERI_API_URL: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Database (Phase 1 — LangGraph PostgresSaver checkpointer)
    # Shared dengan peri-bugi-api (sama-sama akses peri_bugi_db).
    # ai-chat connect ke DB via network `peri-bugi-shared` (container hostname `peri_bugi_db`).
    # ─────────────────────────────────────────────────────────────────────────
    DB_HOST: str = "db"   # default: service name "db" di network peri-bugi-shared (sama dengan api)
    DB_PORT: int = 5432
    DB_NAME: str = "peri_bugi"
    DB_USER: str = "peri_bugi_user"
    DB_PASSWORD: str = ""

    # RnD mode
    RND_MODE: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Langfuse Observability (optional)
    # Pattern 1+2: graceful degradation + explicit toggle.
    # Default OFF supaya safe — Hanif harus eksplisit set true di .env.
    # Lihat docs/LANGFUSE_INTEGRATION.md untuk detail.
    # ─────────────────────────────────────────────────────────────────────────
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse-web:3000"  # internal Docker network

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def llm_model_name(self) -> str:
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
