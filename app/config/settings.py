from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Default sengaja "production" — fail safe. Kalau APP_ENV lupa di-set di
    # suatu host, /docs dan /openapi.json tertutup dan log memakai format JSON.
    # Yang hilang di dev hanyalah kenyamanan; yang dicegah adalah skema seluruh
    # endpoint terbuka karena satu env var terlewat.
    # Dev lokal set APP_ENV=development eksplisit di .env dan .env.example.
    # Mengikuti preseden peri-bugi-api/app/core/config.py.
    APP_ENV: str = "production"
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
    # Text-to-SQL (1 Agustus 2026)
    # Otoritas sebenarnya ada di peri-bugi-api: flag founder + strategi dikirim
    # per request lewat field `data_strategy`. Nilai di sini hanya cadangan
    # untuk jalur yang tidak lewat api (sandbox RnD, tes lokal), dan default-nya
    # "tools" supaya sama sekali tidak mengubah perilaku kalau lupa di-set.
    # ─────────────────────────────────────────────────────────────────────────
    DATA_STRATEGY: str = "tools"

    #: Model khusus penulis SQL. Kosong = ikut model chat.
    #: Generasi SQL lebih menuntut ketepatan sintaks daripada gaya bahasa, jadi
    #: berguna bisa memisahkannya tanpa mengubah model yang menjawab ke orang tua.
    NL_SQL_MODEL: str = ""

    #: Berapa lama katalog semantik di-cache sebelum diambil ulang dari api.
    NL_CATALOG_TTL_SECONDS: int = 300

    # ─────────────────────────────────────────────────────────────────────────
    # Tanya Data Founder (2 Agustus 2026)
    # Jalur terpisah dari chat orang tua: endpoint sendiri, graph sendiri,
    # katalog sendiri. Otoritasnya tetap di peri-bugi-api — feature flag dan
    # role founder diperiksa di sana sebelum request sampai ke sini.
    # ─────────────────────────────────────────────────────────────────────────

    #: Model penulis SQL untuk jalur founder. Kosong = ikut model chat.
    FOUNDER_SQL_MODEL: str = ""

    #: Cache katalog founder. Katalognya jauh lebih besar daripada katalog orang
    #: tua, jadi mengambilnya ulang tiap giliran terasa.
    FOUNDER_CATALOG_TTL_SECONDS: int = 300

    #: Timeout memanggil endpoint eksekusi di api. WAJIB lebih besar daripada
    #: `statement_timeout` di sana (20 detik) — kalau klien menyerah lebih dulu,
    #: query tetap membakar koneksi sampai selesai dan tidak ada yang membaca
    #: hasilnya.
    FOUNDER_EXECUTE_TIMEOUT_SECONDS: int = 30

    #: Berapa kali SQL yang gagal boleh diperbaiki dalam satu giliran.
    FOUNDER_SQL_MAX_ATTEMPTS: int = 3

    #: Berapa giliran terakhir yang dirender ke prompt penulis-ulang pertanyaan.
    #: Kecil dengan sengaja: yang dibutuhkan cuma antecedent terdekat ("bulan
    #: itu", "yang tadi"), dan riwayat panjang justru membuat model menggabung
    #: dua pertanyaan lama jadi satu pertanyaan yang tidak pernah diajukan.
    FOUNDER_REWRITE_HISTORY_TURNS: int = 3

    # ─────────────────────────────────────────────────────────────────────────
    # LLM penulis kode dasbor (7 Agustus 2026)
    #
    # Penyedia KEDUA, terpisah dari `GEMINI_*` di atas, dan pemisahannya bukan
    # soal kerapian: kunci API-nya beda, modelnya beda (penulis kode, bukan
    # penulis SQL), anggaran keluarannya sepuluh kali lipat, dan kalau salah
    # satunya kena batas kuota, yang lain harus tetap jalan.
    #
    # `CODER_LLM_BASE_URL` yang membuat DeepSeek, GLM, Moonshot, dan OpenRouter
    # bisa dipakai nanti tanpa menyentuh kode — semuanya berbicara
    # `/v1/chat/completions` yang sama.
    # ─────────────────────────────────────────────────────────────────────────
    CODER_LLM_PROVIDER: str = "gemini"  # gemini | anthropic | openai
    CODER_LLM_MODEL: str = ""
    CODER_LLM_API_KEY: str = ""
    CODER_LLM_BASE_URL: str = ""  # kosong = bawaan penyedia

    #: `schema` = JSON Schema ketat (OpenAI resmi). `object` = mode JSON longgar,
    #: yang didukung hampir semua endpoint OpenAI-compatible. Satu env var ini
    #: yang membedakan keduanya; sisa modulnya identik.
    CODER_LLM_JSON_MODE: str = "schema"  # schema | object

    CODER_LLM_TEMPERATURE: float = 0.2
    CODER_LLM_MAX_OUTPUT_TOKENS: int = 8000
    CODER_LLM_TIMEOUT_SECONDS: int = 90

    #: SENGAJA tidak dipaku MINIMAL seperti jalur SQL. `DEFAULT_THINKING_LEVEL`
    #: di `gemini_direct.py` ada karena token penalaran bocor ke dalam SQL.
    #: Menulis kode adalah satu-satunya beban kerja di sini yang benar-benar
    #: diuntungkan penalaran — memaksanya minimal berarti membayar model bagus
    #: lalu melarangnya berpikir.
    CODER_LLM_THINKING_LEVEL: str = "LOW"

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

    # Daftar ini adalah ALLOWLIST NAMA MEDAN, bukan aturan umum. Medan baru yang
    # membawa rahasia atau URL dan lupa ditambahkan di sini tidak dibersihkan —
    # dan spasi atau baris baru yang ikut tertempel dari `.env` atau dari mount
    # Secret Manager menghasilkan 401 yang terbaca seperti "penyedia menolak
    # kita", bukan seperti "kuncinya kotor". Itu satu jam yang terbuang.
    @field_validator("INTERNAL_SECRET", "GEMINI_API_KEY", "OPENAI_API_KEY",
                     "QDRANT_API_KEY", "DB_PASSWORD",
                     "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
                     "CODER_LLM_API_KEY", "CODER_LLM_BASE_URL", "CODER_LLM_MODEL")
    @classmethod
    def strip_secrets(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

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
