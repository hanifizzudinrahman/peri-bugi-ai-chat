"""
Autentikasi antar-service untuk AI Chat Backend.

Service ini di-deploy dengan `--allow-unauthenticated`, jadi siapa pun bisa
menjangkaunya di lapisan jaringan. Header `X-Internal-Secret` adalah satu-satunya
yang menahan endpoint chat (biaya LLM) dan knowledge base.

Dipisah dari `main.py` supaya bisa diuji tanpa memuat LangGraph, sentence-
transformers, dan seluruh registry agent.
"""
import hmac

from fastapi import Header, HTTPException, Request

from app.config.settings import settings

# Path yang boleh diakses tanpa autentikasi.
# Sengaja daftar tertutup (allowlist), bukan daftar larangan — route baru
# otomatis TERLINDUNGI kecuali sengaja didaftarkan di sini.
PUBLIC_PATH_PREFIXES: tuple[str, ...] = ("/health",)
PUBLIC_PATHS_EXACT: frozenset[str] = frozenset({"/docs", "/redoc", "/openapi.json"})


def is_public_path(path: str) -> bool:
    """True kalau path boleh diakses tanpa X-Internal-Secret."""
    if path in PUBLIC_PATHS_EXACT:
        return True
    return any(
        path == prefix or path.startswith(prefix + "/") for prefix in PUBLIC_PATH_PREFIXES
    )


def assert_internal_secret_configured() -> None:
    """
    Dipanggil saat startup. Melempar RuntimeError kalau secret tidak ada,
    sehingga service MENOLAK MENYALA.

    Cloud Run menolak revision yang gagal start, jadi revision lama tetap
    melayani traffic dan tidak ada momen di mana endpoint LLM terbuka.
    """
    if not settings.INTERNAL_SECRET:
        raise RuntimeError(
            "INTERNAL_SECRET tidak di-set. Service menolak menyala supaya endpoint "
            "chat dan knowledge base tidak terbuka tanpa autentikasi. Set "
            "INTERNAL_SECRET di .env (lokal) atau Secret Manager (Cloud Run) — "
            "lihat docs/OPERATIONS.md."
        )


def verify_internal_secret(
    request: Request,
    x_internal_secret: str | None = Header(default=None),
) -> None:
    """
    Dependency tingkat aplikasi. FAIL CLOSED tanpa pengecualian.

    Dipasang di `FastAPI(dependencies=[...])`, bukan dipanggil manual di dalam
    tiap fungsi endpoint. Dua alasan:

    1. **Route baru otomatis terlindungi.** Sebelumnya auth berupa panggilan
       fungsi di baris pertama tiap endpoint — satu route baru yang lupa
       memanggilnya langsung terbuka, tanpa ada yang menahan.
    2. **Dependency berjalan SEBELUM validasi body.** Sebelumnya pemanggil tanpa
       kredensial tetap mendapat 422 berisi nama field dan tipe, sehingga seluruh
       kontrak internal bisa dipetakan tanpa autentikasi.

    Cabang 503 semestinya tidak pernah tercapai karena startup sudah menolak
    konfigurasi kosong — ia lapisan kedua.
    """
    if is_public_path(request.url.path):
        return

    if not settings.INTERNAL_SECRET:
        raise HTTPException(status_code=503, detail="Internal secret not configured")

    if not x_internal_secret or not hmac.compare_digest(
        x_internal_secret, settings.INTERNAL_SECRET
    ):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid internal secret")
