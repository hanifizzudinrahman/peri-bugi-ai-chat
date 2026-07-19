"""
Pengunci default konfigurasi yang berdampak keamanan.

`APP_ENV` di service ini menentukan tiga hal (`app/main.py:31`, `:48`, `:54`):

- format log (console berwarna vs JSON)
- `/docs` terbuka atau tidak
- `/openapi.json` terbuka atau tidak

Yang dilindungi adalah skema API: `/openapi.json` membocorkan seluruh bentuk
endpoint beserta model Pydantic-nya. Itu pernah benar-benar terbuka di produksi
karena `openapi_url` tidak pernah di-set — ditutup 19 Juli 2026.

Default `"production"` memastikan host yang lupa menyetel env var jatuh ke mode
paling ketat, bukan paling longgar.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_app_env_default_production():
    source = (REPO_ROOT / "app" / "config" / "settings.py").read_text(encoding="utf-8")

    assert re.search(r'^\s*APP_ENV:\s*str\s*=\s*"production"', source, re.MULTILINE), (
        "default APP_ENV harus 'production' — fail safe kalau env var terlewat"
    )


def test_env_example_menyetel_development_eksplisit():
    """
    Flip default hanya aman kalau dev lokal menyetelnya eksplisit.

    Di-skip saat dijalankan di dalam image: Dockerfile hanya menyalin `app/`,
    jadi `.env.example` memang tidak ada di sana. Pemeriksaan ini berlaku saat
    test dijalankan dari checkout penuh (mis. dengan repo di-mount).
    """
    berkas = REPO_ROOT / ".env.example"
    if not berkas.exists():
        pytest.skip(".env.example tidak ada di konteks ini (kemungkinan di dalam image)")

    contoh = berkas.read_text(encoding="utf-8")
    assert re.search(r"^APP_ENV\s*=\s*development", contoh, re.MULTILINE), (
        ".env.example harus menyetel APP_ENV=development eksplisit"
    )


def test_openapi_dan_docs_tertutup_di_produksi():
    """
    Mengunci perbaikan 19 Juli 2026. `docs_url` saja tidak cukup — kalau
    `openapi_url` tidak di-set, FastAPI memakai default dan skema lengkap
    tetap bisa diambil siapa saja.
    """
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert "openapi_url" in source, "openapi_url harus di-set eksplisit"
    assert re.search(r"docs_url\s*=.*is_production", source)
    assert re.search(r"openapi_url\s*=.*is_production", source)
