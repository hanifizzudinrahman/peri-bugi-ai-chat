"""
Regression test untuk autentikasi antar-service ai-chat.

Tiga cacat yang dikunci di sini (semuanya ditemukan 2026-07-19):

1. `_verify_internal_secret` fail-open kalau INTERNAL_SECRET kosong dan APP_ENV
   bukan "production". APP_ENV defaultnya "development".
2. Auth dipanggil sebagai statement pertama di dalam fungsi endpoint, bukan
   sebagai dependency. Akibatnya validasi body Pydantic berjalan lebih dulu dan
   pemanggil tanpa kredensial menerima 422 yang membocorkan seluruh skema.
3. `openapi_url` tidak pernah dimatikan, sehingga skema lengkap terbuka di
   produksi walau /docs sudah 404.
"""
import ast
import pathlib

import pytest
from fastapi import HTTPException

from app.config.settings import settings
from app.core.security import (
    assert_internal_secret_configured,
    is_public_path,
    verify_internal_secret,
)

SECRET = "s3cr3t-internal-value-untuk-test"


class _FakeURL:
    def __init__(self, path: str):
        self.path = path


class _FakeRequest:
    def __init__(self, path: str):
        self.url = _FakeURL(path)


class TestFailClosed:
    @pytest.mark.parametrize("app_env", ["development", "production", "staging", ""])
    def test_empty_secret_always_rejects(self, monkeypatch, app_env):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", "", raising=False)
        monkeypatch.setattr(settings, "APP_ENV", app_env, raising=False)

        with pytest.raises(HTTPException) as exc:
            verify_internal_secret(_FakeRequest("/chat/stream"), x_internal_secret="apa pun")

        assert exc.value.status_code == 503, (
            f"APP_ENV={app_env!r} dengan secret kosong harus DITOLAK. "
            f"Kalau lolos, cabang fail-open sudah kembali."
        )


class TestVerification:
    def test_correct_secret_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", SECRET, raising=False)
        assert verify_internal_secret(_FakeRequest("/chat/stream"), x_internal_secret=SECRET) is None

    def test_wrong_secret_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", SECRET, raising=False)
        with pytest.raises(HTTPException) as exc:
            verify_internal_secret(_FakeRequest("/chat/stream"), x_internal_secret="salah")
        assert exc.value.status_code == 401

    def test_missing_header_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", SECRET, raising=False)
        with pytest.raises(HTTPException) as exc:
            verify_internal_secret(_FakeRequest("/knowledge/documents"), x_internal_secret=None)
        assert exc.value.status_code == 401


class TestPublicPaths:
    """Health boleh terbuka. Route lain TIDAK, termasuk yang belum ada."""

    @pytest.mark.parametrize(
        "path", ["/health", "/health/agents", "/health/llm", "/health/gpu", "/health/checkpointer"]
    )
    def test_health_is_public(self, path):
        assert is_public_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/chat/stream",
            "/chat/rnd",
            "/knowledge/documents",
            "/memory/summarize",
            "/route/yang/belum/ada",
        ],
    )
    def test_everything_else_is_protected(self, path):
        assert is_public_path(path) is False, (
            f"{path} tidak boleh publik. Allowlist harus tertutup supaya route "
            f"baru otomatis terlindungi."
        )

    def test_health_prefix_does_not_leak_to_similar_names(self):
        # /healthcheck-secret bukan /health/... — jangan sampai lolos
        assert is_public_path("/healthcheck-secret") is False
        assert is_public_path("/health-admin") is False

    def test_health_routes_pass_without_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", SECRET, raising=False)
        assert verify_internal_secret(_FakeRequest("/health"), x_internal_secret=None) is None


class TestStartupValidation:
    def test_refuses_to_start_without_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", "", raising=False)
        with pytest.raises(RuntimeError, match="INTERNAL_SECRET"):
            assert_internal_secret_configured()

    def test_starts_when_secret_present(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_SECRET", SECRET, raising=False)
        assert assert_internal_secret_configured() is None


class TestAppWiring:
    """
    Auth harus terpasang di level aplikasi, dan skema OpenAPI harus bisa
    dimatikan di produksi. Dicek lewat AST supaya tidak perlu meng-import
    main.py (yang menarik LangGraph + sentence-transformers).
    """

    @staticmethod
    def _fastapi_call() -> ast.Call:
        tree = ast.parse(pathlib.Path("app/main.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "FastAPI"
            ):
                return node
        raise AssertionError("Panggilan FastAPI(...) tidak ditemukan di app/main.py")

    def test_app_level_auth_dependency_present(self):
        call = self._fastapi_call()
        kwargs = {kw.arg for kw in call.keywords}
        assert "dependencies" in kwargs, (
            "FastAPI(...) harus punya dependencies=[Depends(verify_internal_secret)]. "
            "Tanpa itu, route baru tidak terlindungi secara default dan validasi "
            "body berjalan sebelum auth."
        )

    def test_openapi_url_is_disabled_in_production(self):
        call = self._fastapi_call()
        kwargs = {kw.arg for kw in call.keywords}
        assert "openapi_url" in kwargs, (
            "openapi_url harus di-set eksplisit (None di produksi). Kalau tidak, "
            "FastAPI memakai default /openapi.json dan skema lengkap terbuka "
            "walau /docs sudah dimatikan."
        )


class TestRouteOrdering:
    """
    Route dengan path literal harus didaftarkan SEBELUM route dengan parameter
    di posisi yang sama. Starlette mencocokkan sesuai urutan pendaftaran.

    Kalau terbalik, `/knowledge/documents/by-source` ditangkap oleh
    `/knowledge/documents/{point_id}` sebagai point_id="by-source". Qdrant
    menghapus id yang tidak ada (no-op) lalu endpoint mengembalikan pesan
    SUKSES PALSU — data tidak terhapus tapi pemanggil mengira berhasil.
    """

    @staticmethod
    def _routes() -> list[tuple[int, str, str]]:
        tree = ast.parse(pathlib.Path("app/main.py").read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                ):
                    found.append((dec.lineno, dec.func.attr.lower(), dec.args[0].value))
        return sorted(found)

    @pytest.mark.parametrize(
        "method,literal,param",
        [
            ("delete", "/knowledge/documents/by-source", "/knowledge/documents/{point_id}"),
            ("patch", "/knowledge/documents/bulk-toggle", "/knowledge/documents/{point_id}"),
        ],
    )
    def test_literal_route_registered_before_parameterised(self, method, literal, param):
        routes = self._routes()
        lit = next((ln for ln, m, p in routes if m == method and p == literal), None)
        par = next((ln for ln, m, p in routes if m == method and p == param), None)

        assert lit is not None, f"{method.upper()} {literal} tidak ditemukan"
        assert par is not None, f"{method.upper()} {param} tidak ditemukan"
        assert lit < par, (
            f"{method.upper()} {literal} (baris {lit}) HARUS didaftarkan sebelum "
            f"{param} (baris {par}), kalau tidak ia tak terjangkau dan yang "
            f"parameterised mengembalikan sukses palsu."
        )
