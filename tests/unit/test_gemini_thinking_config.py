"""Mode `thinking` Gemini harus dimatikan untuk SEMUA keluarga yang punya.

Kenapa ada
----------
Pemeriksaannya dulu `"2.5" in model_name`. Benar saat ditulis, dan diam-diam
salah begitu keluarga 3.x muncul — tidak ada yang gagal, tidak ada yang
berteriak, cuma tagihan yang naik dan keluaran yang berantakan.

Terukur saat membandingkan model untuk Tanya Data Founder:

    gemini-3.1-flash-lite   790 token/panggilan, akurasi 19/20,  96 detik
    gemini-3-flash          4.800 token/panggilan, akurasi 11/20, 370 detik

Separuh kegagalan yang kedua berbunyi "SQL tidak bisa di-parse" — penalarannya
ikut tercetak ke keluaran. Terbaca seperti "model yang lebih besar ternyata
lebih bodoh", padahal itu konfigurasi yang tidak pernah menjangkaunya.

Test ini menahan bentuk kesalahan yang sama terulang saat keluarga berikutnya
keluar.

Cara run:
    docker compose exec ai-chat pytest tests/unit/test_gemini_thinking_config.py -v
"""
import pytest

from app.config.llm import _mendukung_thinking


class TestKeluargaYangHarusDimatikan:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
            "models/gemini-3.1-flash-lite-preview",
        ],
    )
    def test_thinking_dimatikan(self, model):
        assert _mendukung_thinking(model), (
            f"{model} punya mode thinking dan akan menyalakannya sendiri. "
            "Tanpa dimatikan, tokennya berlipat dan penalarannya bocor ke "
            "keluaran — SQL jadi tidak bisa di-parse."
        )


class TestYangTidakBolehIkutDisetel:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    )
    def test_keluarga_lama_dibiarkan(self, model):
        assert not _mendukung_thinking(model), (
            f"{model} tidak punya thinking_config; mengirimkannya bisa ditolak "
            "penyedia."
        )

    def test_bukan_gemini_tidak_ikut(self):
        assert not _mendukung_thinking("gpt-4o-mini")
        assert not _mendukung_thinking("qwen3.5:8b")
