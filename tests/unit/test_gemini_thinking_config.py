"""Kendali `thinking` Gemini — di mana ia bisa dipasang, dan di mana tidak.

Kenapa berkas ini ada
---------------------
`app/config/llm.py` pernah punya baris ini:

    kwargs["model_kwargs"] = {"generation_config":
        {"thinking_config": {"thinking_budget": 0}}}

Baris itu **tidak pernah berlaku**. `ChatGoogleGenerativeAI` tidak punya field
`model_kwargs`, dan pydantic-nya diset `extra="ignore"` — jadi kuncinya dibuang
tanpa galat dan tanpa peringatan. SDK lamanya sendiri tidak punya medan
`thinking_config` sama sekali.

Tidak ada yang gagal, tidak ada yang berteriak. Yang terjadi cuma tagihan naik
dan keluaran berantakan, dan baru ketahuan saat membandingkan model:

    gemini-3.1-flash-lite     790 token/panggilan   akurasi 19/20
    gemini-3.5-flash        3.898 token/panggilan   eval tidak selesai

Kegagalannya berbunyi "SQL tidak bisa di-parse" — isi penalaran ikut tercetak
ke dalam query.

Test ini menahan dua hal:

1. Bentuk kesalahan aslinya — mengirim konfigurasi ke tempat yang menerimanya
   tanpa memakainya. Kalau pustaka suatu saat dinaikkan dan `model_kwargs`
   muncul, test-nya gagal dengan pesan "sekarang sudah bisa, aktifkan lagi".
2. Pemilihan parameter di klien langsung: Gemini 3.x pakai `thinking_level`,
   2.5 pakai `thinking_budget`, dan tidak pernah keduanya sekaligus.

Cara run:
    docker compose exec ai-chat pytest tests/unit/test_gemini_thinking_config.py -v
"""
import pytest

from app.config import gemini_direct as gd


class TestKenapaLewatLangChainTidakBisa:
    """Kalau salah satu ini berubah, jalur pintasnya boleh ditinjau ulang."""

    def test_adaptor_lama_tidak_punya_model_kwargs(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        punya = "model_kwargs" in ChatGoogleGenerativeAI.model_fields
        assert not punya, (
            "ChatGoogleGenerativeAI SEKARANG punya field model_kwargs — "
            "pustakanya naik versi. Tinjau ulang app/config/gemini_direct.py: "
            "mungkin jalur pintas SDK langsung sudah tidak diperlukan."
        )

    def test_extra_masih_ignore_jadi_kunci_asing_hilang_diam_diam(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        assert ChatGoogleGenerativeAI.model_config.get("extra") == "ignore", (
            "Kalau ini berubah jadi 'forbid', kunci asing akan MELEDAK "
            "alih-alih hilang diam-diam — dan itu justru lebih baik."
        )

    def test_sdk_lama_tidak_punya_thinking_config(self):
        import google.generativeai.types as t

        medan = getattr(t.GenerationConfig, "__annotations__", {})
        assert "thinking_config" not in medan, (
            "SDK lama sekarang punya thinking_config. Kendali penalaran bisa "
            "dipasang lewat jalur LangChain lagi."
        )


class TestPemilihanParameterDiKlienLangsung:
    """Gemini 3 memakai thinking_level; 2.5 memakai thinking_budget.

    Mengirim keduanya sekaligus dijawab 400 oleh Google, jadi percabangannya
    harus eksklusif — bukan menumpuk.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3-pro-preview",
        ],
    )
    def test_keluarga_3x_memakai_level(self, model):
        cfg = gd._thinking_config(model, None)
        assert cfg is not None
        assert cfg.thinking_level is not None, f"{model} harus memakai thinking_level"
        assert cfg.thinking_budget is None, (
            f"{model} tidak boleh mengirim thinking_budget bersamaan — "
            "Google menjawab 400."
        )
        assert cfg.include_thoughts is False

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    def test_keluarga_25_memakai_budget(self, model):
        cfg = gd._thinking_config(model, None)
        assert cfg is not None
        assert cfg.thinking_budget == 0, f"{model} harus memakai thinking_budget"
        assert cfg.thinking_level is None, (
            f"{model} tidak mengenal thinking_level."
        )

    @pytest.mark.parametrize("model", ["gemini-2.0-flash", "gemini-1.5-pro"])
    def test_model_tanpa_penalaran_tidak_dikirimi_apa_apa(self, model):
        assert gd._thinking_config(model, None) is None, (
            f"{model} tidak punya mode penalaran; mengirim konfigurasinya "
            "berisiko ditolak penyedia."
        )

    def test_level_bisa_ditimpa(self):
        cfg = gd._thinking_config("gemini-3.5-flash", "HIGH")
        assert cfg.thinking_level == "HIGH"

    def test_default_minimal(self):
        assert gd.DEFAULT_THINKING_LEVEL == "MINIMAL"


class TestPembuangBagianPenalaran:
    """Bagian bertanda `thought` tidak boleh ikut ke jawaban.

    Ini akar kerusakannya: penalaran dan jawaban tergabung jadi satu string,
    sehingga 'SELECT ...' bercampur catatan berpikir dan SQL-nya gagal
    di-parse.
    """

    class _Part:
        def __init__(self, text, thought=False):
            self.text = text
            self.thought = thought

    class _Isi:
        def __init__(self, parts):
            self.parts = parts

    class _Kandidat:
        def __init__(self, parts):
            self.content = TestPembuangBagianPenalaran._Isi(parts)

    class _Resp:
        def __init__(self, parts, text=""):
            self.candidates = [TestPembuangBagianPenalaran._Kandidat(parts)]
            self.text = text

    def test_bagian_thought_dibuang(self):
        resp = self._Resp(
            [
                self._Part("Mari saya pikirkan dulu kolomnya...", thought=True),
                self._Part("SELECT count(*) FROM nlf.v_user"),
            ]
        )
        teks, dibuang = gd._ambil_teks(resp)
        assert teks == "SELECT count(*) FROM nlf.v_user"
        assert dibuang == 1

    def test_tanpa_thought_utuh(self):
        resp = self._Resp([self._Part("SELECT 1")])
        teks, dibuang = gd._ambil_teks(resp)
        assert teks == "SELECT 1"
        assert dibuang == 0

    def test_cadangan_text_agregat_dipakai_hanya_kalau_bersih(self):
        """`.text` agregat sudah tercampur penalaran — jangan dipakai kalau
        ada bagian thought yang dibuang."""
        resp = self._Resp([], text="SELECT 1")
        assert gd._ambil_teks(resp) == ("SELECT 1", 0)

        kotor = self._Resp(
            [self._Part("berpikir...", thought=True)], text="berpikir...SELECT 1"
        )
        teks, dibuang = gd._ambil_teks(kotor)
        assert teks == ""
        assert dibuang == 1
