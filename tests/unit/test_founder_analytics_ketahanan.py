"""Ketahanan jalur founder terhadap BENTUK keluaran model, bukan isinya.

Kenapa berkas ini ada
---------------------
Perbandingan model berkali-kali memberi kesimpulan yang salah karena kegagalan
yang sebenarnya soal bentuk keluaran terbaca sebagai "model ini lebih bodoh":

- Satu model mencetak **0/19** karena node plan mengeluarkan JSON dengan bentuk
  sedikit berbeda, dan pipanya tidak punya jalan mundur sama sekali.
- Satu model mencetak `nol SQL — None` karena keluarannya terpotong di
  `max_output_tokens` — lalu dilaporkan ke founder sebagai *"model menyatakan
  pertanyaan ini tidak terjawab oleh katalog"*. Sebab teknis menyamar jadi
  keputusan model.
- Satu model membungkus SQL-nya dengan kalimat pengantar, dan galatnya diumpan
  ke loop perbaikan SQL — yang menyuruhnya membetulkan semantik query padahal
  masalahnya dia kebanyakan bicara.

Semua di atas kelas kegagalan yang bisa diperiksa TANPA memanggil API. Itu
gunanya berkas ini: klaim "jalan dengan model apa pun" jadi sesuatu yang
dijalankan `pytest`, bukan sesuatu yang diingat orang dari satu putaran eval.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.agents.founder_analytics.graph as g
from app.agents.founder_analytics.state import (
    ChartIntent,
    FounderAnalyticsState,
    KeluaranSQL,
    RencanaKueri,
)
from app.config import gemini_direct


# =============================================================================
# Menyelamatkan SQL dari keluaran yang bertele-tele
# =============================================================================


class TestPenyelamatSQL:
    def test_sql_polos_lewat_apa_adanya(self):
        assert g._selamatkan_sql("SELECT 1") == "SELECT 1"

    def test_titik_koma_dibuang(self):
        assert g._selamatkan_sql("SELECT 1;") == "SELECT 1"

    def test_pagar_kode_dibuang(self):
        assert g._selamatkan_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_kalimat_pengantar_dibuang(self):
        """Ini bentuk kegagalan yang dulu lolos utuh ke validator."""
        teks = "Tentu, berikut querinya:\n\nSELECT count(*) FROM nlf.v_user"
        assert g._selamatkan_sql(teks) == "SELECT count(*) FROM nlf.v_user"

    def test_cte_juga_dikenali(self):
        teks = "Berikut:\nWITH x AS (SELECT 1) SELECT * FROM x"
        assert g._selamatkan_sql(teks) == "WITH x AS (SELECT 1) SELECT * FROM x"

    def test_penjelasan_setelah_sql_ikut_terbawa(self):
        """Batasnya jujur: yang dipotong cuma di DEPAN.

        Ekor penjelasan tetap ikut dan akan ditolak validator — dan itu memang
        benar, karena memotong di belakang menuntut menebak di mana query-nya
        berakhir, dan tebakan yang salah menghasilkan query yang JALAN tapi
        menjawab pertanyaan lain. Lebih baik ditolak daripada diam-diam keliru.
        """
        hasil = g._selamatkan_sql("SELECT 1\n\nQuery ini menghitung jumlah baris.")
        assert hasil.startswith("SELECT 1")
        assert "menghitung" in hasil

    def test_tanpa_sql_sama_sekali_jadi_kosong(self):
        assert g._selamatkan_sql("Maaf, saya tidak bisa menjawab itu.") == ""

    def test_kosong_tetap_kosong(self):
        assert g._selamatkan_sql("") == ""


# =============================================================================
# Parse ke model Pydantic, dan apa yang terjadi kalau gagal
# =============================================================================


class TestParseModel:
    def test_json_valid(self):
        r = g._parse_model('{"dataset_names": ["user"], "time_hint": "bulan ini"}', RencanaKueri)
        assert r.dataset_names == ["user"]
        assert r.time_hint == "bulan ini"

    def test_json_dibungkus_kalimat_tetap_terbaca(self):
        r = g._parse_model('Ini pilihannya: {"dataset_names": ["user"]}', RencanaKueri)
        assert r.dataset_names == ["user"]

    def test_keluaran_sampah_memberi_objek_kosong_bukan_none(self):
        """Bukan `None` — pemanggilnya tidak boleh perlu memeriksa dua hal."""
        r = g._parse_model("saya tidak paham", RencanaKueri)
        assert isinstance(r, RencanaKueri)
        assert r.dataset_names == []

    def test_model_pydantic_memberi_none_saat_gagal(self):
        assert g._model_pydantic("bukan json", KeluaranSQL) is None


# =============================================================================
# Node SQL — tiap sebab kegagalan punya pesannya sendiri
# =============================================================================


def _state() -> FounderAnalyticsState:
    return FounderAnalyticsState(question="berapa orang tua?", catalog_prompt="### nlf.v_user")


@pytest.mark.asyncio
class TestNodeSQLMembedakanSebab:
    async def _jalankan(self, keluaran_llm: str):
        st = _state()
        with patch.object(g, "_llm_text", AsyncMock(return_value=keluaran_llm)):
            sql = await g.node_sql(st, attempt=1)
        return sql, st

    async def test_jalur_normal(self):
        sql, st = await self._jalankan('{"bisa_dijawab": true, "sql": "SELECT 1"}')
        assert sql == "SELECT 1"
        assert st.sql_gagal is None

    async def test_model_menolak_membawa_alasannya_sendiri(self):
        sql, st = await self._jalankan(
            '{"bisa_dijawab": false, "alasan": "katalog tidak punya data pembayaran"}'
        )
        assert sql is None
        assert st.sql_gagal == "katalog tidak punya data pembayaran"

    async def test_sql_berbungkus_prosa_diselamatkan(self):
        sql, st = await self._jalankan(
            '{"bisa_dijawab": true, "sql": "Berikut querinya: SELECT 1"}'
        )
        assert sql == "SELECT 1"

    async def test_bilang_bisa_tapi_sql_kosong_bukan_penolakan(self):
        """Beda pesan, karena beda obatnya."""
        sql, st = await self._jalankan('{"bisa_dijawab": true, "sql": ""}')
        assert sql is None
        assert "tanpa query" in st.sql_gagal

    async def test_keluaran_bukan_json_masih_diperlakukan_sebagai_sql(self):
        """Jalur mundur: tangga bisa sampai ke LangChain, yang tidak menjamin bentuk."""
        sql, st = await self._jalankan("SELECT count(*) FROM nlf.v_user")
        assert sql == "SELECT count(*) FROM nlf.v_user"

    async def test_keluaran_kosong_tidak_dilaporkan_sebagai_penolakan_model(self):
        """Kalimat lama menuduh model menolak. Itu menyesatkan orang ke katalog."""
        sql, st = await self._jalankan("")
        assert sql is None
        assert "menolak" not in (st.sql_gagal or "")
        assert "kosong" in st.sql_gagal


# =============================================================================
# Keluaran terpotong — kelas kegagalan yang dulu tidak punya nama
# =============================================================================


class _FakePart:
    def __init__(self, text, thought=False):
        self.text = text
        self.thought = thought


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeKandidat:
    def __init__(self, parts, finish_reason=None):
        self.content = _FakeContent(parts)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, parts, finish_reason=None):
        self.candidates = [_FakeKandidat(parts, finish_reason)]
        self.text = "".join(p.text for p in parts)
        self.usage_metadata = None


class _AlasanEnum:
    """Meniru enum SDK, yang punya `.name` — bukan string biasa."""

    name = "MAX_TOKENS"


class TestAlasanBerhenti:
    def test_membaca_enum_lewat_name(self):
        r = _FakeResponse([_FakePart("x")], finish_reason=_AlasanEnum())
        assert gemini_direct._alasan_berhenti(r) == "MAX_TOKENS"

    def test_string_biasa_juga_terbaca(self):
        r = _FakeResponse([_FakePart("x")], finish_reason="STOP")
        assert gemini_direct._alasan_berhenti(r) == "STOP"

    def test_tanpa_alasan_jadi_kosong(self):
        assert gemini_direct._alasan_berhenti(_FakeResponse([_FakePart("x")])) == ""


@pytest.mark.asyncio
class TestGenerateMelemparSaatTerpotong:
    async def _panggil(self, response):
        klien = type(
            "K", (), {"aio": type("A", (), {"models": type("M", (), {})()})()}
        )()
        klien.aio.models.generate_content = AsyncMock(return_value=response)
        with (
            patch.object(gemini_direct, "_klien", return_value=klien),
            patch.object(gemini_direct, "_config", return_value=None),
        ):
            return await gemini_direct.generate(
                system="s", user="u", model="gemini-3.1-flash-lite", max_tokens=300
            )

    async def test_max_tokens_melempar_bukan_mengembalikan_kosong(self):
        r = _FakeResponse([_FakePart("SELECT cou")], finish_reason=_AlasanEnum())
        with pytest.raises(gemini_direct.TeksTerpotong) as exc:
            await self._panggil(r)
        # Pesannya harus menyebut anggarannya — itu yang perlu dinaikkan.
        assert "max_output_tokens=300" in str(exc.value)

    async def test_selesai_normal_tidak_melempar(self):
        r = _FakeResponse([_FakePart("SELECT 1")], finish_reason="STOP")
        hasil = await self._panggil(r)
        assert hasil.teks == "SELECT 1"


class TestPembuangBagianPenalaranTetapUtuh:
    def test_seluruhnya_penalaran_memberi_kosong_bukan_teks_tercampur(self):
        r = _FakeResponse([_FakePart("hmm, mungkin...", thought=True)])
        teks, dibuang = gemini_direct._ambil_teks(r)
        assert teks == ""
        assert dibuang == 1


# =============================================================================
# Tangga jalur-mundur di _llm_text
# =============================================================================


@pytest.mark.asyncio
class TestTanggaJalurMundur:
    async def _panggil(self, generate_mock, response_schema=RencanaKueri):
        st = _state()
        with (
            patch.object(g.gemini_direct, "tersedia", return_value=True),
            patch.object(g.gemini_direct, "generate", generate_mock),
        ):
            return await g._llm_text(
                system="s",
                user="u",
                span_name="founder-plan",
                state=st,
                response_schema=response_schema,
            ), st

    async def test_skema_ditolak_diulang_tanpa_skema_bukan_langsung_ke_langchain(self):
        """Jatuh ke LangChain berarti kehilangan kendali penalaran.

        Itu obat yang lebih buruk daripada penyakitnya: bentuk keluaran yang
        longgar cuma menyulitkan parser, sedangkan penalaran tak terkendali
        mencetak isinya ke dalam SQL — dan tidak ada gejalanya di log.
        """
        panggilan = []

        async def fake_generate(**kw):
            panggilan.append(kw.get("response_schema"))
            if kw.get("response_schema") is not None:
                raise ValueError("response_schema tidak didukung")
            return gemini_direct.HasilGenerasi(teks='{"dataset_names": ["user"]}')

        teks, _ = await self._panggil(AsyncMock(side_effect=fake_generate))
        assert panggilan == [RencanaKueri, None]
        assert '"user"' in teks

    async def test_terpotong_tidak_diulang_dan_tidak_jatuh_ke_langchain(self):
        """Mengulanginya dengan anggaran yang sama memberi hasil yang sama."""
        gen = AsyncMock(side_effect=gemini_direct.TeksTerpotong("habis di 300"))
        with patch.object(g, "get_llm") as get_llm:
            teks, st = await self._panggil(gen)
        assert teks == ""
        assert gen.await_count == 1
        get_llm.assert_not_called()
        # Biayanya tetap tercatat, dengan success=False — kalau tidak, node yang
        # gagal terlihat seperti node yang tidak pernah dipanggil.
        assert st.llm_call_logs[-1]["success"] is False

    async def test_tanpa_skema_hanya_satu_anak_tangga(self):
        gen = AsyncMock(return_value=gemini_direct.HasilGenerasi(teks="halo"))
        teks, _ = await self._panggil(gen, response_schema=None)
        assert teks == "halo"
        assert gen.await_count == 1


# =============================================================================
# Pembanding eval — aturannya menentukan vonis tentang model
# =============================================================================


def _run_py():
    """Muat harness eval sebagai modul. Ia di luar paket `app`.

    Didaftarkan ke `sys.modules` SEBELUM di-exec: `@dataclass` mencari modul
    kelasnya di sana, dan tanpa itu gagal dengan
    `'NoneType' object has no attribute '__dict__'` yang tidak menyebut-nyebut
    sys.modules sama sekali.
    """
    import importlib.util
    import sys
    from pathlib import Path

    if "eval_run" in sys.modules:
        return sys.modules["eval_run"]

    p = Path(__file__).resolve().parents[2] / "evals" / "founder_analytics" / "run.py"
    spec = importlib.util.spec_from_file_location("eval_run", p)
    modul = importlib.util.module_from_spec(spec)
    sys.modules["eval_run"] = modul
    spec.loader.exec_module(modul)
    return modul


class TestPenyamaanTanggal:
    """`2026-07-01` dan `2026-07-01T00:00:00+00:00` adalah tanggal yang SAMA.

    Model yang menulis `generate_series(...)` tanpa `::date` menghasilkan kolom
    timestamptz dengan isi yang identik sampai desimal terakhir. Sebelum aturan
    ini ada, itu dihitung 31 baris salah dan terbaca sebagai "modelnya kurang
    akurat" — padahal angkanya sama persis.
    """

    def test_tengah_malam_utc_disamakan_dengan_tanggal(self):
        r = _run_py()
        emas = [["2026-07-01", 5.0], ["2026-07-02", 2.5]]
        model = [["2026-07-01T00:00:00+00:00", 5.0], ["2026-07-02T00:00:00Z", 2.5]]
        assert r._normalkan(emas) == r._normalkan(model)

    def test_jam_selain_tengah_malam_TETAP_dianggap_beda(self):
        """Sengaja sempit. Jam berapa pun selain 00:00:00 memang informasi lain."""
        r = _run_py()
        assert r._normalkan([["2026-07-01", 1]]) != r._normalkan(
            [["2026-07-01T07:00:00+00:00", 1]]
        )

    def test_nilai_beda_tetap_beda_walau_tanggalnya_disamakan(self):
        r = _run_py()
        assert r._normalkan([["2026-07-01", 5.0]]) != r._normalkan(
            [["2026-07-01T00:00:00+00:00", 6.0]]
        )

    def test_beda_baris_menunjukkan_sisi_mana_yang_kelebihan(self):
        r = _run_py()
        beda = r._beda_baris([["a", 1], ["b", 2]], [["a", 1], ["c", 3]])
        assert beda["hanya_emas"] == [["b", 2]]
        assert beda["hanya_model"] == [["c", 3]]


# =============================================================================
# Konfigurasi yang dikirim ke Google
# =============================================================================


class TestKonfigurasiDecodingBerbatas:
    def _kw(self, **kw):
        dipanggil = {}

        class FakeTypes:
            @staticmethod
            def GenerateContentConfig(**k):
                dipanggil.update(k)
                return "cfg"

            @staticmethod
            def ThinkingConfig(**k):
                return ("thinking", k)

        import sys
        from types import ModuleType

        modul = ModuleType("google.genai")
        modul.types = FakeTypes
        with patch.dict(sys.modules, {"google.genai": modul}):
            gemini_direct._config(
                model=kw.pop("model", "gemini-3.1-flash-lite"),
                system="s",
                temperature=0.0,
                max_tokens=100,
                level=None,
                **kw,
            )
        return dipanggil

    def test_tanpa_skema_tidak_menyetel_mime_type(self):
        kw = self._kw()
        assert "response_mime_type" not in kw
        assert "response_schema" not in kw

    def test_dengan_skema_menyetel_keduanya(self):
        kw = self._kw(response_schema=ChartIntent)
        assert kw["response_mime_type"] == "application/json"
        assert kw["response_schema"] is ChartIntent

    def test_keluarga_tak_dikenal_diperingatkan_bukan_didiamkan(self, caplog):
        """Model yang tidak cocok pola jalan TANPA kendali thinking.

        Tidak ada gejalanya sampai tagihannya datang — jadi minimal ada satu
        baris log yang bisa dicari.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            self._kw(model="model-baru-yang-belum-ada")
        assert any("tidak cocok keluarga" in r.message for r in caplog.records)
