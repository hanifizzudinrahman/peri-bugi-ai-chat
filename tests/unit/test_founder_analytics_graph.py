"""Alur Tanya Data Founder dengan LLM dipalsukan.

Kenapa LLM-nya dipalsukan
-------------------------
Bukan karena panggilan sungguhan mahal, tapi karena yang diuji di sini bukan
mutu jawaban model — melainkan **apa yang dilakukan alur ini terhadap keluaran
model**, termasuk keluaran yang buruk: JSON berpagar kode, nama kolom karangan,
niat grafik yang tidak masuk akal, SQL yang ditolak validator. Semua itu tidak
bisa dipanggil sesuka hati dari model sungguhan.

Mutu jawaban diukur terpisah di `evals/founder_analytics/`, yang memang butuh
LLM dan database hidup.

Cara run:
    docker compose exec ai-chat pytest tests/unit/test_founder_analytics_graph.py -v
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.founder_analytics import graph as g
from app.agents.founder_analytics.chart_compile import (
    MAX_WARNA,
    ChartSkipped,
    compile_chart,
)
from app.agents.founder_analytics.state import (
    ChartField,
    ChartIntent,
    FounderAnalyticsState,
)


# =============================================================================
# Pembersih keluaran model
# =============================================================================


class TestPembersihKeluaran:
    @pytest.mark.parametrize(
        "mentah,harap",
        [
            ("```sql\nSELECT 1\n```", "SELECT 1"),
            ("```\nSELECT 1\n```", "SELECT 1"),
            ("SELECT 1", "SELECT 1"),
            ("  SELECT 1  ", "SELECT 1"),
        ],
    )
    def test_pagar_kode_dibuang(self, mentah, harap):
        assert g._strip_fence(mentah) == harap

    def test_json_polos(self):
        assert g._parse_json('{"a": 1}') == {"a": 1}

    def test_json_berpagar(self):
        assert g._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_dengan_kalimat_pengantar(self):
        """Model kadang menambah satu kalimat sebelum JSON-nya."""
        teks = 'Berikut pilihannya:\n{"dataset_names": ["user"]}'
        assert g._parse_json(teks) == {"dataset_names": ["user"]}

    def test_json_rusak_jadi_none(self):
        assert g._parse_json("bukan json sama sekali") is None


# =============================================================================
# Kompilasi grafik
# =============================================================================


def _intent(**kw):
    dasar = dict(
        kind="line",
        x=ChartField(field="tanggal", type="temporal", title="Tanggal"),
        y=ChartField(field="jumlah", type="quantitative", title="Jumlah"),
        title="Tren",
    )
    dasar.update(kw)
    return ChartIntent(**dasar)


KOLOM = ["tanggal", "jumlah", "kategori"]
BARIS = [["2026-07-01", 3, "a"], ["2026-07-02", 5, "b"], ["2026-07-03", 4, "a"]]


class TestKompilasiGrafik:
    def test_deret_waktu_jadi_line(self):
        spec = compile_chart(_intent(), columns=KOLOM, rows=BARIS)
        assert spec["mark"] == "line"
        assert spec["encoding"]["x"]["field"] == "tanggal"
        assert spec["encoding"]["y"]["type"] == "quantitative"
        assert spec["title"] == "Tren"

    def test_tanpa_data_config_dan_width(self):
        """Ketiganya ditambahkan di tempat lain — lihat docstring modulnya."""
        spec = compile_chart(_intent(), columns=KOLOM, rows=BARIS)
        for kunci in ("data", "config", "width", "$schema", "autosize"):
            assert kunci not in spec, (
                f"'{kunci}' tidak boleh ikut: penjaga di peri-bugi-api "
                "membuangnya, dan tema ada di peri-bugi-web."
            )

    def test_kind_none_dilewati(self):
        with pytest.raises(ChartSkipped):
            compile_chart(
                _intent(kind="none", reason="cuma satu angka"),
                columns=KOLOM,
                rows=BARIS,
            )

    def test_satu_baris_dilewati(self):
        """Satu titik bukan tren, satu batang bukan perbandingan."""
        with pytest.raises(ChartSkipped, match="1 baris"):
            compile_chart(_intent(), columns=KOLOM, rows=BARIS[:1])

    def test_kolom_hantu_dilewati(self):
        """Grafik kosong terbaca sebagai 'tidak ada data' — paling menyesatkan."""
        intent = _intent(x=ChartField(field="tidak_ada", type="temporal"))
        with pytest.raises(ChartSkipped, match="tidak ada"):
            compile_chart(intent, columns=KOLOM, rows=BARIS)

    def test_y_bukan_kuantitatif_dilewati(self):
        intent = _intent(y=ChartField(field="kategori", type="nominal"))
        with pytest.raises(ChartSkipped, match="kuantitatif"):
            compile_chart(intent, columns=KOLOM, rows=BARIS)

    def test_temporal_yang_isinya_bukan_tanggal_diturunkan_ke_nominal(self):
        """Sumbu temporal atas teks menghasilkan sumbu kosong tanpa galat."""
        kolom = ["nama", "jumlah"]
        baris = [["Bekasi", 3], ["Depok", 5]]
        intent = ChartIntent(
            kind="bar",
            x=ChartField(field="nama", type="temporal"),
            y=ChartField(field="jumlah", type="quantitative"),
        )
        spec = compile_chart(intent, columns=kolom, rows=baris)
        assert spec["encoding"]["x"]["type"] == "nominal"

    def test_warna_dibuang_kalau_kategorinya_kebanyakan(self):
        kolom = ["k", "v"]
        baris = [[f"kat-{i}", i] for i in range(MAX_WARNA + 5)]
        intent = ChartIntent(
            kind="bar",
            x=ChartField(field="k", type="nominal"),
            y=ChartField(field="v", type="quantitative"),
            color=ChartField(field="k", type="nominal"),
        )
        spec = compile_chart(intent, columns=kolom, rows=baris)
        assert "color" not in spec["encoding"], (
            "Legenda dengan belasan warna lebih tinggi daripada grafiknya."
        )

    def test_warna_dipertahankan_kalau_sedikit(self):
        intent = _intent(
            kind="bar",
            x=ChartField(field="tanggal", type="nominal"),
            color=ChartField(field="kategori", type="nominal"),
        )
        spec = compile_chart(intent, columns=KOLOM, rows=BARIS)
        assert spec["encoding"]["color"]["field"] == "kategori"

    def test_label_tanggal_dimiringkan_kalau_padat(self):
        baris = [[f"2026-07-{d:02d}", d] for d in range(1, 26)]
        spec = compile_chart(_intent(), columns=["tanggal", "jumlah"], rows=baris)
        assert spec["encoding"]["x"]["axis"]["labelAngle"] == -45

    def test_urutan_berdasarkan_y(self):
        intent = _intent(kind="bar", sort="y_desc")
        spec = compile_chart(intent, columns=KOLOM, rows=BARIS)
        assert spec["encoding"]["x"]["sort"]["order"] == "descending"
        assert spec["encoding"]["x"]["sort"]["field"] == "jumlah"


# =============================================================================
# Tabel untuk prompt
# =============================================================================


class TestTabelUntukPrompt:
    def test_baris_berlebih_disebutkan_bukan_dibuang_diam_diam(self):
        state = FounderAnalyticsState(question="q")
        state.columns = ["a"]
        state.rows = [[i] for i in range(g._MAX_ROWS_TO_PROMPT + 20)]
        state.row_count = len(state.rows)
        teks = g._table_for_prompt(state)
        assert "20 baris lagi" in teks
        assert "berkas unduhan" in teks

    def test_tanpa_hasil(self):
        state = FounderAnalyticsState(question="q")
        assert g._table_for_prompt(state) == "(tidak ada hasil)"


# =============================================================================
# Alur penuh, LLM dipalsukan
# =============================================================================


KATALOG = {
    "version": "uji123",
    "index_text": "  - nlf.v_user (akuisisi) — satu baris per akun",
    "prompt_text": "### nlf.v_user",
    "datasets": [{"name": "user", "view": "nlf.v_user"}],
    "max_rows": 5000,
}


async def _kumpulkan(payload: dict) -> list[dict]:
    keluar = []
    async for baris in g.run_founder_analytics(payload):
        keluar.append(json.loads(baris[6:]))
    return keluar


def _urutan(events: list[dict]) -> list[str]:
    nama = [e["event"] for e in events]
    return [n for i, n in enumerate(nama) if i == 0 or nama[i - 1] != n]


@pytest.fixture
def alur_bersih():
    """LLM yang selalu menjawab benar, dan eksekusi yang selalu berhasil."""

    async def fake_llm(*, system, user, span_name, state, **kw):
        if span_name == "founder-plan":
            return '{"dataset_names": ["user"], "time_hint": ""}'
        if span_name.startswith("founder-sql"):
            return "SELECT registered_date_wib AS tanggal, count(*) AS jumlah FROM nlf.v_user GROUP BY 1"
        if span_name == "founder-chart-intent":
            return json.dumps(
                {
                    "kind": "line",
                    "x": {"field": "tanggal", "type": "temporal", "title": "Tanggal"},
                    "y": {"field": "jumlah", "type": "quantitative", "title": "Akun"},
                    "y_aggregate": "none",
                    "title": "Tren pendaftaran",
                    "reason": "deret waktu",
                }
            )
        return ""

    async def fake_execute(state, *, sql, attempt):
        return {
            "ok": True,
            "columns": ["tanggal", "jumlah"],
            "rows": [["2026-07-01", 2], ["2026-07-02", 3]],
            "row_count": 2,
            "truncated": False,
            "datasets": ["user"],
            "pii_datasets": [],
            "elapsed_ms": 12,
            "sql": sql,
        }

    class FakeChunk:
        def __init__(self, c):
            self.content = c

    class FakeLLM:
        async def astream(self, msgs):
            for kata in ["Ada ", "5 ", "akun."]:
                yield FakeChunk(kata)

    with (
        patch.object(g, "_load_catalog", AsyncMock(return_value=KATALOG)),
        patch.object(g, "_prompt_for", AsyncMock(return_value=("### nlf.v_user", "uji123"))),
        patch.object(g, "_llm_text", side_effect=fake_llm),
        patch.object(g, "node_execute", side_effect=fake_execute),
        patch.object(g, "get_llm", return_value=FakeLLM()),
    ):
        yield


class TestAlurBerhasil:
    @pytest.mark.asyncio
    async def test_urutan_event(self, alur_bersih):
        events = await _kumpulkan({"question": "Tren pendaftaran?"})
        urutan = _urutan(events)
        assert urutan == [
            "thinking",
            "sql",
            "data",
            "thinking",
            "chart",
            "thinking",
            "token",
            "meta",
            "done",
        ], urutan

    @pytest.mark.asyncio
    async def test_sql_dan_tabel_datang_sebelum_narasi(self, alur_bersih):
        """Founder bisa memeriksa angkanya sebelum kalimatnya selesai ditulis."""
        nama = [e["event"] for e in await _kumpulkan({"question": "q"})]
        assert nama.index("sql") < nama.index("token")
        assert nama.index("data") < nama.index("token")

    @pytest.mark.asyncio
    async def test_grafik_terkompilasi(self, alur_bersih):
        events = await _kumpulkan({"question": "q"})
        chart = next(e["data"] for e in events if e["event"] == "chart")
        assert chart["mark"] == "line"
        assert "data" not in chart

    @pytest.mark.asyncio
    async def test_done_membawa_llm_call_logs(self, alur_bersih):
        """Kalau ini hilang, angka Pusat Biaya diam-diam mengecil."""
        events = await _kumpulkan({"question": "q"})
        done = next(e["data"] for e in events if e["event"] == "done")
        assert done["content"] == "Ada 5 akun."
        assert isinstance(done["metadata"]["llm_call_logs"], list)


class TestAlurGagal:
    @pytest.mark.asyncio
    async def test_sql_ditolak_dicoba_ulang_lalu_menyerah_dengan_jujur(self):
        panggilan = {"n": 0}

        async def fake_llm(*, system, user, span_name, state, **kw):
            if span_name == "founder-plan":
                return '{"dataset_names": ["user"]}'
            if span_name.startswith("founder-sql"):
                panggilan["n"] += 1
                return "SELECT * FROM public.users"
            return "{}"

        async def fake_execute(state, *, sql, attempt):
            return {
                "ok": False,
                "error_type": "forbidden_table",
                "message": "'public.users' bukan dataset yang tersedia.",
                "sql": sql,
            }

        class FakeChunk:
            def __init__(self, c):
                self.content = c

        class FakeLLM:
            async def astream(self, msgs):
                yield FakeChunk("Pertanyaan itu belum bisa dijawab.")

        with (
            patch.object(g, "_load_catalog", AsyncMock(return_value=KATALOG)),
            patch.object(g, "_prompt_for", AsyncMock(return_value=("x", "v"))),
            patch.object(g, "_llm_text", side_effect=fake_llm),
            patch.object(g, "node_execute", side_effect=fake_execute),
            patch.object(g, "get_llm", return_value=FakeLLM()),
        ):
            events = await _kumpulkan({"question": "data semua orang"})

        assert panggilan["n"] == g.settings.FOUNDER_SQL_MAX_ATTEMPTS
        meta = next(e["data"] for e in events if e["event"] == "meta")
        assert meta["failure"], "Kegagalan harus tercatat, bukan diam-diam kosong."
        assert not any(e["event"] == "data" for e in events)
        assert not any(e["event"] == "chart" for e in events)

    @pytest.mark.asyncio
    async def test_guard_mati_tidak_dicoba_ulang(self):
        """Menulis ulang SQL tidak akan memperbaiki penjaga yang tidak siap."""
        panggilan = {"n": 0}

        async def fake_llm(*, system, user, span_name, state, **kw):
            if span_name == "founder-plan":
                return '{"dataset_names": ["user"]}'
            if span_name.startswith("founder-sql"):
                panggilan["n"] += 1
                return "SELECT 1 FROM nlf.v_user"
            return "{}"

        async def fake_execute(state, *, sql, attempt):
            return {"ok": False, "error_type": "guard_unavailable", "message": "x"}

        class FakeChunk:
            def __init__(self, c):
                self.content = c

        class FakeLLM:
            async def astream(self, msgs):
                yield FakeChunk("Belum bisa.")

        with (
            patch.object(g, "_load_catalog", AsyncMock(return_value=KATALOG)),
            patch.object(g, "_prompt_for", AsyncMock(return_value=("x", "v"))),
            patch.object(g, "_llm_text", side_effect=fake_llm),
            patch.object(g, "node_execute", side_effect=fake_execute),
            patch.object(g, "get_llm", return_value=FakeLLM()),
        ):
            await _kumpulkan({"question": "q"})

        assert panggilan["n"] == 1

    @pytest.mark.asyncio
    async def test_katalog_mati_menghasilkan_pesan_bersih(self):
        with patch.object(g, "_load_catalog", AsyncMock(return_value=None)):
            events = await _kumpulkan({"question": "q"})
        assert events[-1]["event"] == "error"
        teks = events[-1]["data"]
        for jargon in ("nlf", "catalog", "internal", "500", "http"):
            assert jargon not in teks.lower()

    @pytest.mark.asyncio
    async def test_tidak_ada_dataset_cocok(self):
        async def fake_llm(*, system, user, span_name, state, **kw):
            if span_name == "founder-plan":
                return '{"dataset_names": [], "reason": "tidak ada yang cocok"}'
            return "{}"

        class FakeChunk:
            def __init__(self, c):
                self.content = c

        class FakeLLM:
            async def astream(self, msgs):
                yield FakeChunk("Data itu tidak tersedia.")

        with (
            patch.object(g, "_load_catalog", AsyncMock(return_value=KATALOG)),
            patch.object(g, "_llm_text", side_effect=fake_llm),
            patch.object(g, "get_llm", return_value=FakeLLM()),
        ):
            events = await _kumpulkan({"question": "berapa harga saham kita"})

        assert not any(e["event"] == "sql" for e in events)
        meta = next(e["data"] for e in events if e["event"] == "meta")
        assert "dataset" in (meta["failure"] or "")
