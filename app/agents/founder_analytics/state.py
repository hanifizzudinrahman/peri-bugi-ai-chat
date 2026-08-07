"""State graph Tanya Data Founder.

Sengaja jauh lebih kecil daripada `AgentState` di jalur orang tua. Yang tidak
ada di sini — pemilihan tool, klarifikasi gambar, memori jangka panjang — bukan
kelalaian; jalur ini memang tidak punya semuanya, dan state yang memuat field
yang tidak pernah diisi cuma bikin orang menebak-nebak.

**Mode jawaban sekarang ADA** (7 Agustus 2026). Docstring ini sebelumnya
menyebutnya sebagai salah satu yang sengaja tidak ada, dan itu benar sampai
founder memintanya dengan alasan yang tidak terbantahkan: jawaban untuk
"berapa anak aktif" dan jawaban untuk "kenapa kepatuhan turun" tidak seharusnya
sepanjang yang sama. Kosakatanya disamakan persis dengan jalur orang tua —
`simple | medium | detailed` — karena dua vokabuler untuk satu konsep akan
tertukar suatu hari.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChartField(BaseModel):
    """Satu saluran encoding. `field` WAJIB kolom yang ada di hasil query."""

    field: str
    type: Literal["temporal", "nominal", "ordinal", "quantitative"]
    title: str = ""


class ChartIntent(BaseModel):
    """Niat grafik — bukan spesifikasi Vega-Lite.

    Model tidak pernah menulis Vega-Lite langsung. Alasannya bukan gaya:
    spesifikasi bebas yang lolos validasi skema masih bisa jadi grafik yang
    tidak terbaca, dan `data.url` di dalamnya adalah permintaan jaringan
    sungguhan yang dijalankan di browser founder yang sedang login. Bentuk
    berbatas ini tidak punya tempat untuk menaruh hal semacam itu.
    """

    kind: Literal[
        "none", "bar", "bar_grouped", "bar_stacked", "line", "area", "point"
    ] = "none"
    x: ChartField | None = None
    y: ChartField | None = None
    y_aggregate: Literal["none", "sum", "mean", "count", "median"] = "none"
    color: ChartField | None = None
    sort: Literal["x_asc", "x_desc", "y_asc", "y_desc"] | None = None
    title: str = ""
    reason: str = Field(
        default="",
        description="Satu kalimat kenapa bentuk ini dipilih — masuk ke trace",
    )

    dashboard: Literal["none", "simple", "medium", "detailed"] = Field(
        default="none",
        description=(
            "Perlukah dasbor penuh dibuat untuk pertanyaan ini, dan sekaya apa. "
            "`none` = cukup grafik biasa."
        ),
    )
    """Pemicu dasbor menumpang di sini, bukan di `RencanaKueri`, dan alasannya
    ada empat:

    1. Nol panggilan LLM tambahan — node chart sudah memanggil model terstruktur
       yang melihat pertanyaan, nama kolom, dan baris contoh.
    2. Ia jalan SETELAH eksekusi, jadi keputusannya diambil dengan `row_count`
       yang sudah diketahui, bukan ditebak dari pertanyaan saja.
    3. `ChartIntent` memang bersifat usulan — `compile_chart` sudah boleh
       membatalkannya — jadi medan baru di sini tidak bisa merusak jalur yang
       sudah jalan.
    4. Prompt `plan` menanggung eval SQL. Mengutak-atiknya untuk urusan
       tampilan berarti mempertaruhkan angka demi gambar."""


class RencanaKueri(BaseModel):
    """Keluaran node `plan` — dataset mana yang dibaca, dan rentang waktunya.

    Bentuknya dipakai dua kali: sebagai `response_schema` supaya model tidak
    bisa mengarang bentuk lain, dan sebagai pemeriksa hasil parse untuk jalur
    mundur. Sebelum ada ini, keluaran yang tidak terbaca tidak bisa dibedakan
    dari "tidak ada dataset yang cocok" — satu model pernah mencetak 0/19
    karena itu, dan sebabnya baru ketahuan setelah dilacak manual.
    """

    dataset_names: list[str] = Field(default_factory=list)
    time_hint: str = ""
    reason: str = ""


class KeluaranSQL(BaseModel):
    """Keluaran node `sql`.

    `bisa_dijawab` menggantikan sentinel `TIDAK_BISA` yang dulu harus ditulis
    model persis sebagai teks. Sentinel kata ajaib menaruh beban di tempat yang
    salah: model yang menulis "TIDAK BISA" (dengan spasi) atau menjelaskan
    dulu kenapa tidak bisa, keluarannya diperlakukan sebagai SQL.
    """

    bisa_dijawab: bool = True
    sql: str = ""
    alasan: str = ""


class PertanyaanMandiri(BaseModel):
    """Keluaran node `rewrite` — pertanyaan lanjutan yang sudah berdiri sendiri.

    Pola yang dipakai: **tulis-ulang lalu parse**. Pertanyaan lanjutan
    ("rincian per bulannya dong") diubah dulu jadi pertanyaan utuh, baru masuk
    ke node SQL yang memang dirancang untuk pertanyaan tunggal.

    Kenapa bukan menempelkan riwayat mentah ke prompt SQL: prompt itu sudah
    memuat katalog penuh dua dataset, dan menambah percakapan di atasnya membuat
    model MENAFSIRKAN ULANG pertanyaannya. Kegagalan itu persis yang sudah
    tercatat di `docs/FOUNDER_ANALYTICS.md` — model besar menjawab pertanyaan
    yang MIRIP, bukan yang ditanyakan.
    """

    mandiri: bool = Field(
        default=True,
        description=(
            "True kalau pertanyaannya sudah bisa dijawab tanpa melihat "
            "percakapan sebelumnya."
        ),
    )
    pertanyaan: str = Field(
        default="",
        description=(
            "Versi mandiri dari pertanyaan terakhir. Kosongkan kalau `mandiri` "
            "sudah True."
        ),
    )
    alasan: str = ""


class KpiDef(BaseModel):
    """Definisi satu kartu KPI — TANPA angka, dan itu inti rancangannya.

    Model cuma melihat lima baris contoh. Kalau bentuk ini menyediakan tempat
    untuk menulis nilai, model akan mengisinya dari lima baris itu lalu
    menyajikannya sebagai angka seluruh data — dan angka salah yang terlihat
    rapi lebih berbahaya daripada kolom kosong.

    Nilainya dihitung `peri-bugi-api` dari baris sungguhan, dengan mesin yang
    sama yang menulis sheet KPI di berkas Excel. Satu perhitungan, dua tempat
    keluar.
    """

    label: str
    source_column: str
    agg: Literal["sum", "avg", "count", "min", "max", "first", "last"] = "sum"
    baseline: Literal[
        "previous_period", "first_value", "average", "none"
    ] = "none"
    format: Literal[
        "number", "compact", "percent", "currency_idr", "duration_days"
    ] = "number"
    good_direction: Literal["up", "down", "neutral"] = Field(
        default="neutral",
        description=(
            "Arah yang berarti KABAR BAIK. Bukan arah panahnya — 'scan gagal "
            "naik 12%' adalah panah naik dan kabar buruk."
        ),
    )


class DashboardSpec(BaseModel):
    """Keluaran node `dashboard`: satu dasbor lengkap sebagai kode.

    `html` + `css` + `js` ditulis model sepenuhnya — ini memang dasbor buatan
    LLM, bukan template yang diisi. Yang TIDAK boleh ada di dalamnya: angka
    hasil hitungan model, dan URL apa pun.
    """

    # Nol medan `version` di sini. Versi kontrak `ctx` adalah milik shell, dan
    # server yang menstempelnya saat menyimpan — model tidak punya cara tahu
    # kontrak mana yang sedang berlaku. Ada alasan teknisnya juga: `Literal[1]`
    # ditolak Gemini dengan "Literal values must be strings", dan mengubahnya
    # jadi `Literal["1"]` cuma menyembunyikan bahwa medannya memang salah tempat.
    title: str = ""
    subtitle: str = ""
    kpis: list[KpiDef] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    html: str = ""
    css: str = ""
    js: str = ""
    notes: str = Field(
        default="", description="Catatan untuk trace. Tidak pernah dirender."
    )


class SqlAttempt(BaseModel):
    """Satu percobaan menulis SQL, berhasil maupun tidak."""

    attempt: int
    sql: str
    ok: bool
    error_type: str | None = None
    message: str | None = None
    elapsed_ms: int = 0


class FounderAnalyticsState(BaseModel):
    # ── Masukan ─────────────────────────────────────────────────────────────
    question: str
    """Pertanyaan yang DIPAKAI hilir — sudah ditulis ulang kalau perlu."""

    question_asli: str = ""
    """Kalimat yang founder benar-benar ketik.

    Disimpan terpisah karena node jawaban harus menjawab pertanyaan yang DITULIS
    founder, bukan versi tulis-ulangnya. Jawaban yang mengulang kalimat hasil
    tulis-ulang terbaca seperti mesin yang mengoreksi cara bertanya orang."""

    session_id: str | None = None
    founder_user_id: str | None = None
    trace_id: str | None = None
    history: list[dict] = Field(default_factory=list)

    response_mode: Literal["simple", "medium", "detailed"] = "medium"
    """Ditentukan `peri-bugi-api` dari sesi, dikirim per giliran. Bukan diminta
    dari sini — kebijakan tinggal di repo yang memegang database."""

    allow_dashboard: bool = False
    """Izin membuat dasbor, sebagai FAKTA yang dikirim, bukan pertanyaan yang
    ditanyakan. Feature flag, anggaran harian, dan batas per sesi semuanya
    diperiksa di `peri-bugi-api` — ia yang memegang Redis, dan service ini jalan
    multi-instance sehingga penghitung di sini cuma menghitung jatah satu
    instance."""

    rewritten: bool = False
    """Pertanyaannya ditulis ulang di giliran ini. Masuk ke `meta` supaya
    terlihat di jejak — pertanyaan yang berubah tanpa jejak adalah cara paling
    halus untuk membuat jawaban terasa salah."""

    # ── Katalog ─────────────────────────────────────────────────────────────
    catalog_version: str | None = None
    catalog_index: str = ""
    catalog_prompt: str = ""
    dataset_names: list[str] = Field(default_factory=list)
    time_hint: str = ""

    # ── SQL ─────────────────────────────────────────────────────────────────
    sql: str | None = None
    attempts: list[SqlAttempt] = Field(default_factory=list)

    sql_gagal: str | None = None
    """Kenapa node SQL tidak menghasilkan query. Dulu semuanya dilaporkan
    sebagai "model menyatakan pertanyaan ini tidak terjawab oleh katalog" —
    termasuk keluaran terpotong dan keluaran yang bukan SQL. Sebab teknis yang
    menyamar jadi keputusan model membuat orang mencari-cari di katalog padahal
    yang kurang anggaran token."""

    # ── Hasil ───────────────────────────────────────────────────────────────
    columns: list[str] = Field(default_factory=list)
    rows: list[list] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    datasets: list[str] = Field(default_factory=list)
    pii_datasets: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    executed_at: str | None = None

    # ── Grafik ──────────────────────────────────────────────────────────────
    chart_intent: ChartIntent | None = None
    chart_spec: dict[str, Any] | None = None
    chart_skipped_reason: str | None = None

    # ── Narasi ──────────────────────────────────────────────────────────────
    answer: str = ""
    """Jawaban naratif yang sudah selesai di-stream ke layar.

    Diteruskan ke penulis dasbor supaya judul dan insight-nya tidak berselisih
    dengan kalimat yang founder BARU SAJA baca di atasnya. Dua teks tentang
    angka yang sama yang saling bertentangan lebih merusak daripada satu teks
    yang kurang lengkap."""

    # ── Dasbor ──────────────────────────────────────────────────────────────
    dashboard_spec: dict[str, Any] | None = None
    dashboard_mode: str | None = None
    dashboard_skipped_reason: str | None = None
    """Kenapa dasbor tidak dibuat. Bukan galat — sebagian besar pertanyaan
    founder dijawab satu angka dan memang tidak layak digambar. Disimpan supaya
    "kok tidak ada dasbornya" punya jawaban, bukan supaya ditampilkan sebagai
    kegagalan."""

    # ── Kegagalan yang harus disampaikan ke founder ─────────────────────────
    failure: str | None = None
    """Terisi berarti tidak ada hasil. Node jawaban menyampaikannya apa adanya,
    bukan mengarang angka supaya jawabannya terlihat lengkap."""

    llm_call_logs: list[dict] = Field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return len(self.attempts) > 1

    @property
    def has_data(self) -> bool:
        return bool(self.columns) and self.failure is None
