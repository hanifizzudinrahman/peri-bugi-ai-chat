"""State graph Tanya Data Founder.

Sengaja jauh lebih kecil daripada `AgentState` di jalur orang tua. Yang tidak
ada di sini — pemilihan tool, klarifikasi gambar, memori jangka panjang, mode
jawaban — bukan kelalaian; jalur ini memang tidak punya semuanya, dan state
yang memuat field yang tidak pernah diisi cuma bikin orang menebak-nebak.
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
    session_id: str | None = None
    founder_user_id: str | None = None
    trace_id: str | None = None
    history: list[dict] = Field(default_factory=list)

    # ── Katalog ─────────────────────────────────────────────────────────────
    catalog_version: str | None = None
    catalog_index: str = ""
    catalog_prompt: str = ""
    dataset_names: list[str] = Field(default_factory=list)
    time_hint: str = ""

    # ── SQL ─────────────────────────────────────────────────────────────────
    sql: str | None = None
    attempts: list[SqlAttempt] = Field(default_factory=list)

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
