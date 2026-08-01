"""Alur Tanya Data Founder: plan -> sql -> exec -> (perbaiki) -> chart -> jawab.

Kenapa bukan `StateGraph`
-------------------------
Graph chat orang tua memakai LangGraph karena ia punya percabangan nyata,
checkpointer, dan state yang dibagi banyak node. Jalur ini tidak punya satu pun
dari itu: pipanya lurus dengan satu simpul perbaikan, tanpa checkpointer, dan
tiap langkah harus **memancarkan event SSE saat itu juga** supaya founder
melihat SQL dan tabelnya muncul sebelum narasinya selesai ditulis.

Membungkusnya jadi StateGraph berarti mengumpulkan event di state lalu
memancarkannya setelah graph selesai — yang menghapus alasan utama SSE ada.
Jadi bentuknya generator async, dan itu keputusan sadar, bukan jalan pintas.

Pola "bagian deterministik dulu, jawaban di-stream belakangan" sendiri bukan
hal baru di repo ini: `generate_node` di jalur orang tua juga berjalan di luar
graph, dikendalikan `streaming.py`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from app.agents.founder_analytics.chart_compile import ChartSkipped, compile_chart
from app.agents.founder_analytics.prompts import (
    ANSWER_NO_DATA,
    ANSWER_SYSTEM,
    CHART_SYSTEM,
    PLAN_SYSTEM,
    SQL_REPAIR_SUFFIX,
    SQL_SYSTEM,
)
from app.agents.founder_analytics.state import (
    ChartIntent,
    FounderAnalyticsState,
    SqlAttempt,
)
from app.agents.tools._http import call_internal_get, call_internal_post
from app.config.llm import get_llm
from app.config.observability import trace_generation, trace_node
from app.config.settings import settings

logger = logging.getLogger(__name__)

_CATALOG_PATH = "/api/v1/internal/agent/founder-query/catalog"
_PROMPT_PATH = "/api/v1/internal/agent/founder-query/prompt"
_EXECUTE_PATH = "/api/v1/internal/agent/founder-query/execute"

#: Baris yang dilihat LLM saat menyusun jawaban. Sisanya tetap tersimpan dan
#: ikut terunduh. Daftar panjang membuat model memungut satu-dua baris yang
#: menarik dan menceritakannya seolah mewakili keseluruhan.
_MAX_ROWS_TO_PROMPT = 50

#: Katalog di-cache di proses. TTL-nya pendek supaya perubahan katalog tidak
#: perlu menunggu redeploy ai-chat.
_catalog_cache: dict[str, Any] = {"at": 0.0, "data": None}

_FENCE = re.compile(r"^\s*```(?:sql|json)?\s*|\s*```\s*$", re.I | re.M)


def _strip_fence(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


def _parse_json(text: str) -> dict | None:
    bersih = _strip_fence(text)
    try:
        return json.loads(bersih)
    except json.JSONDecodeError:
        # Model kadang membungkus JSON dengan satu kalimat pengantar.
        awal, akhir = bersih.find("{"), bersih.rfind("}")
        if awal >= 0 and akhir > awal:
            try:
                return json.loads(bersih[awal : akhir + 1])
            except json.JSONDecodeError:
                return None
        return None


async def _llm_text(
    *,
    system: str,
    user: str,
    span_name: str,
    state: FounderAnalyticsState,
    temperature: float = 0.0,
    max_tokens: int = 900,
    model: str | None = None,
) -> str:
    """Satu panggilan LLM non-streaming, ter-trace dan ter-catat biayanya."""
    llm = get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False,
        model=model or (settings.FOUNDER_SQL_MODEL or None),
    )
    mulai = time.perf_counter()

    async with trace_generation(
        name=span_name,
        model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
        system_prompt=system,
        user_message=user,
        metadata={"session_id": state.session_id, "trace_id": state.trace_id},
    ) as span:
        hasil = await llm.ainvoke(
            [("system", system), ("human", user)]
        )
        teks = (hasil.content or "").strip() if hasattr(hasil, "content") else ""
        if span:
            span.update(output=teks[:4000])

    # Biaya harus tetap terlihat di dashboard Pusat Biaya. Jalur baru yang
    # lupa mencatat ini bikin angka biaya diam-diam mengecil, dan tidak ada
    # yang akan curiga karena grafiknya tetap naik-turun seperti biasa.
    meta = getattr(hasil, "usage_metadata", None) or {}
    state.llm_call_logs.append(
        {
            "prompt_key": span_name,
            "model": getattr(llm, "model", None)
            or getattr(llm, "model_name", None)
            or "unknown",
            "provider": settings.LLM_PROVIDER,
            "node": span_name,
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "total_tokens": meta.get("total_tokens"),
            "latency_ms": int((time.perf_counter() - mulai) * 1000),
            "success": bool(teks),
        }
    )
    return teks


# =============================================================================
# Katalog
# =============================================================================


async def _load_catalog() -> dict | None:
    sekarang = time.time()
    umur = sekarang - float(_catalog_cache["at"] or 0)
    if _catalog_cache["data"] and umur < settings.FOUNDER_CATALOG_TTL_SECONDS:
        return _catalog_cache["data"]

    data = await call_internal_get(_CATALOG_PATH, timeout=15)
    if not data or data.get("error") or "datasets" not in data:
        logger.error("[founder_analytics] katalog gagal dimuat: %s", data)
        return _catalog_cache["data"]

    _catalog_cache.update({"at": sekarang, "data": data})
    return data


async def _prompt_for(dataset_names: list[str]) -> tuple[str, str | None]:
    data = await call_internal_post(
        _PROMPT_PATH, {"dataset_names": dataset_names}, timeout=15
    )
    if not data or data.get("status") == "failed":
        return "", None
    return data.get("prompt_text", ""), data.get("version")


# =============================================================================
# Node
# =============================================================================


async def node_plan(state: FounderAnalyticsState) -> None:
    """Pilih dataset. Ini yang membuat prompt SQL tetap kecil."""
    async with trace_node(
        name="founder-plan", input_data={"question": state.question}
    ) as span:
        teks = await _llm_text(
            system=PLAN_SYSTEM.format(index=state.catalog_index),
            user=state.question,
            span_name="founder-plan",
            state=state,
            max_tokens=300,
        )
        parsed = _parse_json(teks) or {}
        nama = [str(n) for n in (parsed.get("dataset_names") or [])][:3]
        state.dataset_names = nama
        state.time_hint = str(parsed.get("time_hint") or "")
        if span:
            span.update(
                output={
                    "dataset_names": nama,
                    "time_hint": state.time_hint,
                    "reason": parsed.get("reason"),
                }
            )


async def node_sql(state: FounderAnalyticsState, *, attempt: int) -> str | None:
    """Tulis SQL. Percobaan >1 membawa pesan galat percobaan sebelumnya."""
    system = SQL_SYSTEM.format(catalog=state.catalog_prompt)
    user = state.question
    if state.time_hint:
        user += f"\n\n(rentang waktu yang diminta: {state.time_hint})"

    if attempt > 1 and state.attempts:
        terakhir = state.attempts[-1]
        system += SQL_REPAIR_SUFFIX.format(
            sql=terakhir.sql, error=terakhir.message or terakhir.error_type
        )

    nama_span = "founder-sql-generate" if attempt == 1 else "founder-sql-repair"
    teks = await _llm_text(
        system=system, user=user, span_name=nama_span, state=state, max_tokens=800
    )
    sql = _strip_fence(teks).rstrip(";").strip()

    if not sql or sql.upper().startswith("TIDAK_BISA"):
        return None
    return sql


async def node_execute(state: FounderAnalyticsState, *, sql: str, attempt: int) -> dict:
    return await call_internal_post(
        _EXECUTE_PATH,
        {
            "sql": sql,
            "founder_user_id": state.founder_user_id,
            "session_id": state.session_id,
            "question": state.question,
            "attempt": attempt,
            "trace_id": state.trace_id,
            "catalog_version": state.catalog_version,
        },
        extra_headers=(
            {"X-Request-ID": state.trace_id} if state.trace_id else None
        ),
        timeout=settings.FOUNDER_EXECUTE_TIMEOUT_SECONDS,
    )


async def node_chart(state: FounderAnalyticsState) -> None:
    """Putuskan bentuk grafiknya. Boleh memutuskan tidak ada grafik."""
    if not state.has_data:
        state.chart_skipped_reason = "tidak ada hasil"
        return

    contoh = state.rows[:3]
    async with trace_node(
        name="founder-chart-intent",
        input_data={"columns": state.columns, "row_count": state.row_count},
    ) as span:
        teks = await _llm_text(
            system=CHART_SYSTEM.format(
                columns=", ".join(state.columns),
                sample=json.dumps(contoh, ensure_ascii=False, default=str)[:1200],
                question=state.question,
            ),
            user="Tentukan bentuk grafiknya.",
            span_name="founder-chart-intent",
            state=state,
            max_tokens=350,
        )
        parsed = _parse_json(teks) or {"kind": "none", "reason": "keluaran tak terbaca"}
        try:
            intent = ChartIntent.model_validate(parsed)
        except Exception as e:
            logger.info("[founder_analytics] niat grafik tidak valid: %s", e)
            intent = ChartIntent(kind="none", reason="niat grafik tidak valid")

        state.chart_intent = intent
        try:
            state.chart_spec = compile_chart(
                intent, columns=state.columns, rows=state.rows
            )
        except ChartSkipped as e:
            state.chart_spec = None
            state.chart_skipped_reason = str(e)

        if span:
            span.update(
                output={
                    "kind": intent.kind,
                    "reason": intent.reason,
                    "dikompilasi": state.chart_spec is not None,
                    "dilewati": state.chart_skipped_reason,
                }
            )


def _table_for_prompt(state: FounderAnalyticsState) -> str:
    """Hasil sebagai tabel pipa ringkas — cukup untuk menyusun narasi."""
    if not state.columns:
        return "(tidak ada hasil)"
    baris = state.rows[:_MAX_ROWS_TO_PROMPT]
    keluar = [" | ".join(state.columns), "-" * 40]
    for r in baris:
        keluar.append(" | ".join("" if v is None else str(v) for v in r))
    if state.row_count > len(baris):
        keluar.append(
            f"... {state.row_count - len(baris)} baris lagi tidak ditampilkan "
            "di sini, tapi ikut di tabel dan berkas unduhan"
        )
    return "\n".join(keluar)


# =============================================================================
# Orkestrasi
# =============================================================================


def _sse(event: str, data: Any) -> str:
    return f"data: {json.dumps({'event': event, 'data': data}, default=str)}\n\n"


async def run_founder_analytics(payload: dict) -> AsyncIterator[str]:
    """Jalankan satu giliran, memancarkan event SSE sambil berjalan."""
    state = FounderAnalyticsState(
        question=(payload.get("question") or "").strip(),
        session_id=payload.get("session_id"),
        founder_user_id=payload.get("founder_user_id"),
        trace_id=payload.get("trace_id"),
        history=payload.get("history") or [],
    )

    yield _sse("thinking", {"step": 1, "label": "Memilih data yang perlu dibaca"})

    katalog = await _load_catalog()
    if not katalog:
        yield _sse("error", "Jalur data sedang tidak tersedia. Coba lagi nanti.")
        return

    state.catalog_index = katalog.get("index_text", "")
    state.catalog_version = katalog.get("version")

    await node_plan(state)

    if not state.dataset_names:
        state.failure = "tidak ada dataset yang cocok dengan pertanyaan ini"
    else:
        prompt_text, versi = await _prompt_for(state.dataset_names)
        state.catalog_prompt = prompt_text or katalog.get("prompt_text", "")
        if versi:
            state.catalog_version = versi

        yield _sse(
            "thinking",
            {"step": 2, "label": "Menyusun query", "datasets": state.dataset_names},
        )

        maks = max(1, settings.FOUNDER_SQL_MAX_ATTEMPTS)
        hasil: dict | None = None

        for attempt in range(1, maks + 1):
            sql = await node_sql(state, attempt=attempt)
            if not sql:
                state.failure = (
                    "model menyatakan pertanyaan ini tidak terjawab oleh katalog"
                )
                break

            if attempt > 1:
                yield _sse(
                    "thinking",
                    {"step": 2, "label": f"Memperbaiki query (percobaan {attempt})"},
                )

            hasil = await node_execute(state, sql=sql, attempt=attempt)
            ok = bool(hasil.get("ok"))
            state.attempts.append(
                SqlAttempt(
                    attempt=attempt,
                    sql=sql,
                    ok=ok,
                    error_type=hasil.get("error_type"),
                    message=hasil.get("message"),
                    elapsed_ms=int(hasil.get("elapsed_ms") or 0),
                )
            )

            if ok:
                state.sql = hasil.get("sql") or sql
                break

            # Galat yang tidak bisa diperbaiki dengan menulis ulang SQL —
            # mencoba lagi cuma membakar token dan menunda pesan yang jujur.
            if hasil.get("error_type") in ("guard_unavailable", "timeout"):
                break
        else:
            hasil = hasil or {}

        if hasil and hasil.get("ok"):
            state.columns = hasil.get("columns") or []
            state.rows = hasil.get("rows") or []
            state.row_count = int(hasil.get("row_count") or 0)
            state.truncated = bool(hasil.get("truncated"))
            state.datasets = hasil.get("datasets") or []
            state.pii_datasets = hasil.get("pii_datasets") or []
            state.elapsed_ms = int(hasil.get("elapsed_ms") or 0)
            state.executed_at = datetime.now(timezone.utc).isoformat()
        elif not state.failure:
            terakhir = state.attempts[-1] if state.attempts else None
            state.failure = (
                f"{terakhir.error_type}: {terakhir.message}"
                if terakhir
                else "query tidak bisa dijalankan"
            )

    # ── Kirim SQL dan tabelnya lebih dulu ───────────────────────────────────
    #
    # Sengaja sebelum narasi. Founder melihat query-nya dan angkanya sementara
    # kalimatnya masih ditulis — dan kalau narasinya nanti terasa janggal, ia
    # sudah punya bahan untuk memeriksanya sendiri.
    if state.sql:
        yield _sse(
            "sql",
            {
                "sql": state.sql,
                "datasets": state.datasets,
                "repaired": state.repaired,
                "attempts": len(state.attempts),
                "elapsed_ms": state.elapsed_ms,
            },
        )

    if state.has_data:
        yield _sse(
            "data",
            {
                "columns": state.columns,
                "rows": state.rows,
                "row_count": state.row_count,
                "truncated": state.truncated,
                "datasets": state.datasets,
                "pii_datasets": state.pii_datasets,
                "elapsed_ms": state.elapsed_ms,
                "executed_at": state.executed_at,
            },
        )

        yield _sse("thinking", {"step": 3, "label": "Menyiapkan grafik"})
        await node_chart(state)
        if state.chart_spec:
            yield _sse("chart", state.chart_spec)

    # ── Narasi ──────────────────────────────────────────────────────────────
    yield _sse("thinking", {"step": 4, "label": "Menyusun jawaban"})

    if state.has_data:
        system = ANSWER_SYSTEM
        user = (
            f"Pertanyaan: {state.question}\n\n"
            f"Hasil query ({state.row_count} baris"
            f"{', dipotong' if state.truncated else ''}):\n"
            f"{_table_for_prompt(state)}"
        )
    else:
        system = ANSWER_SYSTEM
        user = ANSWER_NO_DATA.format(reason=state.failure or "tidak diketahui")

    jawaban = ""
    llm = get_llm(temperature=0.3, max_tokens=700, streaming=True)
    async with trace_generation(
        name="founder-answer",
        model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
        system_prompt=system,
        user_message=user,
        metadata={
            "row_count": state.row_count,
            "has_chart": state.chart_spec is not None,
            "repaired": state.repaired,
        },
    ) as span:
        try:
            async for potongan in llm.astream(
                [("system", system), ("human", user)]
            ):
                bagian = getattr(potongan, "content", "") or ""
                if bagian:
                    jawaban += bagian
                    yield _sse("token", bagian)
        except Exception as e:
            logger.exception("[founder_analytics] gagal menyusun jawaban: %s", e)
            if not jawaban:
                jawaban = (
                    "Datanya berhasil diambil, tapi ringkasannya gagal disusun. "
                    "Tabelnya di bawah tetap bisa dipakai."
                )
                yield _sse("token", jawaban)
        if span:
            span.update(output=jawaban[:4000])

    yield _sse(
        "meta",
        {
            "catalog_version": state.catalog_version,
            "datasets": state.datasets,
            "pii_datasets": state.pii_datasets,
            "dataset_names": state.dataset_names,
            "repaired": state.repaired,
            "attempts": len(state.attempts),
            "row_count": state.row_count,
            "truncated": state.truncated,
            "chart_kind": state.chart_intent.kind if state.chart_intent else None,
            "chart_skipped_reason": state.chart_skipped_reason,
            "elapsed_ms": state.elapsed_ms,
            "failure": state.failure,
        },
    )

    yield _sse(
        "done",
        {
            "content": jawaban.strip(),
            "metadata": {
                "catalog_version": state.catalog_version,
                "repaired": state.repaired,
                "attempts": len(state.attempts),
                "chart_skipped_reason": state.chart_skipped_reason,
                "failure": state.failure,
                "llm_call_logs": state.llm_call_logs,
            },
        },
    )
