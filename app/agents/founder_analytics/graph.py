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
from zoneinfo import ZoneInfo

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
    KeluaranSQL,
    RencanaKueri,
    SqlAttempt,
)
from app.agents.tools._http import call_internal_get, call_internal_post
from app.config import gemini_direct
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

#: Satu zona waktu untuk seluruh sistem — sama dengan view di skema `nlf`.
_WIB = ZoneInfo("Asia/Jakarta")


def _strip_fence(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


#: Awalan yang sah untuk sebuah query. Dipakai menyelamatkan keluaran model yang
#: membungkus SQL-nya dengan kalimat pengantar.
_AWAL_SQL = re.compile(r"\b(SELECT|WITH)\b", re.I)


def _model_pydantic(text: str, kelas):
    """Parse teks jadi model Pydantic, atau `None` kalau tidak bisa."""
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        return None
    try:
        return kelas.model_validate(parsed)
    except Exception:
        return None


def _parse_model(text: str, kelas):
    """Sama seperti `_model_pydantic`, tapi selalu memberi objek.

    Dengan `response_schema` aktif, cabang gagalnya nyaris tidak pernah kena.
    Ia tetap ada karena tangga jalur-mundur bisa sampai ke LangChain, dan di
    sana bentuk keluaran tidak dijamin siapa pun.
    """
    hasil = _model_pydantic(text, kelas)
    if hasil is not None:
        return hasil
    logger.info(
        "[founder_analytics] keluaran tidak terbaca sebagai %s: %r",
        kelas.__name__,
        (text or "")[:200],
    )
    return kelas()


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
    response_schema: Any = None,
) -> str:
    """Satu panggilan LLM non-streaming, ter-trace dan ter-catat biayanya.

    `response_schema` (model Pydantic) memaksa bentuk keluaran di sisi Google.
    Kalau modelnya tidak mendukung, panggilannya diulang tanpa skema — bukan
    langsung jatuh ke LangChain, karena jatuh ke LangChain berarti kehilangan
    kendali penalaran, dan itu obat yang lebih buruk daripada penyakitnya.
    """
    nama_model = model or (settings.FOUNDER_SQL_MODEL or None) or settings.GEMINI_MODEL
    mulai = time.perf_counter()

    # Jalur utama: SDK Gemini modern, dengan penalaran ditekan ke MINIMAL.
    #
    # Lewat LangChain, kendali penalaran TIDAK PERNAH sampai — adaptornya tidak
    # punya field `model_kwargs` dan SDK lamanya tidak punya `thinking_config`.
    # Akibatnya terukur: model Flash biasa memakai 3.898 token/panggilan
    # (versus 790) dan isi penalarannya ikut tercetak ke dalam SQL, sehingga
    # query-nya gagal di-parse. Selengkapnya di `app/config/gemini_direct.py`.
    pakai_langsung = gemini_direct.tersedia()
    teks = ""
    pemakaian: dict = {}

    async with trace_generation(
        name=span_name,
        model=nama_model,
        system_prompt=system,
        user_message=user,
        metadata={
            "session_id": state.session_id,
            "trace_id": state.trace_id,
            "klien": "gemini_direct" if pakai_langsung else "langchain",
        },
    ) as span:
        if pakai_langsung:
            # Tangga, dari yang paling terkendali ke yang paling longgar.
            # Skema dilepas duluan, kendali penalaran dilepas paling akhir —
            # urutannya begitu karena kehilangan kendali penalaran adalah
            # kegagalan yang tidak kelihatan sampai tagihannya datang.
            tangga = [response_schema, None] if response_schema is not None else [None]
            for skema in tangga:
                try:
                    hasil = await gemini_direct.generate(
                        system=system,
                        user=user,
                        model=nama_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_schema=skema,
                    )
                    teks = hasil.teks
                    pemakaian = {
                        "input_tokens": hasil.input_tokens,
                        "output_tokens": hasil.output_tokens,
                        "total_tokens": hasil.total_tokens,
                    }
                    break
                except gemini_direct.TeksTerpotong as e:
                    # Ini bukan kegagalan transport dan mengulanginya tanpa
                    # menaikkan anggaran cuma memberi hasil yang sama. Naikkan
                    # `max_tokens` di pemanggilnya; pesannya sudah menyebut
                    # berapa token yang habis dipakai berpikir.
                    logger.warning("[founder_analytics] %s: %s", span_name, e)
                    teks = ""
                    break
                except Exception as e:
                    if skema is not None:
                        logger.info(
                            "[founder_analytics] %s: decoding berbatas ditolak "
                            "(%s), ulangi tanpa skema",
                            span_name,
                            str(e)[:160],
                        )
                        continue
                    # Jangan menjatuhkan giliran gara-gara jalur cepat. Fitur
                    # tetap jalan lewat LangChain, cuma tanpa kendali penalaran.
                    logger.warning(
                        "[founder_analytics] gemini_direct gagal (%s), "
                        "mundur ke LangChain: %s",
                        span_name,
                        str(e)[:200],
                    )
                    pakai_langsung = False

        if not pakai_langsung:
            llm = get_llm(
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=False,
                model=model or (settings.FOUNDER_SQL_MODEL or None),
            )
            hasil_lc = await llm.ainvoke([("system", system), ("human", user)])
            teks = (
                (hasil_lc.content or "").strip()
                if hasattr(hasil_lc, "content")
                else ""
            )
            pemakaian = getattr(hasil_lc, "usage_metadata", None) or {}

        teks = (teks or "").strip()
        if span:
            span.update(output=teks[:4000])

    # Biaya harus tetap terlihat di dashboard Pusat Biaya. Jalur baru yang
    # lupa mencatat ini bikin angka biaya diam-diam mengecil, dan tidak ada
    # yang akan curiga karena grafiknya tetap naik-turun seperti biasa.
    state.llm_call_logs.append(
        {
            "prompt_key": span_name,
            "model": nama_model,
            "provider": settings.LLM_PROVIDER,
            "node": span_name,
            "input_tokens": pemakaian.get("input_tokens"),
            "output_tokens": pemakaian.get("output_tokens"),
            "total_tokens": pemakaian.get("total_tokens"),
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
            # Naik dari 300. Token penalaran memakan `max_output_tokens` yang
            # sama, jadi anggaran yang pas untuk model kecil bisa habis sebelum
            # satu karakter JSON keluar — dan itu pulang sebagai "tidak ada
            # dataset yang cocok". `max_output_tokens` batas atas, bukan target:
            # keluaran yang memang pendek tidak jadi lebih mahal karenanya.
            max_tokens=700,
            response_schema=RencanaKueri,
        )
        rencana = _parse_model(teks, RencanaKueri)
        state.dataset_names = [str(n) for n in rencana.dataset_names][:3]
        state.time_hint = rencana.time_hint
        if span:
            span.update(
                output={
                    "dataset_names": state.dataset_names,
                    "time_hint": state.time_hint,
                    "reason": rencana.reason,
                }
            )


def _hari_ini_wib() -> str:
    """Tanggal hari ini menurut WIB, untuk prompt.

    Tanpa ini, "sejak Maret" dijawab dengan Maret tahun mana pun yang ada di
    kepala model — terbukti menghasilkan tulang punggung tanggal yang mundur
    dua tahun dan grafik berisi 27 baris nol. Model tidak punya jam.
    """
    return datetime.now(_WIB).strftime("%d %B %Y")


def _selamatkan_sql(teks: str) -> str:
    """Ambil SQL dari keluaran yang mungkin dibungkus prosa.

    `_strip_fence` menghapus PENANDA pagar, bukan mengambil isi pagarnya — jadi
    "Berikut kuerinya:" di depan ikut lolos ke validator, lalu galatnya masuk
    loop perbaikan sebagai galat *database*. Loop itu menyuruh model membetulkan
    semantik SQL, padahal masalahnya dia kebanyakan bicara: satu dari tiga
    percobaan terbakar untuk memperbaiki hal yang salah.
    """
    bersih = _strip_fence(teks).strip()
    if not bersih:
        return ""
    if bersih.upper().startswith(("SELECT", "WITH")):
        return bersih.rstrip(";").strip()

    cocok = _AWAL_SQL.search(bersih)
    if not cocok:
        return ""
    logger.info(
        "[founder_analytics] SQL dibungkus prosa, diselamatkan dari posisi %s",
        cocok.start(),
    )
    return bersih[cocok.start() :].rstrip(";").strip()


async def node_sql(state: FounderAnalyticsState, *, attempt: int) -> str | None:
    """Tulis SQL. Percobaan >1 membawa pesan galat percobaan sebelumnya."""
    system = SQL_SYSTEM.format(
        catalog=state.catalog_prompt, today=_hari_ini_wib()
    )
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
        system=system,
        user=user,
        span_name=nama_span,
        state=state,
        max_tokens=1200,
        response_schema=KeluaranSQL,
    )

    keluaran = _model_pydantic(teks, KeluaranSQL)
    if keluaran is None:
        # Jalur mundur: tidak ada JSON, perlakukan seluruh teks sebagai SQL.
        # Ini yang terjadi kalau tangga sampai ke LangChain.
        sql = _selamatkan_sql(teks)
        if sql.upper().startswith("TIDAK_BISA"):
            state.sql_gagal = "model menilai pertanyaan ini di luar katalog"
            return None
        if not sql:
            state.sql_gagal = (
                "model tidak mengeluarkan SQL — keluarannya kosong atau bukan query"
            )
            return None
        return sql

    if not keluaran.bisa_dijawab:
        state.sql_gagal = keluaran.alasan or (
            "model menilai pertanyaan ini tidak terjawab oleh katalog"
        )
        return None

    sql = _selamatkan_sql(keluaran.sql)
    if not sql:
        # Dibedakan dari penolakan di atas dengan sengaja: yang ini cacat bentuk
        # keluaran, dan mengumpankannya ke loop perbaikan SQL cuma membakar
        # percobaan untuk memperbaiki masalah yang bukan masalah SQL.
        state.sql_gagal = "model menjawab tanpa query yang bisa dijalankan"
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
            max_tokens=700,
            response_schema=ChartIntent,
        )
        intent = _model_pydantic(teks, ChartIntent)
        if intent is None:
            logger.info(
                "[founder_analytics] niat grafik tidak terbaca: %r", (teks or "")[:200]
            )
            intent = ChartIntent(kind="none", reason="niat grafik tidak terbaca")

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
    """Jalankan satu giliran, memancarkan event SSE sambil berjalan.

    Seluruh giliran dibungkus satu span induk. Tanpa itu, tiap panggilan LLM
    membuat trace-nya sendiri di Langfuse dan satu pertanyaan tersebar jadi
    lima trace terpisah — span-nya ada, tapi tidak bisa dibaca sebagai satu
    cerita. Diperiksa dengan menanyakan Langfuse, bukan dengan membaca kode.
    """
    async with trace_node(
        name="founder-analytics-turn",
        input_data={
            "question": (payload.get("question") or "")[:500],
            "session_id": payload.get("session_id"),
        },
        metadata={"trace_id": payload.get("trace_id")},
    ):
        async for peristiwa in _jalankan(payload):
            yield peristiwa


async def _jalankan(payload: dict) -> AsyncIterator[str]:
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

    # Node `plan` tidak menghasilkan apa-apa? JANGAN menyerah di sini.
    #
    # "Gagal memilih dataset" dan "tidak ada dataset yang cocok" adalah dua
    # hal berbeda, dan menyamakannya membuat seluruh giliran mati hanya karena
    # satu keluaran JSON yang tidak terbaca. Terbukti mahal: mengganti model
    # menghasilkan 0/19 pada eval — bukan karena model penggantinya lebih
    # buruk, tapi karena bentuk keluarannya sedikit berbeda dan pipa ini tidak
    # punya jalan mundur sama sekali.
    #
    # Yang benar: mundur ke katalog penuh dan biarkan node SQL yang memutuskan.
    # Ia punya cara menyatakan menyerah (TIDAK_BISA) DAN ia melihat definisi
    # kolomnya — jauh lebih siap menilai daripada node pemilih. Harganya cuma
    # token; harga menyerah lebih awal adalah jawaban yang hilang.
    if not state.dataset_names:
        logger.info(
            "[founder_analytics] node plan tidak memilih dataset — "
            "mundur ke katalog penuh"
        )

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
            # Sebabnya datang dari node SQL, tidak lagi diasumsikan di sini.
            # Kalimat lama menyatakan model MENOLAK menjawab, padahal sebabnya
            # bisa keluaran terpotong atau bukan-SQL — dan orang jadi mencari
            # kekurangan di katalog padahal yang kurang anggaran token.
            state.failure = state.sql_gagal or (
                "model tidak menghasilkan query untuk pertanyaan ini"
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
        # SQL-nya ikut, dan itu bukan hiasan: tanpa melihat filternya, model
        # mengarang keterangan yang berlawanan dengan query-nya sendiri —
        # terbukti menulis "termasuk akun uji internal" untuk query yang
        # justru memuat `NOT is_internal`.
        user = (
            f"Pertanyaan: {state.question}\n\n"
            f"Query yang dijalankan:\n{state.sql}\n\n"
            f"Hasil query ({state.row_count} baris"
            f"{', dipotong' if state.truncated else ''}):\n"
            f"{_table_for_prompt(state)}"
        )
    else:
        system = ANSWER_SYSTEM
        user = ANSWER_NO_DATA.format(reason=state.failure or "tidak diketahui")

    jawaban = ""
    mulai_jawab = time.perf_counter()
    pemakaian: dict = {}
    # Sama persis dengan node lain (`_llm_text`). Sebelumnya dipaku ke
    # GEMINI_MODEL, jadi menyetel FOUNDER_SQL_MODEL mengganti tiga node dan
    # meninggalkan node ini di model lain — perbandingan model yang memakai
    # override itu diam-diam mengukur dua model sekaligus.
    model_jawab = (settings.FOUNDER_SQL_MODEL or None) or settings.GEMINI_MODEL
    pakai_langsung = gemini_direct.tersedia()

    async with trace_generation(
        name="founder-answer",
        model=model_jawab,
        system_prompt=system,
        user_message=user,
        metadata={
            "row_count": state.row_count,
            "has_chart": state.chart_spec is not None,
            "repaired": state.repaired,
            "klien": "gemini_direct" if pakai_langsung else "langchain",
        },
    ) as span:
        try:
            if pakai_langsung:
                # SDK modern juga melaporkan pemakaian token di potongan
                # terakhir. Jalur LangChain tidak pernah memberikannya sama
                # sekali, sehingga node ini tercatat nol token di dashboard
                # biaya — padahal ia panggilan terbesar di giliran ini.
                async for bagian, meta_potongan in gemini_direct.stream(
                    system=system, user=user, model=model_jawab
                ):
                    if bagian:
                        jawaban += bagian
                        yield _sse("token", bagian)
                    if meta_potongan:
                        pemakaian = meta_potongan
            else:
                llm = get_llm(temperature=0.3, max_tokens=700, streaming=True)
                async for potongan in llm.astream(
                    [("system", system), ("human", user)]
                ):
                    bagian = getattr(potongan, "content", "") or ""
                    if bagian:
                        jawaban += bagian
                        yield _sse("token", bagian)
                    meta_potongan = getattr(potongan, "usage_metadata", None)
                    if meta_potongan:
                        pemakaian = meta_potongan
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

    # Node jawaban memakai `astream`, bukan `_llm_text`, jadi ia tidak ikut
    # tercatat otomatis. Sempat terlewat — dan yang terlewat justru panggilan
    # paling besar di giliran ini. Ketahuan dari isi tabel llm_call_logs,
    # bukan dari membaca kode.
    state.llm_call_logs.append(
        {
            "prompt_key": "founder-answer",
            "model": model_jawab,
            "provider": settings.LLM_PROVIDER,
            "node": "founder-answer",
            "input_tokens": pemakaian.get("input_tokens"),
            "output_tokens": pemakaian.get("output_tokens"),
            "total_tokens": pemakaian.get("total_tokens"),
            "latency_ms": int((time.perf_counter() - mulai_jawab) * 1000),
            "success": bool(jawaban),
        }
    )

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
