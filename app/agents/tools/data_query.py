"""Tool: query_family_data — Text-to-SQL untuk pertanyaan data ekor panjang.

Kenapa tool ini ada
-------------------
Empat belas tool tetap di folder ini masing-masing menjawab satu bentuk
pertanyaan. Polanya akurat, tapi tiap pertanyaan baru menuntut tool baru,
endpoint baru, dan injector baru — sementara pertanyaan orang tua tidak terbatas
("bandingkan Maret sama April", "hari apa dia paling sering bolong", "sejak scan
terakhir dia sikat berapa hari"). Menambah tool terus-menerus juga punya harga:
makin banyak tool, makin sering LLM salah pilih.

Tool ini membalik arahnya. Satu tool, satu katalog dataset, dan pertanyaan baru
dijawab tanpa kode baru.

Pembagian peran
---------------
Di sini: menulis SQL (LLM) dan mengukur hasilnya (Langfuse).
Di peri-bugi-api: memutuskan SQL itu boleh jalan atau tidak, lalu
menjalankannya di batas read-only yang ter-scope ke satu keluarga.

Service ini tidak punya — dan tidak boleh punya — koneksi ke tabel bisnis.
Satu-satunya pintu adalah `/internal/agent/nl-query/execute`.

Satu kali perbaikan
-------------------
SQL yang ditolak validator atau gagal di database dikembalikan beserta
alasannya, dan model diberi satu kesempatan memperbaiki. Ini pola standar
text-to-SQL produksi (LinkedIn menyebutnya self-correction) dan menutup
sebagian besar kegagalan yang cuma soal nama kolom meleset. Dibatasi satu kali:
percobaan ketiga jarang menolong dan orang tua sudah menunggu.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get, call_internal_post
from app.config.settings import settings

logger = logging.getLogger(__name__)

CATALOG_PATH = "/api/v1/internal/agent/nl-query/catalog"
EXECUTE_PATH = "/api/v1/internal/agent/nl-query/execute"

#: error_type yang layak dicoba perbaiki. `guard_unavailable` tidak masuk —
#: itu masalah infrastruktur, bukan SQL yang salah tulis, dan mencoba lagi cuma
#: menambah latensi.
REPAIRABLE_ERRORS = frozenset(
    {"invalid_sql", "forbidden_table", "forbidden_function", "db_error"}
)

#: Baris yang ikut ke system prompt. Sisanya tetap terhitung di row_count.
#: Angka ini bukan soal biaya token semata — daftar panjang justru membuat LLM
#: memilih-milih baris secara acak saat menyusun kalimat.
MAX_ROWS_TO_PROMPT = 30


# =============================================================================
# Cache katalog
# =============================================================================

_catalog_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}


async def _get_catalog() -> dict[str, Any] | None:
    """Ambil katalog semantik dari api, di-cache singkat.

    Katalog berubah hanya saat ada deploy, tapi cache-nya tetap pendek supaya
    perubahan katalog tidak perlu restart ai-chat.
    """
    now = time.monotonic()
    cached = _catalog_cache.get("data")
    if cached and (now - _catalog_cache["fetched_at"]) < settings.NL_CATALOG_TTL_SECONDS:
        return cached

    data = await call_internal_get(CATALOG_PATH)
    if not data or data.get("error") or not data.get("prompt_text"):
        logger.warning("[data_query] Katalog tidak bisa diambil: %s", data)
        return cached  # pakai cache lama kalau ada; None kalau memang belum pernah

    _catalog_cache["data"] = data
    _catalog_cache["fetched_at"] = now
    return data


# =============================================================================
# Prompt penulis SQL
# =============================================================================

_SQL_SYSTEM_FALLBACK = """Kamu penulis SQL PostgreSQL untuk aplikasi kesehatan gigi anak.

Tugasmu: ubah pertanyaan orang tua menjadi SATU query SELECT.

Aturan mutlak:
- Hanya SELECT. Tidak ada INSERT/UPDATE/DELETE/DDL, tidak ada titik koma ganda.
- Hanya baca view yang ada di katalog. Jangan menyentuh tabel lain.
- JANGAN menulis filter user_id/parent_user_id/child_id milik siapa pun —
  view sudah tersaring otomatis ke keluarga yang bertanya.
- Jangan memakai fungsi di luar agregasi dan fungsi tanggal standar.
- Kembalikan kolom secukupnya untuk menjawab, bukan SELECT * kalau tidak perlu.
- Beri alias kolom yang mudah dibaca.
- Untuk pertanyaan "berapa"/"berapa kali", kembalikan hasil agregat, bukan
  daftar baris mentah.
- Untuk pertanyaan perkembangan harian, urutkan menaik menurut tanggal.

KONTEKS WAKTU: hari ini {today} ({weekday}), zona waktu {timezone}.
Pakai CURRENT_DATE untuk perhitungan relatif.

Jawab HANYA dengan SQL-nya. Tanpa penjelasan, tanpa pagar kode."""


def _strip_sql_fence(raw: str) -> str:
    """Ambil SQL dari jawaban model.

    Model masih sering membungkus jawabannya dengan ```sql walau diminta tidak.
    Menyalahkan prompt untuk hal sekecil ini tidak sepadan — dibersihkan saja.
    """
    text = (raw or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip().rstrip(";").strip()


# =============================================================================
# Factory
# =============================================================================


def make_query_family_data_tool(
    user_id: Optional[str],
    *,
    timezone: str = "Asia/Jakarta",
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    llm_provider_override: Optional[str] = None,
    llm_model_override: Optional[str] = None,
    prompts: Optional[dict] = None,
):
    """Bangun tool query_family_data dengan konteks turn ini terikat closure.

    Keamanan: `user_id` tidak pernah jadi parameter tool, sama seperti seluruh
    tool lain di folder ini. LLM tidak punya cara menyebut pemilik data lain —
    bahkan kalau ia menulis SQL yang mencoba, view di database menyaringnya.
    """
    _prompts = prompts or {}

    @tool
    async def query_family_data(question: str) -> dict[str, Any]:
        """Jawab pertanyaan tentang DATA keluarga ini yang butuh perhitungan,
        perbandingan, atau rincian per periode — dengan query langsung ke data.

        ✅ PAKAI tool ini untuk pertanyaan seperti:
        - "bandingkan sikat gigi bulan ini sama bulan kemarin"
        - "hari apa anak paling sering lupa sikat malam?"
        - "dalam 3 bulan terakhir, berapa hari yang lengkap pagi-malam?"
        - "rincian sikat gigi per hari minggu ini dong"
        - "sejak scan terakhir, dia sudah sikat berapa hari?"
        - "bulan apa yang paling konsisten tahun ini?"
        - pertanyaan data lain yang tidak persis cocok dengan tool khusus

        ❌ JANGAN pakai tool ini kalau:
        - Ada tool khusus yang persis menjawabnya (get_brushing_stats untuk
          streak hari ini, get_brushing_achievements untuk daftar badge,
          get_mata_peri_scan_detail untuk isi temuan scan) — tool khusus lebih
          cepat dan sudah terkurasi.
        - Pertanyaannya tentang pengetahuan gigi umum → search_dental_knowledge
        - Pertanyaannya tentang cara pakai aplikasi → search_app_faq
        - Pertanyaannya minta MENGUBAH data — tool ini hanya membaca.
        - User mengirim foto untuk dianalisa → analyze_chat_image

        Args:
            question: Pertanyaan user apa adanya, dalam Bahasa Indonesia.
                Sertakan konteks waktunya kalau ada ("bulan lalu", "minggu ini").

        Returns dict:
        - has_data: bool — true kalau query berhasil DAN ada baris
        - row_count: int — jumlah baris hasil
        - columns: list[str], rows: list[list] — hasil apa adanya
        - sql: str — SQL yang dijalankan
        - truncated: bool — hasil dipotong karena kebanyakan baris
        - error: str — terisi kalau gagal

        ANTI-HALU:
        - Jawab HANYA dari angka di rows. Jangan menambah angka dari ingatan.
        - row_count = 0 berarti BELUM ADA DATA untuk periode itu — bukan berarti
          anak tidak sikat gigi. Katakan apa adanya.
        - Kalau error terisi, akui datanya belum bisa diambil. Jangan mengarang.
        """
        from app.config.observability import trace_generation, trace_node

        async with trace_node(
            name="tool:query_family_data",
            state=None,
            input_data={"question": question, "has_user_id": bool(user_id)},
        ) as span:
            if not user_id:
                result = {
                    "has_data": False,
                    "error": "no_user_id",
                    "fallback_message": "Data pengguna tidak tersedia.",
                }
                if span:
                    span.update(output=result)
                return result

            catalog = await _get_catalog()
            if not catalog:
                result = {
                    "has_data": False,
                    "error": "catalog_unavailable",
                    "fallback_message": "Katalog data belum bisa diambil.",
                }
                if span:
                    span.update(output=result, level="ERROR")
                return result

            catalog_version = catalog.get("version")
            system_prompt = _build_sql_system_prompt(catalog, _prompts, timezone)

            attempts: list[dict] = []
            sql: str | None = None
            last_error: str | None = None
            repaired = False

            for attempt in range(2):  # satu percobaan + satu perbaikan
                user_msg = _build_sql_user_message(question, sql, last_error)

                sql = await _generate_sql(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    question=question,
                    attempt=attempt,
                    provider=llm_provider_override,
                    model=llm_model_override or (settings.NL_SQL_MODEL or None),
                    trace_generation=trace_generation,
                )
                if not sql:
                    last_error = "model tidak menghasilkan SQL"
                    attempts.append({"attempt": attempt, "error": last_error})
                    break

                exec_result = await call_internal_post(
                    EXECUTE_PATH,
                    {
                        "user_id": user_id,
                        "sql": sql,
                        "question": question,
                        "trace_id": trace_id,
                        "catalog_version": catalog_version,
                    },
                    extra_headers={"X-Request-ID": trace_id} if trace_id else None,
                    timeout=20,
                )

                attempts.append(
                    {
                        "attempt": attempt,
                        "sql": sql,
                        "ok": bool(exec_result.get("ok")),
                        "error_type": exec_result.get("error_type"),
                    }
                )

                if exec_result.get("ok"):
                    rows = exec_result.get("rows") or []
                    result = {
                        "has_data": len(rows) > 0,
                        "row_count": exec_result.get("row_count", len(rows)),
                        "columns": exec_result.get("columns") or [],
                        "rows": rows,
                        "sql": exec_result.get("sql") or sql,
                        "truncated": bool(exec_result.get("truncated")),
                        "datasets": exec_result.get("datasets") or [],
                        "elapsed_ms": exec_result.get("elapsed_ms"),
                        "catalog_version": catalog_version,
                        "repaired": repaired,
                        "question": question,
                        "source": "text2sql",
                    }
                    if span:
                        span.update(
                            output={
                                "row_count": result["row_count"],
                                "sql": result["sql"],
                                "repaired": repaired,
                                "datasets": result["datasets"],
                            }
                        )
                    return result

                error_type = exec_result.get("error_type") or "db_error"
                last_error = exec_result.get("message") or error_type

                if error_type not in REPAIRABLE_ERRORS:
                    break
                repaired = True

            logger.info(
                "[data_query] gagal setelah %d percobaan: %s", len(attempts), last_error
            )
            result = {
                "has_data": False,
                "row_count": 0,
                "error": last_error or "query gagal",
                "sql": sql,
                "attempts": attempts,
                "catalog_version": catalog_version,
                "question": question,
                "source": "text2sql",
            }
            if span:
                span.update(output=result, level="WARNING")
            return result

    return query_family_data


# =============================================================================
# Penyusun prompt + pemanggil LLM
# =============================================================================


def _build_sql_system_prompt(catalog: dict, prompts: dict, timezone: str) -> str:
    """Rangkai prompt penulis SQL: instruksi + katalog.

    Instruksinya diambil dari `prompt_templates` kalau ada (bisa disunting dari
    panel prompt tanpa deploy), jatuh ke teks di berkas ini kalau belum diisi.
    Katalognya selalu dari api — itu satu-satunya sumber kebenaran bentuk data.
    """
    from datetime import datetime, timedelta, timezone as dt_timezone

    # Asia/Jakarta = UTC+7. Dihitung manual supaya tidak menambah dependensi
    # zona waktu hanya untuk satu baris konteks.
    offset = 7 if timezone == "Asia/Jakarta" else 0
    now = datetime.now(dt_timezone.utc) + timedelta(hours=offset)
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][
        now.weekday()
    ]

    template = prompts.get("nl_sql_generate") or _SQL_SYSTEM_FALLBACK
    try:
        instruction = template.format(
            today=now.strftime("%Y-%m-%d"), weekday=hari, timezone=timezone
        )
    except (KeyError, IndexError):
        # Template dari DB yang punya placeholder tak dikenal jangan sampai
        # menjatuhkan tool — pakai apa adanya.
        instruction = template

    return f"{instruction}\n\n{catalog.get('prompt_text', '')}"


def _build_sql_user_message(
    question: str, previous_sql: str | None, error: str | None
) -> str:
    if previous_sql and error:
        return (
            f"Pertanyaan: {question}\n\n"
            f"SQL sebelumnya ditolak:\n{previous_sql}\n\n"
            f"Alasan: {error}\n\n"
            "Tulis ulang SQL-nya supaya lolos. Jawab hanya SQL."
        )
    return f"Pertanyaan: {question}\n\nTulis SQL-nya."


async def _generate_sql(
    *,
    system_prompt: str,
    user_message: str,
    question: str,
    attempt: int,
    provider: str | None,
    model: str | None,
    trace_generation,
) -> str | None:
    """Satu panggilan LLM untuk menulis SQL, ter-trace ke Langfuse."""
    from app.config.llm import get_llm

    # temperature 0: menulis SQL bukan pekerjaan kreatif. Nilai default 0.7
    # yang dipakai jawaban ke orang tua justru menghasilkan variasi kolom dan
    # bentuk query yang tidak perlu.
    llm = get_llm(
        temperature=0,
        max_tokens=512,
        streaming=False,
        provider=provider,
        model=model,
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    async with trace_generation(
        name=f"nl-sql-generate{'-repair' if attempt else ''}",
        model=model or settings.llm_model_name,
        system_prompt=system_prompt,
        user_message=user_message,
        metadata={"question": question, "attempt": attempt},
    ) as gen_span:
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.warning("[data_query] LLM gagal menulis SQL: %s", e)
            if gen_span:
                gen_span.update(output=f"error: {e}", level="ERROR")
            return None

        raw = response.content if isinstance(response.content, str) else str(response.content)
        sql = _strip_sql_fence(raw)

        if gen_span:
            gen_span.update(output=sql)

    return sql or None


# =============================================================================
# ToolSpec — registry pattern (lihat tools/registry.py)
# =============================================================================
from app.agents.tools.registry import (  # noqa: E402
    BridgeContext,
    ToolSpec,
    is_tool_unavailable_result,
    register_tool,
)


def _bridge_data_query(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    if is_tool_unavailable_result(result):
        return
    agent_results["data_query"] = result


def _format_rows_as_table(columns: list, rows: list, limit: int) -> str:
    """Render hasil query jadi tabel teks yang padat.

    Bentuk pipe-table dipilih karena batas kolomnya jelas bagi LLM — daftar JSON
    membuatnya gampang menggeser nilai ke kolom sebelah saat menyusun kalimat.
    """
    head = " | ".join(str(c) for c in columns)
    sep = "-" * len(head)
    body_lines = []
    for row in rows[:limit]:
        cells = ["" if v is None else str(v) for v in row]
        body_lines.append(" | ".join(cells))
    return "\n".join([head, sep, *body_lines])


def _inject_data_query(
    data: dict, child_name: str, prompts: dict, response_mode: str
) -> str:
    """Suntik hasil query ke system prompt, sadar mode jawaban.

    Pagar yang sama dengan tool lain di repo ini: hasil kosong harus dinyatakan
    sebagai "belum ada data", bukan disimpulkan sebagai "anak tidak sikat gigi".
    Bedanya di sini jauh lebih penting — hasil query bisa kosong hanya karena
    periode yang ditanya memang belum terisi.
    """
    if not data:
        return ""

    question = data.get("question") or ""

    if data.get("error"):
        return (
            f"\n\nPertanyaan data \"{question}\": DATANYA TIDAK BISA DIAMBIL "
            f"(kendala teknis). Sampaikan dengan jujur bahwa kamu belum bisa "
            f"mengambil datanya sekarang, tanpa menyebut istilah teknis, "
            f"database, atau SQL. JANGAN mengarang angka."
        )

    row_count = data.get("row_count", 0)
    if not row_count:
        return (
            f"\n\nPertanyaan data \"{question}\": query berhasil tapi HASILNYA "
            f"KOSONG. Artinya BELUM ADA DATA untuk periode itu — BUKAN berarti "
            f"{child_name} tidak sikat gigi. Sampaikan apa adanya, ramah, dan "
            f"jangan menyimpulkan apa pun dari ketiadaan data."
        )

    columns = data.get("columns") or []
    rows = data.get("rows") or []

    # "TERBARU" bukan hiasan. Kalau jawaban sebelumnya di percakapan yang sama
    # keliru, model cenderung mengulanginya — pertanyaan yang sama dijawab benar
    # di sesi baru tapi salah di sesi lanjutan, persis karena jawaban lamanya
    # ikut terbaca sebagai konteks. Konvensi yang sama sudah dipakai injector
    # rapot sikat gigi ("LATEST dari tool call").
    text = (
        f"\n\nHasil data TERBARU dari database untuk pertanyaan \"{question}\" "
        f"({row_count} baris):\n"
        f"{_format_rows_as_table(columns, rows, MAX_ROWS_TO_PROMPT)}"
    )

    if len(rows) > MAX_ROWS_TO_PROMPT:
        text += (
            f"\n(ditampilkan {MAX_ROWS_TO_PROMPT} dari {row_count} baris pertama)"
        )
    if data.get("truncated"):
        text += (
            "\n(hasil dipotong karena datanya banyak — sebutkan bahwa ini "
            "sebagian, jangan mengklaim total)"
        )

    # Panjang jawaban mengikuti mode yang dipilih orang tua di layar chat.
    if response_mode == "simple":
        style = (
            "Ringkas jadi 1-2 kalimat dengan angka kuncinya saja. "
            "Jangan menampilkan tabel."
        )
    elif response_mode == "medium":
        style = (
            "Jawab ringkas dengan angka kunci plus satu kalimat konteks. "
            "Tabel hanya kalau benar-benar membantu."
        )
    else:
        style = (
            "Boleh merinci per hari atau per periode, memakai daftar bertitik "
            "atau tabel kecil, lalu tutup dengan satu kalimat kesimpulan."
        )

    text += (
        f"\n\nINSTRUKSI: Jawab HANYA dari angka di tabel di atas — jangan "
        f"menambah angka dari mana pun.\n"
        # Pagar ini lahir dari kegagalan nyata: untuk "pagi atau malam yang
        # lebih sering kelewat", query mengembalikan pagi=8 dan malam=10, tapi
        # jawaban yang keluar justru "tidak ada yang terlewat". Angkanya ada di
        # prompt, modelnya yang membalik. Kalimat larangan eksplisit di bawah
        # jauh lebih efektif daripada sekadar "jawab sesuai data".
        f"- DILARANG bilang \"tidak ada\", \"nol\", \"belum ada\", atau "
        f"\"semuanya lengkap\" kecuali angka di tabel memang 0. Kalau ada baris "
        f"dengan angka lebih dari 0, sebutkan angkanya.\n"
        f"- Kalau pertanyaannya membandingkan (mana yang lebih sering / paling "
        f"banyak), sebutkan pemenangnya BESERTA angkanya dari tabel.\n"
        f"- Baca nama kolomnya baik-baik: kolom yang menghitung hal yang "
        f"TERLEWAT berlawanan arti dengan kolom yang menghitung hal yang "
        f"TERCATAT. Jangan tertukar.\n"
        f"- Kalau jawabanmu sebelumnya di percakapan ini berbeda dengan tabel "
        f"di atas, TABEL INI yang benar. Jangan mengulang jawaban lama.\n"
        f"- Pakai bahasa sehari-hari untuk Bunda (\"hari berturut-turut\", "
        f"bukan \"streak\"). Jangan menyebut SQL, tabel database, atau nama "
        f"kolom teknis.\n"
        f"- {style}"
    )

    return text


register_tool(
    ToolSpec(
        tool_name="query_family_data",
        agent_key="data_query",
        required_agent="data_query",
        bridge_handler=_bridge_data_query,
        prompt_injector=_inject_data_query,
        thinking_label="Menghitung dari data...",
    )
)
