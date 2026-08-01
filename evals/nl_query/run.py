"""Jalankan eval Text-to-SQL terhadap tumpukan lokal yang benar-benar hidup.

Bukan unit test: skrip ini masuk lewat pintu depan — login sebagai orang tua uji,
kirim pertanyaan ke `POST /api/v1/chat/message`, dan menilai jawaban yang keluar.
Semua yang ada di jalur sungguhan ikut teruji: izin agent, feature flag,
strategi, prompt, katalog, validator, sampai view di database.

Cara menilainya
---------------
**Execution accuracy**, standar yang dipakai benchmark text-to-SQL: SQL buatan
model dijalankan, SQL emas dari `golden.yaml` dijalankan, lalu HASILNYA
dibandingkan. Query yang ditulis dengan cara sama sekali berbeda tapi
mengembalikan angka yang sama tetap lulus — yang dinilai jawabannya, bukan
gayanya. Mencocokkan teks SQL akan menghukum model karena menulis `>= CURRENT_DATE - 6`
alih-alih `> CURRENT_DATE - 7`, padahal keduanya benar.

Ada satu fase lagi sebelum itu: uji negatif keamanan, tanpa LLM sama sekali.
SQL berbahaya ditembakkan langsung ke endpoint eksekusi dan harus ditolak. Fase
ini dijalankan lebih dulu dan kegagalannya bersifat menghentikan — kalau
penjaganya bocor, tidak ada gunanya mengukur akurasi.

Menjalankan:
    docker compose exec ai-chat python evals/nl_query/run.py
    docker compose exec ai-chat python evals/nl_query/run.py --only hari_lengkap_60
    docker compose exec ai-chat python evals/nl_query/run.py --skip-chat   # keamanan saja

Prasyarat:
    1. peri-bugi-api dan ai-chat hidup
    2. `docker compose exec api python scripts/seed_text2sql_demo.py --reset`
    3. Fitur `tanya_peri_text2sql` menyala dan strategi bukan 'tools'
       (skrip ini memeriksanya dan berhenti dengan pesan jelas kalau belum)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, "/app")

from app.config.settings import settings  # noqa: E402

API_URL = os.environ.get("EVAL_API_URL", settings.PERI_API_URL or "http://host.docker.internal:8000")
PHONE = os.environ.get("EVAL_PHONE", "+628119990001")
PASSWORD = os.environ.get("EVAL_PASSWORD", "Text2SQL123!")

GOLDEN_PATH = Path(__file__).parent / "golden.yaml"

INTERNAL_HEADERS = {"X-Internal-Secret": settings.INTERNAL_SECRET}

# Satu pertanyaan bisa memicu dua panggilan LLM (tulis SQL, lalu susun jawaban)
# plus perbaikan. Longgar, karena yang diuji kebenaran, bukan kecepatan.
CHAT_TIMEOUT = 120


# =============================================================================
# Hasil
# =============================================================================


@dataclass
class CaseResult:
    id: str
    passed: bool
    reason: str = ""
    used_sql: bool = False
    sql: str | None = None
    elapsed_s: float = 0.0
    answer: str = ""


@dataclass
class Report:
    security: list[CaseResult] = field(default_factory=list)
    chat: list[CaseResult] = field(default_factory=list)
    #: Kasus di `known_gaps` — dilaporkan apa adanya, tidak menentukan lulus.
    gaps: list[CaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.security + self.chat)


# =============================================================================
# Pembantu HTTP
# =============================================================================


async def login(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        f"{API_URL}/api/v1/auth/login/phone",
        json={"phone_number": PHONE, "password": PASSWORD, "device_type": "web"},
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"Login orang tua uji gagal ({resp.status_code}). "
            f"Sudah jalankan scripts/seed_text2sql_demo.py?\n{resp.text[:300]}"
        )
    return resp.json()["data"]["access_token"]


async def run_sql(client: httpx.AsyncClient, user_id: str, sql: str) -> dict:
    """Jalankan SQL lewat endpoint internal — jalur yang sama dengan tool."""
    resp = await client.post(
        f"{API_URL}/api/v1/internal/agent/nl-query/execute",
        json={"user_id": user_id, "sql": sql, "question": "[eval]"},
        headers=INTERNAL_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


async def clear_rate_limit(user_id: str) -> str:
    """Bersihkan penghitung rate limit chat untuk user uji saja.

    Satu putaran eval mengirim 21 pesan; batasnya 50 per jam. Dua putaran
    berturut-turut — hal biasa saat menyetel katalog — langsung menabrak
    dinding, dan seluruh hasilnya jadi 429 yang tidak ada hubungannya dengan
    kualitas jawaban.

    Yang dibersihkan hanya kunci milik user uji. Batasnya sendiri TIDAK diubah:
    itu pengaman biaya untuk orang tua sungguhan, bukan sesuatu yang boleh
    dilonggarkan demi kenyamanan eval.
    """
    # Redis yang dipakai rate limit adalah milik peri-bugi-api, bukan milik
    # ai-chat — REDIS_URL di sini memang sering kosong. Karena itu ada nilai
    # bawaan yang menunjuk ke host, dan bisa ditimpa lewat EVAL_REDIS_URL.
    redis_url = (
        os.environ.get("EVAL_REDIS_URL")
        or settings.REDIS_URL
        or "redis://host.docker.internal:6379"
    )

    # RESP mentah lewat asyncio: ai-chat tidak memasang paket redis, dan
    # menambah dependensi hanya demi kenyamanan eval tidak sepadan.
    # Kuncinya deterministik (lihat peri-bugi-api/app/middleware/chat_rate_limit.py),
    # jadi tidak perlu SCAN.
    keys = [
        f"rate:chat:user:{user_id}:hourly",
        f"rate:chat:user:{user_id}:daily",
        f"rate:chat:user:{user_id}:concurrent",
    ]

    try:
        from urllib.parse import urlparse

        parsed = urlparse(redis_url)
        reader, writer = await asyncio.open_connection(
            parsed.hostname or "localhost", parsed.port or 6379
        )
        command = f"*{len(keys) + 1}\r\n$3\r\nDEL\r\n" + "".join(
            f"${len(k)}\r\n{k}\r\n" for k in keys
        )
        writer.write(command.encode())
        await writer.drain()
        reply = (await reader.readline()).decode().strip()
        writer.close()
        await writer.wait_closed()
        return f"{reply.lstrip(':')} penghitung dihapus"
    except Exception as e:
        return f"gagal ({e}) — lanjut apa adanya"


async def whoami(client: httpx.AsyncClient, token: str) -> dict:
    resp = await client.get(
        f"{API_URL}/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    return resp.json()["data"]


async def preflight(client: httpx.AsyncClient, token: str) -> None:
    """Pastikan jalur Text-to-SQL memang hidup untuk user uji.

    Kalau agent `data_query` mati, setiap kasus `must_use_sql` gagal dengan
    "tidak memakai Text-to-SQL" — dan itu terbaca seperti kualitas jawaban yang
    memburuk, padahal cuma saklar. Persis itu yang terjadi setelah menjalankan
    ulang migration: `downgrade` menghapus baris agent_configs, `upgrade`
    memasangnya kembali dalam keadaan MATI (memang disengaja), dan angka eval
    terjun tanpa ada yang salah dengan fiturnya.

    Lebih baik berhenti di sini dengan penjelasan daripada menghasilkan laporan
    yang menyesatkan.
    """
    resp = await client.get(
        f"{API_URL}/api/v1/chat/agents", headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    agents = resp.json()["data"]["agents"]
    entry = next((a for a in agents if a["agent_key"] == "data_query"), None)

    if entry is None or not entry.get("is_active"):
        raise SystemExit(
            "Agent 'data_query' tidak aktif untuk user uji — semua kasus SQL akan\n"
            "gagal karena saklarnya, bukan karena jawabannya. Perbaiki dulu:\n"
            "  1. FEATURE_TANYA_PERI_TEXT2SQL_ENABLED=true di peri-bugi-api/.env\n"
            "  2. TANYA_PERI_DATA_STRATEGY=hybrid (atau sql)\n"
            "  3. UPDATE agent_configs SET is_globally_active = true\n"
            "       WHERE agent_key = 'data_query';\n"
            "  4. jalankan ulang scripts/seed_text2sql_demo.py supaya izinnya ada\n"
            "  5. restart api, lalu kosongkan cache:user_agents:* di Redis\n"
        )


async def new_session(client: httpx.AsyncClient, token: str, mode: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.post(f"{API_URL}/api/v1/chat/session/new", headers=headers)
    resp.raise_for_status()
    # Bentuk respons: {data: {session: {...}, messages: [...]}} — bukan session
    # di akar. Lihat chat_service.get_session_detail_dict.
    session_id = resp.json()["data"]["session"]["id"]

    await client.patch(
        f"{API_URL}/api/v1/chat/session/{session_id}/mode",
        json={"response_mode": mode},
        headers=headers,
    )
    return session_id


async def ask(
    client: httpx.AsyncClient, token: str, session_id: str, question: str
) -> tuple[str, dict]:
    """Kirim pertanyaan, ikuti SSE-nya, kembalikan (jawaban, metadata)."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
    content = ""
    metadata: dict = {}

    async with client.stream(
        "POST",
        f"{API_URL}/api/v1/chat/message",
        json={"content": question, "session_id": session_id, "source": "web"},
        headers=headers,
        timeout=CHAT_TIMEOUT,
    ) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode()[:300]
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("event") == "token":
                content += event.get("data") or ""
            elif event.get("event") == "done":
                data = event.get("data") or {}
                content = data.get("content") or content
                metadata = data.get("metadata") or {}

    return content, metadata


# =============================================================================
# Perbandingan hasil
# =============================================================================


def _normalize(rows: list) -> list:
    """Samakan bentuk supaya perbandingan tidak tersandung hal remeh.

    Angka desimal yang keluar sebagai 30 vs 30.0, dan urutan baris yang berbeda
    padahal isinya sama, bukan kesalahan menjawab.
    """
    out = []
    for row in rows:
        norm = []
        for v in row:
            if isinstance(v, float) and v.is_integer():
                norm.append(int(v))
            elif isinstance(v, str):
                norm.append(v.strip())
            else:
                norm.append(v)
        out.append(tuple(norm))
    return sorted(out, key=lambda r: json.dumps(r, default=str, sort_keys=True))


def compare(mode: str, gold: dict, got: dict) -> tuple[bool, str]:
    if mode == "none":
        return True, ""

    gold_rows = gold.get("rows") or []
    got_rows = got.get("rows") or []

    if mode == "value":
        if not gold_rows or not got_rows:
            return False, f"baris kosong (emas {len(gold_rows)}, model {len(got_rows)})"
        g, m = gold_rows[0][0], got_rows[0][0]
        if isinstance(g, float) and g.is_integer():
            g = int(g)
        if isinstance(m, float) and m.is_integer():
            m = int(m)
        if g != m:
            return False, f"nilai beda: emas {g!r}, model {m!r}"
        return True, ""

    if _normalize(gold_rows) != _normalize(got_rows):
        return (
            False,
            f"hasil beda: emas {len(gold_rows)} baris, model {len(got_rows)} baris",
        )
    return True, ""


# =============================================================================
# Fase 1 — keamanan
# =============================================================================


async def run_security(client: httpx.AsyncClient, user_id: str, cases: list) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        try:
            res = await run_sql(client, user_id, case["sql"])
        except Exception as e:
            results.append(CaseResult(case["id"], False, f"permintaan gagal: {e}"))
            continue

        elapsed = time.perf_counter() - started

        if "expect_error" in case:
            if res.get("ok"):
                results.append(
                    CaseResult(case["id"], False, "DITERIMA padahal harus ditolak", elapsed_s=elapsed)
                )
            elif res.get("error_type") != case["expect_error"]:
                results.append(
                    CaseResult(
                        case["id"],
                        False,
                        f"ditolak dengan alasan {res.get('error_type')!r}, "
                        f"diharapkan {case['expect_error']!r}",
                        elapsed_s=elapsed,
                    )
                )
            else:
                results.append(CaseResult(case["id"], True, elapsed_s=elapsed))
            continue

        if "expect_rows" in case:
            if not res.get("ok"):
                results.append(
                    CaseResult(case["id"], False, f"gagal jalan: {res.get('message')}", elapsed_s=elapsed)
                )
            elif res.get("row_count") != case["expect_rows"]:
                results.append(
                    CaseResult(
                        case["id"],
                        False,
                        f"dapat {res.get('row_count')} baris, diharapkan {case['expect_rows']} "
                        "— scoping keluarga BOCOR",
                        elapsed_s=elapsed,
                    )
                )
            else:
                results.append(CaseResult(case["id"], True, elapsed_s=elapsed))

    return results


# =============================================================================
# Fase 2 — pertanyaan
# =============================================================================


async def run_chat(
    client: httpx.AsyncClient, token: str, user_id: str, cases: list
) -> list[CaseResult]:
    results: list[CaseResult] = []

    for case in cases:
        case_id = case["id"]
        started = time.perf_counter()
        try:
            session_id = await new_session(client, token, case.get("mode", "medium"))
            answer, metadata = await ask(client, token, session_id, case["question"])
        except Exception as e:
            # 429 bukan kegagalan kualitas — melanjutkan hanya menghasilkan
            # deretan ✗ palsu yang menutupi hasil sesungguhnya.
            if "429" in str(e):
                print()
                print("BERHENTI: rate limit chat tercapai (50 pesan/jam per user).")
                print("Tunggu sebentar, atau jalankan ulang — skrip ini otomatis")
                print("membersihkan penghitungnya di awal.")
                raise SystemExit(3)
            results.append(CaseResult(case_id, False, f"chat gagal: {e}"))
            print(f"  ✗ {case_id}: chat gagal: {e}")
            continue

        elapsed = time.perf_counter() - started
        nl = metadata.get("nl_query") or {}
        used_sql = metadata.get("answer_source") in ("text2sql", "mixed")
        sql = nl.get("sql")

        def fail(reason: str) -> CaseResult:
            return CaseResult(case_id, False, reason, used_sql, sql, elapsed, answer)

        # Jalur yang dipakai
        if case.get("forbid_sql") and used_sql:
            results.append(fail("memakai Text-to-SQL padahal ini bukan pertanyaan data"))
            print(f"  ✗ {case_id}: {results[-1].reason}")
            continue
        if case.get("must_use_sql") and not used_sql:
            results.append(fail("tidak memakai Text-to-SQL padahal seharusnya"))
            print(f"  ✗ {case_id}: {results[-1].reason}")
            continue
        if used_sql and nl.get("error"):
            results.append(fail(f"query gagal: {nl.get('error')}"))
            print(f"  ✗ {case_id}: {results[-1].reason}")
            continue

        # Ketepatan hasil
        compare_mode = case.get("compare", "none")
        if compare_mode != "none" and case.get("gold_sql") and sql:
            gold = await run_sql(client, user_id, case["gold_sql"])
            got = await run_sql(client, user_id, sql)
            if not gold.get("ok"):
                results.append(fail(f"SQL emas sendiri gagal: {gold.get('message')}"))
                print(f"  ✗ {case_id}: {results[-1].reason}")
                continue
            same, why = compare(compare_mode, gold, got)
            if not same:
                results.append(fail(why))
                print(f"  ✗ {case_id}: {why}")
                continue

        # Isi jawaban
        needle = case.get("answer_contains")
        if needle and needle.lower() not in (answer or "").lower():
            results.append(fail(f"jawaban tidak memuat {needle!r}"))
            print(f"  ✗ {case_id}: {results[-1].reason}")
            continue

        results.append(CaseResult(case_id, True, "", used_sql, sql, elapsed, answer))
        jalur = "SQL " if used_sql else "tool"
        print(f"  ✓ {case_id} [{jalur}] {elapsed:.1f}s")

    return results


# =============================================================================
# Utama
# =============================================================================


async def main(
    only: str | None, skip_chat: bool, verbose: bool, report_path: str | None = None
) -> int:
    spec = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    chat_cases = spec.get("cases") or []
    security_cases = spec.get("security") or []
    gap_cases = spec.get("known_gaps") or []

    if only:
        chat_cases = [c for c in chat_cases if c["id"] == only]
        security_cases = [c for c in security_cases if c["id"] == only]
        gap_cases = [c for c in gap_cases if c["id"] == only]

    report = Report()

    async with httpx.AsyncClient(timeout=30) as client:
        token = await login(client)
        me = await whoami(client, token)
        user_id = me["id"]
        print(f"Orang tua uji: {me.get('full_name')} ({user_id})")
        print(f"API: {API_URL}")
        print(f"Rate limit: {await clear_rate_limit(user_id)}")
        if not skip_chat:
            await preflight(client, token)
            print("Preflight : agent data_query aktif")
        print()

        # ── Fase 1 ───────────────────────────────────────────────────────────
        print(f"[1/2] Uji negatif keamanan ({len(security_cases)} kasus)")
        report.security = await run_security(client, user_id, security_cases)
        for r in report.security:
            print(f"  {'✓' if r.passed else '✗'} {r.id}{'' if r.passed else ': ' + r.reason}")

        if any(not r.passed for r in report.security):
            print()
            print("BERHENTI: penjaga keamanan bocor. Perbaiki itu dulu —")
            print("mengukur akurasi di atas penjaga yang bolong tidak ada gunanya.")
            return 2

        if skip_chat:
            print("\n(--skip-chat) Fase pertanyaan dilewati.")
            return 0

        # ── Fase 2 ───────────────────────────────────────────────────────────
        print()
        print(f"[2/2] Pertanyaan ({len(chat_cases)} kasus) — ini memanggil LLM sungguhan")
        report.chat = await run_chat(client, token, user_id, chat_cases)

        # ── Celah yang diketahui ─────────────────────────────────────────────
        if gap_cases:
            print()
            print(f"Celah yang diketahui ({len(gap_cases)}) — dilaporkan, tidak dihitung")
            report.gaps = await run_chat(client, token, user_id, gap_cases)

    # ── Ringkasan ────────────────────────────────────────────────────────────
    passed = sum(1 for r in report.chat if r.passed)
    total = len(report.chat)
    pct = (passed / total * 100) if total else 100.0

    print()
    print("=" * 62)
    print(f"Keamanan   : {sum(1 for r in report.security if r.passed)}/{len(report.security)} lolos")
    print(f"Pertanyaan : {passed}/{total} lolos ({pct:.0f}%)")
    if report.gaps:
        gaps_ok = sum(1 for r in report.gaps if r.passed)
        print(f"Celah      : {gaps_ok}/{len(report.gaps)} membaik (tidak dihitung)")
    print("=" * 62)

    gagal = [r for r in report.chat if not r.passed]
    if gagal:
        print()
        print("Yang gagal:")
        for r in gagal:
            print(f"  · {r.id}: {r.reason}")
            if verbose and r.sql:
                print(f"      SQL: {r.sql}")
            if verbose and r.answer:
                print(f"      Jawaban: {r.answer[:200]}")

    if report_path:
        from dataclasses import asdict

        Path(report_path).write_text(
            json.dumps(
                {
                    "security": [asdict(r) for r in report.security],
                    "chat": [asdict(r) for r in report.chat],
                    "gaps": [asdict(r) for r in report.gaps],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nLaporan ditulis ke {report_path}")

    # Ambang 90%: di bawah itu bukan "kurang rapi", tapi ada yang rusak.
    return 0 if pct >= 90 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Jalankan satu kasus saja, sebut id-nya")
    parser.add_argument("--skip-chat", action="store_true", help="Uji keamanan saja")
    parser.add_argument("-v", "--verbose", action="store_true", help="Tampilkan SQL + jawaban")
    parser.add_argument("--report", help="Tulis hasil lengkap ke berkas JSON")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.only, args.skip_chat, args.verbose, args.report)))
