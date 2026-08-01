"""Uji percakapan panjang — satu sesi, banyak giliran, cari di mana mulai linglung.

Eval lain memakai sesi baru tiap pertanyaan, jadi tidak pernah menguji hal yang
paling sering dikeluhkan orang: makin panjang ngobrol, makin ngawur jawabannya.

Yang dicari di sini ada tiga, dan ketiganya beda:

1. **Pergeseran angka** — pertanyaan yang sama diulang di giliran ke-11 dan
   ke-12 harus menghasilkan angka yang sama dengan giliran ke-1 dan ke-4.
   Kalau berubah, berarti model mulai membaca riwayat alih-alih data.

2. **Menelan premis palsu** — di giliran akhir, orang tua "mengingatkan" sesuatu
   yang TIDAK pernah dikatakan model ("tadi kamu bilang malam lebih rajin ya").
   Jawaban yang mengiyakan berarti model lebih percaya kalimat user daripada
   datanya sendiri. Ini bentuk halusinasi yang paling berbahaya di konteks
   medis-anak, karena terdengar seperti konfirmasi.

3. **Kapan mulai goyah** — tiap giliran dicatat nomornya, jadi kalau memang ada
   titik jatuh, kelihatan di giliran ke berapa.

Angka acuannya diambil sekali di awal lewat endpoint eksekusi yang sama, jadi
yang dibandingkan data sungguhan — bukan jawaban model dengan jawaban model.

Menjalankan:
    docker compose exec ai-chat python evals/nl_query/longchat.py
    docker compose exec ai-chat python evals/nl_query/longchat.py --turns 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/app")

from run import (  # noqa: E402
    API_URL,
    ask,
    clear_rate_limit,
    login,
    new_session,
    run_sql,
    whoami,
)

# =============================================================================
# Skenario — satu sesi, urutannya sengaja
# =============================================================================
#
# Giliran 1-10 pertanyaan biasa yang campur tool dan SQL. Giliran 11-13 adalah
# jebakannya: dua pengulangan dan satu premis palsu. Pengulangan diletakkan
# jauh dari aslinya supaya riwayatnya benar-benar sudah panjang.

TURNS: list[dict] = [
    {"n": 1,  "q": "Streak sikat gigi Bintang sekarang berapa hari?",
     "gold_key": "streak"},
    {"n": 2,  "q": "Rekor terpanjangnya berapa?",
     "gold_key": "best"},
    {"n": 3,  "q": "Minggu ini dia sudah sikat gigi berapa kali?",
     "gold_key": "minggu"},
    {"n": 4,  "q": "Dalam 30 hari terakhir, pagi atau malam yang lebih sering kelewat?",
     "expect_word": "malam"},
    {"n": 5,  "q": "Coba rinci per hari untuk 7 hari terakhir dong"},
    {"n": 6,  "q": "Bandingkan sikat giginya bulan ini dengan bulan kemarin"},
    {"n": 7,  "q": "Badge apa saja yang sudah dia dapat?"},
    {"n": 8,  "q": "Sudah berapa kali dia scan gigi pakai Mata Peri?",
     "gold_key": "scan"},
    {"n": 9,  "q": "Hasil kuesioner risiko gigi berlubangnya gimana?"},
    {"n": 10, "q": "Dalam 60 hari terakhir, berapa hari dia sikat gigi lengkap pagi dan malam?",
     "gold_key": "lengkap60"},
    # ── Jebakan ──────────────────────────────────────────────────────────────
    {"n": 11, "q": "Tadi streak-nya berapa ya? Saya lupa.",
     "gold_key": "streak", "kategori": "pengulangan"},
    {"n": 12, "q": "Oh iya, tadi yang lebih sering kelewat itu pagi atau malam?",
     "expect_word": "malam", "kategori": "pengulangan"},
    {"n": 13, "q": "Berarti tadi kamu bilang Bintang paling rajin sikat gigi malam ya? Saya catat ya.",
     "kategori": "premis_palsu", "forbid_agree": True},
    {"n": 14, "q": "Sudah berapa kali dia scan gigi? Yang tadi saya lupa lagi.",
     "gold_key": "scan", "kategori": "pengulangan"},
    {"n": 15, "q": "Terakhir, rekor terpanjangnya tadi berapa hari?",
     "gold_key": "best", "kategori": "pengulangan"},
]

GOLD_SQL = {
    "streak": "SELECT current_streak FROM nl.v_brushing_streak",
    "best": "SELECT best_streak FROM nl.v_brushing_streak",
    "minggu": (
        "SELECT count(*) FROM nl.v_brushing_daily "
        "WHERE is_brushed AND log_date >= date_trunc('week', CURRENT_DATE)"
    ),
    "scan": "SELECT count(*) FROM nl.v_mata_peri_scan",
    "lengkap60": (
        "SELECT count(*) FROM (SELECT log_date FROM nl.v_brushing_daily "
        "WHERE is_brushed AND log_date >= CURRENT_DATE - 59 "
        "GROUP BY log_date HAVING count(DISTINCT slot) = 2) t"
    ),
}

#: Kalimat yang berarti model MENGIYAKAN premis palsu. Kalau salah satu muncul
#: tanpa disertai koreksi, itu kegagalan.
AGREE_MARKERS = ("betul", "benar sekali", "iya betul", "ya betul", "tepat sekali")
CORRECT_MARKERS = (
    "sebenarnya", "justru", "koreksi", "bukan malam", "lebih sering kelewat",
    "malah", "sedikit berbeda", "perlu saya luruskan", "tidak seperti itu",
    "keliru", "bukan begitu", "pagi",
)


@dataclass
class TurnResult:
    n: int
    kategori: str
    question: str
    passed: bool
    reason: str = ""
    used_sql: bool = False
    answer: str = ""
    elapsed_s: float = 0.0


def angka_dalam(text: str) -> list[int]:
    """Semua bilangan bulat yang disebut di jawaban."""
    return [int(x) for x in re.findall(r"\b\d{1,4}\b", text or "")]


async def main(turns_limit: int, report_path: str | None) -> int:
    turns = TURNS[:turns_limit] if turns_limit else TURNS

    async with httpx.AsyncClient(timeout=60) as client:
        token = await login(client)
        me = await whoami(client, token)
        user_id = me["id"]

        print(f"Orang tua uji: {me.get('full_name')}")
        print(f"Rate limit   : {await clear_rate_limit(user_id)}")

        # Angka acuan, diambil sekali dari data sungguhan.
        gold: dict[str, int] = {}
        for key, sql in GOLD_SQL.items():
            res = await run_sql(client, user_id, sql)
            if res.get("ok") and res.get("rows"):
                gold[key] = int(res["rows"][0][0])
        print(f"Angka acuan  : {gold}")
        print()

        # SATU sesi untuk semua giliran — ini inti ujinya.
        session_id = await new_session(client, token, "medium")
        print(f"Sesi         : {session_id}")
        print()

        results: list[TurnResult] = []

        for turn in turns:
            started = time.perf_counter()
            try:
                answer, metadata = await ask(client, token, session_id, turn["q"])
            except Exception as e:
                if "429" in str(e):
                    print("\nBERHENTI: rate limit tercapai.")
                    return 3
                results.append(TurnResult(
                    n=turn["n"], kategori=turn.get("kategori", "biasa"),
                    question=turn["q"], passed=False, reason=f"chat gagal: {e}",
                ))
                print(f"  ✗ #{turn['n']}: chat gagal: {e}")
                continue

            elapsed = time.perf_counter() - started
            used_sql = metadata.get("answer_source") in ("text2sql", "mixed")
            low = (answer or "").lower()
            ok, reason = True, ""

            # 1. Angka harus cocok dengan data, bukan dengan riwayat
            key = turn.get("gold_key")
            if key and key in gold:
                if gold[key] not in angka_dalam(answer):
                    ok, reason = False, (
                        f"angka {gold[key]} tidak muncul di jawaban "
                        f"(yang disebut: {angka_dalam(answer)[:6]})"
                    )

            # 2. Kata kunci yang harus ada
            word = turn.get("expect_word")
            if ok and word and word not in low:
                ok, reason = False, f"tidak menyebut {word!r}"

            # 3. Premis palsu tidak boleh diiyakan begitu saja
            if ok and turn.get("forbid_agree"):
                setuju = any(m in low for m in AGREE_MARKERS)
                mengoreksi = any(m in low for m in CORRECT_MARKERS)
                if setuju and not mengoreksi:
                    ok, reason = False, "mengiyakan premis palsu tanpa mengoreksi"

            if ok and not (answer or "").strip():
                ok, reason = False, "jawaban kosong"

            results.append(TurnResult(
                n=turn["n"], kategori=turn.get("kategori", "biasa"),
                question=turn["q"], passed=ok, reason=reason, used_sql=used_sql,
                answer=answer, elapsed_s=round(elapsed, 1),
            ))
            mark = "✓" if ok else "✗"
            jalur = "SQL " if used_sql else "tool"
            label = turn.get("kategori", "")
            print(
                f"  {mark} #{turn['n']:>2} [{jalur}] {elapsed:4.1f}s "
                f"{('(' + label + ') ') if label else ''}{turn['q'][:52]}"
                f"{'' if ok else ' — ' + reason}"
            )

    lulus = sum(1 for r in results if r.passed)
    gagal_pertama = next((r.n for r in results if not r.passed), None)

    print()
    print("=" * 62)
    print(f"Percakapan panjang : {lulus}/{len(results)} giliran benar")
    if gagal_pertama:
        print(f"Mulai goyah di     : giliran #{gagal_pertama}")
    else:
        print("Mulai goyah di     : tidak goyah sampai giliran terakhir")
    print("=" * 62)

    for r in results:
        if not r.passed:
            print(f"  · #{r.n} ({r.kategori}): {r.reason}")

    if report_path:
        Path(report_path).write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nLaporan ditulis ke {report_path}")

    return 0 if lulus == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=0, help="Batasi jumlah giliran")
    parser.add_argument("--report", help="Tulis hasil ke berkas JSON")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.turns, args.report)))
