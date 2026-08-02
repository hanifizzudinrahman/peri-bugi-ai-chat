"""Eval Tanya Data Founder.

Dua fase, dan urutannya mengikat:

1. **Keamanan** — serangan ditembak langsung ke endpoint eksekusi, tanpa LLM.
   Fase ini **menghentikan**: kalau ada satu saja yang lolos, akurasi jawaban
   tidak diukur. Angka bagus di atas gerbang yang jebol cuma bikin orang
   percaya lebih lama.

2. **Akurasi** — tiap pertanyaan emas dikirim ke `/founder-analytics/stream`,
   SQL yang ditulis model dijalankan, SQL emas dijalankan, lalu **hasilnya**
   yang dibandingkan. Query yang ditulis berbeda tapi mengembalikan angka sama
   tetap lulus.

Plus penilaian grafik: apakah model memilih bentuk yang masuk akal, dan —
sama pentingnya — apakah ia memilih TIDAK menggambar saat pertanyaannya cuma
satu angka.

Cara run:
    docker compose exec ai-chat python evals/founder_analytics/run.py
    docker compose exec ai-chat python evals/founder_analytics/run.py --skip-chat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import yaml

from app.config.settings import settings

GOLDEN = Path(__file__).parent / "golden.yaml"
ADVERSARIAL = Path(__file__).parent / "adversarial.yaml"

API = settings.PERI_API_URL.rstrip("/") if settings.PERI_API_URL else ""
AI_CHAT = "http://localhost:8003"
EXECUTE = f"{API}/api/v1/internal/agent/founder-query/execute"
CATALOG = f"{API}/api/v1/internal/agent/founder-query/catalog"

H_API = {"X-Internal-Secret": settings.INTERNAL_SECRET}
H_CHAT = {"X-Internal-Secret": settings.INTERNAL_SECRET}


@dataclass
class Hasil:
    id: str
    lulus: bool
    catatan: str = ""

    # Di bawah ini bukan hiasan. Sebelum ada medan-medan ini, sebuah kasus merah
    # cuma berbunyi "baris emas=6 model=6" — enam baris di dua sisi, isinya beda,
    # beda di mana tidak pernah dicetak. Perbandingan empat model jadi tiga angka
    # yang tidak bisa dipertanggungjawabkan.
    pertanyaan: str = ""
    mode: str = ""
    sql_emas: str = ""
    sql_model: str = ""
    beda: dict = field(default_factory=dict)
    chart_spec: dict | None = None


@dataclass
class Laporan:
    keamanan: list[Hasil] = field(default_factory=list)
    akurasi: list[Hasil] = field(default_factory=list)
    grafik: list[Hasil] = field(default_factory=list)
    gaps: list[Hasil] = field(default_factory=list)


# =============================================================================
# Eksekusi SQL lewat api
# =============================================================================


async def jalankan(client: httpx.AsyncClient, sql: str) -> dict:
    """Jalankan SQL lewat api. Tidak pernah melempar — galat jadi hasil biasa.

    Sama alasannya dengan `tanya()`: satu galat transport tidak boleh membuang
    seluruh putaran. Bedanya di sini juga menyangkut FASE KEAMANAN, dan fase itu
    menghentikan — kalau ia mati karena jaringan, yang terbaca "gerbang jebol".
    """
    try:
        r = await client.post(
            EXECUTE,
            json={"sql": sql, "question": "[eval]", "attempt": 1},
            headers=H_API,
            timeout=60,
        )
    except Exception as e:
        return {"ok": False, "error_type": f"transport_{type(e).__name__}"}
    if r.status_code != 200:
        return {"ok": False, "error_type": f"http_{r.status_code}"}
    return r.json()


def _satu_baris(sql: str, batas: int = 220) -> str:
    """SQL multi-baris jadi satu baris, dipendekkan — buat dicetak di terminal.

    Yang utuh tetap ada di artefak `--keluar`; ini cuma supaya ringkasannya
    terbaca tanpa menggulung layar.
    """
    rapat = " ".join((sql or "").split())
    return rapat if len(rapat) <= batas else rapat[: batas - 1] + "…"


#: Tanggal yang pulang sebagai timestamp tengah malam UTC. Terjadi saat model
#: memakai `generate_series` tanpa `::date` — kolomnya jadi timestamptz, isinya
#: tetap tanggal yang sama.
_TENGAH_MALAM = re.compile(r"^(\d{4}-\d{2}-\d{2})T00:00:00(\+00:00|Z)?$")


def _normalkan(baris: list) -> list:
    """Urutkan dan bulatkan supaya beda gaya query tidak dihitung beda hasil.

    Termasuk menyamakan `2026-07-01T00:00:00+00:00` dengan `2026-07-01`.
    Keduanya tanggal yang SAMA — yang berbeda cuma tipe kolomnya, dan itu
    urusan gaya query, bukan kebenaran jawaban. Tanpa ini, model yang menulis
    `generate_series(...)` tanpa `::date` kehilangan poin untuk 31 baris yang
    angkanya identik sampai desimal terakhir. Itu pernah terjadi dan sempat
    terbaca sebagai "modelnya kurang akurat".

    Sengaja sempit: HANYA tengah malam UTC persis. Jam berapa pun selain itu
    memang informasi yang berbeda dan tetap dihitung berbeda.
    """

    def sel(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, str):
            cocok = _TENGAH_MALAM.match(v)
            if cocok:
                return cocok.group(1)
        return v

    return sorted(
        [[sel(v) for v in b] for b in baris],
        key=lambda b: json.dumps(b, default=str, sort_keys=True),
    )


#: Berapa baris beda yang dicetak per sisi. Cukup untuk mengenali polanya
#: (nilai geser, kolom ketuker, NULL lawan 0) tanpa membanjiri terminal.
MAKS_BARIS_BEDA = 3


def _beda_baris(be: list, bm: list) -> dict:
    """Baris mana yang cuma ada di satu sisi, setelah dinormalkan.

    Multiset, bukan himpunan: baris kembar yang jumlahnya beda tetap terlihat.
    """
    ne, nm = _normalkan(be), _normalkan(bm)
    sisa = list(nm)
    hanya_emas = []
    for b in ne:
        if b in sisa:
            sisa.remove(b)
        else:
            hanya_emas.append(b)
    return {
        "hanya_emas": hanya_emas[:MAKS_BARIS_BEDA],
        "hanya_model": sisa[:MAKS_BARIS_BEDA],
        "jumlah_hanya_emas": len(hanya_emas),
        "jumlah_hanya_model": len(sisa),
    }


def bandingkan(mode: str, emas: dict, model: dict) -> tuple[bool, str, dict]:
    if not model.get("ok"):
        return False, f"SQL model gagal: {model.get('error_type')}", {
            "pesan_model": model.get("error") or model.get("message") or ""
        }
    if not emas.get("ok"):
        return (
            False,
            f"SQL EMAS gagal: {emas.get('error_type')} — perbaiki golden.yaml",
            {"pesan_emas": emas.get("error") or emas.get("message") or ""},
        )

    be, bm = emas.get("rows") or [], model.get("rows") or []

    if mode == "scalar":
        if not be or not bm:
            return (
                (be == bm),
                "dua-duanya kosong" if be == bm else "salah satu kosong",
                {"emas": be[:1], "model": bm[:1]},
            )
        a, b = be[0][0], bm[0][0]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            cocok = abs(float(a) - float(b)) < 0.05
        else:
            cocok = a == b
        return cocok, f"emas={a} model={b}", {} if cocok else {"emas": a, "model": b}

    if mode == "shape":
        cocok = len(be) == len(bm)
        catatan = f"baris emas={len(be)} model={len(bm)}"
        return cocok, catatan, {} if cocok else _beda_baris(be, bm)

    cocok = _normalkan(be) == _normalkan(bm)
    catatan = f"baris emas={len(be)} model={len(bm)}"
    if cocok:
        return True, catatan, {}

    beda = _beda_baris(be, bm)
    # Jumlah baris sama tapi isinya beda adalah kasus yang paling sering
    # disalahartikan sebagai "modelnya salah" — katakan terang-terangan.
    if len(be) == len(bm):
        catatan = f"{catatan} — jumlah sama, isi beda di {beda['jumlah_hanya_emas']} baris"
    return False, catatan, beda


# =============================================================================
# Fase 1 — keamanan
# =============================================================================


async def fase_keamanan(client: httpx.AsyncClient) -> list[Hasil]:
    data = yaml.safe_load(ADVERSARIAL.read_text(encoding="utf-8"))
    hasil: list[Hasil] = []

    print("\n=== Fase 1 — keamanan (tanpa LLM) ===")

    for kasus in data.get("cases", []):
        r = await jalankan(client, kasus["sql"])
        tipe = r.get("error_type") or ""
        # Galat jaringan BUKAN penolakan. Tanpa baris ini, api yang mati
        # membuat seluruh fase keamanan hijau — 28/28 karena tidak ada yang
        # sampai ke penjaganya. Hijau paling berbahaya adalah hijau yang
        # didapat karena tidak ada yang diuji.
        ditolak = (not r.get("ok")) and not tipe.startswith("transport_")
        sesuai = ditolak and (
            not kasus.get("expect") or tipe in kasus["expect"]
        )
        hasil.append(
            Hasil(
                kasus["id"],
                sesuai,
                "LOLOS — GERBANG JEBOL" if not ditolak else f"ditolak {tipe}",
            )
        )
        tanda = "OK  " if sesuai else "GAGAL"
        print(f"  {tanda} {kasus['id']:38} {hasil[-1].catatan}")

    for kasus in data.get("harus_lolos", []):
        r = await jalankan(client, kasus["sql"])
        lolos = bool(r.get("ok"))
        hasil.append(
            Hasil(
                kasus["id"],
                lolos,
                f"{r.get('row_count', 0)} baris"
                if lolos
                else f"DITOLAK {r.get('error_type')} — penjaganya terlalu lebar",
            )
        )
        tanda = "OK  " if lolos else "GAGAL"
        print(f"  {tanda} {kasus['id']:38} {hasil[-1].catatan}")

    return hasil


# =============================================================================
# Fase 2 — akurasi lewat stream
# =============================================================================


async def tanya(client: httpx.AsyncClient, pertanyaan: str) -> dict:
    """Kirim satu pertanyaan, kumpulkan SQL, grafik, dan jawabannya.

    TIDAK PERNAH melempar. Galat transport dikembalikan sebagai `meta.failure`.

    Sebelumnya `httpx.ReadError` di satu pertanyaan **menjatuhkan seluruh
    putaran** — 14 kasus yang sudah selesai ikut hilang, ringkasannya tidak
    pernah tercetak, dan artefaknya tidak pernah ditulis. Dari luar itu terbaca
    seperti model yang diam-diam dilewati. Satu pertanyaan yang bermasalah
    adalah satu kasus merah, bukan alasan membuang 20 kasus lainnya.
    """
    keluar = {
        "sql": None,
        "chart": None,
        "jawaban": "",
        "meta": {},
        "pii": [],
        "row_count": 0,
    }
    try:
        return await _tanya(client, pertanyaan, keluar)
    except Exception as e:
        keluar["meta"] = {"failure": f"{type(e).__name__}: {str(e)[:160]}".strip(": ")}
        return keluar


async def _tanya(client: httpx.AsyncClient, pertanyaan: str, keluar: dict) -> dict:
    async with client.stream(
        "POST",
        f"{AI_CHAT}/founder-analytics/stream",
        json={"question": pertanyaan, "history": []},
        headers=H_CHAT,
        timeout=180,
    ) as r:
        if r.status_code != 200:
            keluar["meta"] = {"failure": f"http_{r.status_code}"}
            return keluar
        async for baris in r.aiter_lines():
            if not baris.startswith("data: "):
                continue
            try:
                ev = json.loads(baris[6:])
            except json.JSONDecodeError:
                continue
            t, d = ev.get("event"), ev.get("data")
            if t == "sql":
                keluar["sql"] = (d or {}).get("sql")
            elif t == "chart":
                keluar["chart"] = d
            elif t == "token":
                keluar["jawaban"] += d or ""
            elif t == "data":
                keluar["pii"] = (d or {}).get("pii_datasets") or []
                keluar["row_count"] = int((d or {}).get("row_count") or 0)
            elif t == "meta":
                keluar["meta"] = d or {}
    return keluar


#: Di bawah ini grafik memang TIDAK boleh digambar — satu titik bukan tren,
#: satu batang bukan perbandingan. Sama dengan `MIN_BARIS` di `chart_compile`.
MIN_BARIS_GRAFIK = 2


def nilai_grafik(
    diharapkan: str, spec: dict | None, jumlah_baris: int
) -> tuple[bool, str]:
    ada = spec is not None
    if diharapkan == "none":
        return (not ada), "tidak digambar" if not ada else "digambar padahal tidak perlu"

    # `chart` di golden.yaml menyatakan bentuk yang MASUK AKAL untuk pertanyaan
    # itu — tapi apakah grafiknya layak digambar sama sekali bergantung pada
    # data yang kebetulan ada. Di lingkungan dengan satu sekolah dan satu
    # fasilitas, penjaga menolak menggambar, dan itu perilaku yang BENAR.
    # Menghitungnya gagal berarti menghukum penjaga karena bekerja.
    if not ada and jumlah_baris < MIN_BARIS_GRAFIK:
        return True, f"tidak digambar — cuma {jumlah_baris} baris, memang tidak layak"

    if not ada:
        return False, "tidak digambar padahal seharusnya"
    mark = spec.get("mark")
    # `area` dan `line` sama-sama sah untuk deret waktu; `bar` untuk kategori.
    setara = {"line": {"line", "area"}, "area": {"line", "area"}, "bar": {"bar"}}
    cocok = mark in setara.get(diharapkan, {diharapkan})
    return cocok, f"harap {diharapkan}, dapat {mark}"


async def fase_akurasi(client: httpx.AsyncClient) -> tuple[list[Hasil], list[Hasil], list[Hasil]]:
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    akurasi: list[Hasil] = []
    grafik: list[Hasil] = []
    gaps: list[Hasil] = []

    print("\n=== Fase 2 — akurasi (butuh LLM) ===")

    for kasus in data.get("cases", []):
        jawaban = await tanya(client, kasus["question"])
        sql_model = jawaban["sql"]

        mode = kasus.get("mode", "exact")

        if not sql_model:
            akurasi.append(
                Hasil(
                    kasus["id"],
                    False,
                    f"nol SQL — {jawaban['meta'].get('failure')}",
                    pertanyaan=kasus["question"],
                    mode=mode,
                    sql_emas=kasus["sql"],
                )
            )
            print(f"  GAGAL {kasus['id']:34} {akurasi[-1].catatan}")
            continue

        emas = await jalankan(client, kasus["sql"])
        model = await jalankan(client, sql_model)
        lulus, catatan, beda = bandingkan(mode, emas, model)
        akurasi.append(
            Hasil(
                kasus["id"],
                lulus,
                catatan,
                pertanyaan=kasus["question"],
                mode=mode,
                sql_emas=kasus["sql"],
                sql_model=sql_model,
                beda=beda,
                chart_spec=jawaban["chart"],
            )
        )
        print(f"  {'OK  ' if lulus else 'GAGAL'} {kasus['id']:34} {catatan}")

        g_lulus, g_catatan = nilai_grafik(
            kasus.get("chart", "none"), jawaban["chart"], jawaban["row_count"]
        )
        grafik.append(
            Hasil(
                kasus["id"],
                g_lulus,
                g_catatan,
                pertanyaan=kasus["question"],
                chart_spec=jawaban["chart"],
            )
        )

        if kasus.get("expect_pii"):
            benar = bool(jawaban["pii"])
            akurasi.append(
                Hasil(
                    kasus["id"] + "-pii",
                    benar,
                    "dataset ber-PII dipilih" if benar else "TIDAK memilih dataset ber-PII",
                )
            )

    for kasus in data.get("known_gaps", []):
        jawaban = await tanya(client, kasus["question"])
        # Tidak dihitung lulus atau gagal — cuma diukur supaya angkanya jujur.
        gaps.append(
            Hasil(
                kasus["id"],
                False,
                "model menulis SQL" if jawaban["sql"] else "model menolak (diharapkan)",
            )
        )

    return akurasi, grafik, gaps


# =============================================================================
# main
# =============================================================================


async def main(skip_chat: bool, keluar: str | None = None) -> int:
    if not API:
        print("PERI_API_URL kosong — eval butuh peri-bugi-api hidup.")
        return 2

    laporan = Laporan()
    katalog: dict = {}

    async with httpx.AsyncClient() as client:
        r = await client.get(CATALOG, headers=H_API, timeout=30)
        if r.status_code != 200:
            print(f"Katalog tidak terjangkau (HTTP {r.status_code}). Berhenti.")
            return 2
        katalog = r.json()
        print(f"katalog versi {katalog['version']}, {len(katalog['datasets'])} dataset")

        laporan.keamanan = await fase_keamanan(client)
        bocor = [h for h in laporan.keamanan if not h.lulus]
        if bocor:
            print(f"\nFASE KEAMANAN GAGAL — {len(bocor)} kasus. Akurasi TIDAK diukur.")
            for h in bocor:
                print(f"  - {h.id}: {h.catatan}")
            if keluar:
                tulis_artefak(keluar, laporan, katalog)
            return 1

        if skip_chat:
            print("\n--skip-chat: fase akurasi dilewati.")
            if keluar:
                tulis_artefak(keluar, laporan, katalog)
            return 0

        (
            laporan.akurasi,
            laporan.grafik,
            laporan.gaps,
        ) = await fase_akurasi(client)

    print("\n=== Ringkasan ===")
    for nama, daftar in (
        ("keamanan", laporan.keamanan),
        ("akurasi", laporan.akurasi),
        ("grafik", laporan.grafik),
    ):
        lulus = sum(1 for h in daftar if h.lulus)
        print(f"  {nama:9} {lulus}/{len(daftar)}")

    # Angka ringkasan tanpa daftar yang gagal tidak bisa ditindaklanjuti —
    # "16/19" tidak memberi tahu grafik mana yang perlu diperbaiki. Dan daftar
    # yang gagal tanpa SQL model juga tidak: "baris emas=6 model=6" tidak
    # memberi tahu siapa yang salah, modelnya atau soal emasnya.
    for nama, daftar in (("akurasi", laporan.akurasi), ("grafik", laporan.grafik)):
        gagal = [h for h in daftar if not h.lulus]
        if gagal:
            print(f"\n  {nama} yang gagal:")
            for h in gagal:
                print(f"    - {h.id}: {h.catatan}")
                if h.sql_model:
                    print(f"        SQL model : {_satu_baris(h.sql_model)}")
                    print(f"        SQL emas  : {_satu_baris(h.sql_emas)}")
                for sisi, kunci in (("hanya emas ", "hanya_emas"), ("hanya model", "hanya_model")):
                    baris = h.beda.get(kunci)
                    if baris:
                        jumlah = h.beda.get(f"jumlah_{kunci}", len(baris))
                        lebih = f" (+{jumlah - len(baris)} lagi)" if jumlah > len(baris) else ""
                        print(f"        {sisi}: {json.dumps(baris, default=str)}{lebih}")
                if h.beda.get("emas") is not None and "hanya_emas" not in h.beda:
                    print(f"        nilai     : emas={h.beda['emas']} model={h.beda.get('model')}")

    if laporan.gaps:
        print("\n  celah yang diketahui (diukur, tidak dihitung):")
        for h in laporan.gaps:
            print(f"    - {h.id}: {h.catatan}")

    if keluar:
        tulis_artefak(keluar, laporan, katalog)
        print(f"\n  artefak ditulis: {keluar}")

    gagal = [h for h in laporan.akurasi if not h.lulus]
    return 1 if gagal else 0


def tulis_artefak(path: str, laporan: Laporan, katalog: dict) -> None:
    """Simpan hasil per kasus ke JSON.

    Perbandingan model sebelumnya cuma hidup di terminal dan di transkrip sesi;
    sekali sesinya hilang, tidak ada yang bisa diaudit. Ini yang membuat
    kesimpulan "model besar kalah" bertahan berhari-hari tanpa ada yang bisa
    memeriksanya ulang.
    """

    def satu(h: Hasil) -> dict:
        d = asdict(h)
        return {k: v for k, v in d.items() if v not in ("", {}, None)} | {
            "id": h.id,
            "lulus": h.lulus,
        }

    isi = {
        "model": settings.FOUNDER_SQL_MODEL or settings.GEMINI_MODEL,
        "katalog_versi": katalog.get("version"),
        "ringkasan": {
            nama: {
                "lulus": sum(1 for h in daftar if h.lulus),
                "total": len(daftar),
            }
            for nama, daftar in (
                ("keamanan", laporan.keamanan),
                ("akurasi", laporan.akurasi),
                ("grafik", laporan.grafik),
            )
        },
        "keamanan": [satu(h) for h in laporan.keamanan],
        "akurasi": [satu(h) for h in laporan.akurasi],
        "grafik": [satu(h) for h in laporan.grafik],
        "celah": [satu(h) for h in laporan.gaps],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(isi, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-chat", action="store_true", help="keamanan saja")
    p.add_argument(
        "--keluar",
        metavar="PATH",
        help="tulis hasil per kasus ke JSON (SQL model, baris yang beda, niat grafik)",
    )
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.skip_chat, args.keluar)))
