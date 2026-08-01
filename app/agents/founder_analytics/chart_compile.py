"""Kompilasi niat grafik jadi inti spesifikasi Vega-Lite.

Model mengeluarkan `ChartIntent` — bentuk kecil dan berbatas. Berkas ini
mengubahnya jadi `mark` + `encoding` + `title`, dan tidak lebih.

Kenapa begitu, dan bukan "biar model menulis Vega-Lite lalu divalidasi":

1. Spesifikasi yang lolos validasi skema masih bisa jadi grafik yang tidak
   terbaca — sumbu tertukar, kategori 200 warna, deret waktu digambar sebagai
   batang. Validasi skema tidak menangkap satu pun dari itu.
2. `{"data": {"url": "..."}}` adalah permintaan jaringan sungguhan yang
   dijalankan di browser founder yang sedang login. Bahasa ekspresi Vega
   ter-sandbox; pemuat datanya tidak.
3. Loop perbaikan spesifikasi berarti satu panggilan LLM tambahan per giliran,
   plus skema Vega-Lite ~1 MB ikut ke dalam image ai-chat.

Kenapa tanpa `config`, `data`, dan `width`
------------------------------------------
Ketiganya ditambahkan di tempat lain, dan masing-masing punya alasan:

- **`data`** disuntik browser dari `result_json`. Kalau ikut di sini, baris
  yang sama tersimpan dua kali di database dan salah satunya bisa menyimpang.
- **`config`** (warna, font, kisi) ditambahkan `peri-bugi-web`, yang **sudah**
  memiliki token brand di `lib/brand.ts`. Menyalinnya ke sini berarti dua
  sumber kebenaran untuk satu palet, dan yang kedua akan tertinggal.
- **`width`** ikut lebar kontainer, dan itu urusan tata letak, bukan data.

Efek sampingnya bagus: penjaga di `peri-bugi-api` bisa tetap ketat. Ia menolak
`config`, `data`, `transform`, dan `params` tanpa perlu tahu mana yang buatan
server dan mana yang buatan model — karena tidak ada satu pun yang boleh lewat.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.founder_analytics.state import ChartIntent

logger = logging.getLogger(__name__)

#: Palet kategori di `peri-bugi-web` punya **lima** warna, dan jumlah itu bukan
#: selera: palet aslinya diuji dengan validator dataviz terhadap latar terang,
#: dan lima langkah itu satu-satunya susunan yang lolos seluruh pemeriksaan —
#: pita terang, lantai kroma, pemisahan buta warna, dan kontras — tanpa satu
#: pun peringatan.
#:
#: Di atas lima, saluran warna dibuang seluruhnya (grafiknya jadi satu warna).
#: Alternatifnya adalah membangkitkan hue keenam, dan hue yang dibangkitkan
#: tidak pernah lolos pemeriksaan yang sama — ia cuma terlihat berbeda bagi
#: yang menulis kodenya.
MAX_WARNA = 5

#: Satu titik bukan tren, dan satu batang bukan perbandingan.
MIN_BARIS = 2

_MARK = {
    "bar": "bar",
    "bar_grouped": "bar",
    "bar_stacked": "bar",
    "line": "line",
    "area": "area",
    "point": "point",
}

_SORT = {
    "x_asc": "ascending",
    "x_desc": "descending",
    "y_asc": "ascending",
    "y_desc": "descending",
}


class ChartSkipped(Exception):
    """Grafiknya dibatalkan. Tabelnya tetap tampil — jawaban tanpa grafik masih
    berguna, grafik yang salah tidak."""


def _terurai_sebagai_tanggal(nilai: Any) -> bool:
    if not isinstance(nilai, str) or len(nilai) < 8:
        return False
    kepala = nilai[:10]
    return kepala.count("-") == 2 and kepala[:4].isdigit()


def _channel(f, *, default_type: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"field": f.field, "type": f.type or default_type}
    if f.title:
        out["title"] = f.title
    return out


def compile_chart(
    intent: ChartIntent | None,
    *,
    columns: list[str],
    rows: list[list],
) -> dict[str, Any]:
    """Kembalikan inti spesifikasi Vega-Lite, atau lempar `ChartSkipped`."""
    if intent is None or intent.kind == "none":
        raise ChartSkipped(
            (intent.reason if intent else "") or "model memilih tanpa grafik"
        )

    if len(rows) < MIN_BARIS:
        raise ChartSkipped(
            f"cuma {len(rows)} baris — satu titik bukan tren, satu batang "
            "bukan perbandingan"
        )

    if not intent.x or not intent.y:
        raise ChartSkipped("sumbu x atau y tidak lengkap")

    kolom = set(columns)
    for nama, f in (("x", intent.x), ("y", intent.y), ("color", intent.color)):
        if f is None:
            continue
        if f.field not in kolom:
            # Grafik dengan kolom hantu tampil kosong, dan kosong terbaca
            # sebagai "tidak ada data" — kesalahan paling menyesatkan yang bisa
            # dibuat fitur ini.
            raise ChartSkipped(
                f"sumbu {nama} menunjuk kolom '{f.field}' yang tidak ada"
            )

    if intent.y.type != "quantitative":
        raise ChartSkipped("sumbu y harus kuantitatif")

    intent = intent.model_copy(deep=True)

    # Sumbu temporal yang isinya ternyata bukan tanggal menghasilkan sumbu
    # kosong tanpa galat apa pun — turunkan saja jadi nominal.
    if intent.x.type == "temporal" and rows:
        idx = columns.index(intent.x.field)
        if not _terurai_sebagai_tanggal(rows[0][idx]):
            logger.info(
                "[founder_chart] kolom '%s' ditandai temporal tapi isinya bukan "
                "tanggal — diturunkan ke nominal",
                intent.x.field,
            )
            intent.x.type = "nominal"

    encoding: dict[str, Any] = {
        "x": _channel(intent.x),
        "y": _channel(intent.y, default_type="quantitative"),
    }

    if intent.y_aggregate and intent.y_aggregate != "none":
        encoding["y"]["aggregate"] = intent.y_aggregate

    if intent.color:
        idx = columns.index(intent.color.field)
        kardinalitas = len({str(r[idx]) for r in rows})
        if kardinalitas > MAX_WARNA:
            logger.info(
                "[founder_chart] warna dibuang, kardinalitas %s > %s",
                kardinalitas,
                MAX_WARNA,
            )
        else:
            encoding["color"] = _channel(intent.color)
            if intent.kind == "bar_grouped":
                encoding["xOffset"] = {"field": intent.color.field}
            if intent.kind == "bar_stacked":
                encoding["y"]["stack"] = "zero"

    if intent.sort:
        arah = _SORT[intent.sort]
        if intent.sort.startswith("y"):
            encoding["x"]["sort"] = {
                "field": intent.y.field,
                "op": "sum",
                "order": arah,
            }
        else:
            encoding["x"]["sort"] = arah

    # Label tanggal yang menumpuk lebih baik dimiringkan daripada saling
    # menimpa sampai tidak terbaca.
    if intent.x.type == "temporal" and len(rows) > 12:
        encoding["x"].setdefault("axis", {})["labelAngle"] = -45

    spec: dict[str, Any] = {
        "mark": _MARK[intent.kind],
        "encoding": encoding,
        "height": 260,
    }
    if intent.title:
        spec["title"] = intent.title[:160]

    return spec
