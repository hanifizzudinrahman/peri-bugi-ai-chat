"""Klien Gemini langsung untuk jalur Tanya Data Founder.

Kenapa tidak lewat LangChain seperti sisa repo ini
--------------------------------------------------
Karena lewat LangChain, kendali penalaran tidak sampai — dan itu bukan dugaan,
melainkan hasil pemeriksaan terhadap pustaka yang terpasang:

    ChatGoogleGenerativeAI punya field model_kwargs : False
    model_config extra                              : ignore
    GenerationConfig SDK lama punya thinking_config : False

`app/config/llm.py` selama ini mengirim konfigurasi penonaktif `thinking` lewat
`model_kwargs`. Medan itu tidak ada, dan pydantic-nya diset `extra="ignore"` —
jadi kuncinya **dibuang diam-diam**, tanpa galat dan tanpa peringatan. SDK
lamanya sendiri memang tidak punya medan `thinking_config` sama sekali. Artinya
penonaktifan itu tidak pernah berlaku, untuk keluarga model mana pun.

Akibatnya terukur saat membandingkan model — angka di bawah ini **sebelum**
modul ini ada:

    gemini-3.1-flash-lite    ~990 token/panggilan   akurasi 19/20
    gemini-3.5-flash        3.898 token/panggilan   eval tidak selesai
    gemini-3-flash-preview  4.843 token/panggilan   akurasi 10/20

Kegagalan model besar berbunyi "SQL tidak bisa di-parse" — isi penalaran ikut
tercetak ke dalam query. `gemini-3.5-flash` bahkan menyalin potongan prompt ke
tengah SQL. Terbaca seperti "model besar ternyata lebih bodoh", padahal yang
terjadi kita tidak pernah punya tuasnya.

Catatan angka: `790` yang sempat tercatat untuk flash-lite adalah **kekurangan
hitung**, bukan penghematan — jalur LangChain tidak melaporkan pemakaian token
saat streaming, jadi node jawaban tercatat nol. Yang sebenarnya selalu ~990.

Dan catatan yang lebih penting, ditemukan belakangan: setelah kendali penalaran
benar-benar sampai, **model besar berhenti gagal karena format**. Sisa selisih
skornya soal menalar data, sebagian malah cacat di katalog dan soal emas kita
sendiri. Jangan pakai berkas ini sebagai bukti "model besar lebih buruk".

Kenapa cuma jalur founder
-------------------------
Memperbaikinya lewat LangChain menuntut `langchain-google-genai >= 3.1`, yang
menuntut `langchain-core >= 1.5` — kita di `0.3.28`. Menaikkannya menyeret
`langgraph`, checkpointer, dan seluruh adaptor, dan itu menyentuh graph Tanya
Peri yang sudah live. Itu proyek tersendiri, dan sudah dicatat sebagai utang di
`docs/OPEN_ITEMS.md`.

Modul ini jalan pintas yang jujur: SDK modern (`google.genai`) dipakai **hanya**
oleh jalur founder, hidup berdampingan dengan yang lama karena namespace-nya
berbeda. Tanya Peri tidak berubah satu baris pun.

Kalau apa pun di sini gagal, pemanggilnya jatuh kembali ke `get_llm()`. Fitur
tetap jalan, cuma tanpa kendali penalaran.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.config.settings import settings

logger = logging.getLogger(__name__)

#: Keluarga yang memakai `thinking_level` (Gemini 3.x ke atas).
_KELUARGA_LEVEL = ("gemini-3", "gemini-4")

#: Keluarga yang masih memakai `thinking_budget` (Gemini 2.5).
_KELUARGA_BUDGET = ("2.5",)

#: Untuk beban kerja ini — satu SELECT, JSON berbentuk pasti — penalaran panjang
#: bukan cuma mubazir, ia merusak: isinya ikut tercetak ke keluaran.
DEFAULT_THINKING_LEVEL = "MINIMAL"


@dataclass
class HasilGenerasi:
    teks: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thought_tokens: int | None = None
    model: str = ""
    #: Bagian bertanda `thought` yang dibuang. Kosong berarti bersih.
    thought_dibuang: int = 0
    catatan: list[str] = field(default_factory=list)


def tersedia() -> bool:
    """SDK modern terpasang dan kuncinya ada?"""
    if not settings.GEMINI_API_KEY:
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _thinking_config(model: str, level: str | None):
    """Bangun ThinkingConfig yang benar untuk keluarga model ini.

    JANGAN pernah mengirim `thinking_level` dan `thinking_budget` bersamaan —
    Google menjawab 400. Karena itu percabangannya eksplisit, bukan menumpuk.
    """
    from google.genai import types

    nama = model.lower()

    if any(k in nama for k in _KELUARGA_LEVEL):
        return types.ThinkingConfig(
            thinking_level=(level or DEFAULT_THINKING_LEVEL),
            include_thoughts=False,
        )

    if any(k in nama for k in _KELUARGA_BUDGET):
        # Keluarga 2.5 tidak mengenal thinking_level.
        return types.ThinkingConfig(thinking_budget=0, include_thoughts=False)

    # Model tanpa mode penalaran — kirim apa pun justru berisiko ditolak.
    #
    # Tapi katakan, jangan diam. Pencocokannya substring, jadi nama model baru
    # yang tidak mengandung pola di atas akan jalan TANPA kendali penalaran sama
    # sekali — persis keadaan yang modul ini dibuat untuk mengakhiri, dan tidak
    # ada satu pun gejala yang kelihatan sampai tagihannya datang.
    logger.warning(
        "[gemini_direct] model '%s' tidak cocok keluarga mana pun — "
        "jalan TANPA kendali thinking. Tambahkan polanya ke _KELUARGA_LEVEL "
        "atau _KELUARGA_BUDGET.",
        model,
    )
    return None


class TeksTerpotong(RuntimeError):
    """Model berhenti karena kehabisan `max_output_tokens`, bukan karena selesai.

    Dibedakan dari galat lain karena obatnya beda dan sebabnya ada di sisi kita.
    Sebelum ada kelas ini, respons terpotong pulang sebagai string kosong, lalu
    dilaporkan ke founder sebagai *"model menyatakan pertanyaan ini tidak
    terjawab oleh katalog"* — sebab teknis menyamar jadi keputusan model.
    """


#: Alasan berhenti yang berarti keluarannya tidak utuh.
_BERHENTI_TERPOTONG = ("MAX_TOKENS", "MAX_OUTPUT_TOKENS")


def _alasan_berhenti(response: Any) -> str:
    for kandidat in getattr(response, "candidates", None) or []:
        fr = getattr(kandidat, "finish_reason", None)
        if fr is None:
            continue
        return getattr(fr, "name", None) or str(fr)
    return ""


def _ambil_teks(response: Any) -> tuple[str, int]:
    """Ambil teks jawaban, BUANG bagian yang bertanda `thought`.

    Ini inti masalahnya. Pustaka lama menggabungkan penalaran dan jawaban jadi
    satu string, sehingga "SELECT ..." bercampur dengan catatan berpikir model
    dan SQL-nya gagal di-parse. Di sini keduanya dipisah di tingkat `part`,
    seperti yang memang dikembalikan API.
    """
    potongan: list[str] = []
    dibuang = 0

    for kandidat in getattr(response, "candidates", None) or []:
        isi = getattr(kandidat, "content", None)
        for part in getattr(isi, "parts", None) or []:
            teks = getattr(part, "text", None)
            if not teks:
                continue
            if getattr(part, "thought", False):
                dibuang += 1
                continue
            potongan.append(teks)

    if potongan:
        return "".join(potongan), dibuang

    # Sebagian respons cuma punya `.text` agregat. Dipakai sebagai cadangan,
    # dan kalau ada bagian thought yang dibuang di atas, jangan pakai ini —
    # nilainya sudah tercampur.
    if dibuang == 0:
        return (getattr(response, "text", "") or ""), 0
    return "", dibuang


def _pemakaian(response: Any) -> dict:
    u = getattr(response, "usage_metadata", None)
    if not u:
        return {}
    return {
        "input_tokens": getattr(u, "prompt_token_count", None),
        "output_tokens": getattr(u, "candidates_token_count", None),
        "total_tokens": getattr(u, "total_token_count", None),
        "thought_tokens": getattr(u, "thoughts_token_count", None),
    }


def _klien():
    from google import genai

    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _config(
    *,
    model: str,
    system: str,
    temperature: float,
    max_tokens: int,
    level: str | None,
    response_schema: Any = None,
):
    from google.genai import types

    kw: dict[str, Any] = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    tc = _thinking_config(model, level)
    if tc is not None:
        kw["thinking_config"] = tc

    # Decoding berbatas. Ini yang membuat pipa tahan model apa pun: bentuk
    # keluarannya dijamin di sisi Google, bukan diharapkan dari prompt lalu
    # ditebak-tebak parser kita. Prompt yang meminta JSON cuma imbauan; model
    # yang lebih cerewet tetap boleh membungkusnya dengan kalimat pengantar.
    if response_schema is not None:
        kw["response_mime_type"] = "application/json"
        kw["response_schema"] = response_schema

    return types.GenerateContentConfig(**kw)


async def generate(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 900,
    thinking_level: str | None = None,
    response_schema: Any = None,
) -> HasilGenerasi:
    """Satu panggilan non-streaming dengan penalaran terkendali.

    `response_schema` boleh model Pydantic; SDK menerjemahkannya sendiri. Kalau
    diisi, keluarannya dijamin JSON yang cocok skema itu.
    """
    nama_model = model or settings.GEMINI_MODEL
    client = _klien()

    response = await client.aio.models.generate_content(
        model=nama_model,
        contents=user,
        config=_config(
            model=nama_model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            level=thinking_level,
            response_schema=response_schema,
        ),
    )

    teks, dibuang = _ambil_teks(response)
    if dibuang:
        logger.info(
            "[gemini_direct] %s bagian bertanda thought dibuang (model %s)",
            dibuang,
            nama_model,
        )

    # Diperiksa SETELAH teks diambil: potongan yang sempat keluar sebelum
    # anggaran habis tetap berguna untuk log, tapi tidak boleh dipakai seolah
    # jawaban utuh. Kalau modelnya berpikir banyak, penalaran itu memakan
    # `max_output_tokens` yang sama — jadi anggaran yang pas untuk model kecil
    # bisa habis sebelum satu karakter jawaban keluar.
    berhenti = _alasan_berhenti(response)
    if berhenti in _BERHENTI_TERPOTONG:
        pakai = _pemakaian(response)
        raise TeksTerpotong(
            f"keluaran terpotong di max_output_tokens={max_tokens} "
            f"(model {nama_model}, penalaran {pakai.get('thought_tokens') or 0} token, "
            f"teks yang sempat keluar {len(teks.strip())} karakter)"
        )

    return HasilGenerasi(
        teks=teks.strip(),
        model=nama_model,
        thought_dibuang=dibuang,
        **_pemakaian(response),
    )


async def stream(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 700,
    thinking_level: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Streaming. Menghasilkan (potongan_teks, pemakaian).

    `pemakaian` terisi hanya di potongan terakhir — di situlah API
    melaporkannya. Jalur LangChain tidak pernah memberikannya sama sekali,
    sehingga node jawaban tercatat dengan token nol di dashboard biaya.
    """
    nama_model = model or settings.GEMINI_MODEL
    client = _klien()

    iterator = await client.aio.models.generate_content_stream(
        model=nama_model,
        contents=user,
        config=_config(
            model=nama_model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            level=thinking_level,
        ),
    )

    async for potongan in iterator:
        teks, _ = _ambil_teks(potongan)
        # Di sini terpotong TIDAK dilempar: jawaban yang separuh jadi tetap
        # lebih berguna bagi founder daripada pesan galat, dan tabel hasilnya
        # tetap utuh di bawahnya. Tapi jangan diam — kalau ini sering muncul,
        # anggaran node jawabannya yang perlu dinaikkan.
        if _alasan_berhenti(potongan) in _BERHENTI_TERPOTONG:
            logger.warning(
                "[gemini_direct] jawaban terpotong di max_output_tokens=%s (model %s)",
                max_tokens,
                nama_model,
            )
        yield teks, _pemakaian(potongan)
