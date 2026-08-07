"""Klien LLM penulis kode dasbor.

Modul sendiri, bersebelahan dengan `gemini_direct.py`, dan pemisahannya bukan
soal kerapian: kunci API-nya beda, modelnya beda (penulis kode, bukan penulis
SQL), anggaran keluarannya sepuluh kali lipat, dan kalau salah satunya kena
batas kuota, yang lain harus tetap jalan. Menumpangkannya di modul yang sama
berarti satu `settings.GEMINI_API_KEY` melayani dua beban kerja yang biayanya
harus bisa dipisahkan saat salah satunya meledak.

Kenapa httpx polos, bukan `langchain-anthropic`
-----------------------------------------------
Tiga alasan, berurut bobot:

1. **Pin-nya.** `langchain-core` dipaku `0.3.28`, dan `gemini_direct.py` sudah
   mencatat apa yang harus dinaikkan untuk melepasnya. Adaptor yang cocok
   dengan pin itu adalah adaptor lama, dan setiap knob API baru sesudahnya
   tergantung bump dependensi yang sudah diputuskan tidak bisa dilakukan.

2. **Repo ini sudah pernah kena persis kegagalan itu, dan tak terlihat.**
   `model_kwargs` dibuang diam-diam oleh adaptor ber-`extra="ignore"` — nol
   error, nol peringatan, tagihan lima kali lipat, dan eval yang terbaca
   "model besar lebih bodoh". Menambah lapisan adaptor lagi antara kita dan
   penyedia berarti menyiapkan ulang panggung untuk kesalahan yang sama.

3. **`httpx` sudah jadi dependensi.** Nol paket baru. Tiga format kabelnya
   masing-masing sekitar empat puluh baris dan tidak menyembunyikan apa pun.

Keluaran terstruktur, per penyedia
----------------------------------
Ketiganya diberi JSON Schema yang sama, tapi mekanismenya berbeda dan
perbedaannya penting:

- **Gemini** — `response_schema` di `GenerateContentConfig`. Inilah alasan jalur
  founder bisa diandalkan sejak awal; prompt yang meminta JSON cuma imbauan.
- **Anthropic** — **tool call yang DIPAKSA** (`tool_choice` bertipe `tool`),
  bukan "tolong balas JSON". Itu satu-satunya mekanisme Claude yang
  sungguh-sungguh membatasi bentuk, dan hasilnya datang sudah ter-parse di
  `content[i].input`.
- **OpenAI-compatible** — `response_format` `json_schema` dengan `strict: true`
  kalau endpointnya mendukung, `json_object` kalau tidak.
  `CODER_LLM_JSON_MODE` yang memilih, dan itulah satu-satunya perbedaan antara
  memakai OpenAI resmi dan memakai DeepSeek, GLM, Moonshot, atau OpenRouter.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)

NAMA_TOOL = "emit_dashboard"

_DEFAULT_BASE = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
}

#: Keluarga Gemini yang memakai `thinking_level`. Disalin dari
#: `gemini_direct._KELUARGA_LEVEL` dengan sengaja, BUKAN diimpor: modul itu
#: memakai MINIMAL sebagai patokan karena penalaran merusak SQL, dan di sini
#: patokannya justru sebaliknya. Mengimpor konstantanya akan mengundang orang
#: menyatukan dua kebijakan yang memang harus berbeda.
_KELUARGA_LEVEL = ("gemini-3", "gemini-4")
_KELUARGA_BUDGET = ("2.5",)


@dataclass
class HasilKode:
    spec: dict | None = None
    raw: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    terpotong: bool = False
    catatan: list[str] = field(default_factory=list)


def tersedia() -> bool:
    """Kunci dan nama model dua-duanya terisi?

    Nama model sengaja ikut dihitung. Kunci tanpa model menghasilkan panggilan
    ke model kosong yang gagal dengan pesan penyedia — jauh dari sebabnya.
    """
    return bool(settings.CODER_LLM_API_KEY and settings.CODER_LLM_MODEL)


def penyedia() -> str:
    return (settings.CODER_LLM_PROVIDER or "gemini").strip().lower()


def nama_model() -> str:
    return (settings.CODER_LLM_MODEL or "").strip()


def _base_url() -> str:
    if settings.CODER_LLM_BASE_URL:
        return settings.CODER_LLM_BASE_URL.rstrip("/")
    return _DEFAULT_BASE.get(penyedia(), "")


def _ambil_json(teks: str) -> dict | None:
    """Selamatkan objek JSON dari keluaran yang mungkin berbungkus prosa.

    Bukan pengganti keluaran terstruktur — ketiga jalur di bawah sudah memaksa
    bentuknya. Ini jaring untuk endpoint OpenAI-compatible yang mengaku
    mendukung `json_object` lalu tetap membungkusnya dengan pagar ```json.
    """
    if not teks:
        return None
    try:
        return json.loads(teks)
    except json.JSONDecodeError:
        pass

    tanpa_pagar = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", teks.strip())
    try:
        return json.loads(tanpa_pagar)
    except json.JSONDecodeError:
        pass

    mulai = tanpa_pagar.find("{")
    akhir = tanpa_pagar.rfind("}")
    if mulai >= 0 and akhir > mulai:
        try:
            return json.loads(tanpa_pagar[mulai : akhir + 1])
        except json.JSONDecodeError:
            return None
    return None


# ------------------------------------------------------------------- Gemini


async def _gemini(*, system: str, user: str, schema: Any, max_tokens: int,
                  timeout: float) -> HasilKode:
    """SDK `google-genai`, bukan REST mentah.

    `response_schema` adalah alasan seluruh jalur founder bisa diandalkan, dan
    SDK-nya sudah terpasang. Tapi kliennya dibangun SENDIRI dengan
    `CODER_LLM_API_KEY` — `gemini_direct._klien()` memaku `GEMINI_API_KEY`, dan
    memakainya ulang di sini berarti dua beban kerja berbagi satu kuota dan satu
    tagihan yang tidak bisa dipisahkan.
    """
    from google import genai
    from google.genai import types

    model = nama_model()
    hasil = HasilKode(provider="gemini", model=model)

    thinking = None
    nama = model.lower()
    if any(k in nama for k in _KELUARGA_LEVEL):
        thinking = types.ThinkingConfig(
            thinking_level=(settings.CODER_LLM_THINKING_LEVEL or "LOW"),
            include_thoughts=False,
        )
    elif any(k in nama for k in _KELUARGA_BUDGET):
        thinking = types.ThinkingConfig(thinking_budget=1024, include_thoughts=False)
    else:
        hasil.catatan.append(f"model '{model}' tanpa kendali penalaran")

    klien = genai.Client(api_key=settings.CODER_LLM_API_KEY)
    konfig = types.GenerateContentConfig(
        system_instruction=system,
        temperature=settings.CODER_LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=schema,
        **({"thinking_config": thinking} if thinking else {}),
    )

    respons = await klien.aio.models.generate_content(
        model=model, contents=user, config=konfig
    )

    # Bagian bertanda `thought` dibuang di tingkat part, sama seperti di
    # `gemini_direct._ambil_teks` — kalau ikut, JSON-nya gagal di-parse.
    potongan: list[str] = []
    for kandidat in getattr(respons, "candidates", None) or []:
        for part in getattr(getattr(kandidat, "content", None), "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                potongan.append(part.text)
    hasil.raw = "".join(potongan) or (getattr(respons, "text", "") or "")

    for kandidat in getattr(respons, "candidates", None) or []:
        fr = getattr(kandidat, "finish_reason", None)
        nama_fr = getattr(fr, "name", None) or str(fr or "")
        if nama_fr in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS"):
            hasil.terpotong = True
        break

    u = getattr(respons, "usage_metadata", None)
    if u:
        hasil.input_tokens = getattr(u, "prompt_token_count", None)
        hasil.output_tokens = getattr(u, "candidates_token_count", None)

    hasil.spec = _ambil_json(hasil.raw)
    return hasil


# ---------------------------------------------------------------- Anthropic


async def _anthropic(*, system: str, user: str, schema: dict, max_tokens: int,
                     timeout: float) -> HasilKode:
    model = nama_model()
    hasil = HasilKode(provider="anthropic", model=model)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": settings.CODER_LLM_TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        # Tool yang DIPAKSA. "Tolong balas JSON" adalah imbauan; ini kontrak.
        "tools": [
            {
                "name": NAMA_TOOL,
                "description": "Keluarkan spesifikasi dasbor.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": NAMA_TOOL},
    }

    async with httpx.AsyncClient(timeout=timeout) as klien:
        r = await klien.post(
            f"{_base_url()}/v1/messages",
            json=payload,
            headers={
                "x-api-key": settings.CODER_LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        r.raise_for_status()
        data = r.json()

    hasil.terpotong = data.get("stop_reason") == "max_tokens"
    pakai = data.get("usage") or {}
    hasil.input_tokens = pakai.get("input_tokens")
    hasil.output_tokens = pakai.get("output_tokens")

    for blok in data.get("content") or []:
        if blok.get("type") == "tool_use" and blok.get("name") == NAMA_TOOL:
            hasil.spec = blok.get("input")
            hasil.raw = json.dumps(hasil.spec, ensure_ascii=False)
            break
    else:
        hasil.raw = "".join(
            b.get("text", "") for b in (data.get("content") or [])
        )
        hasil.spec = _ambil_json(hasil.raw)

    return hasil


# ------------------------------------------------- OpenAI dan yang kompatibel


async def _openai(*, system: str, user: str, schema: dict, max_tokens: int,
                  timeout: float) -> HasilKode:
    model = nama_model()
    hasil = HasilKode(provider=penyedia(), model=model)

    mode = (settings.CODER_LLM_JSON_MODE or "schema").strip().lower()
    if mode == "schema":
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "dashboard",
                "strict": True,
                "schema": schema,
            },
        }
    else:
        # Endpoint yang cuma mengenal mode JSON longgar. Bentuknya tidak
        # dijamin, jadi `_ambil_json` yang menanggung sisanya.
        response_format = {"type": "json_object"}

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": settings.CODER_LLM_TEMPERATURE,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as klien:
        r = await klien.post(
            f"{_base_url()}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {settings.CODER_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
        data = r.json()

    pakai = data.get("usage") or {}
    hasil.input_tokens = pakai.get("prompt_tokens")
    hasil.output_tokens = pakai.get("completion_tokens")

    pilihan = (data.get("choices") or [{}])[0]
    hasil.terpotong = pilihan.get("finish_reason") == "length"
    hasil.raw = ((pilihan.get("message") or {}).get("content")) or ""
    hasil.spec = _ambil_json(hasil.raw)
    return hasil


# --------------------------------------------------------------------- muka


async def generate_dashboard(*, system: str, user: str, schema_model: Any) -> HasilKode:
    """Satu muka untuk tiga penyedia. Melempar apa adanya kalau transportnya
    gagal — pemanggilnya yang memutuskan bahwa dasbor boleh tidak ada.

    `schema_model` adalah KELAS Pydantic, bukan dict, dan itu penting:

    - **Gemini** menerima kelasnya langsung, dan SDK-nya yang menerjemahkan.
      Memberinya `model_json_schema()` gagal, karena skema itu memuat `$defs`
      dan `$ref` untuk `KpiDef` sementara `response_schema` Gemini tidak
      mengenal referensi.
    - **Anthropic dan OpenAI** memang meminta JSON Schema, dan keduanya
      mengerti `$defs`. Jadi konversinya dilakukan di sini, di titik yang
      membutuhkannya.

    Sengaja TIDAK punya jalur mundur ke penyedia lain. Jatuh diam-diam ke model
    yang berbeda berarti `llm_call_logs` mencatat penyedia yang salah, dan biaya
    Anthropic muncul sebagai Gemini di Pusat Biaya — angkanya tetap masuk akal,
    cuma di kolom yang keliru, dan nol orang akan curiga.
    """
    p = penyedia()
    umum = dict(
        system=system,
        user=user,
        max_tokens=settings.CODER_LLM_MAX_OUTPUT_TOKENS,
        timeout=float(settings.CODER_LLM_TIMEOUT_SECONDS),
    )

    if p == "gemini":
        return await _gemini(schema=schema_model, **umum)

    schema_dict = (
        schema_model.model_json_schema()
        if hasattr(schema_model, "model_json_schema")
        else schema_model
    )
    if p == "anthropic":
        return await _anthropic(schema=schema_dict, **umum)
    if p in ("openai", "openai_compatible", "compatible"):
        return await _openai(schema=schema_dict, **umum)

    raise ValueError(
        f"CODER_LLM_PROVIDER tidak dikenal: '{p}'. "
        "Pilih: gemini | anthropic | openai"
    )
