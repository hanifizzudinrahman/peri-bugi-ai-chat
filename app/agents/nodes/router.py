"""
Node: router
Klasifikasi intent user dengan pendekatan hybrid:
1. Rule-based keyword matching (instant, zero latency)
2. Fallback ke LLM classify jika tidak ada keyword match

Intent yang didukung:
- dental_qa           : pertanyaan umum kesehatan gigi
- context_query       : tanya tentang data anak/brushing/scan milik user
- image               : user kirim gambar atau minta analisis foto
- clarification_answer: user menjawab checkpoint klarifikasi sebelumnya
- smalltalk           : sapaan, pertanyaan umum, basa-basi
"""
import re
import time
from typing import AsyncIterator

from app.schemas.chat import AgentState, LLMCallLogPayload, make_thinking_event

# =============================================================================
# Rule-based keyword maps
# Order matters: lebih spesifik di atas
# =============================================================================

_RULES: list[tuple[str, list[str]]] = [
    # clarification_answer — deteksi jawaban checkpoint (huruf/angka tunggal atau "pilih X")
    ("clarification_answer", [
        r"^[a-d]$",
        r"^[1-4]$",
        r"^pilih\s+[a-d1-4]",
        r"^jawaban",
        r"^opsi",
    ]),
    # image — user kirim gambar atau sebut foto/gambar/foto gigi
    ("image", [
        r"foto",
        r"gambar",
        r"image",
        r"scan",
        r"kirim foto",
        r"lihat foto",
    ]),
    # context_query — tanya tentang data personal user
    ("context_query", [
        r"streak",
        r"sikat gigi (anak|dia|si kecil)",
        r"rapot",
        r"hasil scan",
        r"mata peri",
        r"riwayat",
        r"kebiasaan (anak|dia)",
        r"progress",
        r"pencapaian",
        r"achievement",
        r"berapa (hari|streak|sesi)",
    ]),
    # smalltalk — sapaan dan basa-basi
    ("smalltalk", [
        r"^(halo|hai|hi|hey|assalam|pagi|siang|sore|malam)\b",
        r"^apa kabar",
        r"^siapa kamu",
        r"^kamu (siapa|apa|bisa)",
        r"^terima kasih",
        r"^makasih",
        r"^thanks",
        r"^oke$",
        r"^ok$",
    ]),
    # dental_qa — catch-all untuk pertanyaan gigi (paling bawah)
    ("dental_qa", [
        r"gigi",
        r"karies",
        r"berlubang",
        r"plak",
        r"karang gigi",
        r"sikat",
        r"pasta gigi",
        r"fluoride",
        r"dokter gigi",
        r"orthodon",
        r"behel",
        r"gusi",
        r"mulut",
        r"nafas",
        r"bau mulut",
        r"susu",  # gigi susu
        r"permanen",
    ]),
]


def _classify_by_rules(message: str) -> str | None:
    """
    Coba klasifikasi dengan keyword rules.
    Return intent string jika match, None jika tidak ada yang cocok.
    """
    text = message.lower().strip()
    for intent, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return None


async def _classify_by_llm(
    message: str,
    prompt_template: str,
    provider: str | None = None,
    model: str | None = None,
    state: dict | None = None,
) -> tuple[str, LLMCallLogPayload]:
    """
    Fallback: klasifikasi via LLM jika rule-based tidak match.
    Pakai model kecil/cepat, temperature=0 untuk output deterministik.
    Return (intent, log_payload).

    Args:
        state: AgentState — dipakai untuk Langfuse trace metadata.
               Optional: kalau None, trace tetap masuk tapi tanpa
               session_id / user_id metadata.
    """
    from langchain_core.messages import HumanMessage
    from app.config.llm import get_llm, get_model_name, get_provider_name
    from app.config.observability import build_trace_config

    llm = get_llm(
        temperature=0,
        max_tokens=10,
        streaming=False,
        provider=provider,
        model=model,
    )

    prompt = prompt_template.replace("{user_message}", message)

    start_time = time.monotonic()
    ttft: float | None = None
    success = True
    error_msg: str | None = None
    intent = "dental_qa"

    # Per-call trace config (no-op kalau Langfuse disabled)
    trace_config = build_trace_config(state=state, agent_name="router")

    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)], config=trace_config)
        ttft = (time.monotonic() - start_time) * 1000
        intent_raw = result.content.strip().lower()

        # Validasi output LLM — fallback ke dental_qa jika tidak valid
        valid_intents = {"dental_qa", "context_query", "image", "clarification_answer", "smalltalk"}
        intent = intent_raw if intent_raw in valid_intents else "dental_qa"

    except Exception as e:
        success = False
        error_msg = str(e)
        ttft = (time.monotonic() - start_time) * 1000

    latency_ms = int((time.monotonic() - start_time) * 1000)

    log = LLMCallLogPayload(
        prompt_key="router_classify",
        model=get_model_name(provider=provider, model=model),
        provider=get_provider_name(provider=provider),
        node="router",
        latency_ms=latency_ms,
        ttft_ms=int(ttft) if ttft else None,
        output_tokens=1,  # router hanya output 1 token (intent)
        success=success,
        error_message=error_msg,
        metadata={"intent_result": intent},
    )

    return intent, log


async def router_node(state: AgentState) -> AsyncIterator[str]:
    """
    LangGraph node: router.
    Emit thinking event, lalu klasifikasi intent.
    Update state dengan intent yang diklasifikasi.

    FIX: handle message dict yang tidak punya key 'role'
    (defensive check untuk avoid KeyError).
    """
    # Ambil pesan terakhir dari user — defensive: skip message tanpa 'role'
    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    last_message = user_messages[-1].get("content", "") if user_messages else ""

    # Emit thinking step
    yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=False)

    # Check force_intent (RnD mode — skip router sepenuhnya)
    force_intent = state.get("force_intent")
    if force_intent:
        intent = force_intent
        yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=True)
        state["intent"] = intent
        state["thinking_steps"].append({"step": 1, "label": "Memahami pertanyaanmu...", "done": True})
        return

    # 1. Coba rule-based dulu
    intent = _classify_by_rules(last_message)

    if intent is None:
        # 2. Fallback ke LLM
        router_prompt = state.get("prompts", {}).get("router_classify", "")
        if router_prompt:
            intent, log = await _classify_by_llm(
                last_message,
                router_prompt,
                provider=state.get("llm_provider_override"),
                model=state.get("llm_model_override"),
                state=state,  # untuk Langfuse trace metadata
            )
            # Simpan log untuk dikirim ke api
            state["llm_call_logs"].append(log.model_dump())
        else:
            # Tidak ada prompt → default ke dental_qa
            intent = "dental_qa"

    yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=True)

    # Update state
    state["intent"] = intent
    state["thinking_steps"].append({"step": 1, "label": "Memahami pertanyaanmu...", "done": True})
