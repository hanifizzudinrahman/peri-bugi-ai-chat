"""
Node: generate
Generate response dari LLM dengan streaming token per token.
"""
import json
import time
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.config.llm import get_llm, get_model_name, get_provider_name
from app.schemas.chat import (
    AgentState,
    LLMCallLogPayload,
    make_clarify_event,
    make_thinking_event,
    make_token_event,
)


def _build_system_prompt(state: AgentState) -> str:
    # Cek apakah ada override langsung (dari RnD)
    if "_override_system" in state.get("prompts", {}):
        return state["prompts"]["_override_system"]

    persona = state.get("prompts", {}).get(
        "persona_system",
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak dari aplikasi Peri Bugi. "
        "Kamu ramah, sabar, dan berbicara dengan bahasa yang mudah dipahami orang tua Indonesia. "
        "Jangan pernah memberikan diagnosis medis langsung. "
        "Selalu sarankan konsultasi dokter gigi untuk masalah serius.",
    )

    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"
    child_age = f"{child.get('age_years', '?')} tahun" if child.get("age_years") else "?"

    system = persona.replace("{user_name}", user_name)
    system = system.replace("{child_name}", child_name)
    system = system.replace("{child_age}", child_age)

    brushing = ctx.get("brushing")
    if brushing:
        system += (
            f"\n\nData sikat gigi {child_name} saat ini: "
            f"streak {brushing.get('current_streak', 0)} hari, "
            f"rekor terbaik {brushing.get('best_streak', 0)} hari."
        )

    mata_peri = ctx.get("mata_peri_last_result")
    if mata_peri and mata_peri.get("summary_text"):
        system += (
            f"\n\nHasil scan gigi terakhir {child_name} "
            f"({mata_peri.get('scan_date', 'tidak diketahui')}): "
            f"{mata_peri.get('summary_text')}. "
            f"Status: {mata_peri.get('summary_status', 'tidak diketahui')}."
        )

    docs = state.get("retrieved_docs", [])
    if docs:
        docs_text = "\n\n".join(docs[:3])
        system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

    image_analysis = state.get("image_analysis")
    if image_analysis:
        system += (
            f"\n\nHasil analisis gambar gigi yang dikirim user: "
            f"{image_analysis.get('summary', 'tidak tersedia')}."
        )

    return system


def _build_messages(state: AgentState, system_prompt: str) -> list:
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in state.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            lc_messages.append(AIMessage(content=content))
    return lc_messages


async def check_clarification_node(state: AgentState) -> None:
    if state.get("clarification_selected"):
        state["needs_clarification"] = False
        return

    clarify_prompt_template = state.get("prompts", {}).get("clarify_decision", "")
    if not clarify_prompt_template:
        state["needs_clarification"] = False
        return

    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    last_message = user_messages[-1].get("content", "") if user_messages else ""

    history_preview = " | ".join([
        f"{m.get('role', '?')}: {m.get('content', '')[:50]}"
        for m in state.get("messages", [])[-4:]
        if isinstance(m, dict)
    ])

    prompt = clarify_prompt_template.replace("{user_message}", last_message)
    prompt = prompt.replace("{conversation_history}", history_preview)

    llm = get_llm(
        temperature=0,
        max_tokens=200,
        streaming=False,
        provider=state.get("llm_provider_override"),
        model=state.get("llm_model_override"),
    )

    start_time = time.monotonic()
    success = True
    error_msg: str | None = None

    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        parsed = json.loads(result.content.strip())
        if parsed.get("needs_clarification") and parsed.get("options"):
            state["needs_clarification"] = True
            state["clarification_data"] = parsed
        else:
            state["needs_clarification"] = False
    except (json.JSONDecodeError, Exception) as e:
        state["needs_clarification"] = False
        success = False
        error_msg = str(e)

    latency_ms = int((time.monotonic() - start_time) * 1000)

    log = LLMCallLogPayload(
        prompt_key="clarify_decision",
        model=get_model_name(
            provider=state.get("llm_provider_override"),
            model=state.get("llm_model_override"),
        ),
        provider=get_provider_name(provider=state.get("llm_provider_override")),
        node="check_clarification",
        latency_ms=latency_ms,
        success=success,
        error_message=error_msg,
    )
    state["llm_call_logs"].append(log.model_dump())


async def generate_node(state: AgentState) -> AsyncIterator[str]:
    thinking_step = len(state.get("thinking_steps", [])) + 1

    intent = state.get("intent", "dental_qa")
    thinking_label = {
        "dental_qa": "Mencari informasi kesehatan gigi...",
        "context_query": f"Mengecek data {_get_child_name(state)}...",
        "image": "Menganalisis gambar gigi...",
        "clarification_answer": "Memproses jawabanmu...",
        "smalltalk": "Menyiapkan respons...",
    }.get(intent, "Menyiapkan respons...")

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=False)

    if state.get("needs_clarification") and state.get("clarification_data"):
        clarify = state["clarification_data"]
        yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
        yield make_clarify_event(
            question=clarify.get("question", ""),
            options=clarify.get("options", []),
            allow_multiple=clarify.get("allow_multiple", False),
        )
        state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})
        return

    system_prompt = _build_system_prompt(state)
    lc_messages = _build_messages(state, system_prompt)

    if state.get("include_prompt_debug"):
        state["prompt_debug"] = {
            "system": system_prompt,
            "messages": [
                {"role": type(m).__name__, "content": m.content}
                for m in lc_messages
            ],
        }

    _provider = state.get("llm_provider_override") or None
    _model = state.get("llm_model_override") or None

    llm = get_llm(
        streaming=True,
        provider=_provider,
        model=_model,
        temperature=state.get("llm_temperature_override"),
        max_tokens=state.get("llm_max_tokens_override"),
    )

    # ── Timing breakdown ──────────────────────────────────────────────────
    # t_start      : saat kita mulai kirim request ke LLM
    # t_first_token: saat token pertama keluar (TTFT)
    # t_end        : saat token terakhir keluar (total latency)
    #
    # TTFT tinggi = model sedang load ke VRAM (cold start Ollama)
    # TTFT rendah tapi total tinggi = model jalan tapi output panjang
    t_start = time.monotonic()
    t_first_token: float | None = None
    full_response = ""
    output_tokens = 0
    success = True
    error_msg: str | None = None

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
    state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})

    try:
        async for chunk in llm.astream(lc_messages):
            token = chunk.content
            if not token:
                continue

            if t_first_token is None:
                t_first_token = time.monotonic()

            full_response += token
            output_tokens += 1
            yield make_token_event(token)

    except Exception as e:
        success = False
        error_msg = str(e)

    t_end = time.monotonic()

    # Semua timing dalam ms
    total_latency_ms = int((t_end - t_start) * 1000)
    ttft_ms = int((t_first_token - t_start) * 1000) if t_first_token else None
    # generation_ms = waktu dari token pertama sampai token terakhir
    # = pure LLM throughput tanpa overhead cold start
    generation_ms = int((t_end - t_first_token) * 1000) if t_first_token else None
    # tokens_per_second = throughput generate (hanya bagian setelah token pertama)
    tps = round(output_tokens / (generation_ms / 1000), 1) if generation_ms and generation_ms > 0 else None

    state["final_response"] = full_response
    state["llm_metadata"] = {
        "model": get_model_name(provider=_provider, model=_model),
        "provider": get_provider_name(provider=_provider),
        "intent": intent,
        # Total: dari kirim request sampai token terakhir
        "latency_ms": total_latency_ms,
        # TTFT: dari kirim request sampai token PERTAMA
        # Tinggi = cold start (model baru di-load ke VRAM)
        # Rendah = model sudah warm di VRAM
        "ttft_ms": ttft_ms,
        # Generation: dari token pertama sampai token terakhir
        # Ini adalah pure LLM throughput
        "generation_ms": generation_ms,
        # Tokens per second setelah warm (useful untuk compare model)
        "tokens_per_second": tps,
        "output_tokens_approx": output_tokens,
    }

    log = LLMCallLogPayload(
        prompt_key="generate",
        model=get_model_name(provider=_provider, model=_model),
        provider=get_provider_name(provider=_provider),
        node="generate",
        output_tokens=output_tokens,
        latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        success=success,
        error_message=error_msg,
        metadata={
            "intent": intent,
            "generation_ms": generation_ms,
            "tokens_per_second": tps,
            "temperature": state.get("llm_temperature_override"),
            "max_tokens": state.get("llm_max_tokens_override"),
        },
    )
    state["llm_call_logs"].append(log.model_dump())


def _get_child_name(state: AgentState) -> str:
    ctx = state.get("user_context", {})
    child = ctx.get("child") or {}
    return child.get("nickname") or child.get("full_name") or "si kecil"
