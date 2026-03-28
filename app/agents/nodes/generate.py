"""
Node: generate
Generate response dari LLM dengan streaming token per token.
Menggabungkan semua konteks: user data, retrieved docs, image analysis.
Juga handle clarification checkpoint sebelum generate.
"""
import time
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.config.llm import get_llm, get_model_name, get_provider_name
from app.schemas.chat import (
    AgentState,
    make_clarify_event,
    make_thinking_event,
    make_token_event,
)


def _build_system_prompt(state: AgentState) -> str:
    """
    Bangun system prompt dengan inject variabel user context.
    Template dari DB, variabel dari state.
    Fallback ke default hardcoded jika prompt tidak tersedia di DB.
    """
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

    # Tambahkan konteks brushing jika tersedia
    brushing = ctx.get("brushing")
    if brushing:
        system += (
            f"\n\nData sikat gigi {child_name} saat ini: "
            f"streak {brushing.get('current_streak', 0)} hari, "
            f"rekor terbaik {brushing.get('best_streak', 0)} hari."
        )

    # Tambahkan hasil scan terakhir jika tersedia
    mata_peri = ctx.get("mata_peri_last_result")
    if mata_peri and mata_peri.get("summary_text"):
        system += (
            f"\n\nHasil scan gigi terakhir {child_name} "
            f"({mata_peri.get('scan_date', 'tidak diketahui')}): "
            f"{mata_peri.get('summary_text')}. "
            f"Status: {mata_peri.get('summary_status', 'tidak diketahui')}."
        )

    # Tambahkan retrieved docs jika ada (dari RAG)
    docs = state.get("retrieved_docs", [])
    if docs:
        docs_text = "\n\n".join(docs[:3])  # max 3 chunks
        system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

    # Tambahkan hasil image analysis jika ada
    image_analysis = state.get("image_analysis")
    if image_analysis:
        system += (
            f"\n\nHasil analisis gambar gigi yang dikirim user: "
            f"{image_analysis.get('summary', 'tidak tersedia')}."
        )

    return system


def _build_messages(state: AgentState, system_prompt: str) -> list:
    """Convert conversation history ke format LangChain messages."""
    lc_messages = [SystemMessage(content=system_prompt)]

    for msg in state["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            lc_messages.append(AIMessage(content=content))

    return lc_messages


async def check_clarification_node(state: AgentState) -> AsyncIterator[str]:
    """
    Node: check_clarification
    Tentukan apakah perlu klarifikasi dari user sebelum generate.
    Hanya dijalankan untuk intent dental_qa dan context_query.
    """
    # Skip jika sudah ada jawaban klarifikasi
    if state.get("clarification_selected"):
        state["needs_clarification"] = False
        return

    clarify_prompt_template = state.get("prompts", {}).get("clarify_decision", "")
    if not clarify_prompt_template:
        state["needs_clarification"] = False
        return

    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    last_message = user_messages[-1]["content"] if user_messages else ""

    # Ambil history sebagai string singkat
    history_preview = " | ".join([
        f"{m['role']}: {m['content'][:50]}"
        for m in state["messages"][-4:]
    ])

    prompt = clarify_prompt_template.replace("{user_message}", last_message)
    prompt = prompt.replace("{conversation_history}", history_preview)

    llm = get_llm(temperature=0, max_tokens=200, streaming=False)
    import json
    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        parsed = json.loads(result.content.strip())
        if parsed.get("needs_clarification") and parsed.get("options"):
            state["needs_clarification"] = True
            state["clarification_data"] = parsed
        else:
            state["needs_clarification"] = False
    except (json.JSONDecodeError, Exception):
        state["needs_clarification"] = False


async def generate_node(state: AgentState) -> AsyncIterator[str]:
    """
    Node: generate
    Stream response LLM token per token.
    Emit clarification event jika perlu, atau stream tokens langsung.
    """
    thinking_step = len(state.get("thinking_steps", [])) + 1

    # Emit thinking sebelum generate
    intent = state.get("intent", "dental_qa")
    thinking_label = {
        "dental_qa": "Mencari informasi kesehatan gigi...",
        "context_query": f"Mengecek data {_get_child_name(state)}...",
        "image": "Menganalisis gambar gigi...",
        "clarification_answer": "Memproses jawabanmu...",
        "smalltalk": "Menyiapkan respons...",
    }.get(intent, "Menyiapkan respons...")

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=False)

    # Cek apakah perlu clarification
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

    # Build prompt dan messages
    system_prompt = _build_system_prompt(state)
    lc_messages = _build_messages(state, system_prompt)

    llm = get_llm(streaming=True)

    start_time = time.monotonic()
    ttft: float | None = None
    full_response = ""
    input_tokens = 0
    output_tokens = 0

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
    state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})

    # Stream tokens
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if not token:
            continue

        if ttft is None:
            ttft = (time.monotonic() - start_time) * 1000  # ms

        full_response += token
        output_tokens += 1  # estimasi — token sebenarnya dari usage metadata
        yield make_token_event(token)

    latency_ms = int((time.monotonic() - start_time) * 1000)

    # Simpan ke state
    state["final_response"] = full_response
    state["llm_metadata"] = {
        "model": get_model_name(),
        "provider": get_provider_name(),
        "intent": intent,
        "latency_ms": latency_ms,
        "ttft_ms": int(ttft) if ttft else None,
        "output_tokens_approx": output_tokens,
    }


def _get_child_name(state: AgentState) -> str:
    ctx = state.get("user_context", {})
    child = ctx.get("child") or {}
    return child.get("nickname") or child.get("full_name") or "si kecil"