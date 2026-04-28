"""
Node: supervisor

Phase 1 refactor: return dict (LangGraph signature) instead of yield SSE events.
Logic preserved 1:1 dari supervisor.py original — hanya signature berubah.

SSE thinking events di-emit oleh stream_adapter (app/agents/streaming.py)
based on `thinking_steps` field di state update yang di-return.

Approach: hybrid routing
1. Force intent (RnD only) — short-circuit
2. Image auto-route — kalau ada image_url, route ke mata_peri
3. Rule-based regex (~80 patterns, fast, free)
4. LLM fallback (kalau supervisor_route prompt ada di DB — currently absent)
5. Default ke kb_dental kalau allowed
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from app.agents.state import AgentState, ThinkingStep, LLMCallLog

logger = logging.getLogger(__name__)


# =============================================================================
# Rule-based routing (preserved dari supervisor.py original)
# =============================================================================

_AGENT_RULES: list[tuple[str, list[str]]] = [
    ("mata_peri", [
        r"foto gigi",
        r"scan gigi",
        r"kirim (foto|gambar)",
        r"lihat (foto|gambar)",
        r"image",
        r"\bscan\b",
    ]),
    ("janji_peri", [
        r"(buat|jadwal|booking|book|daftar|reservasi) janji",
        r"(cari|cek|lihat) (dokter|jadwal|slot)",
        r"dokter (terdekat|dekat|mana)",
        r"puskesmas",
        r"klinik gigi",
        r"janji (dokter|gigi)",
        r"appointment",
    ]),
    ("rapot_peri", [
        r"\bstreak\b",
        r"rapot",
        r"(sikat gigi anak|sikat gigi (hari|kali))",
        r"(riwayat|progress|rekapan) (sikat|brushing)",
        r"berapa (hari|kali) sikat",
        r"pencapaian",
        r"achievement",
    ]),
    ("cerita_peri", [
        r"cerita peri",
        r"(modul|literatur|materi) (ke-?\d+|berapa|mana)",
        r"(sudah|belum) baca",
        r"(progress|sampai) (modul|cerita)",
        r"quiz (cerita|peri)",
    ]),
    ("user_profile", [
        r"(nama|umur|usia) (saya|anak|si kecil)",
        r"(profil|data) (saya|anak)",
        r"anak saya (siapa|berapa|bernama)",
        r"tanggal lahir",
        r"info (akun|user)",
    ]),
    ("app_faq", [
        r"(cara|bagaimana|gimana) (pakai|menggunakan|daftar|login|install)",
        r"(fitur|menu|fungsi) (apa|peri bugi)",
        r"aplikasi (ini|peri bugi)",
        r"(tidak bisa|gagal|error) (login|daftar|buka)",
        r"peri bugi (itu|ini)",
        r"(apa itu|apa sih) (mata|rapot|cerita|janji|tanya) peri",
    ]),
    ("kb_dental", [
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
        r"\bmulut\b",
        r"nafas",
        r"bau mulut",
        r"gigi susu",
        r"gigi permanen",
        r"gigi berlubang",
        r"gigi anak",
        r"(sakit|nyeri) gigi",
    ]),
]

_SMALLTALK_PATTERNS = [
    r"^(halo|hai|hi|hey|assalam|pagi|siang|sore|malam)\b",
    r"^apa kabar",
    r"^(terima kasih|makasih|thanks)",
    r"^(oke|ok|baik|siap)$",
    r"^(siapa kamu|kamu (siapa|apa))",
]

_CLARIFICATION_ANSWER_PATTERNS = [
    r"^[a-d]$",
    r"^[1-4]$",
    r"^pilih\s+[a-d1-4]",
    r"^(jawaban|opsi|pilihan)\s+",
]


def _classify_rule_based(message: str, allowed_agents: list[str]) -> Optional[list[str]]:
    """Try rule-based classification. Return list of agents or None if no match."""
    text = message.lower().strip()

    # Smalltalk → empty (no agent needed)
    for pattern in _SMALLTALK_PATTERNS:
        if re.search(pattern, text):
            return []

    # Clarification answer → empty (handle di generate)
    for pattern in _CLARIFICATION_ANSWER_PATTERNS:
        if re.search(pattern, text):
            return []

    # Multi-match: collect all matching agents
    matched = []
    for agent_key, patterns in _AGENT_RULES:
        if agent_key not in allowed_agents:
            continue
        for pattern in patterns:
            if re.search(pattern, text):
                if agent_key not in matched:
                    matched.append(agent_key)
                break

    return matched if matched else None


async def _classify_by_llm(
    message: str,
    allowed_agents: list[str],
    prompt_template: str,
    state: AgentState,
) -> tuple[list[str], LLMCallLog]:
    """LLM fallback: return list agent yang perlu dipanggil + log payload."""
    from langchain_core.messages import HumanMessage
    from app.config.llm import get_llm, get_model_name, get_provider_name
    from app.config.observability import build_trace_config

    provider = state.rnd.llm_provider
    model = state.rnd.llm_model

    llm = get_llm(temperature=0, max_tokens=50, streaming=False, provider=provider, model=model)

    agents_str = ", ".join(allowed_agents)
    prompt = prompt_template.replace("{user_message}", message).replace("{allowed_agents}", agents_str)

    start = time.monotonic()
    success = True
    error_msg = None
    agents_selected: list[str] = []
    raw = ""

    # Per-call trace config (no-op kalau Langfuse disabled)
    trace_config = build_trace_config(
        state={"session_id": state.session.session_id, "user_context": state.user_context.model_dump()},
        agent_name="supervisor",
    )

    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)], config=trace_config)
        raw = result.content.strip().lower()
        if raw and raw != "none":
            candidates = [a.strip() for a in raw.split(",")]
            agents_selected = [a for a in candidates if a in allowed_agents]
        if not agents_selected and "kb_dental" in allowed_agents:
            agents_selected = ["kb_dental"]
    except Exception as e:
        success = False
        error_msg = str(e)
        agents_selected = ["kb_dental"] if "kb_dental" in allowed_agents else []

    latency_ms = int((time.monotonic() - start) * 1000)
    log = LLMCallLog(
        prompt_key="supervisor_route",
        model=get_model_name(provider=provider, model=model),
        provider=get_provider_name(provider=provider),
        node="supervisor",
        latency_ms=latency_ms,
        output_tokens=len(raw.split(",")) if raw else 1,
        success=success,
        error_message=error_msg,
        metadata={"agents_selected": agents_selected},
    )
    return agents_selected, log


# =============================================================================
# LangGraph node
# =============================================================================


async def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node signature: receives AgentState, returns partial state update dict.

    Returns dict dengan:
    - agents_selected: list[str]
    - execution_plan: dict
    - thinking_steps: appended ThinkingStep
    - llm_call_logs: appended LLMCallLog (kalau LLM fallback dipakai)
    """
    # Phase 4: wrap node body with trace span (preserve Langfuse)
    from app.config.observability import trace_node

    # Extract last user message
    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    last_message = user_messages[-1].content if user_messages else ""
    if not isinstance(last_message, str):
        last_message = str(last_message)

    allowed = state.control.allowed_agents

    # Pseudo-state untuk trace_node (pakai legacy dict format untuk kompat)
    legacy_state = {
        "session_id": state.session.session_id,
        "user_context": state.user_context.model_dump(),
        "response_mode": state.session.response_mode,
    }

    async with trace_node(
        name="supervisor",
        state=legacy_state,
        input_data={
            "user_message": last_message[:500],
            "has_image": state.image is not None,
            "force_intent": state.rnd.force_intent,
            "allowed_agents": allowed,
        },
    ) as span:
        thinking_done = ThinkingStep(step=1, label="Memahami pertanyaanmu...", done=True)

        # ── Force intent (RnD only) ────────────────────────────────────────
        if state.rnd.force_intent:
            agents = [state.rnd.force_intent] if state.rnd.force_intent in allowed else []
            plan = {"mode": "sequential", "reasoning": "force_intent", "decision_path": "force_intent"}
            if span:
                span.update(output={
                    "agents_selected": agents,
                    "execution_plan": plan,
                    "decision_path": "force_intent",
                })
            return {
                "agents_selected": agents,
                "execution_plan": plan,
                "thinking_steps": [*state.thinking_steps, thinking_done],
            }

        # ── Image auto-route ────────────────────────────────────────────────
        if state.image is not None and "mata_peri" in allowed:
            agents = ["mata_peri"]
            plan = {"mode": "sequential", "reasoning": "image_detected", "decision_path": "image_auto"}
            if span:
                span.update(output={
                    "agents_selected": agents,
                    "execution_plan": plan,
                    "decision_path": "image_auto",
                })
            return {
                "agents_selected": agents,
                "execution_plan": plan,
                "thinking_steps": [*state.thinking_steps, thinking_done],
            }

        # ── Rule-based ──────────────────────────────────────────────────────
        agents = _classify_rule_based(last_message, allowed)
        decision_path = "rule_based"
        new_logs: list[LLMCallLog] = []

        if agents is None:
            # ── LLM fallback ────────────────────────────────────────────────
            decision_path = "llm_fallback"
            supervisor_prompt = state.prompts.get("supervisor_route", "")
            if supervisor_prompt:
                agents, log = await _classify_by_llm(
                    message=last_message,
                    allowed_agents=allowed,
                    prompt_template=supervisor_prompt,
                    state=state,
                )
                new_logs.append(log)
            else:
                # Prompt tidak ada di DB (current state) → default ke kb_dental
                agents = ["kb_dental"] if "kb_dental" in allowed else []
                decision_path = "no_prompt_default"

        # ── Determine execution mode ────────────────────────────────────────
        mode = "parallel" if len(agents) > 1 else "sequential"
        plan = {"mode": mode, "reasoning": "auto", "decision_path": decision_path}

        if span:
            span.update(output={
                "agents_selected": agents,
                "execution_plan": plan,
                "decision_path": decision_path,
            })

        return {
            "agents_selected": agents,
            "execution_plan": plan,
            "thinking_steps": [*state.thinking_steps, thinking_done],
            "llm_call_logs": [*state.llm_call_logs, *new_logs],
        }
