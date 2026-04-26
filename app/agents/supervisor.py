"""
Node: supervisor
Planner yang menentukan agent mana yang perlu dipanggil untuk menjawab pertanyaan user.

Approach: hybrid
1. Rule-based fast-path (instant, tanpa LLM)
2. LLM fallback untuk kasus ambigu

Output:
- agents_selected: list agent yang perlu dipanggil
- execution_plan: {mode: parallel|sequential, reasoning}
"""
import re
import time
from typing import AsyncIterator

from app.agents.state import AgentState
from app.schemas.chat import LLMCallLogPayload, make_thinking_event

# =============================================================================
# Rule-based routing
# =============================================================================

# Pattern per agent — ordered by specificity
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
    # kb_dental sebagai catch-all dental questions
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

# Intent yang tidak butuh agent khusus (smalltalk/clarification)
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


def _classify_rule_based(message: str, allowed_agents: list[str]) -> list[str] | None:
    """
    Coba klasifikasi dengan rules. Return list agent atau None jika tidak match.
    Hanya return agents yang ada di allowed_agents.
    """
    text = message.lower().strip()

    # Smalltalk → tidak butuh agent
    for pattern in _SMALLTALK_PATTERNS:
        if re.search(pattern, text):
            return []  # empty = smalltalk, tidak perlu agent

    # Clarification answer → tidak butuh agent baru
    for pattern in _CLARIFICATION_ANSWER_PATTERNS:
        if re.search(pattern, text):
            return []  # handle di generate node

    # Multi-match: kumpulkan semua agent yang match, filter by allowed
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
    provider: str | None,
    model: str | None,
    state: dict | None = None,
) -> tuple[list[str], LLMCallLogPayload]:
    """LLM fallback: return list agent yang perlu dipanggil.

    Args:
        state: AgentState — untuk Langfuse trace metadata. Optional.
    """
    from langchain_core.messages import HumanMessage
    from app.config.llm import get_llm, get_model_name, get_provider_name
    from app.config.observability import build_trace_config

    llm = get_llm(temperature=0, max_tokens=50, streaming=False,
                  provider=provider, model=model)

    agents_str = ", ".join(allowed_agents)
    prompt = prompt_template \
        .replace("{user_message}", message) \
        .replace("{allowed_agents}", agents_str)

    start = time.monotonic()
    success = True
    error_msg = None
    agents_selected: list[str] = []

    # Per-call trace config (no-op kalau Langfuse disabled)
    trace_config = build_trace_config(state=state, agent_name="supervisor")

    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)], config=trace_config)
        raw = result.content.strip().lower()
        # Parse: "kb_dental,user_profile" atau "kb_dental" atau "none"
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
    log = LLMCallLogPayload(
        prompt_key="supervisor_route",
        model=get_model_name(provider=provider, model=model),
        provider=get_provider_name(provider=provider),
        node="supervisor",
        latency_ms=latency_ms,
        output_tokens=len(raw.split(",")) if "raw" in dir() else 1,
        success=success,
        error_message=error_msg,
        metadata={"agents_selected": agents_selected},
    )
    return agents_selected, log


async def supervisor_node(state: AgentState) -> AsyncIterator[str]:
    """
    LangGraph node: supervisor
    Determine which agents to call for this message.
    Updates state with agents_selected and execution_plan.
    """
    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    last_message = user_messages[-1].get("content", "") if user_messages else ""
    allowed_agents = state.get("allowed_agents", [])

    yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=False)

    # Force intent dari RnD mode
    force_intent = state.get("force_intent")
    if force_intent:
        state["agents_selected"] = [force_intent] if force_intent in allowed_agents else []
        state["execution_plan"] = {"mode": "sequential", "reasoning": "force_intent"}
        yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=True)
        state["thinking_steps"].append({"step": 1, "label": "Memahami pertanyaanmu...", "done": True})
        return

    # Juga cek image_url — otomatis tambah mata_peri
    if state.get("image_url") and "mata_peri" in allowed_agents:
        state["agents_selected"] = ["mata_peri"]
        state["execution_plan"] = {"mode": "sequential", "reasoning": "image_detected"}
        yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=True)
        state["thinking_steps"].append({"step": 1, "label": "Memahami pertanyaanmu...", "done": True})
        return

    # Rule-based
    agents = _classify_rule_based(last_message, allowed_agents)

    if agents is None:
        # LLM fallback
        supervisor_prompt = state.get("prompts", {}).get("supervisor_route", "")
        if supervisor_prompt:
            agents, log = await _classify_by_llm(
                message=last_message,
                allowed_agents=allowed_agents,
                prompt_template=supervisor_prompt,
                provider=state.get("llm_provider_override"),
                model=state.get("llm_model_override"),
                state=state,  # untuk Langfuse trace metadata
            )
            state["llm_call_logs"].append(log.model_dump())
        else:
            # No prompt → default ke kb_dental jika allowed
            agents = ["kb_dental"] if "kb_dental" in allowed_agents else []

    # Tentukan mode eksekusi
    # Sequential jika satu agent butuh hasil agent lain (saat ini belum ada case)
    # Parallel untuk semua kasus lainnya
    mode = "parallel" if len(agents) > 1 else "sequential"

    state["agents_selected"] = agents
    state["execution_plan"] = {"mode": mode, "reasoning": "auto"}

    yield make_thinking_event(step=1, label="Memahami pertanyaanmu...", done=True)
    state["thinking_steps"].append({"step": 1, "label": "Memahami pertanyaanmu...", "done": True})
