"""
Agent node — Phase 2 Step 2a (Single-Pass Tool Selection)

Replaces supervisor + agent_dispatcher from Phase 1.

DESIGN: Single-pass tool selection (per Hanif's "Opsi D" decision).
- LLM gets ONE invocation with bind_tools
- LLM decides which tools to call (parallel)
- After tools execute, response goes to generate_node DIRECTLY
- NO loop back to agent_node

This sidesteps Gemini multi-turn tool calling bug (issue #29418) which
manifests when ToolMessage.name fails to propagate across turns.

DEFAULT INLINE PROMPT: Step 2a uses a hardcoded persona prompt. Step 2b
will migrate to DB-stored agent_system_simple/medium/detailed prompts.

FORCED ROUTING: pre_router_node may set state.forced_tool_calls. If non-empty,
agent_node SKIPS LLM invocation entirely — emits AIMessage with the forced
tool_calls. This guarantees deterministic image-flow behavior without LLM risk.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState, LLMCallLog, ThinkingStep
from app.agents.tools import make_tools

logger = logging.getLogger(__name__)


# =============================================================================
# Inline default system prompt (Step 2b will migrate to DB)
# =============================================================================

_PERSONA_BASE = """Kamu adalah ROUTER untuk asisten Peri Bugi (kesehatan gigi anak).

TUGAS KAMU HANYA SATU: Pilih tool yang TEPAT untuk menjawab pertanyaan orang tua.
KAMU BUKAN yang menyusun jawaban final — itu tugas node lain.

ATURAN MUTLAK:
1. JANGAN tulis content/jawaban apa pun. Kosongkan content.
2. Untuk pertanyaan yang butuh data/info → panggil tool yang sesuai.
3. Untuk smalltalk/sapaan ("halo", "terima kasih", "ok", "gimana kabar") → JANGAN panggil tool, JANGAN tulis jawaban. Cukup return tanpa apa-apa (empty content + no tool_calls). Node lain yang akan menyapa user dengan hangat.
4. Kalau ragu antara tool A dan B, pilih yang paling cocok berdasarkan KATA KUNCI di pertanyaan.

Output kamu HANYA berupa:
- tool_calls (kalau perlu cari info/data), ATAU
- empty (kalau smalltalk atau pertanyaan yang tidak perlu tool)

JANGAN PERNAH menulis jawaban dalam content — itu akan di-discard.
"""

_TOOL_GUIDANCE = """
PEDOMAN MEMILIH TOOL:

Untuk pertanyaan KESEHATAN GIGI umum (kenapa, gimana, apa itu, dll):
→ Pakai search_dental_knowledge dengan query yang fokus

Untuk pertanyaan tentang aplikasi PERI BUGI (cara pakai, fitur, dll):
→ Pakai search_app_faq

Untuk pertanyaan SPESIFIK ANAK (streak, progress, riwayat scan):
→ Pakai get_brushing_stats untuk streak/rapot/progress sikat
→ Pakai get_cerita_progress untuk progress modul Cerita Peri
→ Pakai get_scan_history untuk riwayat scan Mata Peri lama

Untuk pertanyaan multi-aspek (e.g. "anak streak 5 tapi gigi sakit"):
→ Boleh pilih beberapa tool sekaligus (parallel)

Untuk SMALLTALK / SAPAAN ("halo", "terima kasih", "ok", "halo peri", dll):
→ JANGAN panggil tool. JANGAN tulis jawaban di content.
→ Cukup return empty (tidak perlu tool, tidak perlu content).
→ Node lain (generate) yang akan menyapa user dengan hangat memakai persona Peri.

Profil orang tua dan anak sudah otomatis tersedia — JANGAN panggil get_user_profile
kecuali kamu butuh detail spesifik yang tidak ada di system context.
"""

_MODE_LIMITS = {
    "simple": "MODE SIMPLE: Maksimal 1 tool. Prioritaskan jawaban cepat dan ringkas.",
    "medium": "MODE MEDIUM: Maksimal 2 tools. Boleh kombinasi untuk jawaban lebih lengkap.",
    "detailed": "MODE DETAILED: Maksimal 5 tools. Eksploratif untuk jawaban komprehensif.",
}


def _build_system_prompt(state: AgentState) -> str:
    """Build inline system prompt for agent_node.

    Step 2b: Pull prompt template from `state.prompts['agent_router_v2']` (DB).
    Graceful fallback to inline default if DB prompt missing (Q3A decision).

    Variables di template (di-render via str.replace):
      {parent_name} {sapaan} {child_name}{age_str} {mode_limit}
    """
    # Build context vars (same logic regardless of DB or fallback prompt)
    ctx = state.user_context.model_dump() if state.user_context else {}
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    parent_name = (user.get("nickname") or user.get("full_name") or "Bunda/Ayah") if user else "Bunda/Ayah"
    gender = user.get("gender") if user else None
    sapaan = "Ayah" if gender == "M" else ("Bunda" if gender == "F" else "Bunda/Ayah")

    child_name = (child.get("nickname") or child.get("full_name") or "anak") if child else "anak"
    child_age = child.get("age_years") if child else None
    age_str = f", umur {child_age} tahun" if child_age else ""

    mode = state.session.response_mode
    mode_limit = _MODE_LIMITS.get(mode, _MODE_LIMITS["simple"])

    # Try DB prompt first (Step 2b)
    prompts = state.prompts or {}
    db_template = prompts.get("agent_router_v2")

    if db_template:
        # DB-driven path — render template variables
        rendered = db_template
        rendered = rendered.replace("{parent_name}", parent_name)
        rendered = rendered.replace("{sapaan}", sapaan)
        rendered = rendered.replace("{child_name}", child_name)
        rendered = rendered.replace("{age_str}", age_str)
        rendered = rendered.replace("{mode_limit}", mode_limit)

        # Append memory context if available (kept dynamic — not in DB template
        # because it's per-session data, not static config)
        mem = state.memory.model_dump() if state.memory else {}
        if mem.get("session_summaries") or mem.get("user_facts"):
            mem_parts = ["\n\nMEMORI PERCAKAPAN SEBELUMNYA:"]
            for s in (mem.get("session_summaries") or [])[:2]:
                mem_parts.append(f"- {s}")
            for f in (mem.get("user_facts") or [])[:5]:
                if isinstance(f, dict):
                    mem_parts.append(f"- {f.get('fact', '')}")
            rendered += "\n".join(mem_parts)

        return rendered

    # ────────────────────────────────────────────────────────────────────────
    # FALLBACK PATH (Step 2b — Q3A: graceful fallback)
    # Triggered when DB seeder belum jalan / prompt key missing.
    # Sistem tetap jalan dengan inline default + warning log.
    # ────────────────────────────────────────────────────────────────────────
    logger.warning(
        "[agent_node] 'agent_router_v2' prompt not in state.prompts — "
        "using inline fallback. Run scripts/seed_prompts.py to populate DB."
    )

    parts = [_PERSONA_BASE, _TOOL_GUIDANCE]
    parts.append(mode_limit)

    # Auto-inject user context (Hanif's Q2 decision: auto-kasih)
    if user or child:
        parts.append("\nKONTEKS USER (auto-tersedia, sudah pasti benar):")
        if user:
            parts.append(f"- Orang tua: {parent_name} ({sapaan})")
        if child:
            parts.append(f"- Anak: {child_name}{age_str}")

    # Memory context (kalau ada)
    mem = state.memory.model_dump() if state.memory else {}
    if mem.get("session_summaries") or mem.get("user_facts"):
        parts.append("\nMEMORI PERCAKAPAN SEBELUMNYA:")
        for s in (mem.get("session_summaries") or [])[:2]:
            parts.append(f"- {s}")
        for f in (mem.get("user_facts") or [])[:5]:
            if isinstance(f, dict):
                parts.append(f"- {f.get('fact', '')}")

    return "\n".join(parts)


def _extract_user_message_for_llm(state: AgentState) -> str:
    """Extract last user message text from state.messages."""
    for msg in reversed(state.messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


# =============================================================================
# LangGraph node
# =============================================================================


async def agent_node(state: AgentState) -> dict[str, Any]:
    """
    Single-pass agent: decide which tools to call (or skip if forced).

    Returns partial state update dict containing:
    - messages: appended AIMessage (with tool_calls if any)
    - thinking_steps: appended ThinkingStep
    - llm_call_logs: appended LLMCallLog (kalau LLM dipakai)
    """
    from app.config.observability import trace_node, build_trace_config
    from app.config.llm import get_llm, get_model_name, get_provider_name

    # Build pseudo-state for trace_node compat
    legacy_state_for_trace = {
        "session_id": state.session.session_id,
        "user_context": state.user_context.model_dump() if state.user_context else {},
        "response_mode": state.session.response_mode,
    }

    user_message = _extract_user_message_for_llm(state)
    forced = state.forced_tool_calls or []
    is_smalltalk = state.is_smalltalk

    # FIX (Langfuse audit Bagian B3): Enrich agent_node trace with diagnostic metadata.
    # Compute tools_in_context info BEFORE entering trace_node — supaya bisa di-input.
    # Note: make_tools(state) idempotent + cheap (just builds factory closures).
    _mode = state.session.response_mode
    _max_tools_map = {"simple": 1, "medium": 2, "detailed": 5}
    _max_tools_for_mode = _max_tools_map.get(_mode, 1)

    async with trace_node(
        name="node:agent",
        state=legacy_state_for_trace,
        input_data={
            "user_message": user_message[:500],
            "has_image": state.image is not None,
            "forced_tool_calls": [t.get("name") for t in forced],
            "is_smalltalk": is_smalltalk,
            "response_mode": _mode,
            # FIX (Langfuse audit Bagian B3): Add allowed_agents + tool count info
            # supaya bisa debug "kenapa LLM tidak pilih tool X" — mungkin tidak in scope.
            "allowed_agents": state.control.allowed_agents,
            "max_tools_for_mode": _max_tools_for_mode,
        },
    ) as span:
        # ────────────────────────────────────────────────────────────────────
        # Path A1: Smalltalk (Step 2b fix) — skip LLM entirely.
        # pre_router sudah detect smalltalk via strict regex+length+keyword.
        # No tools needed, no LLM call needed. Emit empty AIMessage.
        # generate_node will respond using lean smalltalk prompt (state.is_smalltalk
        # propagated via _build_legacy_dict_state in agent_dispatcher.py).
        #
        # Why skip LLM here even though Gemini "should" follow no-tool rule:
        # - Gemini compliance for negative instructions ("JANGAN call tool")
        #   unreliable. Defense-in-depth.
        # - Cost saving: ~2,500 prompt tokens + 1.3s latency avoided per smalltalk.
        # - Cleaner trace: no spurious tool_calls in Langfuse.
        # ────────────────────────────────────────────────────────────────────
        if is_smalltalk and not forced:
            ai_msg = AIMessage(content="", tool_calls=[])

            thinking_done = ThinkingStep(
                step=2,
                label="Menyiapkan jawaban...",
                done=True,
            )

            if span:
                span.update(output={
                    "decision_path": "smalltalk_skip_llm",
                    "tool_calls": [],
                    "tool_calls_count": 0,
                    "is_smalltalk": True,
                    "content_was_stripped": False,
                    "stripped_content": None,
                })

            logger.info(
                f"[agent_node] Smalltalk path — skip LLM, emit empty AIMessage. "
                f"user_message={user_message[:50]!r}"
            )

            return {
                "messages": [ai_msg],
                "thinking_steps": [*state.thinking_steps, thinking_done],
            }

        # ────────────────────────────────────────────────────────────────────
        # Path A: Forced routing (image flow) — skip LLM, emit AIMessage with
        # the forced tool_calls. ToolNode will execute them.
        # ────────────────────────────────────────────────────────────────────
        if forced:
            tool_calls = []
            for f in forced:
                tool_calls.append({
                    "name": f["name"],
                    "args": f.get("args") or {},
                    "id": f"forced_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })

            ai_msg = AIMessage(content="", tool_calls=tool_calls)

            thinking_done = ThinkingStep(
                step=2,
                label="Memproses foto kamu...",
                done=True,
            )

            if span:
                span.update(output={
                    "decision_path": "forced",
                    "tool_calls": [tc["name"] for tc in tool_calls],
                })

            return {
                "messages": [ai_msg],
                "thinking_steps": [*state.thinking_steps, thinking_done],
            }

        # ────────────────────────────────────────────────────────────────────
        # Path B: Free routing — LLM with bind_tools decides
        # ────────────────────────────────────────────────────────────────────
        tools = make_tools(state)

        # No tools allowed (defensive — should not happen in practice) → skip LLM,
        # emit empty AIMessage. tools_node will be no-op, generate_node will
        # respond using just system prompt + history.
        if not tools:
            logger.warning(
                "[agent_node] No tools available (allowed_agents likely empty). "
                "Returning empty AIMessage; generate will respond from context only."
            )
            ai_msg = AIMessage(content="")
            thinking_done = ThinkingStep(step=2, label="Menyusun jawaban...", done=True)
            if span:
                span.update(output={"decision_path": "no_tools_skip"})
            return {
                "messages": [ai_msg],
                "thinking_steps": [*state.thinking_steps, thinking_done],
            }

        # Resolve LLM provider/model (RnD overrides if set)
        provider = state.rnd.llm_provider
        model = state.rnd.llm_model
        # Lower temp for tool selection (more deterministic)
        llm = get_llm(temperature=0.2, max_tokens=1024, streaming=False,
                      provider=provider, model=model)
        llm_with_tools = llm.bind_tools(tools)

        # Build messages: system prompt + last user message
        # We don't pass full history here — generate_node has full context.
        # agent_node's job is just tool selection, not response synthesis.
        system_prompt = _build_system_prompt(state)
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        trace_config = build_trace_config(
            state=legacy_state_for_trace,
            agent_name="agent_node",
        )

        start = time.monotonic()
        success = True
        error_msg = None
        ai_msg: AIMessage = AIMessage(content="")

        try:
            result = await llm_with_tools.ainvoke(messages_for_llm, config=trace_config)
            # result IS an AIMessage (LangChain ChatModel returns BaseMessage)
            if isinstance(result, AIMessage):
                ai_msg = result
            else:
                # Defensive: convert if returns plain BaseMessage subclass
                ai_msg = AIMessage(
                    content=getattr(result, "content", "") or "",
                    tool_calls=getattr(result, "tool_calls", []) or [],
                )
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(f"[agent_node] LLM ainvoke failed: {e}", exc_info=True)
            # Fallback: empty AIMessage; generate will run with no tool results
            ai_msg = AIMessage(content="")

        # ────────────────────────────────────────────────────────────────────
        # PHASE 2A FIX2: STRIP CONTENT (single-responsibility principle)
        # ────────────────────────────────────────────────────────────────────
        # agent_node = TOOL SELECTOR ONLY. Tidak boleh menjawab.
        # generate_node = RESPONSE SYNTHESIZER. Yang berwenang menjawab user.
        #
        # Background bug: Untuk smalltalk ("halo peri"), Gemini bind_tools
        # kadang patuh prompt "respond hangat" lalu kasih content "Halo Ayah!..."
        # tanpa tool_calls. Content ini ke-append ke state.messages oleh
        # add_messages reducer. generate_node lalu lihat conversation history
        # punya assistant turn yang sudah jawab → LLM thinks done → 0 tokens.
        #
        # Fix: discard content dari agent_node, regardless apa pun. Tool selection
        # (tool_calls) preserved. generate_node always responsible for synthesis.
        #
        # Saved content untuk audit/diagnostic via observability span saja.
        agent_content_stripped = ai_msg.content if (ai_msg.content and str(ai_msg.content).strip()) else None
        if agent_content_stripped:
            logger.info(
                f"[agent_node] Stripped LLM content (len={len(agent_content_stripped)}) — "
                f"agent_node is tool selector only. content preserved in span for audit."
            )

        ai_msg = AIMessage(
            content="",
            tool_calls=ai_msg.tool_calls or [],
        )

        # Apply mode_behavior tool limit (defensive trim)
        if ai_msg.tool_calls:
            mode = state.session.response_mode
            max_tools_map = {"simple": 1, "medium": 2, "detailed": 5}
            max_tools = max_tools_map.get(mode, 1)
            if len(ai_msg.tool_calls) > max_tools:
                logger.info(
                    f"[agent_node] LLM emitted {len(ai_msg.tool_calls)} tool_calls, "
                    f"trimming to mode={mode} max={max_tools}"
                )
                ai_msg = AIMessage(
                    content=ai_msg.content,
                    tool_calls=ai_msg.tool_calls[:max_tools],
                )

        latency_ms = int((time.monotonic() - start) * 1000)
        log = LLMCallLog(
            prompt_key="agent_inline_v2a",
            model=get_model_name(provider=provider, model=model),
            provider=get_provider_name(provider=provider),
            node="agent_node",
            latency_ms=latency_ms,
            success=success,
            error_message=error_msg,
            metadata={
                "tool_calls_count": len(ai_msg.tool_calls or []),
                "tool_calls_names": [tc.get("name") for tc in (ai_msg.tool_calls or [])],
                "response_mode": state.session.response_mode,
                # Phase 2a fix2: track whether Gemini violated the "no content"
                # rule so we can monitor prompt compliance via DB queries.
                "content_was_stripped": bool(agent_content_stripped),
                "stripped_content_len": len(agent_content_stripped) if agent_content_stripped else 0,
            },
        )

        thinking_done = ThinkingStep(
            step=2,
            label="Memilih cara menjawab terbaik..." if ai_msg.tool_calls else "Menyusun jawaban...",
            done=True,
        )

        if span:
            span.update(output={
                "decision_path": "llm",
                "tool_calls": [tc.get("name") for tc in (ai_msg.tool_calls or [])],
                "tool_calls_count": len(ai_msg.tool_calls or []),
                "latency_ms": latency_ms,
                "success": success,
                # FIX (Langfuse audit Bagian B3): Capture tool_args yang dipilih LLM
                # supaya bisa debug "tool dipanggil dengan args yang aneh".
                "tool_args": [tc.get("args") for tc in (ai_msg.tool_calls or [])],
                # FIX (Langfuse audit Bagian B3): Capture tools_in_context count
                # supaya bisa correlate "ada N tools available, LLM pilih X".
                "tools_available_count": len(tools),
                "tools_available_names": [getattr(t, "name", "") for t in tools],
                # Phase 2a fix2: log stripped content for audit/debugging.
                # Content is discarded from state.messages but kept here so
                # we can verify Gemini prompt compliance via Langfuse trace.
                "stripped_content": (agent_content_stripped[:500] if agent_content_stripped else None),
                "content_was_stripped": bool(agent_content_stripped),
            })

        return {
            "messages": [ai_msg],
            "thinking_steps": [*state.thinking_steps, thinking_done],
            "llm_call_logs": [*state.llm_call_logs, log],
        }
