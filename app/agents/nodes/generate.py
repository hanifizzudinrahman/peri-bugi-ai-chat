"""
Node: generate

Phase 1 refactor: return dict (LangGraph signature) instead of yield SSE events.
ALL logic dari generate_node original PRESERVED 1:1:
- Mode-specific system prompt (simple/medium/detailed)
- Image analysis prompt injection (DB or hardcoded fallback)
- Memory injection (L2 summaries + L3 facts)
- Agent results injection
- Clarification / quick_reply early return
- LLM streaming with metrics (TTFT, latency, tokens)
- Langfuse trace_generation wrapping
- LLM call log

Token-level streaming TIDAK lagi yield dari node ini (LangGraph node return dict).
Stream adapter (app/agents/streaming.py) capture token events via
graph.astream_events() pattern → emit SSE token events ke FE.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState, LLMCallLog, ThinkingStep

logger = logging.getLogger(__name__)


# =============================================================================
# System prompt builder (preserved 1:1 dari generate.py original)
# =============================================================================


def _build_system_prompt(state: AgentState) -> str:
    """
    Build system prompt dengan komposisi:
    1. Persona Tanya Peri (DB atau default)
    2. Data user & anak dari user_context
    3. Memory context (L2 summary + L3 facts)
    4. Hasil agent (docs, profile, image analysis, dll)
    5. Response mode instructions
    """
    prompts = state.prompts

    # RnD override
    if "_override_system" in prompts:
        return prompts["_override_system"]

    # ── Persona base ──────────────────────────────────────────────────────────
    persona = prompts.get(
        "persona_system",
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak dari aplikasi Peri Bugi. "
        "Kamu ramah, sabar, dan berbicara dengan bahasa yang mudah dipahami orang tua Indonesia. "
        "Jangan pernah memberikan diagnosis medis langsung. "
        "Selalu sarankan konsultasi dokter gigi untuk masalah serius.",
    )

    ctx = state.user_context.model_dump()
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"
    child_age = f"{child.get('age_years')} tahun" if child.get("age_years") else "?"

    system = persona.replace("{user_name}", user_name)
    system = system.replace("{child_name}", child_name)
    system = system.replace("{child_age}", child_age)

    # ── Brushing context ──────────────────────────────────────────────────────
    brushing = ctx.get("brushing")
    if brushing:
        system += (
            f"\n\nData sikat gigi {child_name}: "
            f"streak {brushing.get('current_streak', 0)} hari, "
            f"rekor terbaik {brushing.get('best_streak', 0)} hari."
        )

    # ── Mata Peri last result (from user_context) ─────────────────────────────
    mata_peri = ctx.get("mata_peri_last_result")
    if mata_peri and mata_peri.get("summary_text"):
        system += (
            f"\n\nHasil scan gigi terakhir {child_name} "
            f"({mata_peri.get('scan_date', 'tidak diketahui')}): "
            f"{mata_peri.get('summary_text')}. "
            f"Status: {mata_peri.get('summary_status', 'tidak diketahui')}."
        )

    # ── Memory context (L2 summaries + L3 facts) ──────────────────────────────
    memory = state.memory.model_dump()

    summaries = memory.get("session_summaries", [])
    if summaries:
        summaries_text = "\n".join(f"- {s}" for s in summaries)
        system += f"\n\nRingkasan percakapan sebelumnya:\n{summaries_text}"

    facts = memory.get("user_facts", [])
    if facts:
        facts_text = "\n".join(
            f"- {f.get('value', '')}" for f in facts if f.get("value")
        )
        system += f"\n\nYang kamu ketahui tentang user ini:\n{facts_text}"

    # ── Agent results ─────────────────────────────────────────────────────────
    agent_results = state.agent_results

    # KB Dental docs
    kb_result = agent_results.get("kb_dental", {})
    kb_docs = kb_result.get("docs", []) or state.retrieved_docs
    if kb_docs:
        docs_text = "\n\n".join(kb_docs[:3])
        system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

    # App FAQ docs
    faq_result = agent_results.get("app_faq", {})
    faq_docs = faq_result.get("docs", [])
    if faq_docs:
        faq_text = "\n\n".join(faq_docs[:3])
        system += f"\n\nInfo aplikasi yang relevan:\n{faq_text}"

    # Image analysis (Mata Peri image flow)
    image_analysis = state.image_analysis
    if image_analysis:
        session_summary = (
            image_analysis.get("session_summary")
            if isinstance(image_analysis, dict)
            else None
        )

        if session_summary:
            sum_status = session_summary.get("summary_status") or "tidak diketahui"
            sum_text = session_summary.get("summary_text") or ""
            rec_text = session_summary.get("recommendation_text") or ""
            requires_review = session_summary.get("requires_dentist_review", False)

            status_emoji = {
                "ok": "✅",
                "perlu_perhatian": "⚠️",
                "perlu_dokter": "🚨",
                "gagal": "❌",
            }.get(sum_status, "📋")

            response_mode = state.session.response_mode
            image_prompt_key = f"tanya_peri_image_response_{response_mode}"
            image_prompt_template = prompts.get(image_prompt_key)

            if image_prompt_template:
                try:
                    rendered = image_prompt_template.format(
                        summary_status=sum_status,
                        summary_text=sum_text,
                        recommendation_text=rec_text,
                        child_name=child_name,
                        requires_dentist_review=str(requires_review).lower(),
                        status_emoji=status_emoji,
                    )
                    system += "\n\n" + rendered
                except KeyError as e:
                    logger.warning(
                        f"[generate] Prompt {image_prompt_key} pakai placeholder "
                        f"yang tidak dikenal: {e}. Fallback ke hardcoded."
                    )
                    system += _build_image_analysis_fallback_prompt(
                        sum_status, sum_text, rec_text, child_name,
                        requires_review, status_emoji,
                    )
            else:
                system += _build_image_analysis_fallback_prompt(
                    sum_status, sum_text, rec_text, child_name,
                    requires_review, status_emoji,
                )
        else:
            system += (
                f"\n\nHasil analisis gambar gigi: "
                f"{image_analysis.get('summary', 'tidak tersedia')}."
            )

    # ── Response mode instructions ────────────────────────────────────────────
    response_mode = state.session.response_mode
    mode_key = f"generate_{response_mode}"
    mode_instruction = prompts.get(mode_key)

    if not mode_instruction:
        if response_mode == "simple":
            mode_instruction = (
                "Jawab dengan singkat, ramah, dan mudah dipahami. "
                "Gunakan bahasa sehari-hari. Boleh pakai emoji yang relevan. "
                "Maksimal 3-5 kalimat."
            )
        elif response_mode == "medium":
            mode_instruction = (
                "Jawab dengan cukup detail dan informatif. "
                "Gunakan 1-2 paragraf yang jelas. "
                "Tidak perlu cantumkan sumber atau referensi."
            )
        elif response_mode == "detailed":
            mode_instruction = (
                "Jawab secara lengkap dan ilmiah. Sertakan penjelasan mekanisme, "
                "pencegahan, dan penanganan. Cantumkan referensi atau sumber "
                "jika relevan (WHO, jurnal gigi, dll). "
                "Gunakan struktur yang jelas."
            )

    if mode_instruction:
        system += f"\n\nCara menjawab: {mode_instruction}"

    return system


def _build_image_analysis_fallback_prompt(
    sum_status: str,
    sum_text: str,
    rec_text: str,
    child_name: str,
    requires_review: bool,
    status_emoji: str,
) -> str:
    """Hardcoded fallback kalau DB belum punya prompt key tanya_peri_image_response_*."""
    return (
        f"\n\n=== KONTEKS PENTING ===\n"
        f"USER SUDAH MENGIRIM FOTO GIGI ANAK dan SUDAH dianalisis oleh sistem AI. "
        f"JANGAN minta user upload foto lagi.\n\n"
        f"📸 Hasil Analisis Foto Gigi {child_name}:\n"
        f"{status_emoji} Status: {sum_status}\n"
        f"📝 Ringkasan: {sum_text}\n"
        f"💡 Saran: {rec_text}\n\n"
        f"INSTRUKSI MERESPON:\n"
        f"1. JANGAN minta user upload foto — foto sudah ada dan sudah dianalisis.\n"
        f"2. Acknowledge hasil analisis singkat (1-2 kalimat empati).\n"
        f"3. Jawab pertanyaan user kalau ada.\n"
        f"4. Hasil sudah ditampilkan di card terpisah — cukup komentar singkat.\n"
        f"5. {'Anjurkan konsultasi dokter gigi.' if requires_review else 'Tetap pantau dan jaga kebiasaan sikat gigi.'}\n"
        f"=== AKHIR KONTEKS ===\n"
    )


def _build_messages(state: AgentState, system_prompt: str) -> list:
    """Build LangChain messages: SystemMessage + history dari state.messages."""
    lc_messages = [SystemMessage(content=system_prompt)]
    # state.messages already in BaseMessage format (from build_initial_state)
    lc_messages.extend(state.messages)
    return lc_messages


def _get_llm_for_state(state: AgentState):
    """Get LLM dengan possible per-agent override."""
    from app.config.llm import get_llm

    agents_selected = state.agents_selected
    agent_configs = state.control.agent_configs

    provider_override = state.rnd.llm_provider
    model_override = state.rnd.llm_model

    # Single-agent mode → use agent-specific config kalau ada
    if len(agents_selected) == 1 and not provider_override:
        agent_key = agents_selected[0]
        agent_conf = agent_configs.get(agent_key, {})
        if agent_conf.get("llm_provider"):
            provider_override = agent_conf["llm_provider"]
        if agent_conf.get("llm_model"):
            model_override = agent_conf["llm_model"]

    return get_llm(
        streaming=True,
        provider=provider_override,
        model=model_override,
        temperature=state.rnd.llm_temperature,
        max_tokens=state.rnd.llm_max_tokens,
    ), provider_override, model_override


def _get_child_name(state: AgentState) -> str:
    ctx = state.user_context.model_dump()
    child = ctx.get("child") or {}
    return child.get("nickname") or child.get("full_name") or "si kecil"


# =============================================================================
# LangGraph node
# =============================================================================


async def generate_node(state: AgentState) -> dict:
    """
    LangGraph node: generate final response from LLM.

    Returns state update dict. Token streaming captured via LangGraph
    astream_events() di stream_adapter, NOT yield dari node.

    Early return cases (no LLM call):
    - needs_clarification + clarification_data: stream adapter emit clarify event
    - quick_reply_data: stream adapter emit quick_reply event
    """
    from app.config.llm import get_model_name, get_provider_name
    from app.config.observability import build_trace_config, trace_generation, _safe_dict_for_trace

    thinking_step_num = len(state.thinking_steps) + 1

    # Build thinking label based on agents
    intent_label = {
        "kb_dental": "Menyiapkan jawaban kesehatan gigi...",
        "user_profile": f"Membaca data {_get_child_name(state)}...",
        "rapot_peri": f"Mengecek rapot sikat gigi {_get_child_name(state)}...",
        "mata_peri": "Menganalisis hasil scan gigi...",
        "cerita_peri": "Mengecek progress Cerita Peri...",
        "app_faq": "Mencari info aplikasi...",
        "janji_peri": "Mencari informasi dokter...",
    }

    if len(state.agents_selected) > 1:
        thinking_label = "Merangkum informasi untuk kamu..."
    elif state.agents_selected:
        thinking_label = intent_label.get(state.agents_selected[0], "Menyiapkan respons...")
    else:
        thinking_label = "Menyiapkan respons..."

    thinking_done = ThinkingStep(step=thinking_step_num, label=thinking_label, done=True)

    # ── Early return: clarification ───────────────────────────────────────────
    if state.needs_clarification and state.clarification_data:
        return {
            "thinking_steps": [*state.thinking_steps, thinking_done],
            # clarification_data tetap di state — stream adapter akan emit clarify event
        }

    # ── Early return: quick_reply ─────────────────────────────────────────────
    if state.quick_reply_data:
        return {
            "thinking_steps": [*state.thinking_steps, thinking_done],
        }

    # ── Build prompt + messages ────────────────────────────────────────────────
    system_prompt = _build_system_prompt(state)
    lc_messages = _build_messages(state, system_prompt)

    prompt_debug = None
    if state.rnd.include_prompt_debug:
        prompt_debug = {
            "system": system_prompt,
            "messages": [
                {"role": type(m).__name__, "content": m.content}
                for m in lc_messages
            ],
        }

    llm, provider, model = _get_llm_for_state(state)

    # ── LLM streaming call ────────────────────────────────────────────────────
    t_start = time.monotonic()
    t_first_token: Optional[float] = None
    full_response = ""
    output_tokens = 0
    success = True
    error_msg: Optional[str] = None

    # Per-call trace config (Langfuse)
    trace_config = build_trace_config(
        state={
            "session_id": state.session.session_id,
            "user_context": state.user_context.model_dump(),
            "response_mode": state.session.response_mode,
        },
        agent_name="generate",
    )

    # Extract last user message untuk trace input
    user_msg_text = ""
    last_user_msg = next(
        (m for m in reversed(state.messages) if hasattr(m, "type") and m.type == "human"),
        None,
    )
    if last_user_msg:
        content = last_user_msg.content
        if isinstance(content, str):
            user_msg_text = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_msg_text = part.get("text", "")
                    break

    # Phase 4.5: full agent_results capture untuk Langfuse diagnostic
    agent_results_snapshot = state.agent_results
    image_analysis_snapshot = state.image_analysis
    memory_snapshot = state.memory.model_dump()

    async with trace_generation(
        name="generate",
        model=get_model_name(provider=provider, model=model),
        system_prompt=system_prompt,
        messages=lc_messages,
        user_message=user_msg_text,
        metadata={
            "agents_used": list(agent_results_snapshot.keys()),
            "response_mode": state.session.response_mode,
            "has_image_analysis": bool(image_analysis_snapshot),
            "child_name": _get_child_name(state),
            "provider": get_provider_name(provider=provider),
            "agent_results_keys": list(agent_results_snapshot.keys()),
            "image_analysis_present": bool(image_analysis_snapshot),
            "kb_docs_count": len(state.retrieved_docs),
            "memory_summaries_count": len(memory_snapshot.get("session_summaries", [])),
            "memory_facts_count": len(memory_snapshot.get("user_facts", [])),
        },
    ) as gen_span:
        try:
            async for chunk in llm.astream(lc_messages, config=trace_config):
                token = chunk.content
                if not token:
                    continue
                if t_first_token is None:
                    t_first_token = time.monotonic()
                full_response += token
                output_tokens += 1
                # NOTE: token-level emission ke FE happens via LangGraph
                # astream_events('on_chat_model_stream') captured di stream_adapter.
                # Node itself doesn't yield.
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(f"[generate_node] LLM streaming error: {e}", exc_info=True)
            if gen_span:
                gen_span.update(
                    output=f"[ERROR] {str(e)[:500]}",
                    level="ERROR",
                    status_message=str(e)[:200],
                )

        # Capture output ke generation span
        if gen_span and success:
            try:
                gen_span.update(
                    output=full_response[:5000] if full_response else "(empty)",
                    usage_details={"output": output_tokens},
                    metadata={
                        "agent_results_summary": _safe_dict_for_trace(agent_results_snapshot),
                        "image_analysis_summary": _safe_dict_for_trace(image_analysis_snapshot)
                            if image_analysis_snapshot else None,
                        "memory_context_summary": _safe_dict_for_trace(memory_snapshot)
                            if memory_snapshot else None,
                    },
                )
            except Exception:
                pass  # defensive

    t_end = time.monotonic()
    total_latency_ms = int((t_end - t_start) * 1000)
    ttft_ms = int((t_first_token - t_start) * 1000) if t_first_token else None
    generation_ms = int((t_end - t_first_token) * 1000) if t_first_token else None
    tps = round(output_tokens / (generation_ms / 1000), 1) if generation_ms and generation_ms > 0 else None

    # Build llm_metadata untuk done event (FE consume)
    metadata: dict[str, Any] = {
        "model": get_model_name(provider=provider, model=model),
        "provider": get_provider_name(provider=provider),
        "agents_used": list(agent_results_snapshot.keys()),
        "response_mode": state.session.response_mode,
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "tokens_per_second": tps,
        "output_tokens_approx": output_tokens,
    }

    # Tag has_image_analysis kalau ada session_summary (FE render compact card)
    if isinstance(image_analysis_snapshot, dict) and image_analysis_snapshot.get("session_summary"):
        metadata["has_image_analysis"] = True
        if state.scan_session_id:
            metadata["scan_session_id"] = state.scan_session_id
        ss = image_analysis_snapshot.get("session_summary", {})
        metadata["image_analysis_summary"] = {
            "summary_status": ss.get("summary_status"),
            "summary_text": ss.get("summary_text"),
            "recommendation_text": ss.get("recommendation_text"),
            "requires_dentist_review": ss.get("requires_dentist_review", False),
        }
        # Forward overlay artifact URLs (5-view scan)
        image_results = image_analysis_snapshot.get("results", []) or []
        if image_results and isinstance(image_results, list):
            artifacts_list = []
            for ir in image_results:
                if not isinstance(ir, dict):
                    continue
                artifacts = ir.get("artifacts") or {}
                if not isinstance(artifacts, dict):
                    continue
                view_type = ir.get("view_type")
                crop_url = artifacts.get("crop_image_url")
                overlay_url = artifacts.get("overlay_image_url")
                if crop_url or overlay_url:
                    artifacts_list.append({
                        "view_type": view_type,
                        "crop_image_url": crop_url,
                        "overlay_image_url": overlay_url,
                    })
            if artifacts_list:
                metadata["image_artifacts"] = artifacts_list

    # Build LLM call log
    call_log = LLMCallLog(
        prompt_key=f"generate_{state.session.response_mode}",
        model=get_model_name(provider=provider, model=model),
        provider=get_provider_name(provider=provider),
        node="generate",
        output_tokens=output_tokens,
        latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        success=success,
        error_message=error_msg,
        metadata={
            "agents_used": list(agent_results_snapshot.keys()),
            "response_mode": state.session.response_mode,
            "generation_ms": generation_ms,
            "tokens_per_second": tps,
        },
    )

    # Build state update dict
    update: dict[str, Any] = {
        "final_response": full_response,
        "llm_metadata": metadata,
        "thinking_steps": [*state.thinking_steps, thinking_done],
        "llm_call_logs": [*state.llm_call_logs, call_log],
    }

    if prompt_debug is not None:
        update["prompt_debug"] = prompt_debug

    return update
