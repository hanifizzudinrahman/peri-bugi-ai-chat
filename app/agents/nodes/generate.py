"""
Node: generate
Generate response dari LLM dengan streaming token per token.

Update v2:
- Support response_mode (simple/medium/detailed) → pilih prompt template berbeda
- Build context dari agent_results (multi-agent)
- Inject memory context (L2+L3) ke system prompt
- Support quick_reply event
"""
import json
import time
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.config.llm import get_llm, get_model_name, get_provider_name
from app.schemas.chat import (
    LLMCallLogPayload,
    make_clarify_event,
    make_quick_reply_event,
    make_thinking_event,
    make_token_event,
)


def _build_system_prompt(state: AgentState) -> str:
    """
    Build system prompt dengan:
    1. Persona Tanya Peri (dari DB atau default)
    2. Data user & anak dari user_context
    3. Memory context (L2 summary + L3 facts)
    4. Hasil agent (docs, profile data, dll)
    5. Response mode instructions
    """
    prompts = state.get("prompts", {})

    # Override system prompt penuh (RnD mode)
    if "_override_system" in prompts:
        return prompts["_override_system"]

    # Persona base
    persona = prompts.get(
        "persona_system",
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak dari aplikasi Peri Bugi. "
        "Kamu ramah, sabar, dan berbicara dengan bahasa yang mudah dipahami orang tua Indonesia. "
        "Jangan pernah memberikan diagnosis medis langsung. "
        "Selalu sarankan konsultasi dokter gigi untuk masalah serius.",
    )

    # Inject user & child data
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"
    child_age = f"{child.get('age_years')} tahun" if child.get("age_years") else "?"

    system = persona.replace("{user_name}", user_name)
    system = system.replace("{child_name}", child_name)
    system = system.replace("{child_age}", child_age)

    # Brushing data dari user_context (juga bisa dari rapot_peri agent di Phase 2)
    brushing = ctx.get("brushing")
    if brushing:
        system += (
            f"\n\nData sikat gigi {child_name}: "
            f"streak {brushing.get('current_streak', 0)} hari, "
            f"rekor terbaik {brushing.get('best_streak', 0)} hari."
        )

    # Mata Peri scan result
    mata_peri = ctx.get("mata_peri_last_result")
    if mata_peri and mata_peri.get("summary_text"):
        system += (
            f"\n\nHasil scan gigi terakhir {child_name} "
            f"({mata_peri.get('scan_date', 'tidak diketahui')}): "
            f"{mata_peri.get('summary_text')}. "
            f"Status: {mata_peri.get('summary_status', 'tidak diketahui')}."
        )

    # ── Memory context (L2 + L3) ──────────────────────────────────────────────
    memory = state.get("memory_context", {})

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
    agent_results = state.get("agent_results", {})

    # KB Dental docs
    kb_result = agent_results.get("kb_dental", {})
    kb_docs = kb_result.get("docs", []) or state.get("retrieved_docs", [])
    if kb_docs:
        docs_text = "\n\n".join(kb_docs[:3])
        system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

    # App FAQ docs
    faq_result = agent_results.get("app_faq", {})
    faq_docs = faq_result.get("docs", [])
    if faq_docs:
        faq_text = "\n\n".join(faq_docs[:3])
        system += f"\n\nInfo aplikasi yang relevan:\n{faq_text}"

    # Image analysis (Mata Peri agent - Phase 2 / Tanya Peri image - Phase 4 Batch B)
    # Phase 6: prompts pindah ke DB — dipilih berdasarkan response_mode.
    image_analysis = state.get("image_analysis")
    if image_analysis:
        # AnalyzeResponse dari ai-cv punya struktur:
        # { session_summary: { summary_status, summary_text, recommendation_text, ... }, results: [...] }
        # Backward compat: kalau dari _analyze_image lama (deprecated), fallback ke field 'summary'.
        session_summary = image_analysis.get("session_summary") if isinstance(image_analysis, dict) else None

        if session_summary:
            sum_status = session_summary.get("summary_status") or "tidak diketahui"
            sum_text = session_summary.get("summary_text") or ""
            rec_text = session_summary.get("recommendation_text") or ""
            requires_review = session_summary.get("requires_dentist_review", False)

            # Status emoji untuk visual reinforcement
            status_emoji = {
                "ok": "✅",
                "perlu_perhatian": "⚠️",
                "perlu_dokter": "🚨",
                "gagal": "❌",
            }.get(sum_status, "📋")

            # Pull prompt template dari DB berdasarkan response_mode aktif.
            # Key: tanya_peri_image_response_{simple|medium|detailed}
            response_mode = state.get("response_mode", "simple")
            image_prompt_key = f"tanya_peri_image_response_{response_mode}"
            image_prompt_template = prompts.get(image_prompt_key)

            if image_prompt_template:
                # Render template dengan placeholder
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
                    # Prompt ada placeholder yang tidak kita support — log + fallback
                    logger.warning(
                        f"[generate] Prompt {image_prompt_key} pakai placeholder "
                        f"yang tidak dikenal: {e}. Fallback ke hardcoded format."
                    )
                    system += _build_image_analysis_fallback_prompt(
                        sum_status, sum_text, rec_text, child_name,
                        requires_review, status_emoji,
                    )
            else:
                # DB tidak punya prompt (seeder belum jalan / disabled) → hardcoded fallback
                system += _build_image_analysis_fallback_prompt(
                    sum_status, sum_text, rec_text, child_name,
                    requires_review, status_emoji,
                )
        else:
            # Backward compat — old _analyze_image format (Phase 2)
            system += (
                f"\n\nHasil analisis gambar gigi: "
                f"{image_analysis.get('summary', 'tidak tersedia')}."
            )

    # ── Response mode instructions ────────────────────────────────────────────
    response_mode = state.get("response_mode", "simple")
    mode_key = f"generate_{response_mode}"
    mode_instruction = prompts.get(mode_key)

    if not mode_instruction:
        # Default instructions per mode
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


def _build_messages(state: AgentState, system_prompt: str) -> list:
    """Build list LangChain messages dari state."""
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in state.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def _get_llm_for_state(state: AgentState):
    """Dapatkan LLM — bisa pakai override per-agent atau default global."""
    # Cek apakah ada agent-specific config
    agents_selected = state.get("agents_selected", [])
    agent_configs = state.get("agent_configs", {})

    provider_override = state.get("llm_provider_override")
    model_override = state.get("llm_model_override")

    # Jika ada satu agent dengan config khusus dan tidak ada multi-agent
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
        temperature=state.get("llm_temperature_override"),
        max_tokens=state.get("llm_max_tokens_override"),
    ), provider_override, model_override


async def generate_node(state: AgentState) -> AsyncIterator[str]:
    """Node: generate — stream response dari LLM."""
    thinking_step = len(state.get("thinking_steps", [])) + 1

    agents_selected = state.get("agents_selected", [])
    intent_label = {
        "kb_dental": "Menyiapkan jawaban kesehatan gigi...",
        "user_profile": f"Membaca data {_get_child_name(state)}...",
        "rapot_peri": f"Mengecek rapot sikat gigi {_get_child_name(state)}...",
        "mata_peri": "Menganalisis hasil scan gigi...",
        "cerita_peri": "Mengecek progress Cerita Peri...",
        "app_faq": "Mencari info aplikasi...",
        "janji_peri": "Mencari informasi dokter...",
    }

    if len(agents_selected) > 1:
        thinking_label = "Merangkum informasi untuk kamu..."
    elif agents_selected:
        thinking_label = intent_label.get(agents_selected[0], "Menyiapkan respons...")
    else:
        thinking_label = "Menyiapkan respons..."

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=False)

    # Handle clarification (sudah ada di state dari check sebelumnya)
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

    # Handle quick reply
    if state.get("quick_reply_data"):
        qr = state["quick_reply_data"]
        yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
        yield make_quick_reply_event(
            qr_type=qr.get("type", "single_select"),
            question=qr.get("question"),
            options=qr.get("options", []),
            allow_multiple=qr.get("allow_multiple", False),
            dismissible=qr.get("dismissible", True),
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

    llm, _provider, _model = _get_llm_for_state(state)

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
    total_latency_ms = int((t_end - t_start) * 1000)
    ttft_ms = int((t_first_token - t_start) * 1000) if t_first_token else None
    generation_ms = int((t_end - t_first_token) * 1000) if t_first_token else None
    tps = round(output_tokens / (generation_ms / 1000), 1) if generation_ms and generation_ms > 0 else None

    state["final_response"] = full_response
    # Build llm_metadata for client. Tambahan field has_image_analysis +
    # scan_session_id untuk FE render compact card di chat bubble.
    metadata: dict[str, Any] = {
        "model": get_model_name(provider=_provider, model=_model),
        "provider": get_provider_name(provider=_provider),
        "agents_used": list(state.get("agent_results", {}).keys()),
        "response_mode": state.get("response_mode", "simple"),
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "tokens_per_second": tps,
        "output_tokens_approx": output_tokens,
    }
    # Tag has_image_analysis kalau ada session_summary structure
    # (Tanya Peri image analysis dari Phase 4 Batch B).
    image_analysis_state = state.get("image_analysis")
    if isinstance(image_analysis_state, dict) and image_analysis_state.get("session_summary"):
        metadata["has_image_analysis"] = True
        scan_session_id = state.get("scan_session_id")
        if scan_session_id:
            metadata["scan_session_id"] = scan_session_id
        # Include compact subset of session_summary buat FE render card tanpa fetch ulang
        ss = image_analysis_state.get("session_summary", {})
        metadata["image_analysis_summary"] = {
            "summary_status": ss.get("summary_status"),
            "summary_text": ss.get("summary_text"),
            "recommendation_text": ss.get("recommendation_text"),
            "requires_dentist_review": ss.get("requires_dentist_review", False),
        }
        # Phase 6 — Forward overlay artifact URLs (mask gigi hijau + karies merah).
        # Konsisten dengan Mata Peri 5-view: same pattern dari ai-cv pipeline.
        # FE detail screen pakai ini untuk render foto dengan annotation.
        # NOTE: URL ini signed dengan TTL 1 jam. Untuk akses ulang setelah expired,
        # FE harus fetch detail dari /mata-peri/sessions/{id} yang regenerate URL fresh.
        image_results = image_analysis_state.get("results", []) or []
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
    state["llm_metadata"] = metadata

    log = LLMCallLogPayload(
        prompt_key=f"generate_{state.get('response_mode', 'simple')}",
        model=get_model_name(provider=_provider, model=_model),
        provider=get_provider_name(provider=_provider),
        node="generate",
        output_tokens=output_tokens,
        latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        success=success,
        error_message=error_msg,
        metadata={
            "agents_used": list(state.get("agent_results", {}).keys()),
            "response_mode": state.get("response_mode", "simple"),
            "generation_ms": generation_ms,
            "tokens_per_second": tps,
        },
    )
    state["llm_call_logs"].append(log.model_dump())

    # Emit suggestion chips jika ada
    if state.get("suggestion_chips"):
        from app.schemas.chat import make_suggestions_event
        yield make_suggestions_event(state["suggestion_chips"])


def _get_child_name(state: AgentState) -> str:
    ctx = state.get("user_context", {})
    child = ctx.get("child") or {}
    return child.get("nickname") or child.get("full_name") or "si kecil"


def _build_image_analysis_fallback_prompt(
    sum_status: str,
    sum_text: str,
    rec_text: str,
    child_name: str,
    requires_review: bool,
    status_emoji: str,
) -> str:
    """
    Hardcoded fallback prompt untuk image analysis kalau DB belum punya
    prompt key `tanya_peri_image_response_{simple|medium|detailed}`.

    Pattern: konteks + instruction berbasis poin (bukan paragraph).
    Pakai bahasa explicit "JANGAN minta upload" karena small LLM mudah miss.

    Catatan: setelah seeder jalan, function ini tidak akan ke-trigger.
    Sengaja simpan untuk safety net + dev tanpa DB seed.
    """
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
