"""
Node: generate
Generate response dari LLM dengan streaming token per token.

Update v2:
- Support response_mode (simple/medium/detailed) → pilih prompt template berbeda
- Build context dari agent_results (multi-agent)
- Inject memory context (L2+L3) ke system prompt
- Support quick_reply event

Update v3 (Step 2b):
- Smalltalk path: jika state.is_smalltalk=True (set by pre_router), pakai
  lean prompt minimal (Q2B: keep nama anak doang). SKIP injection of
  brushing, mata_peri_last_result, memory_context, agent_results.
"""
import json
import logging
import time
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.config.llm import get_llm, get_model_name, get_provider_name
from app.config.observability import build_trace_config
from app.schemas.chat import (
    LLMCallLogPayload,
    make_clarify_event,
    make_quick_reply_event,
    make_thinking_event,
    make_token_event,
)

logger = logging.getLogger(__name__)


def _build_smalltalk_system_prompt(state: AgentState, prompts: dict) -> str:
    """
    Lean smalltalk system prompt (Step 2b — Q2B decision).

    Triggered when state.is_smalltalk=True (set by pre_router_node via strict
    regex+length+keyword check).

    SKIP injection: brushing, mata_peri_last_result, memory_context, agent_results.
    KEEP: persona + nama orang tua + nama anak.

    Q3A graceful fallback: kalau 'generate_smalltalk' tidak ada di state.prompts
    (DB seeder belum jalan / key inactive), pakai inline default + warning log.
    """
    SMALLTALK_FALLBACK = (
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak yang ramah dan hangat.\n\n"
        "User baru kasih sapaan singkat. Bales dengan hangat dan natural — JANGAN dump info detail.\n\n"
        "ATURAN:\n"
        "- Sapa user dengan nama orang tua-nya\n"
        "- Sebut nama anak (kalau diketahui) untuk personalisasi\n"
        "- Tanyakan ada yang bisa kamu bantu hari ini\n"
        "- Maksimal 1-2 kalimat\n"
        "- 1-2 emoji wajar (🧚 ✨ 👋 💙)\n"
        "- JANGAN sebutkan hasil scan, streak, atau detail teknis lain\n"
        "- JANGAN tanya pertanyaan medis spesifik\n\n"
        "KONTEKS USER:\n"
        "- Orang tua: {user_name}\n"
        "- Anak: {child_name}\n"
    )

    template = prompts.get("generate_smalltalk")
    if not template:
        logger.warning(
            "[generate] 'generate_smalltalk' prompt not in state.prompts — "
            "using inline fallback. Run scripts/seed_prompts.py to populate DB."
        )
        template = SMALLTALK_FALLBACK

    # Inject {user_name} + {child_name}
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"

    rendered = template.replace("{user_name}", user_name)
    rendered = rendered.replace("{child_name}", child_name)

    return rendered


def _sanitize_session_summaries(
    summaries: list,
    state: AgentState,
) -> tuple[list[str], bool]:
    """Sanitize session summaries — detect entries yang reference feature OFF.
    
    Bagian C v3 — Part C2: Memory summaries dari background job kadang punya
    detail data spesifik (e.g., "Modul 4 masih terkunci"). Kalau feature
    sekarang OFF tapi summary bahas fitur itu, LLM bisa halusinasi data
    "current" pakai detail dari summary lama.
    
    Strategy: detect summaries yang mention feature OFF via keyword match.
    Kalau detected, kembalikan flag `has_stale=True` supaya generate.py
    inject disclaimer.
    
    Args:
        summaries: List of summary strings dari memory.
        state: AgentState — used untuk derive features yang OFF.
    
    Returns:
        Tuple (sanitized_summaries, has_stale_summary):
        - sanitized_summaries: Same list (we don't filter — just flag)
        - has_stale_summary: True kalau ada summary yang reference feature OFF
    
    Note:
        We DON'T remove summaries — masih useful sebagai topical context.
        Cuma flag has_stale supaya disclaimer di-inject.
        Ada risk false-positive (keyword match), tapi disclaimer tone-nya
        soft enough supaya LLM cuma extra hati-hati, bukan ignore total.
    """
    if not summaries:
        return [], False
    
    # Get unavailable features (kalau ada)
    try:
        from app.agents.tools.registry import iter_tool_specs, AGENT_KEY_TO_FEATURE_NAME
        all_specs = iter_tool_specs()
        allowed_agents = set(state.control.allowed_agents or []) if hasattr(state, 'control') else set()
        if not allowed_agents and hasattr(state, 'get'):
            # Fallback: try state.get for dict-like access
            user_ctx = state.get("user_context", {})
            allowed_agents = set(user_ctx.get("allowed_agents") or [])
        
        unavailable_keywords: list[str] = []
        for spec in all_specs:
            if spec.required_agent and spec.required_agent not in allowed_agents:
                # Add feature name + agent_key ke keyword list
                feature = AGENT_KEY_TO_FEATURE_NAME.get(spec.agent_key, "")
                if feature:
                    unavailable_keywords.append(feature.lower())
                # Also add common keyword variants per feature
                if spec.agent_key == "cerita_peri" or spec.agent_key == "cerita_module_detail":
                    unavailable_keywords.extend(["modul", "cerita peri"])
                elif spec.agent_key == "mata_peri" or spec.agent_key == "mata_peri_scan_detail":
                    unavailable_keywords.extend(["scan gigi", "mata peri", "foto gigi"])
                elif spec.agent_key == "rapot_peri" or spec.agent_key.startswith("rapot_peri_"):
                    unavailable_keywords.extend(["sikat gigi", "rapot peri", "streak", "badge"])
                elif spec.agent_key == "caries_risk":
                    unavailable_keywords.extend(["kuesioner", "risiko karies"])
                elif spec.agent_key == "tips":
                    unavailable_keywords.extend(["tips parenting", "tips harian"])
        
        unavailable_keywords = list(set(unavailable_keywords))
    except Exception as e:
        logger.warning(f"[_sanitize_session_summaries] Failed to derive keywords: {e}")
        return list(summaries), False
    
    if not unavailable_keywords:
        # All features ON — no sanitization needed
        return list(summaries), False
    
    # Check each summary untuk reference ke feature OFF
    has_stale = False
    for summary in summaries:
        if not isinstance(summary, str):
            continue
        summary_lower = summary.lower()
        if any(kw in summary_lower for kw in unavailable_keywords):
            has_stale = True
            logger.info(
                f"[_sanitize_session_summaries] Stale summary detected (references "
                f"feature OFF): {summary[:80]!r}"
            )
            break  # one match enough — disclaimer will cover all
    
    return list(summaries), has_stale


def _build_system_prompt(state: AgentState) -> str:
    """
    Build system prompt dengan:
    1. Persona Tanya Peri (dari DB atau default)
    2. Data user & anak dari user_context
    3. Memory context (L2 summary + L3 facts)
    4. Hasil agent (docs, profile data, dll)
    5. Response mode instructions

    Step 2b: Early branch ke lean smalltalk prompt jika state.is_smalltalk=True
    (set by pre_router via strict regex check). Skip injection brushing/scan/memory.
    """
    prompts = state.get("prompts", {})

    # Override system prompt penuh (RnD mode) — UNCHANGED
    if "_override_system" in prompts:
        return prompts["_override_system"]

    # ─────────────────────────────────────────────────────────────────────────
    # SMALLTALK PATH (Step 2b — Q2B): early return with lean prompt
    # ─────────────────────────────────────────────────────────────────────────
    if state.get("is_smalltalk"):
        return _build_smalltalk_system_prompt(state, prompts)

    # ─────────────────────────────────────────────────────────────────────────
    # NORMAL PATH (existing logic — UNCHANGED below)
    # ─────────────────────────────────────────────────────────────────────────
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
        # Bagian C v3 — Part C2: Sanitize summaries.
        # Memory summaries dari background job. Kalau session lama bahas feature
        # yang sekarang OFF, summary bisa leak info "modul terkunci" yang bikin
        # LLM halusinasi data pas user tanya feature OFF.
        # Strategy: detect summaries yang reference unavailable features, lalu
        # PREPEND disclaimer ("info historis, fitur sekarang OFF").
        sanitized_summaries, has_stale = _sanitize_session_summaries(
            summaries, state
        )

        if sanitized_summaries:
            summaries_text = "\n".join(f"- {s}" for s in sanitized_summaries)
            system += f"\n\nRingkasan percakapan sebelumnya:\n{summaries_text}"

            if has_stale:
                # Loud disclaimer ke LLM bahwa summaries punya info dari feature OFF
                system += (
                    f"\n\n⚠️ DISCLAIMER untuk Ringkasan di atas:\n"
                    f"Beberapa entry di ringkasan membahas feature yang SAAT INI TIDAK AKTIF "
                    f"di akun user. Detail data spesifik (status modul, badge, scan, dll) "
                    f"di ringkasan itu adalah info HISTORIS dari session lama.\n"
                    f"\n"
                    f"PERATURAN:\n"
                    f"- Pakai ringkasan untuk konteks topik percakapan saja.\n"
                    f"- JANGAN treat angka/status/data spesifik di ringkasan sebagai "
                    f"data CURRENT user.\n"
                    f"- Untuk data current, HANYA gunakan hasil tool call di session ini.\n"
                    f"- Kalau tidak ada tool call, JANGAN karang data."
                )

    facts = memory.get("user_facts", [])
    if facts:
        facts_text = "\n".join(
            f"- {f.get('value', '')}" for f in facts if f.get("value")
        )
        system += f"\n\nYang kamu ketahui tentang user ini:\n{facts_text}"

    # ── Agent results (Bagian C: registry-driven prompt injection) ────────────
    agent_results = state.get("agent_results", {})

    # FIX (Bagian C): Loop semua ToolSpec yang punya prompt_injector dan inject
    # data dari agent_results ke system prompt. Ini fix CRITICAL BUG dimana
    # tool result sebelumnya HILANG (di-bridge tapi tidak di-consume), bikin
    # LLM halusinasi. Sekarang setiap tool inject data-nya sendiri.
    #
    # KENAPA pakai loop instead of inline if/elif:
    # - Tambah tool baru = update 1 file (tool itu sendiri), tidak perlu sentuh generate.py
    # - Impossible to forget — registry validate at startup
    # - Pattern consistent untuk all tools
    from app.agents.tools.registry import iter_tool_specs

    response_mode = state.get("response_mode", "simple")
    
    # SAFETY: ensure registry populated. iter_tool_specs() returns empty list
    # kalau tools/__init__.py belum di-import (edge case in test/standalone). 
    # Force-import to trigger registration.
    _all_specs = iter_tool_specs()
    if not _all_specs:
        try:
            import app.agents.tools  # noqa: F401 — trigger __init__ side effects
            _all_specs = iter_tool_specs()
            if not _all_specs:
                logger.error(
                    "[generate] Tool registry KOSONG setelah force-import. "
                    "Tool injection skipped — LLM akan halusinasi tanpa data!"
                )
        except Exception as e:
            logger.error(f"[generate] Failed to force-load tools registry: {e}")
            _all_specs = []
    
    for spec in _all_specs:
        if spec.prompt_injector is None:
            # Tool tidak butuh prompt injection (e.g., analyze_chat_image
            # pakai template terpisah di bawah)
            continue

        agent_data = agent_results.get(spec.agent_key)
        if not agent_data:
            # Tool tidak di-call this turn, atau result-nya empty
            continue

        try:
            injection_text = spec.prompt_injector(
                agent_data, child_name, prompts, response_mode
            )
        except Exception as e:
            logger.error(
                f"[generate] prompt_injector for '{spec.tool_name}' raised: {e}",
                exc_info=True,
            )
            injection_text = ""

        if injection_text:
            system += injection_text

    # Bagian C v2 — Layer 4: Inject "tool unavailable" warning ke system prompt
    # Kalau LLM panggil tool yang tidak available (gated off via allowed_agents),
    # tool_bridge_node sudah mark di state.unavailable_tools. Inject penjelasan
    # ke LLM supaya kasih honest answer, bukan halusinasi data.
    #
    # Bagian C v3: Combine 2 sources untuk feature unavailable detection:
    # 1. state.unavailable_tools — LLM HALUSINASI tool name (Layer 3 catch)
    # 2. state.detected_unavailable_features — LLM CORRECTLY skip tool call,
    #    tapi user message reference feature OFF (agent_node detect via
    #    keyword match in user message)
    unavailable_tools = state.get("unavailable_tools", []) or []
    detected_features = state.get("detected_unavailable_features", []) or []
    no_tools_reason = state.get("no_tools_reason")

    # Bagian C v3 RESILIENCY FIX: Fallback detection in generate_node.
    # Kalau state.detected_unavailable_features empty (e.g., AgentState belum
    # punya field karena state.py belum di-apply), do detection di sini juga.
    # Pake field yang SUDAH ADA: state.messages (last user msg) + 
    # state.control.allowed_agents + registry.
    if not detected_features:
        try:
            from app.agents.tools.registry import iter_tool_specs, AGENT_KEY_TO_FEATURE_NAME
            
            # Get last user message
            last_user_msg = ""
            messages = state.get("messages", [])
            for msg in reversed(messages):
                msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
                if msg_type == "human":
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                    if isinstance(content, str):
                        last_user_msg = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                last_user_msg = part.get("text", "")
                                break
                    break
            
            # Get unavailable features
            allowed = set()
            ctrl = state.get("control", None)
            if ctrl and hasattr(ctrl, "allowed_agents"):
                allowed = set(ctrl.allowed_agents or [])
            elif ctrl and isinstance(ctrl, dict):
                allowed = set(ctrl.get("allowed_agents", []) or [])
            
            unavailable_features_set = set()
            for spec in iter_tool_specs():
                if spec.required_agent and spec.required_agent not in allowed:
                    feature = AGENT_KEY_TO_FEATURE_NAME.get(
                        spec.agent_key, spec.agent_key.replace("_", " ").title()
                    )
                    unavailable_features_set.add(feature)
            
            # Keyword matching (mirror agent_node logic)
            feature_keywords = {
                "Cerita Peri": ["cerita peri", "cerita-cerita peri", "modul peri", "modul cerita", "story peri"],
                "Modul Cerita Peri": ["modul peri", "modul cerita", "modul ke-", "isi modul"],
                "Mata Peri (Scan Gigi)": ["mata peri", "scan gigi", "foto gigi", "analisa gigi"],
                "Detail Hasil Scan": ["scan terakhir", "hasil scan", "detail scan"],
                "Rapot Sikat Gigi": ["rapot peri", "rapot sikat", "stats sikat"],
                "Riwayat Sikat Gigi": ["riwayat sikat", "history sikat", "kapan sikat"],
                "Badge Sikat Gigi": ["badge", "achievement", "milestone sikat"],
                "Kuesioner Risiko Karies": ["kuesioner", "risiko karies", "caries risk"],
                "Tips Parenting Harian": ["tips parenting", "tips hari ini"],
            }
            
            msg_lower = (last_user_msg or "").lower()
            detected_fallback: list[str] = []
            seen_fallback = set()
            for feature in sorted(unavailable_features_set):
                keywords = feature_keywords.get(feature, [])
                if not keywords:
                    continue
                if any(kw in msg_lower for kw in keywords):
                    if feature not in seen_fallback:
                        detected_fallback.append(feature)
                        seen_fallback.add(feature)
            
            if detected_fallback:
                logger.info(
                    f"[generate] Fallback detection of unavailable features: "
                    f"{detected_fallback} (state field was empty, possibly state.py "
                    f"not deployed)"
                )
                detected_features = detected_fallback
                # Also infer no_tools_reason kalau LLM tidak panggil tool
                if not unavailable_tools and not no_tools_reason:
                    # Check if tool_calls empty in last AI message
                    no_tools_reason = "feature_unavailable"
        except Exception as e:
            logger.warning(f"[generate] Fallback detection failed: {e}")

    # Combine kedua sources jadi 1 list features (deduped)
    all_unavailable_features: list[str] = []
    seen_features = set()

    # Source 1: dari unavailable_tools (Layer 3) — convert tool_name to feature
    if unavailable_tools:
        from app.agents.tools.registry import get_friendly_feature_name
        for tool_name in unavailable_tools:
            feature = get_friendly_feature_name(tool_name)
            if feature and feature not in seen_features:
                all_unavailable_features.append(feature)
                seen_features.add(feature)

    # Source 2: dari detected_unavailable_features (agent_node detect)
    for feature in detected_features:
        if feature and feature not in seen_features:
            all_unavailable_features.append(feature)
            seen_features.add(feature)

    if all_unavailable_features:
        features_text = ", ".join(all_unavailable_features)

        # Determine context based on signal source
        if no_tools_reason == "feature_unavailable":
            context_intro = (
                f"User bertanya tentang fitur yang TIDAK AKTIF di akun mereka: "
                f"**{features_text}**.\n"
                f"\nLLM CORRECT TIDAK panggil tool (karena tool tidak ada di list available)."
            )
        else:
            context_intro = (
                f"User mencoba mengakses fitur berikut yang TIDAK AKTIF di akun mereka: "
                f"**{features_text}**.\n"
                f"\nTool yang dipanggil: {unavailable_tools or '(none)'}."
            )

        system += (
            f"\n\n⚠️ INSTRUKSI PENTING — FITUR TIDAK AKTIF:\n"
            f"{context_intro}\n"
            f"\n"
            f"WAJIB:\n"
            f"1. JANGAN karang/halusinasi data (jumlah modul selesai, badge, hasil scan, "
            f"progress, dll). Kamu TIDAK PUNYA data tentang fitur tersebut.\n"
            f"2. Beri tahu user dengan ramah bahwa fitur '{features_text}' "
            f"belum tersedia untuk akun mereka saat ini.\n"
            f"3. Sarankan alternative: pertanyaan tentang kesehatan gigi umum, "
            f"tips sikat gigi, atau fitur lain yang tersedia.\n"
            f"4. JANGAN sebut alasan teknis (admin, allowed_agents, gated, dll) — "
            f"cukup bilang 'belum tersedia'.\n"
            f"5. PENTING: Walaupun di 'Ringkasan percakapan sebelumnya' (di atas) "
            f"ada info tentang fitur ini dari session lama, JANGAN gunakan info itu "
            f"sebagai DATA CURRENT. Itu cuma context historis, fitur sekarang OFF.\n"
            f"\n"
            f"Contoh respons baik:\n"
            f"\"Maaf Hanif, saat ini fitur {features_text} belum tersedia di akun "
            f"{child_name}. Tapi Peri masih bisa bantu jawab pertanyaan seputar "
            f"kesehatan gigi atau tips sikat gigi anak ya! 🧚✨\""
        )

    # NOTE: KB dental & FAQ docs inject lewat registry sekarang. Behavior 1:1
    # preserved — _inject_dental_kb dan _inject_app_faq di knowledge.py mirror
    # exact format string sebelumnya ("Referensi dari knowledge base:\n...").
    # Backward compat untuk legacy retrieved_docs (pre-Phase 2) handled below.

    # Backward compat — kalau ada retrieved_docs di state tapi tidak ke-inject
    # via kb_dental (e.g., dari old retrieve_node yang masih ada), inject manual.
    if not agent_results.get("kb_dental") and state.get("retrieved_docs"):
        legacy_docs = state.get("retrieved_docs", [])[:3]
        if legacy_docs:
            docs_text = "\n\n".join(legacy_docs)
            system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

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
    """Build list LangChain messages dari state.

    PHASE 2A DEFENSIVE FIX: Skip empty assistant messages.
    Background: Sebelum Phase 2a fix1, agent_node mengappend empty AIMessage
    ke state.messages, yang ke-save ke DB via chat_messages.content="".
    Saat user kirim turn berikutnya, history dari DB include empty assistant
    rows. LLM dapat empty AIMessage → respond empty.
    Defensive filter ini protect future turns kalau ada empty assistant
    rows di DB (artifact dari turn yang gagal sebelum fix).
    """
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in state.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            # Skip empty assistant messages (artifact dari Phase 2a bug atau
            # turn yang gagal sebelum fix shipped).
            if not content or not str(content).strip():
                continue
            lc_messages.append(AIMessage(content=content))
        elif role == "user":
            # Always keep user messages (even empty — image-only upload valid)
            lc_messages.append(HumanMessage(content=content))
        # Other roles (system, tool) skipped — system prompt built separately,
        # tool results consumed via agent_results.
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

    # Per-call trace config (no-op kalau Langfuse disabled)
    trace_config = build_trace_config(state=state, agent_name="generate")

    # Phase 4: extract last user message untuk trace input
    user_msg_text = ""
    last_user_msg = next(
        (m for m in reversed(state.get("messages", []))
         if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if last_user_msg:
        content = last_user_msg.get("content", "")
        if isinstance(content, str):
            user_msg_text = content
        elif isinstance(content, list):
            # Multimodal — extract text part
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_msg_text = part.get("text", "")
                    break

    # Phase 4: generation span — capture full prompt + output untuk diagnostic
    # Phase 4.5: tambah agent_results context yang membentuk system_prompt
    from app.config.observability import trace_generation, _safe_dict_for_trace

    # Phase 4.5: extract agent_results context — yang ini di-inject ke system_prompt
    agent_results = state.get("agent_results", {})
    image_analysis = state.get("image_analysis")
    memory_context = state.get("memory_context", {})
    retrieved_docs = state.get("retrieved_docs", []) or []

    # FIX (Langfuse audit): Filter SystemMessage dari messages untuk trace
    # supaya tidak duplicate dengan system_prompt arg yang sudah di-pass.
    # Sebelumnya: system prompt muncul 2x di Langfuse UI (sebagai arg + sebagai messages[0]).
    messages_for_trace = [m for m in lc_messages if not isinstance(m, SystemMessage)]

    # FIX (Langfuse audit): Track actual input/output token counts dari LLM response.
    # Sebelumnya: usage_details cuma report `output: chunk_count` (bukan token count).
    # Akibat: trace report "0 prompt → 7 completion" padahal actual 539 prompt → 119 completion.
    # Gemini API expose usage_metadata di chunks (terutama last chunk).
    input_tokens_actual = 0
    output_tokens_actual = 0

    async with trace_generation(
        name="generate",
        model=get_model_name(provider=_provider, model=_model),
        system_prompt=system_prompt,
        messages=messages_for_trace,  # FIX: tanpa SystemMessage (avoid duplicate)
        user_message=user_msg_text,
        metadata={
            "agents_used": list(agent_results.keys()),
            "response_mode": state.get("response_mode", "simple"),
            "has_image_analysis": bool(image_analysis),
            "child_name": _get_child_name(state),
            "provider": get_provider_name(provider=_provider),
            # Phase 4.5: cross-reference info supaya Hanif bisa correlate
            # context yang masuk → system_prompt yang di-render
            "agent_results_keys": list(agent_results.keys()),
            "image_analysis_present": bool(image_analysis),
            "kb_docs_count": len(retrieved_docs),
            "memory_summaries_count": len(memory_context.get("session_summaries", [])),
            "memory_facts_count": len(memory_context.get("user_facts", [])),
        },
    ) as gen_span:
        try:
            async for chunk in llm.astream(lc_messages, config=trace_config):
                # FIX (Langfuse audit): Capture token counts dari API response.
                # Gemini chunks may carry usage_metadata (terutama last chunk).
                # Pakai max() supaya kalau muncul di multiple chunks, ambil yang terbesar.
                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta:
                    input_tokens_actual = max(
                        input_tokens_actual,
                        usage_meta.get("input_tokens", 0) or 0,
                    )
                    output_tokens_actual = max(
                        output_tokens_actual,
                        usage_meta.get("output_tokens", 0) or 0,
                    )

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
            if gen_span:
                gen_span.update(
                    output=f"[ERROR] {str(e)[:500]}",
                    level="ERROR",
                    status_message=str(e)[:200],
                )

        # Phase 4: capture output ke generation span (sebelum exit context)
        # Phase 4.5: tambah agent_results di metadata untuk full diagnostic
        if gen_span and success:
            try:
                # FIX (Langfuse audit): Report actual token counts from API response.
                # Fallback ke chunk count kalau API tidak expose usage_metadata.
                # Naming follows Langfuse convention: "input"/"output" tokens.
                usage_payload: dict = {}
                if input_tokens_actual > 0:
                    usage_payload["input"] = input_tokens_actual
                if output_tokens_actual > 0:
                    usage_payload["output"] = output_tokens_actual
                else:
                    # Fallback: chunk count (better than nothing, kalau API tidak expose)
                    usage_payload["output"] = output_tokens

                gen_span.update(
                    output=full_response[:5000] if full_response else "(empty)",
                    usage_details=usage_payload,
                    # Phase 4.5: capture agent_results in update metadata —
                    # diakses via "Additional Input" panel di Langfuse UI
                    metadata={
                        "agent_results_summary": _safe_dict_for_trace(agent_results),
                        "image_analysis_summary": _safe_dict_for_trace(image_analysis) if image_analysis else None,
                        "memory_context_summary": _safe_dict_for_trace(memory_context) if memory_context else None,
                        # FIX (Langfuse audit): Track output_tokens accuracy untuk debug
                        "output_chunks_yielded": output_tokens,
                        "output_tokens_from_api": output_tokens_actual,
                        "input_tokens_from_api": input_tokens_actual,
                    },
                )
            except Exception:
                pass  # defensive — jangan break flow kalau Langfuse error

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
