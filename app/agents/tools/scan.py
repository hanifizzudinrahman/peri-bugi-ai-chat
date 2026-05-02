"""
Tools: get_scan_history, analyze_chat_image

Mata Peri (dental scan via CV) tools — replaces mata_peri_agent.

Two modes preserved 1:1 from legacy mata_peri_agent:
- Mode A (image-based): user uploaded photo this turn → analyze via CV pipeline
- Mode B (history-based): user asks about past scans → fetch from API

CRITICAL — Clarification flow (analyze_chat_image):
When the view angle (front/upper/lower/left/right) cannot be determined from
the user's text caption, the tool returns a special envelope:
    {"needs_clarification": True, "clarification_data": {...}, ...}

Step 2 (graph wiring) will detect this envelope after the tool call and:
1. Emit make_clarify_event SSE so the FE renders ClarificationCard
2. Skip the LLM final-response generation (FE waits for user's choice)
3. On next turn, user's selected option arrives as state.image.clarification_selected,
   which the tool re-uses as the view_hint (skipping detection).

Replaces: app/agents/sub_agents/phase2_agents.py::mata_peri_agent
        + _analyze_chat_image (private helper)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get, call_internal_post

logger = logging.getLogger(__name__)


# View options untuk ClarificationCard saat hint ambigu (preserved persis dari legacy)
_VIEW_CLARIFICATION_OPTIONS = [
    {"id": "front", "label": "Tampak Depan", "icon": "🦷"},
    {"id": "upper", "label": "Rahang Atas", "icon": "⬆️"},
    {"id": "lower", "label": "Rahang Bawah", "icon": "⬇️"},
    {"id": "left", "label": "Sisi Kiri", "icon": "◀"},
    {"id": "right", "label": "Sisi Kanan", "icon": "▶"},
    {"id": "skip", "label": "Lewati (depan saja)", "icon": "🤷"},
]

_CLARIFICATION_QUESTION = (
    "Foto ini tampak gigi bagian mana ya, Bu? Biar Peri analisa lebih tepat 😊"
)


# =============================================================================
# Tool 1: get_scan_history (Mode B)
# =============================================================================


def make_get_scan_history_tool(
    user_id: Optional[str],
    user_context: dict,
):
    """
    Factory: build get_scan_history tool with user_id + user_context closure.

    Args:
        user_id: Current user ID (closure-captured)
        user_context: Full user_context dict — used to short-circuit if
                      mata_peri_last_result already injected.

    Returns:
        @tool decorated async function.
    """
    ctx = user_context or {}
    mata_peri_ctx = ctx.get("mata_peri_last_result")

    @tool
    async def get_scan_history(
        since: str = "all",
        scan_type: str = "all",
    ) -> dict[str, Any]:
        """Get the child's Mata Peri (dental scan) history, with optional period + source filtering.

        ⚠️ TWO TYPES OF SCANS in Peri Bugi:
        1. **Scan Lengkap (Mata Peri)** = 5 view foto resmi (depan, atas, bawah, kiri, kanan).
           Dilakukan via menu Mata Peri di aplikasi. `feature_source = "mata_peri"`.
        2. **Scan via Chat (Tanya Peri)** = 1 foto cepat di-upload via chat ke Tanya Peri.
           Quick analysis, bukan scan lengkap. `feature_source = "tanya_peri"`.

        ✅ USE THIS TOOL when user asks about:
        - "hasil scan terakhir gimana?" → since="all", scan_type="all" — show latest dari source apapun
        - "scan minggu lalu pernah?" → since="last_week", scan_type="all"
        - "Mata Peri terakhir kapan?" → scan_type="full_scan" (cuma scan lengkap)
        - "scan lengkap berapa kali?" → scan_type="full_scan"
        - "tadi foto chat hasilnya gimana?" → scan_type="chat_scan"
        - "berapa kali scan lengkap bulan ini?" → since="last_month", scan_type="full_scan"
        - "tahun ini scan total berapa?" → since="last_year", scan_type="all"

        ❌ DO NOT use this tool when:
        - User minta detail 1 scan tertentu → use get_mata_peri_scan_detail
        - User mau analisis foto BARU → analyze_chat_image (auto-routed)

        Args:
            since: Period filter. Valid values:
                - "all" (DEFAULT) — 5 scan terakhir
                - "last_week" — 1 minggu terakhir
                - "last_month" — 1 bulan terakhir
                - "last_3_months" — 3 bulan terakhir
                - "last_6_months" — 6 bulan terakhir
                - "last_year" — 1 tahun terakhir

            scan_type: Source filter. Valid values:
                - "all" (DEFAULT) — Mata Peri + Tanya Peri (semua sumber)
                - "full_scan" — hanya Mata Peri 5-view (scan resmi/lengkap)
                - "chat_scan" — hanya Tanya Peri 1-view (foto via chat)

                INSTRUKSI MEMILIH:
                - User sebut "Mata Peri" / "scan lengkap" / "5 angle" → "full_scan"
                - User sebut "foto chat" / "tadi kirim foto" / "Tanya Peri" → "chat_scan"
                - Ambigu / "scan apapun" / general → "all"

        Returns a dict with keys:
        - has_data: bool
        - period: str | period_label: str
        - scan_type: str (echo) | scan_type_label: str
        - scan_count: int — total scan setelah filter
        - count_full_scan: int — breakdown: jumlah Mata Peri 5-view
        - count_chat_scan: int — breakdown: jumlah Tanya Peri 1-view
        - latest_scan: dict | null
        - scans: list of {scan_date, session_id, summary_status, summary_text,
                  recommendation_text, requires_dentist_review,
                  feature_source ("mata_peri"|"tanya_peri"),
                  view_count_expected (5 | 1),
                  source_label (Indonesian)}
        - source: "user_context" | "api"

        ANTI-HALU & DIFFERENTIATION:
        - SETIAP scan punya field `feature_source` — pakai ini untuk label di response.
          Mata Peri = "Scan Lengkap (5-view)", Tanya Peri = "Scan via Chat (1-view)".
        - Kalau user tanya "scan berapa kali" tanpa specify type, sebut breakdown:
          "X scan total — Y scan lengkap + Z dari foto chat".
        - Kalau scan_count=0 dengan filter spesifik, sebut filter explicitly:
          "Tidak ada scan lengkap dalam 1 bulan terakhir" (jangan generic "tidak ada scan").
        - Kalau user nanya tren, perhatikan: scan lengkap (5-view) lebih comprehensive
          daripada scan dari chat (1-view). Untuk tren proper, sebut limitation kalau perlu.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_scan_history",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "has_context_snapshot": mata_peri_ctx is not None,
                "since": since,
                "scan_type": scan_type,
            },
        ) as span:
            # Fast path: user_context sudah punya snapshot.
            # HANYA dipakai kalau since='all' DAN scan_type='all' — kalau ada filter,
            # harus call API untuk dapat data lengkap dengan breakdown.
            if mata_peri_ctx and since == "all" and scan_type == "all":
                result = {
                    "has_data": True,
                    "period": "all",
                    "period_label": "scan terakhir",
                    "scan_type": "all",
                    "scan_type_label": "(scan apapun)",
                    "scan_count": 1,
                    "count_full_scan": 0,  # tidak tahu dari snapshot
                    "count_chat_scan": 0,  # tidak tahu dari snapshot
                    "latest_scan": mata_peri_ctx,
                    "scans": [mata_peri_ctx],
                    "source": "user_context",
                }
                if span:
                    span.update(output={
                        "mode": "history",
                        "has_data": True,
                        "source": "user_context",
                        "since": since,
                        "scan_type": scan_type,
                        "latest_scan": _safe_dict_for_trace(mata_peri_ctx),
                    })
                return result

            if not user_id:
                if span:
                    span.update(
                        output={"has_data": False, "error": "user_id_missing"},
                        level="ERROR",
                    )
                return {"has_data": False, "error": "user_id tidak tersedia"}

            data = await call_internal_get(
                f"/api/v1/internal/agent/mata-peri-history/{user_id}"
                f"?since={since}&scan_type={scan_type}"
            )

            if span:
                span.update(output={
                    "mode": "history",
                    "has_data": data.get("has_data", False),
                    "scan_count": data.get("scan_count", 0),
                    "count_full_scan": data.get("count_full_scan", 0),
                    "count_chat_scan": data.get("count_chat_scan", 0),
                    "period": data.get("period"),
                    "scan_type": data.get("scan_type"),
                    "source": "api",
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_scan_history


# =============================================================================
# Tool 2: analyze_chat_image (Mode A — most complex tool)
# =============================================================================


def make_analyze_chat_image_tool(
    user_id: Optional[str],
    image_url: Optional[str],
    chat_message_id: Optional[str],
    trace_id: Optional[str],
    user_text: str,
    clarification_selected: Optional[list[str]],
    agent_configs: dict,
    prompts: dict,
    rnd_llm_provider: Optional[str],
    rnd_llm_model: Optional[str],
    session_id: Optional[str] = None,  # FIX (Langfuse audit Bagian B5): thread session_id
):
    """
    Factory: build analyze_chat_image tool with all auth/turn context as closures.

    Args:
        user_id: Current user ID
        image_url: Image URL uploaded this turn (closure — LLM cannot supply
                   arbitrary URLs)
        chat_message_id: For trace propagation
        trace_id: For trace propagation
        user_text: User's text caption alongside the image (for view_hint detection)
        clarification_selected: Pre-selected view options if user just answered
                                a ClarificationCard (skips view_hint detection)
        agent_configs: Per-agent LLM overrides (agent_configs.mata_peri.llm_provider)
        prompts: Prompts dict (looks up "view_hint_classification")
        rnd_llm_provider: RnD-mode LLM provider override (top priority)
        rnd_llm_model: RnD-mode LLM model override (top priority)
        session_id: Current session ID (FIX Langfuse audit Bagian B5) — used
                    to thread session metadata into view_hint_detector trace.

    Returns:
        @tool decorated async function.
    """

    @tool
    async def analyze_chat_image() -> dict[str, Any]:
        """Analyze the dental photo the user just uploaded in this chat turn.

        This tool runs the Mata Peri CV pipeline on the photo:
        1. Detect which view angle the photo shows (front/upper/lower/left/right)
           — derived from user's text caption or asks the user to clarify
        2. Run the CV model to detect potential dental issues
        3. Return analysis result for response generation

        IMPORTANT — when to call this tool:
        - The system PRE-ROUTES image inputs to this tool automatically.
          You typically do NOT need to call it manually — when the user
          uploads a photo, this tool will already be in your tool_calls.
        - Only call manually if you need to re-analyze with a different view hint.

        Returns one of:
        - SUCCESS: {has_data: True, mode: "new_scan_tanya_peri", image_analysis: {...},
                   scan_session_id: str, view_hint: str, decision_source: str}
        - CLARIFICATION NEEDED: {needs_clarification: True, clarification_data: {...}}
          → System will emit ClarificationCard, do not write a response yet
        - FAILED: {has_data: False, mode: "new_scan_failed", error: str, fallback_text: str}
          → Use the fallback_text in your response or fallback to general advice
        """
        return await _analyze_chat_image_impl(
            user_id=user_id,
            image_url=image_url,
            chat_message_id=chat_message_id,
            trace_id=trace_id,
            user_text=user_text,
            clarification_selected=clarification_selected,
            agent_configs=agent_configs,
            prompts=prompts,
            rnd_llm_provider=rnd_llm_provider,
            rnd_llm_model=rnd_llm_model,
            session_id=session_id,  # FIX (Langfuse audit Bagian B5): forward session_id
        )

    return analyze_chat_image


# =============================================================================
# Pure async helper (testable) — wraps Langfuse trace
# =============================================================================


async def _analyze_chat_image_impl(
    *,
    user_id: Optional[str],
    image_url: Optional[str],
    chat_message_id: Optional[str],
    trace_id: Optional[str],
    user_text: str,
    clarification_selected: Optional[list[str]],
    agent_configs: dict,
    prompts: dict,
    rnd_llm_provider: Optional[str],
    rnd_llm_model: Optional[str],
    session_id: Optional[str] = None,  # FIX (Langfuse audit Bagian B5)
) -> dict[str, Any]:
    """
    Pure helper — analyze chat image with view_hint detection + CV pipeline.

    Logic preserved 1:1 from legacy _analyze_chat_image (phase2_agents.py:454).
    """
    from app.agents.utils.view_hint_detector import detect_view_hint
    from app.config.observability import trace_node, _safe_dict_for_trace

    async with trace_node(
        name="tool:analyze_chat_image",
        state=None,
        input_data={
            "has_image": bool(image_url),
            "has_clarification_selected": bool(clarification_selected),
            "user_text_preview": (user_text or "")[:200],
        },
    ) as span:
        # Defensive: tidak ada image_url ya tidak bisa analyze
        if not image_url:
            result = {
                "has_data": False,
                "mode": "new_scan_failed",
                "error": "no_image_url",
                "fallback_text": (
                    "Maaf, Peri tidak melihat foto di pesan ini. Coba kirim ulang ya."
                ),
            }
            if span:
                span.update(output=result, level="ERROR")
            return result

        # ===== Step 1: Check clarification reply =====
        view_hint: Optional[str] = None
        decision_source: str = "ambiguous"

        if clarification_selected:
            selected = (
                clarification_selected[0]
                if isinstance(clarification_selected, list)
                else clarification_selected
            )
            if selected == "skip":
                view_hint = "front"
                decision_source = "user_skip"
            elif selected in {"front", "upper", "lower", "left", "right"}:
                view_hint = selected
                decision_source = "user_clarification"

        # ===== Step 2: Detect via Hybrid Tiered (kalau belum dapat) =====
        if view_hint is None:
            # Resolve LLM override priority: RnD > agent_configs.mata_peri > default config
            llm_provider = rnd_llm_provider
            llm_model = rnd_llm_model
            mata_peri_conf = (agent_configs or {}).get("mata_peri", {}) or {}
            if not llm_provider and mata_peri_conf.get("llm_provider"):
                llm_provider = mata_peri_conf["llm_provider"]
            if not llm_model and mata_peri_conf.get("llm_model"):
                llm_model = mata_peri_conf["llm_model"]

            view_hint_prompt = (prompts or {}).get("view_hint_classification")

            # FIX (Langfuse audit Bagian B5): Build state dict untuk detect_view_hint
            # supaya LLM call view_hint_detector punya session_id + user_id metadata.
            # Sebelumnya: state=None → orphan trace tanpa session linking.
            # Setelah fix: trace muncul dengan session_id + user_id, searchable di Langfuse.
            view_hint_state = None
            if session_id or user_id:
                view_hint_state = {
                    "session_id": session_id,
                    "user_context": {"user_id": user_id} if user_id else {},
                }

            view_hint, decision_source = await detect_view_hint(
                text=user_text or "",
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_template=view_hint_prompt,
                state=view_hint_state,  # FIX B5: pass state instead of None
            )

        # ===== Step 3a: Ambiguous → return clarification envelope =====
        if view_hint is None:
            logger.info(
                f"[tool:analyze_chat_image] View hint ambiguous, "
                f"return clarification: text={(user_text or '')[:60]}"
            )
            result = {
                "has_data": False,
                "needs_clarification": True,
                "mode": "clarification_pending",
                "clarification_data": {
                    "type": "single_select",
                    "question": _CLARIFICATION_QUESTION,
                    "options": _VIEW_CLARIFICATION_OPTIONS,
                },
                "decision_source": "ambiguous",
            }
            if span:
                span.update(output={
                    "needs_clarification": True,
                    "decision_source": "ambiguous",
                })
            return result

        # ===== Step 3b: Call internal API untuk full analyze =====
        if not user_id:
            logger.error("[tool:analyze_chat_image] user_id tidak tersedia")
            result = {
                "has_data": False,
                "mode": "new_scan_failed",
                "error": "user_id_missing",
                "fallback_text": (
                    "Maaf, Peri tidak bisa proses foto kamu sekarang. "
                    "Coba beberapa saat lagi ya."
                ),
            }
            if span:
                span.update(output=result, level="ERROR")
            return result

        payload = {
            "user_id": user_id,
            "image_url": image_url,
            "view_hint": view_hint,
            "chat_message_id": chat_message_id,
            "trace_id": trace_id,
        }

        headers = {}
        if trace_id:
            headers["X-Request-ID"] = trace_id

        # Image analysis butuh waktu — pakai timeout 120s (cold start ai-cv)
        response = await call_internal_post(
            "/api/v1/internal/agent/tanya-peri-analyze-image",
            json_body=payload,
            extra_headers=headers,
            timeout=120,
        )

        response_status = response.get("status")
        error_code = response.get("error_code")
        scan_session_id = response.get("scan_session_id")
        ai_response = response.get("ai_response")

        if response_status != "success" or not ai_response:
            fallback_msg = response.get("error_message") or (
                "Maaf, Peri lagi belum bisa cek foto kamu detail. "
                "Tapi dari pertanyaanmu Peri tetap bisa bantu."
            )
            logger.warning(
                f"[tool:analyze_chat_image] Analyze failed: "
                f"code={error_code} msg={fallback_msg[:100]}"
            )
            result = {
                "has_data": False,
                "mode": "new_scan_failed",
                "error": error_code or "analysis_failed",
                "fallback_text": fallback_msg,
                "scan_session_id": scan_session_id,
                "view_hint": view_hint,
                "decision_source": decision_source,
            }
            if span:
                span.update(output=_safe_dict_for_trace(result), level="ERROR")
            return result

        # ===== Success =====
        result = {
            "has_data": True,
            "mode": "new_scan_tanya_peri",
            "image_analysis": ai_response,
            "scan_session_id": scan_session_id,
            "view_hint": view_hint,
            "decision_source": decision_source,
        }
        if span:
            span.update(output={
                "mode": "new_scan_tanya_peri",
                "has_data": True,
                "view_hint": view_hint,
                "decision_source": decision_source,
                "scan_session_id": scan_session_id,
                "image_analysis": _safe_dict_for_trace(ai_response),
            })
        return result


# =============================================================================
# Tool: get_mata_peri_scan_detail (Phase 2 Tools Expansion)
# =============================================================================

def make_get_mata_peri_scan_detail_tool(user_id: Optional[str]):
    """
    Factory: build get_mata_peri_scan_detail tool.

    Returns per-view scan findings (front/upper/lower/left/right) extracted from
    AI pipeline result_json. Token-efficient: only user-relevant fields, no bbox/pixel coords.
    """

    @tool
    async def get_mata_peri_scan_detail(session_id: Optional[str] = None) -> dict[str, Any]:
        """Get DETAIL of a Mata Peri scan session — per-view findings (5 views) + overall cleanliness score.

        Mata Peri scans 5 views: front (depan), upper (rahang atas), lower (rahang bawah),
        left (kiri), right (kanan). Each view analyzed by AI for caries severity.

        ✅ USE THIS TOOL when the user asks about:
        - "hasil scan terakhir gimana?" (latest scan)
        - "scan rahang atas anak saya gimana?" → check views[view_type='upper']
        - "tanggal X scannya hasilnya apa?" (specific session, panggil get_scan_history dulu untuk dapat session_id)
        - "rekomendasi dari scan terakhir apa?" → check recommendation_text
        - "score bersihnya berapa persen?" → check overall_clean_ratio (Phase 2)
        - "kebersihan rahang atas berapa?" → check views[].clean_ratio per view (Phase 2)
        - "ada cavity ga di scan terakhir?" → check views[].caries_detected_count
        - "bagian gigi mana yang ada masalah?" → filter views[] by severity != 'clean'

        ❌ DO NOT use this tool when:
        - User just asks how many scans done → use get_scan_history (lighter, summary)
        - User wants to do a NEW scan → tool is for existing scans only
        - User asks about scan that doesn't exist → tool returns has_data=False

        Args:
            session_id: UUID string of the scan session, OR None to get latest scan
                       (if None, tool will fetch user's most recent completed scan)

        Returns a dict with keys:
        - has_data: bool
        - session: dict with:
          - session_id, child_name, performed_at, completed_at
          - is_processing: bool — True kalau scan masih dianalisis
          - summary_status: "ok" | "perlu_perhatian" | "segera_ke_dokter"
          - summary_text, recommendation_text (already user-friendly Indonesian)
          - confidence_overall: 0.0-1.0
          - requires_dentist_review: bool (CRITICAL flag)
          - worst_view_severity: "clean" | "moderate" | "severe" | "unknown"
          - overall_clean_ratio: float 0.0-1.0 | null — average dari valid views (Phase 2)
          - views: list of {
              view_type ("front"|"upper"|"lower"|"left"|"right"),
              view_label ("Tampak Depan", etc),
              is_valid: bool,
              severity ("clean"|"moderate"|"severe"|"unknown"),
              severity_label (Indonesian),
              clean_ratio (0.0-1.0, rounded 2 decimal) | null,
              teeth_detected_count, caries_detected_count,
              invalid_reason (only if !is_valid)
            }

        ANTI-HALU:
        - Kalau is_processing=True, bilang scan masih dianalisis. JANGAN halu hasil.
        - Kalau requires_dentist_review=True, sebut critical info ini.
        - Kalau overall_clean_ratio null, JANGAN sebut angka kebersihan — bilang 
          "AI belum bisa hitung rata-rata kebersihan untuk scan ini".
        - Kalau view.is_valid=False, sebut "AI tidak bisa analisis bagian itu, 
          mungkin foto kurang jelas". JANGAN guess severity.
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_mata_peri_scan_detail",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "session_id_provided": bool(session_id),
                "session_id": session_id if session_id else "latest",
            },
        ) as span:
            if not user_id:
                result = {
                    "has_data": False,
                    "reason": "no_user_id",
                    "fallback_message": "User ID tidak tersedia.",
                }
                if span:
                    span.update(output={
                        "has_data": False,
                        "reason": "no_user_id",
                    })
                return result

            # Build path
            if session_id:
                path = f"/api/v1/internal/agent/mata-peri-scan/{session_id}?user_id={user_id}"
            else:
                # "latest" sentinel value
                path = f"/api/v1/internal/agent/mata-peri-scan/latest?user_id={user_id}"

            data = await call_internal_get(path)

            if span:
                session = data.get("session") or {}
                views = session.get("views") or []
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "session_id": session.get("session_id"),
                    "is_processing": session.get("is_processing"),
                    "summary_status": session.get("summary_status"),
                    "worst_view_severity": session.get("worst_view_severity"),
                    "requires_dentist_review": session.get("requires_dentist_review"),
                    "views_count": len(views),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_mata_peri_scan_detail


# =============================================================================
# ToolSpec registrations — Bagian C: registry pattern
# =============================================================================
from app.agents.tools.registry import ToolSpec, register_tool, BridgeContext


# ── get_scan_history ────────────────────────────────────────────────────────

def _bridge_scan_history(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: scan history → agent_results['mata_peri']."""
    agent_results["mata_peri"] = result


def _inject_scan_history(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject scan history (list of recent scans) to system prompt.
    
    Phase 3: tambah period info kalau user filter by `since`.
    Phase 3.1: tambah source differentiation (Mata Peri full vs Tanya Peri chat).
    """
    if not data or not data.get("has_data"):
        # Loud signal kalau tool return empty for specific filter
        if data:
            period = data.get("period", "all")
            scan_type = data.get("scan_type", "all")
            if period != "all" or scan_type != "all":
                period_label = data.get("period_label", period)
                scan_type_label = data.get("scan_type_label", "")
                return (
                    f"\n\nData history scan {child_name}: "
                    f"TIDAK ADA scan {scan_type_label} dalam {period_label}. "
                    f"Bilang user dengan jujur."
                )
        return ""
    
    scans = data.get("scans") or data.get("sessions") or []
    if not scans:
        return ""
    
    period = data.get("period", "all")
    period_label = data.get("period_label", "scan terakhir")
    scan_type = data.get("scan_type", "all")
    scan_type_label = data.get("scan_type_label", "")
    scan_count = data.get("scan_count", len(scans))
    count_full_scan = data.get("count_full_scan", 0)
    count_chat_scan = data.get("count_chat_scan", 0)
    
    text = f"\n\nData history scan {child_name} (dari tool call):"
    
    # Period info
    if period != "all":
        text += f"\n- Period: {period_label}"
    
    # Scan type filter info
    if scan_type != "all":
        text += f"\n- Filter: {scan_type_label}"
    
    # Total + breakdown
    text += f"\n- Total scan: {scan_count}"
    if scan_type == "all" and (count_full_scan > 0 or count_chat_scan > 0):
        # Show breakdown supaya LLM bisa context-aware response
        breakdown_parts = []
        if count_full_scan > 0:
            breakdown_parts.append(f"{count_full_scan} Scan Lengkap (5-view via Mata Peri)")
        if count_chat_scan > 0:
            breakdown_parts.append(f"{count_chat_scan} Scan Chat (1-view via Tanya Peri)")
        text += f"\n  Breakdown: {' + '.join(breakdown_parts)}"
    
    # Show scans (cap di 8)
    for i, sess in enumerate(scans[:8]):
        scan_date = sess.get("scan_date") or sess.get("performed_at") or sess.get("completed_at", "-")
        status = sess.get("summary_status", "-")
        source_label = sess.get("source_label", "")
        feature_source = sess.get("feature_source", "")
        requires_review = sess.get("requires_dentist_review")
        warning = " 🚨 PERLU DOKTER" if requires_review else ""
        
        # Format: "1. 2026-05-01 — Status: ok [Scan Lengkap (5-view)]"
        source_suffix = f" [{source_label}]" if source_label else ""
        text += f"\n  {i+1}. {scan_date} — Status: {status}{source_suffix}{warning}"
    
    if len(scans) > 8:
        text += f"\n  (dan {len(scans) - 8} scan lainnya tidak ditampilkan)"
    
    text += (
        f"\n\nINSTRUKSI: "
        f"Pakai data ini untuk jawab pertanyaan tentang riwayat / tren. "
        f"Setiap scan punya source label — sebut explicit kalau user tanya specific "
        f"(\"scan lengkap berapa kali\" → count_full_scan; \"foto chat berapa\" → count_chat_scan). "
        f"Untuk pertanyaan generic \"scan berapa\", sebut breakdown: "
        f"\"Total X scan, terdiri dari Y scan lengkap dan Z dari foto chat\". "
        f"Kalau ada requires_dentist_review=True, sebut info itu."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_scan_history",
    agent_key="mata_peri",
    required_agent="mata_peri",
    bridge_handler=_bridge_scan_history,
    prompt_injector=_inject_scan_history,
    thinking_label="Mengecek riwayat scan...",
))


# ── analyze_chat_image (special — updates ctx fields) ───────────────────────

def _bridge_analyze_chat_image(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: analyze_chat_image — special handling untuk clarification + success + failure.
    
    Mirror existing tool_bridge.py logic line 157-175.
    """
    if result.get("needs_clarification"):
        # User butuh pilih view dulu
        ctx.needs_clarification = True
        ctx.clarification_data = result.get("clarification_data")
        agent_results["mata_peri"] = result
    elif result.get("has_data"):
        # Success — extract image_analysis + scan_session_id
        ctx.image_analysis = result.get("image_analysis")
        ctx.scan_session_id = result.get("scan_session_id")
        agent_results["mata_peri"] = result
    else:
        # Failure mode — preserve fallback_text
        agent_results["mata_peri"] = result


# NOTE: analyze_chat_image TIDAK punya prompt_injector — image_analysis di-inject
# via separate logic di generate.py (template-based, lebih kompleks). Setting
# prompt_injector=None disengaja.

register_tool(ToolSpec(
    tool_name="analyze_chat_image",
    agent_key="mata_peri",
    required_agent="mata_peri",
    bridge_handler=_bridge_analyze_chat_image,
    prompt_injector=None,  # image_analysis path di generate.py, bukan via injector
    thinking_label="Menganalisis foto gigi...",
))


# ── get_mata_peri_scan_detail ───────────────────────────────────────────────

def _bridge_scan_detail(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: scan detail → agent_results['mata_peri_scan_detail']."""
    agent_results["mata_peri_scan_detail"] = result


def _inject_scan_detail(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject specific scan detail to system prompt.
    
    Phase 2: tambah overall_clean_ratio (rata-rata clean_ratio dari valid views)
    dan clean_ratio per-view supaya LLM bisa jawab "score bersihnya berapa persen".
    Phase 3.1: tambah source_label (Mata Peri 5-view vs Tanya Peri 1-view).
    """
    if not data or not data.get("has_data"):
        return ""
    
    session = data.get("session") or {}
    is_processing = session.get("is_processing", False)
    
    # Phase 3.1: source label
    source_label = session.get("source_label")
    feature_source = session.get("feature_source")
    view_count_expected = session.get("view_count_expected", 5)
    
    text = f"\n\nDetail Scan {child_name} (dari tool call):"
    if source_label:
        text += f"\n- Tipe scan: {source_label}"
    text += f"\n- Session ID: {session.get('session_id', '-')}"
    text += f"\n- Tanggal: {session.get('performed_at') or session.get('completed_at', '-')}"
    
    if is_processing:
        text += (
            f"\n- ⚠️ STATUS: Masih dalam proses analisis"
            f"\n  Beri tahu user untuk tunggu sebentar."
        )
        return text
    
    summary_status = session.get("summary_status", "-")
    summary_text = session.get("summary_text", "-")
    rec_text = session.get("recommendation_text", "-")
    requires_review = session.get("requires_dentist_review", False)
    worst_severity = session.get("worst_view_severity", "-")
    overall_clean_ratio = session.get("overall_clean_ratio")
    
    text += f"\n- Status: {summary_status}"
    text += f"\n- Severity terburuk: {worst_severity}"
    text += f"\n- Ringkasan: {summary_text}"
    text += f"\n- Rekomendasi: {rec_text}"
    
    if overall_clean_ratio is not None:
        clean_pct = round(overall_clean_ratio * 100)
        text += f"\n- Rata-rata kebersihan dari semua view: {clean_pct}% (clean_ratio: {overall_clean_ratio})"
    
    if requires_review:
        text += f"\n- 🚨 PERLU PEMERIKSAAN DOKTER GIGI"
    
    views = session.get("views") or []
    if views:
        text += f"\n- Detail per view ({len(views)} views):"
        for v in views[:5]:
            view_label = v.get("view_label", v.get("view_type", "-"))
            severity = v.get("severity_label") or v.get("severity", "-")
            cr = v.get("clean_ratio")
            cr_text = f", kebersihan {round(cr * 100)}%" if cr is not None else ""
            invalid_reason = v.get("invalid_reason")
            if invalid_reason:
                text += f"\n  • {view_label}: TIDAK BISA DIANALISA ({invalid_reason})"
            else:
                text += f"\n  • {view_label}: {severity}{cr_text}"
    
    # Phase 3.1: Context-aware instruction berdasarkan source
    text += "\n\nINSTRUKSI: "
    if feature_source == "tanya_peri":
        text += (
            "Ini adalah analisis dari foto chat (1-view, bukan scan lengkap), "
            "jadi cuma cover bagian yang ke-foto. "
            "Sarankan user untuk Mata Peri (5-view scan resmi) untuk hasil lebih lengkap. "
        )
    elif feature_source == "mata_peri":
        text += "Ini adalah scan lengkap 5-view dari Mata Peri — cover semua sisi gigi. "
    
    text += (
        f"Kalau user tanya 'score bersih berapa persen', pakai overall_clean_ratio. "
        f"Kalau view tidak valid (invalid_reason), JANGAN halu — bilang 'AI tidak bisa "
        f"analisis bagian itu, mungkin foto kurang jelas'. "
        f"Kalau ada bagian dengan severity selain 'clean', sebut bagian mana yang affected."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_mata_peri_scan_detail",
    agent_key="mata_peri_scan_detail",
    required_agent="mata_peri",
    bridge_handler=_bridge_scan_detail,
    prompt_injector=_inject_scan_detail,
    thinking_label="Membaca detail hasil scan...",
))
