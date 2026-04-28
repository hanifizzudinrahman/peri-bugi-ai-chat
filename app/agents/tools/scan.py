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
    async def get_scan_history() -> dict[str, Any]:
        """Get the child's Mata Peri (dental scan) history, including the latest scan result.

        Use this tool when the user asks about:
        - Past scan results ("hasil scan terakhir gimana?")
        - Whether a previous scan detected anything
        - Trends over time ("gigi anak saya makin baik gak?")

        Returns a dict with keys:
        - has_data: bool
        - latest_scan: dict with scan summary (or null)
        - scans: list of recent scans
        - source: "user_context" | "api"
        - error: str (only on failure)

        IMPORTANT: This tool is for history queries ONLY.
        For analyzing a NEW photo uploaded in the current turn, use analyze_chat_image instead.
        The image flow is auto-routed by the system — you typically don't need to call analyze_chat_image manually.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_scan_history",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "has_context_snapshot": mata_peri_ctx is not None,
            },
        ) as span:
            # Fast path: user_context sudah punya snapshot
            if mata_peri_ctx:
                result = {
                    "has_data": True,
                    "latest_scan": mata_peri_ctx,
                    "scans": [mata_peri_ctx],
                    "source": "user_context",
                }
                if span:
                    span.update(output={
                        "mode": "history",
                        "has_data": True,
                        "source": "user_context",
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
            )

            if span:
                span.update(output={
                    "mode": "history",
                    "has_data": data.get("has_data", False),
                    "scan_count": data.get("scan_count", 0),
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

            view_hint, decision_source = await detect_view_hint(
                text=user_text or "",
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_template=view_hint_prompt,
                state=None,  # tool layer tidak punya AgentState; trace masih jalan via parent
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
