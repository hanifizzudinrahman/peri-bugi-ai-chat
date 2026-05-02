"""
Sub-agents Phase 2: rapot_peri, cerita_peri, mata_peri

Setiap agent call internal API dari peri-bugi-api via HTTP.
ai-chat tidak akses DB langsung — semua data lewat api.
"""
import logging
from typing import Any

import httpx

from app.agents.state import AgentState
from app.config.settings import settings

logger = logging.getLogger(__name__)

_INTERNAL_HEADERS = {"X-Internal-Secret": settings.INTERNAL_SECRET}
_TIMEOUT = 10  # seconds


async def _call_internal_api(path: str) -> dict:
    """Helper: call peri-bugi-api internal endpoint."""
    if not settings.PERI_API_URL:
        return {"error": "PERI_API_URL tidak diset", "has_data": False}
    url = f"{settings.PERI_API_URL.rstrip('/')}{path}"

    # Phase 3: trace HTTP call as child observation
    from app.config.observability import trace_http_call, _safe_dict_for_trace
    span_name = f"http-internal-get-{path.split('/')[-1].split('?')[0] or 'root'}"

    async with trace_http_call(
        name=span_name,
        method="GET",
        url=url,
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, headers=_INTERNAL_HEADERS)
                resp.raise_for_status()
                result = resp.json()
            if span:
                # Phase 4.5: capture full response body for diagnostic
                span.update(output={
                    "status_code": resp.status_code,
                    "has_data": result.get("has_data"),
                    "response_body": _safe_dict_for_trace(result),  # Phase 4.5
                })
            return result
        except httpx.TimeoutException:
            if span:
                span.update(
                    output={"error": "timeout"},
                    level="ERROR",
                    status_message="HTTP timeout",
                )
            return {"error": "timeout", "has_data": False}
        except Exception as e:
            if span:
                span.update(
                    output={"error": str(e)[:200]},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
            return {"error": str(e), "has_data": False}


async def _call_internal_api_post(
    path: str,
    json_body: dict,
    extra_headers: dict | None = None,
    timeout: int | None = None,
) -> dict:
    """
    Helper: POST ke peri-bugi-api internal endpoint.

    Berbeda dengan _call_internal_api yang GET-only:
    - Support JSON body
    - Support extra headers (e.g. X-Request-ID untuk trace propagation)
    - Configurable timeout (default _TIMEOUT, untuk image analysis perlu lebih lama)
    """
    if not settings.PERI_API_URL:
        return {"status": "failed", "error": "PERI_API_URL tidak diset"}

    url = f"{settings.PERI_API_URL.rstrip('/')}{path}"
    headers = dict(_INTERNAL_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    actual_timeout = timeout or _TIMEOUT

    # Phase 3: trace HTTP call as child observation
    # Phase 4.5: pass full body + capture full response for diagnostic visibility
    from app.config.observability import trace_http_call, _safe_dict_for_trace
    span_name = f"http-internal-post-{path.split('/')[-1].split('?')[0] or 'root'}"

    async with trace_http_call(
        name=span_name,
        method="POST",
        url=url,
        body=json_body,  # Phase 4.5: full body values (auto-redacted)
        body_keys=list(json_body.keys()) if json_body else None,  # backward compat
        metadata={"timeout_sec": actual_timeout},
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=actual_timeout) as client:
                resp = await client.post(url, json=json_body, headers=headers)
                resp.raise_for_status()
                result = resp.json()
            if span:
                # Phase 4.5: capture FULL response body for full diagnostic visibility.
                # _safe_dict_for_trace handle redact secret + truncate + skip base64.
                span.update(output={
                    "status_code": resp.status_code,
                    "status": result.get("status"),
                    "scan_session_id": result.get("scan_session_id"),
                    "error_code": result.get("error_code"),
                    "response_body": _safe_dict_for_trace(result),  # Phase 4.5
                })
            return result
        except httpx.TimeoutException:
            logger.warning(f"[internal_api_post] Timeout: {path}")
            if span:
                span.update(
                    output={"error": "timeout"},
                    level="ERROR",
                    status_message=f"timeout after {actual_timeout}s",
                )
            return {"status": "failed", "error": "timeout", "error_code": "api_timeout"}
        except httpx.HTTPStatusError as e:
            logger.warning(f"[internal_api_post] HTTP {e.response.status_code}: {path}")
            if span:
                span.update(
                    output={"status_code": e.response.status_code, "error_code": "api_http_error"},
                    level="ERROR",
                    status_message=f"HTTP {e.response.status_code}",
                )
            return {
                "status": "failed",
                "error": f"HTTP {e.response.status_code}",
                "error_code": "api_http_error",
            }
        except Exception as e:
            logger.exception(f"[internal_api_post] Error: {path}: {e}")
            if span:
                span.update(
                    output={"error": str(e)[:200], "error_code": "api_unknown_error"},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
            return {
                "status": "failed",
                "error": str(e)[:200],
                "error_code": "api_unknown_error",
            }


# =============================================================================
# Rapot Peri Agent
# =============================================================================

async def rapot_peri_agent(state: AgentState) -> dict[str, Any]:
    """
    Ambil data brushing lengkap dari api.
    Return: {streak, achievements, child_name, has_data}
    """
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    user_id = user.get("id")

    # Phase 4: wrap agent body with trace_node
    from app.config.observability import trace_node

    async with trace_node(
        name="rapot_peri_agent",
        state=state,
        input_data={
            "has_user_id": bool(user_id),
            "has_brushing_in_context": ctx.get("brushing") is not None,
        },
    ) as span:
        if not user_id:
            # Fallback ke user_context yang sudah di-inject
            brushing = ctx.get("brushing")
            child = ctx.get("child") or {}
            state["tool_calls"].append({
                "tool": "get_brushing_stats",
                "agent": "rapot_peri",
                "input": {"source": "user_context_fallback"},
                "result": {"has_data": brushing is not None},
            })
            result = {
                "child_name": child.get("nickname") or child.get("full_name", "si kecil"),
                "streak": brushing,
                "has_data": brushing is not None,
                "source": "user_context",
            }
            if span:
                span.update(output={
                    "has_data": result["has_data"],
                    "source": "user_context_fallback",
                    "current_streak": brushing.get("current_streak") if brushing else None,
                })
            return result

        data = await _call_internal_api(f"/api/v1/internal/agent/brushing-stats/{user_id}")
        state["tool_calls"].append({
            "tool": "get_brushing_stats",
            "agent": "rapot_peri",
            "input": {"user_id": user_id},
            "result": {"has_data": data.get("has_data", False), "error": data.get("error")},
        })

        # Merge dengan user_context streak (lebih fresh, sudah di-inject saat request)
        if not data.get("has_data") and ctx.get("brushing"):
            data["streak"] = ctx["brushing"]
            data["has_data"] = True
            data["source"] = "user_context"

        if span:
            streak_data = data.get("streak") if isinstance(data.get("streak"), dict) else None
            # Phase 4.5: capture full data so generate context visible
            from app.config.observability import _safe_dict_for_trace
            span.update(output={
                "has_data": data.get("has_data", False),
                "source": data.get("source", "api"),
                "current_streak": streak_data.get("current_streak") if streak_data else None,
                "best_streak": streak_data.get("best_streak") if streak_data else None,
                "data": _safe_dict_for_trace(data),  # Phase 4.5: full result
            })

        return data


# =============================================================================
# Cerita Peri Agent
# =============================================================================

async def cerita_peri_agent(state: AgentState) -> dict[str, Any]:
    """
    Ambil progress modul Cerita Peri user dari api.
    Return: {total_modules, completed_count, current_module_id, total_stars, modules}
    """
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    user_id = user.get("id")

    # Phase 4: wrap agent body with trace_node
    from app.config.observability import trace_node

    async with trace_node(
        name="cerita_peri_agent",
        state=state,
        input_data={"has_user_id": bool(user_id)},
    ) as span:
        if not user_id:
            state["tool_calls"].append({
                "tool": "get_cerita_progress",
                "agent": "cerita_peri",
                "input": {},
                "result": {"error": "user_id tidak tersedia"},
            })
            if span:
                span.update(output={"has_data": False, "error": "user_id_missing"})
            return {"has_data": False, "error": "user_id tidak tersedia"}

        data = await _call_internal_api(f"/api/v1/internal/agent/cerita-progress/{user_id}")
        state["tool_calls"].append({
            "tool": "get_cerita_progress",
            "agent": "cerita_peri",
            "input": {"user_id": user_id},
            "result": {
                "has_data": data.get("has_data", False),
                "completed_count": data.get("completed_count", 0),
                "total_stars": data.get("total_stars", 0),
            },
        })
        if span:
            # Phase 4.5: capture full data so generate context visible
            from app.config.observability import _safe_dict_for_trace
            span.update(output={
                "has_data": data.get("has_data", False),
                "completed_count": data.get("completed_count", 0),
                "total_stars": data.get("total_stars", 0),
                "current_module_id": data.get("current_module_id"),
                "data": _safe_dict_for_trace(data),  # Phase 4.5: full result (modules, etc)
            })
        return data


# =============================================================================
# Mata Peri Agent
# =============================================================================

async def mata_peri_agent(state: AgentState) -> dict[str, Any]:
    """
    Mata Peri agent — handle 2 mode:
      Mode A (image-based): user kirim foto di chat → analyze via Tanya Peri pipeline
      Mode B (history-based): user tanya soal scan history → ambil dari api

    Mode A flow (Phase 4 Batch B):
      1. Cek state["clarification_selected"] — kalau user just answered ClarificationCard,
         pakai value itu sebagai view_hint
      2. Detect view_hint via Hybrid Tiered (rule + LLM)
      3a. Detected → call internal /agent/tanya-peri-analyze-image
      3b. Ambiguous → emit clarification_data, return early (generate node skip image)

    Returns:
      Mode A success: {has_data, mode='new_scan_tanya_peri', image_analysis,
                       scan_session_id, view_hint, decision_source}
      Mode A clarification: {needs_clarification: True, clarification_data: {...}}
      Mode A failed: {has_data: False, mode='new_scan_failed', fallback_text}
      Mode B: {has_data, latest_scan, scans, source}
    """
    image_url = state.get("image_url")

    # Phase 4: wrap agent body with trace_node
    # NOTE: _analyze_chat_image dan internal API calls sudah punya child spans sendiri
    # (trace_http_call dari Phase 3) — mereka akan nest di bawah span ini.
    from app.config.observability import trace_node

    mode_input = "image" if image_url else "history"
    has_clarification = bool(state.get("clarification_selected"))

    async with trace_node(
        name="mata_peri_agent",
        state=state,
        input_data={
            "mode": mode_input,
            "has_image": bool(image_url),
            "has_clarification_selected": has_clarification,
        },
    ) as span:
        # Mode A: image-based analysis
        if image_url:
            result = await _analyze_chat_image(state, image_url)
            if span:
                # Phase 4.5: capture image_analysis JSON yang masuk ke generate prompt.
                # Ini THE MOST IMPORTANT data flow — apa yang ai-cv return = apa yang
                # akan jadi context di system prompt LLM. Tanpa ini Hanif tidak bisa
                # debug kenapa LLM merespon dengan tone tertentu.
                from app.config.observability import _safe_dict_for_trace
                image_analysis = state.get("image_analysis")
                span.update(output={
                    "mode": result.get("mode"),
                    "has_data": result.get("has_data", False),
                    "view_hint": result.get("view_hint"),
                    "decision_source": result.get("decision_source"),
                    "scan_session_id": result.get("scan_session_id"),
                    "needs_clarification": result.get("needs_clarification", False),
                    # Phase 4.5: full image_analysis content (session_summary + results)
                    "image_analysis": _safe_dict_for_trace(image_analysis) if image_analysis else None,
                    # Phase 4.5: full result dict (mode, fallback_text kalau gagal, dll)
                    "result_data": _safe_dict_for_trace(result),
                })
            return result

        # Mode B: history-based query (existing flow — preserved)
        ctx = state.get("user_context", {})
        user = ctx.get("user") or {}
        user_id = user.get("id")

        # Cek user_context dulu (lebih cepat, sudah di-inject)
        mata_peri_ctx = ctx.get("mata_peri_last_result")
        if mata_peri_ctx:
            state["tool_calls"].append({
                "tool": "get_mata_peri_history",
                "agent": "mata_peri",
                "input": {"source": "user_context"},
                "result": {"has_data": True},
            })
            result = {
                "has_data": True,
                "latest_scan": mata_peri_ctx,
                "scans": [mata_peri_ctx],
                "source": "user_context",
            }
            if span:
                # Phase 4.5: capture full latest_scan content
                from app.config.observability import _safe_dict_for_trace
                span.update(output={
                    "mode": "history",
                    "has_data": True,
                    "source": "user_context",
                    "latest_scan": _safe_dict_for_trace(mata_peri_ctx),  # Phase 4.5
                })
            return result

        if not user_id:
            if span:
                span.update(
                    output={"has_data": False, "error": "user_id_missing"},
                    level="ERROR",
                )
            return {"has_data": False, "error": "user_id tidak tersedia"}

        data = await _call_internal_api(f"/api/v1/internal/agent/mata-peri-history/{user_id}")
        state["tool_calls"].append({
            "tool": "get_mata_peri_history",
            "agent": "mata_peri",
            "input": {"user_id": user_id},
            "result": {"has_data": data.get("has_data", False), "scan_count": data.get("scan_count", 0)},
        })
        if span:
            # Phase 4.5: capture full history data (latest_scan, scans list)
            from app.config.observability import _safe_dict_for_trace
            span.update(output={
                "mode": "history",
                "has_data": data.get("has_data", False),
                "scan_count": data.get("scan_count", 0),
                "source": "api",
                "data": _safe_dict_for_trace(data),  # Phase 4.5: full result
            })
        return data


# =============================================================================
# Mode A: Analyze chat image (Phase 4 Batch B)
# =============================================================================

# View options untuk ClarificationCard saat hint ambigu
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


def _get_last_user_message(state: AgentState) -> str:
    """Ambil text pesan user terakhir dari state."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        # Format: {role: 'user', content: '...'}
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                # content bisa list of parts (text + image) — extract text
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
                return ""
    return ""


async def _analyze_chat_image(state: AgentState, image_url: str) -> dict[str, Any]:
    """
    Multi-step image analysis flow.

    Step 1: Cek apakah ini retry setelah user pilih clarification
    Step 2: Detect view_hint via Hybrid Tiered
    Step 3a: Ambiguous → emit clarification, return early
    Step 3b: Detected → call internal API → save result ke state
    """
    from app.agents.utils.view_hint_detector import detect_view_hint

    user_text = _get_last_user_message(state)

    # ===== Step 1: Check clarification reply =====
    clarification_selected = state.get("clarification_selected") or []
    view_hint: str | None = None
    decision_source: str = "ambiguous"

    if clarification_selected:
        # User just answered ClarificationCard
        selected = clarification_selected[0] if isinstance(clarification_selected, list) else clarification_selected
        if selected == "skip":
            view_hint = "front"  # default fallback
            decision_source = "user_skip"
        elif selected in {"front", "upper", "lower", "left", "right"}:
            view_hint = selected
            decision_source = "user_clarification"

    # ===== Step 2: Detect via Hybrid Tiered (kalau belum dapat dari clarification) =====
    if view_hint is None:
        # Per-agent LLM override
        llm_provider = state.get("llm_provider_override")
        llm_model = state.get("llm_model_override")
        agent_configs = state.get("agent_configs", {})
        mata_peri_conf = agent_configs.get("mata_peri", {})
        if not llm_provider and mata_peri_conf.get("llm_provider"):
            llm_provider = mata_peri_conf["llm_provider"]
        if not llm_model and mata_peri_conf.get("llm_model"):
            llm_model = mata_peri_conf["llm_model"]

        # Pull prompt template dari DB (key=view_hint_classification).
        # Kalau tidak ada (e.g. seeder belum jalan), view_hint_detector akan
        # fallback ke hardcoded constant.
        prompts_dict = state.get("prompts", {}) or {}
        view_hint_prompt = prompts_dict.get("view_hint_classification")

        view_hint, decision_source = await detect_view_hint(
            text=user_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_template=view_hint_prompt,
            state=state,  # untuk Langfuse trace metadata
        )

    # ===== Step 3a: Ambiguous → emit clarification =====
    if view_hint is None:
        logger.info(
            f"[mata_peri_agent] View hint ambiguous, emit clarification: "
            f"text={user_text[:60]}"
        )
        state["needs_clarification"] = True
        state["clarification_data"] = {
            "type": "single_select",
            "question": _CLARIFICATION_QUESTION,
            "options": _VIEW_CLARIFICATION_OPTIONS,
        }
        state["tool_calls"].append({
            "tool": "view_hint_detection",
            "agent": "mata_peri",
            "input": {"text": user_text[:100]},
            "result": {"decision_source": "ambiguous", "needs_clarification": True},
        })
        return {
            "has_data": False,
            "mode": "clarification_pending",
            "needs_clarification": True,
        }

    # ===== Step 3b: Call internal API untuk full analyze =====
    user_ctx = state.get("user_context", {}) or {}
    user_obj = user_ctx.get("user") or {}
    user_id = user_obj.get("id")
    chat_message_id = state.get("chat_message_id")
    trace_id = state.get("trace_id") or state.get("request_id")

    if not user_id:
        logger.error("[mata_peri_agent] user_id tidak tersedia di state")
        return {
            "has_data": False,
            "mode": "new_scan_failed",
            "error": "user_id_missing",
            "fallback_text": (
                "Maaf, Peri tidak bisa proses foto kamu sekarang. Coba beberapa saat lagi ya."
            ),
        }

    payload = {
        "user_id": user_id,
        "image_url": image_url,
        "view_hint": view_hint,
        "chat_message_id": chat_message_id,
        "trace_id": trace_id,
        # Phase 4.1.2 fix (Bug B1): forward source ('web'|'mobile') supaya
        # MataPeriScanSession.source label benar di Rapot Peri / Mata Peri list.
        # Sebelum fix: ai-chat tidak forward state['source'] → service fallback
        # ke 'mobile' (lihat chat_image_analysis_service.py:266) → semua web
        # session ke-tag 'mobile'.
        # state['source'] sudah di-set oleh API (chat.py:688) saat /chat/message,
        # jadi kita tinggal forward.
        "source": state.get("source"),
    }

    headers = {}
    if trace_id:
        headers["X-Request-ID"] = trace_id

    # Image analysis butuh waktu — pakai timeout lebih panjang
    response = await _call_internal_api_post(
        "/api/v1/internal/agent/tanya-peri-analyze-image",
        json_body=payload,
        extra_headers=headers,
        timeout=120,  # ai-cv pipeline could take up to 2 min cold start
    )

    response_status = response.get("status")
    error_code = response.get("error_code")
    scan_session_id = response.get("scan_session_id")
    ai_response = response.get("ai_response")

    state["tool_calls"].append({
        "tool": "tanya_peri_analyze_image",
        "agent": "mata_peri",
        "input": {
            "view_hint": view_hint,
            "decision_source": decision_source,
            "image_url_preview": image_url[:80],
        },
        "result": {
            "status": response_status,
            "scan_session_id": scan_session_id,
            "error_code": error_code,
        },
    })

    if response_status != "success" or not ai_response:
        # Graceful failure — pesan empati ke user via fallback_text
        fallback_msg = response.get("error_message") or (
            "Maaf, Peri lagi belum bisa cek foto kamu detail. "
            "Tapi dari pertanyaanmu Peri tetap bisa bantu."
        )
        logger.warning(
            f"[mata_peri_agent] Analyze failed: code={error_code} msg={fallback_msg[:100]}"
        )
        return {
            "has_data": False,
            "mode": "new_scan_failed",
            "error": error_code or "analysis_failed",
            "fallback_text": fallback_msg,
            "scan_session_id": scan_session_id,  # might exist meski failed (audit)
        }

    # ===== Success — save result ke state untuk generate node =====
    state["image_analysis"] = ai_response
    state["scan_session_id"] = scan_session_id

    return {
        "has_data": True,
        "mode": "new_scan_tanya_peri",
        "image_analysis": ai_response,
        "scan_session_id": scan_session_id,
        "view_hint": view_hint,
        "decision_source": decision_source,
    }
