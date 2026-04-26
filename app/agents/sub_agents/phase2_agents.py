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
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=_INTERNAL_HEADERS)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        return {"error": "timeout", "has_data": False}
    except Exception as e:
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

    try:
        async with httpx.AsyncClient(timeout=actual_timeout) as client:
            resp = await client.post(url, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        logger.warning(f"[internal_api_post] Timeout: {path}")
        return {"status": "failed", "error": "timeout", "error_code": "api_timeout"}
    except httpx.HTTPStatusError as e:
        logger.warning(f"[internal_api_post] HTTP {e.response.status_code}: {path}")
        return {
            "status": "failed",
            "error": f"HTTP {e.response.status_code}",
            "error_code": "api_http_error",
        }
    except Exception as e:
        logger.exception(f"[internal_api_post] Error: {path}: {e}")
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
        return {
            "child_name": child.get("nickname") or child.get("full_name", "si kecil"),
            "streak": brushing,
            "has_data": brushing is not None,
            "source": "user_context",
        }

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

    if not user_id:
        state["tool_calls"].append({
            "tool": "get_cerita_progress",
            "agent": "cerita_peri",
            "input": {},
            "result": {"error": "user_id tidak tersedia"},
        })
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

    # Mode A: image-based analysis
    if image_url:
        return await _analyze_chat_image(state, image_url)

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
        return {
            "has_data": True,
            "latest_scan": mata_peri_ctx,
            "scans": [mata_peri_ctx],
            "source": "user_context",
        }

    if not user_id:
        return {"has_data": False, "error": "user_id tidak tersedia"}

    data = await _call_internal_api(f"/api/v1/internal/agent/mata-peri-history/{user_id}")
    state["tool_calls"].append({
        "tool": "get_mata_peri_history",
        "agent": "mata_peri",
        "input": {"user_id": user_id},
        "result": {"has_data": data.get("has_data", False), "scan_count": data.get("scan_count", 0)},
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
        # Bug A fix — forward source dari request supaya MataPeriScanSession.source
        # akurat (web/mobile), bukan hardcoded 'mobile'.
        "source": state.get("source"),
    }

    headers = {}
    if trace_id:
        headers["X-Request-ID"] = trace_id

    # Image analysis butuh waktu — tapi timeout ini harus LEBIH KECIL
    # daripada AI_CHAT_TIMEOUT_SECONDS (default 120s di api side) supaya
    # kita fail dulu dengan error message yang user-friendly, bukan dapet
    # generic "Tanya Peri tidak merespons" dari proxy timeout di api.
    #
    # Chain timeout:
    #   web SSE read    : tidak ada hard limit
    #   api → ai-chat   : AI_CHAT_TIMEOUT_SECONDS = 120s
    #   ai-chat → api   : 90s (di sini) ← LEBIH KECIL, kasih buffer 30s
    #   api → ai-cv     : AI_BACKEND_TIMEOUT_SECONDS = 60s
    # Bug #5 fix — turunin dari 120 jadi 90.
    response = await _call_internal_api_post(
        "/api/v1/internal/agent/tanya-peri-analyze-image",
        json_body=payload,
        extra_headers=headers,
        timeout=90,
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
        # Graceful failure — pesan empati ke user via fallback_text.
        # Bug #5 fix — pesan disesuaikan dengan error_code supaya user
        # tahu apa yang salah (timeout vs ai down vs no_active_child).
        error_message_overrides = {
            "api_timeout": (
                "Maaf, analisis foto butuh waktu lebih lama dari biasanya. "
                "Coba kirim foto lagi sebentar lagi ya, atau tanya tanpa foto dulu."
            ),
            "ai_timeout": (
                "Maaf, AI lagi sibuk menganalisis. Coba kirim foto lagi sebentar "
                "ya — biasanya 1-2 menit udah lancar lagi."
            ),
            "ai_http_error": (
                "Maaf, ada gangguan di sistem analisis foto. "
                "Coba lagi nanti ya, atau langsung tanya tanpa foto dulu."
            ),
            "no_active_child": (
                "Untuk analisis foto, profil anak harus dibuat dulu ya. "
                "Silakan buka menu Profil > Tambah Anak."
            ),
        }
        fallback_msg = (
            response.get("error_message")
            or error_message_overrides.get(error_code or "")
            or (
                "Maaf, Peri lagi belum bisa cek foto kamu detail. "
                "Tapi dari pertanyaanmu Peri tetap bisa bantu."
            )
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
