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
    Ambil riwayat scan Mata Peri dan narasi hasil.

    Dua mode:
    1. Image baru dikirim → forward ke ai-cv untuk analisis
    2. Tanya riwayat scan → ambil dari api

    Return: {has_data, latest_scan, scans, image_analysis}
    """
    image_url = state.get("image_url")

    # Mode 1: ada gambar baru → analisis via ai-cv
    if image_url and settings.AI_CV_URL:
        result = await _analyze_image(state, image_url)
        return result

    # Mode 2: tanya riwayat → ambil dari api
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


async def _analyze_image(state: AgentState, image_url: str) -> dict[str, Any]:
    """Forward image ke ai-cv untuk analisis YOLO."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{settings.AI_CV_URL}/analyze",
                json={"image_url": image_url},
                headers={"X-Internal-Secret": settings.INTERNAL_SECRET},
            )
            resp.raise_for_status()
            result = resp.json()

        state["image_analysis"] = result
        state["tool_calls"].append({
            "tool": "image_analysis",
            "agent": "mata_peri",
            "input": {"image_url": image_url[:80]},
            "result": {"status": "success", "summary_status": result.get("summary_status")},
        })
        return {
            "has_data": True,
            "image_analysis": result,
            "mode": "new_scan",
        }
    except Exception as e:
        state["tool_calls"].append({
            "tool": "image_analysis",
            "agent": "mata_peri",
            "input": {"image_url": image_url[:80]},
            "result": {"error": str(e)},
        })
        return {"has_data": False, "error": str(e), "mode": "new_scan"}
