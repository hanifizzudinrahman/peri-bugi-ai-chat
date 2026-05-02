"""
Tool: get_cerita_progress

Fetches Cerita Peri educational module progress for the current user.
Wraps GET /api/v1/internal/agent/cerita-progress/{user_id}.

Replaces: app/agents/sub_agents/phase2_agents.py::cerita_peri_agent
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get

logger = logging.getLogger(__name__)


def make_get_cerita_progress_tool(user_id: Optional[str]):
    """
    Factory: build get_cerita_progress tool with user_id closure.

    Args:
        user_id: Current user ID (closure-captured, not LLM-controlled)

    Returns:
        @tool decorated async function.
    """

    @tool
    async def get_cerita_progress() -> dict[str, Any]:
        """Get the child's progress through Cerita Peri (interactive dental education stories).

        Cerita Peri = 6 modul edukasi gigi yang dikerjakan oleh ORANG TUA (bukan anak).
        Tiap modul punya slides + quiz. Skor quiz nentuin stars_earned (1-3 bintang).

        ✅ USE THIS TOOL when the user asks about:
        - "anak saya udah belajar apa aja di Cerita Peri?" → list modules dengan status
        - "modul apa aja yang udah selesai?" → filter modules by status='completed'
        - "total bintang berapa?" → check total_stars
        - "modul mana yang anak paling jago?" → check best_module (Phase 2)
        - "skor terbaik di modul mana?" → check best_module.best_score (Phase 2)
        - "modul apa yang harus dikerjain selanjutnya?" → check current_module_id
        - "berapa modul yang udah dikerjain?" → check completed_count

        ❌ DO NOT use this tool when:
        - User asks isi modul tertentu → use get_cerita_module_detail (lock-aware)
        - General education content questions (use search_dental_knowledge instead)
        - User asks for certificate / sertifikat → fitur belum ada, bilang jujur

        Returns a dict with keys:
        - has_data: bool
        - total_modules: int — total modul available (saat ini 6)
        - completed_count: int — modul yang completed
        - total_stars: int — total bintang dari semua modul
        - current_module_id: int | null — modul yang sedang aktif / berikutnya
        - current_module_status: str — status modul current
        - modules: list of {module_id, status, stars_earned, best_score, completed_at}
        - best_module: {module_id, title, subtitle, stars_earned, best_score} | null (Phase 2)
        - error: str (only if has_data=False)

        ANTI-HALU:
        - YANG NGERJAIN MODUL ADALAH ORANG TUA, bukan anak. Pakai bahasa
          "Bunda sudah selesaikan modul X", BUKAN "anak sudah selesaikan".
        - Kalau best_module null, artinya belum ada modul completed — JANGAN
          karang "modul X paling jago".
        - Kalau modul status='locked', JANGAN spoiler isi modul. Kasih tahu user
          modul belum unlock + saran selesaikan modul sebelumnya.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_cerita_progress",
            state=None,
            input_data={"has_user_id": bool(user_id)},
        ) as span:
            if not user_id:
                if span:
                    span.update(output={"has_data": False, "error": "user_id_missing"})
                return {"has_data": False, "error": "user_id tidak tersedia"}

            data = await call_internal_get(
                f"/api/v1/internal/agent/cerita-progress/{user_id}"
            )

            if span:
                best_mod = data.get("best_module") or {}
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "completed_count": data.get("completed_count", 0),
                    "total_stars": data.get("total_stars", 0),
                    "current_module_id": data.get("current_module_id"),
                    "best_module_id": best_mod.get("module_id"),
                    "best_module_stars": best_mod.get("stars_earned"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_cerita_progress


# =============================================================================
# Tool: get_cerita_module_detail (Phase 2 Tools Expansion)
# =============================================================================

def make_get_cerita_module_detail_tool(user_id: Optional[str]):
    """
    Factory: build get_cerita_module_detail tool.

    CRITICAL: Tool RESPECT LOCK STATUS.
    Cerita Peri = edukasi literasi bertahap (mingguan).
    Tool tidak boleh kasih content modul yang masih LOCKED untuk user (no spoiler!).
    """

    @tool
    async def get_cerita_module_detail(module_id: int) -> dict[str, Any]:
        """Get DETAIL of a specific Cerita Peri module (literature content + quiz info).

        Cerita Peri = staged literacy education on dental health.
        6 modules per season, unlocking weekly after previous module completion.
        Module unlock logic is RESPECTED — tool returns no content for locked modules.

        ✅ USE THIS TOOL when the user asks about:
        - "modul Tanda Masalah isinya apa?" (specific module by name)
        - "cerita modul X ngajarin apa?"
        - "ringkasan modul Cerita yang lagi anak baca?"
        - "modul yang baru unlock topiknya apa?"
        - "modul 4 isinya apa?" (specific module by number)

        ❌ DO NOT use this tool when:
        - User asks about progress (how many modules completed) → use get_cerita_progress
        - User asks general dental topics → use search_dental_knowledge
        - User does NOT specify which module → ask user to clarify which module

        Args:
            module_id: integer 1-6 (which module). MUST be int, not UUID or string.

        Returns a dict with keys:
        - has_data: bool
        - module: dict with:
          - module_id, title, subtitle, estimated_minutes
          - is_locked: bool — IMPORTANT! Check this first
          - user_status: "locked" | "available" | "in_progress" | "completed"
          - IF locked:
            - unlock_at: ISO datetime (when module unlocks)
            - fallback_message: user-friendly message about unlock
            - NO slides_summary (no spoiler!)
          - IF unlocked (available/in_progress/completed):
            - best_score, stars_earned, attempt_count, completed_at
            - total_slides, total_quiz_questions
            - slides_summary: list of {slide_order, headline, body_text}

        IMPORTANT: When module is locked, tell user when it will unlock — DO NOT
        provide any content from slides_summary.
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_cerita_module_detail",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "module_id": module_id,
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

            if not isinstance(module_id, int) or module_id < 1 or module_id > 6:
                result = {
                    "has_data": False,
                    "reason": "invalid_module_id",
                    "fallback_message": "Modul Cerita Peri tersedia 1-6 saja.",
                }
                if span:
                    span.update(output={
                        "has_data": False,
                        "reason": "invalid_module_id",
                        "module_id_received": module_id,
                    })
                return result

            data = await call_internal_get(
                f"/api/v1/internal/agent/cerita-module/{module_id}?user_id={user_id}"
            )

            if span:
                module = data.get("module") or {}
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "module_id": module.get("module_id"),
                    "module_title": module.get("title"),
                    "is_locked": module.get("is_locked"),
                    "user_status": module.get("user_status"),
                    # IMPORTANT: Don't dump full content if locked (sensitive)
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_cerita_module_detail


# =============================================================================
# ToolSpec registrations — Bagian C: registry pattern
# =============================================================================
from app.agents.tools.registry import ToolSpec, register_tool, BridgeContext


# ── get_cerita_progress ─────────────────────────────────────────────────────

def _bridge_cerita_progress(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: progress result → agent_results['cerita_peri']."""
    agent_results["cerita_peri"] = result


def _inject_cerita_progress(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject Cerita Peri progress (modul completed + current modul + best_module) to system prompt.
    
    Phase 2: tambah best_module — modul yang skornya paling baik.
    Bug fix: sebelumnya pakai key wrong (completed_modules vs API completed_count).
    Sekarang sinkron dengan response shape internal endpoint.
    """
    if not data or not data.get("has_data"):
        return ""
    
    # Sinkron dengan response /agent/cerita-progress/{user_id}
    completed = data.get("completed_count", 0)
    total = data.get("total_modules", 6)
    total_stars = data.get("total_stars", 0)
    current_module_id = data.get("current_module_id")
    current_module_status = data.get("current_module_status")
    modules = data.get("modules") or []
    best_module = data.get("best_module")  # Phase 2
    
    text = (
        f"\n\nData progress Cerita Peri {child_name} (dari tool call):"
        f"\n- Modul selesai: {completed} dari {total}"
        f"\n- Total bintang: {total_stars}"
    )
    
    if current_module_id is not None:
        text += (
            f"\n- Modul yang sedang aktif: Modul {current_module_id} "
            f"(status: {current_module_status or '-'})"
        )
    
    # Phase 2: Best module — yang skornya paling baik
    if best_module:
        bm_title = best_module.get("title", f"Modul {best_module.get('module_id', '?')}")
        bm_stars = best_module.get("stars_earned", 0)
        bm_score = best_module.get("best_score")
        text += f"\n- Modul yang Bunda paling jago: \"{bm_title}\" — {bm_stars} bintang"
        if bm_score is not None:
            text += f" (skor: {bm_score})"
    
    # Per-module status snapshot (cap untuk hemat token, max 6 modul)
    if modules:
        text += f"\n- Detail per modul:"
        for m in modules[:6]:
            mid = m.get("module_id")
            st = m.get("status", "-")
            stars = m.get("stars_earned", 0)
            score = m.get("best_score")
            score_text = f", skor {score}" if score is not None else ""
            text += f"\n  • Modul {mid}: {st}, {stars} bintang{score_text}"
    
    text += (
        f"\n\nINSTRUKSI: "
        f"Yang ngerjain modul adalah ORANG TUA (Bunda), bukan anak. Pakai bahasa "
        f"\"Bunda sudah selesaikan modul X\". "
        f"Kalau best_module null, JANGAN karang modul mana yang paling jago — "
        f"belum ada modul yang completed. "
        f"Untuk detail isi modul tertentu, panggil tool get_cerita_module_detail."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_cerita_progress",
    agent_key="cerita_peri",
    required_agent="cerita_peri",
    bridge_handler=_bridge_cerita_progress,
    prompt_injector=_inject_cerita_progress,
    thinking_label="Mengecek progress Cerita Peri...",
))


# ── get_cerita_module_detail (THE BUG TOOL — was completely missing!) ───────

def _bridge_cerita_module_detail(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: module detail result → agent_results['cerita_module_detail'].
    
    Pakai key terpisah dari 'cerita_peri' (progress) supaya bisa coexist.
    """
    agent_results["cerita_module_detail"] = result


def _inject_cerita_module_detail(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject specific Cerita module detail to system prompt.
    
    CRITICAL: Sebelum fix, tool ini di-call tapi data dibuang. LLM halusinasi
    karang title module. Sekarang inject data REAL ke prompt.
    """
    if not data or not data.get("has_data"):
        return ""
    
    module = data.get("module") or {}
    module_id = module.get("module_id")
    title = module.get("title", "-")
    subtitle = module.get("subtitle", "-")
    is_locked = module.get("is_locked", False)
    user_status = module.get("user_status", "unknown")
    
    text = (
        f"\n\nDetail Modul Cerita Peri (dari tool call — DATA TEPAT, JANGAN KARANG):"
        f"\n- Modul ID: {module_id}"
        f"\n- Judul: \"{title}\""
        f"\n- Subtitle: \"{subtitle}\""
        f"\n- Status: {user_status}"
    )
    
    if is_locked:
        fallback_msg = module.get("fallback_message", "")
        unlock_at = module.get("unlock_at")
        text += (
            f"\n\n⚠️ MODUL INI MASIH TERKUNCI."
            f"\n- {fallback_msg}"
        )
        if unlock_at:
            text += f"\n- Unlock pada: {unlock_at}"
        text += (
            f"\n\nINSTRUKSI: Beri tahu user bahwa modul masih terkunci. "
            f"JANGAN spoiler isi modul. Sarankan selesaikan modul sebelumnya."
        )
    else:
        # Unlocked — boleh include slides_summary kalau ada
        slides_summary = module.get("slides_summary") or []
        estimated_minutes = module.get("estimated_minutes")
        best_score = module.get("best_score")
        stars_earned = module.get("stars_earned")
        
        if estimated_minutes:
            text += f"\n- Estimasi waktu baca: {estimated_minutes} menit"
        if best_score is not None:
            text += f"\n- Skor terbaik: {best_score}"
        if stars_earned is not None:
            text += f"\n- Bintang didapat: {stars_earned}"
        
        if slides_summary:
            slides_text = "\n".join(
                f"  Slide {s.get('slide_order')}: {s.get('headline', '-')} — "
                f"{(s.get('body_text') or '')[:150]}"
                for s in slides_summary[:5]  # cap di 5 slides untuk hemat token
            )
            text += f"\n- Ringkasan konten:\n{slides_text}"
        
        text += (
            f"\n\nINSTRUKSI: Pakai data title + subtitle + ringkasan konten "
            f"untuk jawab user. JANGAN karang isi modul yang tidak ada di list ini."
        )
    
    return text


register_tool(ToolSpec(
    tool_name="get_cerita_module_detail",
    agent_key="cerita_module_detail",
    required_agent="cerita_peri",
    bridge_handler=_bridge_cerita_module_detail,
    prompt_injector=_inject_cerita_module_detail,
    thinking_label="Mengecek detail modul Cerita...",
))


# =============================================================================
# Phase 3 — Tool: get_cerita_modules_summary
# =============================================================================

def make_get_cerita_modules_summary_tool(user_id: Optional[str]):
    """Factory: build get_cerita_modules_summary tool."""

    @tool
    async def get_cerita_modules_summary() -> dict[str, Any]:
        """Get summary of COMPLETED Cerita Peri modules (rekap singkat untuk parent).

        ✅ USE THIS TOOL when user asks about:
        - "rekap modul yang udah dibaca dong"
        - "udah belajar apa aja dari Cerita Peri?"
        - "ringkasan modul yang udah selesai"
        - "key takeaway tiap modul yang dah selesai"
        - "modul apa aja yang udah saya selesaikan?"

        ❌ DO NOT use this tool when:
        - User minta progress umum (semua modul) → use get_cerita_progress
        - User minta detail isi modul tertentu → use get_cerita_module_detail
        - User minta best/jago module → use get_cerita_progress (best_module field)
        - User minta materi modul yang BELUM selesai → ANTI-SPOILER, jangan kasih

        Returns dict with keys:
        - has_data: bool
        - completed_count: int
        - total_count: int (saat ini 6)
        - completed_modules: list of {
            module_id, title, key_takeaway (= subtitle dari MODULE_METADATA),
            estimated_minutes, stars_earned, best_score, completed_at
          }
        - fallback_message: kalau belum ada modul completed

        ANTI-SPOILER & ANTI-HALU:
        - Tool HANYA return modul yang sudah completed. Modul locked / available
          tidak include — supaya user yang minta "rekap" tidak ke-spoiler.
        - Kalau completed_modules empty, bilang user "belum ada modul yang Bunda
          selesaikan", JANGAN halu rekap modul yang belum selesai.
        - Yang ngerjain adalah ORANG TUA, bukan anak. Pakai bahasa "Bunda sudah
          selesaikan modul X", BUKAN "anak sudah belajar".
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_cerita_modules_summary",
            state=None,
            input_data={"has_user_id": bool(user_id)},
        ) as span:
            if not user_id:
                if span:
                    span.update(output={"has_data": False, "error": "user_id_missing"})
                return {"has_data": False, "error": "user_id tidak tersedia"}

            data = await call_internal_get(
                f"/api/v1/internal/agent/cerita-modules-summary/{user_id}"
            )

            if span:
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "completed_count": data.get("completed_count", 0),
                    "total_count": data.get("total_count", 0),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_cerita_modules_summary


def _bridge_cerita_modules_summary(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: summary → agent_results['cerita_modules_summary']."""
    agent_results["cerita_modules_summary"] = result


def _inject_cerita_modules_summary(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject cerita completed modules summary to system prompt."""
    if not data or not data.get("has_data"):
        return ""
    
    completed_count = data.get("completed_count", 0)
    total_count = data.get("total_count", 6)
    completed_modules = data.get("completed_modules") or []
    
    if completed_count == 0:
        return (
            f"\n\nRekap modul Cerita Peri yang sudah Bunda selesaikan:"
            f"\n- BELUM ADA modul yang completed dari {total_count} modul total."
            f"\nBilang user dengan ramah: \"Belum ada modul yang Bunda selesaikan, "
            f"yuk mulai dari modul 1!\""
        )
    
    text = (
        f"\n\nRekap modul Cerita Peri yang sudah Bunda selesaikan "
        f"({completed_count} dari {total_count} modul):"
    )
    
    for m in completed_modules:
        title = m.get("title", f"Modul {m.get('module_id', '?')}")
        takeaway = m.get("key_takeaway", "")
        stars = m.get("stars_earned", 0)
        score = m.get("best_score")
        
        text += f"\n\n**{title}** — {stars} bintang"
        if score is not None:
            text += f" (skor: {score})"
        if takeaway:
            text += f"\n  _Inti: {takeaway}_"
    
    text += (
        f"\n\nINSTRUKSI: Pakai data ini untuk rekap natural ke user. "
        f"YANG MENYELESAIKAN MODUL ADALAH ORANG TUA (Bunda), bukan anak. "
        f"JANGAN bocorkan modul yang belum selesai."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_cerita_modules_summary",
    agent_key="cerita_modules_summary",
    required_agent="cerita_peri",
    bridge_handler=_bridge_cerita_modules_summary,
    prompt_injector=_inject_cerita_modules_summary,
    thinking_label="Merangkum modul Cerita Peri...",
))
