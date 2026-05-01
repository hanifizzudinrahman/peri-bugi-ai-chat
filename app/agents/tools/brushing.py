"""
Tool: get_brushing_stats

Fetches brushing/streak data for the current child user.
Wraps GET /api/v1/internal/agent/brushing-stats/{user_id}
with fallback to user_context.brushing if API fails.

Replaces: app/agents/sub_agents/phase2_agents.py::rapot_peri_agent
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get

logger = logging.getLogger(__name__)


def make_get_brushing_stats_tool(
    user_id: Optional[str],
    user_context: dict,
):
    """
    Factory: build get_brushing_stats tool with user_id closure.

    Security: user_id is closure-captured (NOT a tool param), so LLM
    cannot spoof another user's data.

    Args:
        user_id: Current user ID (from state.user_context.user.id)
        user_context: Full user_context dict for fallback (state.user_context.model_dump())

    Returns:
        @tool decorated async function.
    """
    ctx = user_context or {}
    child = ctx.get("child") or {}
    brushing_in_context = ctx.get("brushing")

    @tool
    async def get_brushing_stats() -> dict[str, Any]:
        """Get the child's tooth brushing statistics, including current streak, best streak, today's status, and progress towards next milestone badge.

        ✅ USE THIS TOOL when user asks:
        - "streak anak berapa hari?" → check streak.current_streak
        - "rekor terbaik anak?" → check streak.best_streak
        - "kapan terakhir anak sikat?" → check streak.last_brushing_date (any slot)
          ATAU streak.last_complete_date (sikat pagi+malam dalam 1 hari)
        - "anak udah sikat hari ini belum?" → check today_status.morning + evening
        - "pagi tadi udah sikat?" → check today_status.morning.is_checked + is_window_open
        - "malam ini udah sikat?" → check today_status.evening.is_checked + is_window_open
        - "berapa hari lagi menuju badge baru?" → check streak.days_to_next_milestone
        - "target streak berikutnya berapa?" → check streak.next_milestone

        ❌ DO NOT use this tool for:
        - Riwayat sikat tanggal spesifik / minggu / bulan (use get_brushing_history)
        - List badge yang sudah didapat (use get_brushing_achievements)
        - Pertanyaan WHY brushing matters (use search_dental_knowledge)
        - Setting reminder (no tool — arahkan user ke aplikasi)

        Returns dict dengan keys:
        - has_data: bool — true kalau brushing data tersedia
        - streak: dict (current_streak, best_streak, last_complete_date,
                  last_brushing_date, last_brushing_slots,
                  next_milestone, days_to_next_milestone, progress_percentage)
        - today_status: dict (date, morning{is_checked,is_window_open},
                        evening{is_checked,is_window_open}, is_complete) — atau null
        - child_name: nama anak untuk personalisasi response
        - source: "api" | "user_context" | "user_context_fallback"

        ANTI-HALU:
        - Bedakan dua field tanggal:
          • last_brushing_date = terakhir ada catatan sikat (slot apapun)
          • last_complete_date = terakhir sikat pagi+malam dalam 1 hari yang sama
        - Kalau user nanya "kapan terakhir sikat" tanpa qualifier, jawab pakai
          last_brushing_date (lebih general). Sebut juga slot mana yang sikat
          (last_brushing_slots: ["morning"], ["evening"], atau dua-duanya).
        - Kalau today_status null, JANGAN guess "anak sudah sikat hari ini" — bilang
          "datanya belum tersedia".
        - Kalau next_milestone null, artinya streak sudah max — bilang sudah 
          capai semua target, bukan "tidak ada milestone".
        - Kalau is_window_open false dan is_checked false, slot itu sudah lewat /
          belum buka — sebut konteks waktu, bukan langsung "belum sikat".
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_brushing_stats",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "has_brushing_in_context": brushing_in_context is not None,
            },
        ) as span:
            # Fallback: tidak ada user_id → coba pakai snapshot dari user_context
            if not user_id:
                result = {
                    "child_name": child.get("nickname") or child.get("full_name", "si kecil"),
                    "streak": brushing_in_context,
                    "has_data": brushing_in_context is not None,
                    "source": "user_context_fallback",
                }
                if span:
                    span.update(output={
                        "has_data": result["has_data"],
                        "source": "user_context_fallback",
                        "current_streak": (
                            brushing_in_context.get("current_streak")
                            if brushing_in_context else None
                        ),
                    })
                return result

            # Primary: call internal API
            data = await call_internal_get(
                f"/api/v1/internal/agent/brushing-stats/{user_id}"
            )

            # Merge: kalau API gagal tapi context punya brushing, pakai itu
            if not data.get("has_data") and brushing_in_context:
                data["streak"] = brushing_in_context
                data["has_data"] = True
                data["source"] = "user_context"

            # Pastikan child_name selalu ada untuk generate prompt
            if "child_name" not in data:
                data["child_name"] = (
                    child.get("nickname") or child.get("full_name", "si kecil")
                )

            if span:
                streak_data = data.get("streak") if isinstance(data.get("streak"), dict) else None
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "source": data.get("source", "api"),
                    "current_streak": streak_data.get("current_streak") if streak_data else None,
                    "best_streak": streak_data.get("best_streak") if streak_data else None,
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_brushing_stats


# =============================================================================
# Tool: get_brushing_history (Phase 2 Tools Expansion)
# =============================================================================

def make_get_brushing_history_tool(user_id: Optional[str]):
    """
    Factory: build get_brushing_history tool.

    Fetches monthly calendar + weekly stats for brushing history.
    Reuses existing BrushingStatsService.get_calendar + .get_weekly_stats.
    """

    @tool
    async def get_brushing_history(month: Optional[str] = None) -> dict[str, Any]:
        """Get the child's BRUSHING HISTORY — monthly calendar + this week's stats + slot compliance.

        ✅ USE THIS TOOL when the user asks about:
        - "anak kemarin sikat ga?" (yesterday — filter calendar.days by date)
        - "minggu ini anak sikat berapa hari?" → check this_week.completed_sessions
        - "tanggal 15 anak sikat?" → filter calendar.days[date='YYYY-MM-15']
        - "bulan ini progressnya gimana?" → calendar overview
        - "bulan kemarin gimana?" → call dengan month="YYYY-MM" (bulan kemarin)
        - "pagi atau malam yang lebih sering kelewat?" → check slot_compliance.morning_percentage vs evening_percentage
        - "konsistensi anak gimana?" → check slot_compliance overview

        ❌ DO NOT use this tool when:
        - User only asks about today's status / current streak → use get_brushing_stats (lighter)
        - User asks about achievements/badges → use get_brushing_achievements
        - User asks how to log brushing → use search_app_faq
        - User asks tren multi-bulan / tahun → tool tidak support multi-month aggregation, just bulan tertentu

        Args:
            month: format "YYYY-MM" (optional, default current month).
                   Untuk "bulan kemarin", compute dari KONTEKS WAKTU di system prompt.

        Returns a dict with keys:
        - has_data: bool
        - child_name: str
        - month: "YYYY-MM" string
        - calendar: dict (year, month, days[{date, morning, evening, complete}])
        - this_week: dict (week_start, week_end, completed_sessions, total_sessions,
                     compliance_percentage, current_streak)
        - slot_compliance: dict (total_days, morning_count, evening_count,
                          morning_percentage, evening_percentage) — atau null
        - reason / fallback_message: kalau has_data=False

        ANTI-HALU:
        - Kalau tanggal yang ditanya tidak ada di calendar.days, artinya BELUM ADA DATA,
          BUKAN belum sikat. Bilang "datanya belum tersedia di tanggal itu".
        - Kalau slot_compliance null atau total_days=0, JANGAN compare pagi vs malam.
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_brushing_history",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
                "month": month,
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

            path = f"/api/v1/internal/agent/brushing-history/{user_id}"
            if month:
                path += f"?month={month}"

            data = await call_internal_get(path)

            if span:
                this_week = data.get("this_week") or {}
                calendar = data.get("calendar") or {}
                days_list = calendar.get("days") or []
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "month": data.get("month"),
                    "days_in_calendar": len(days_list),
                    "this_week_completed": this_week.get("completed_sessions"),
                    "this_week_total": this_week.get("total_sessions"),
                    "current_streak": this_week.get("current_streak"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_brushing_history


# =============================================================================
# Tool: get_brushing_achievements (Phase 2 Tools Expansion)
# =============================================================================

def make_get_brushing_achievements_tool(user_id: Optional[str]):
    """
    Factory: build get_brushing_achievements tool.

    Fetches list of all achievements (unlocked + locked) + next target.
    Reuses existing BrushingService.get_achievements.
    """

    @tool
    async def get_brushing_achievements() -> dict[str, Any]:
        """Get the child's BRUSHING ACHIEVEMENTS / BADGES (milestone-based).

        Achievements based on consecutive brushing days: 1, 3, 7, 14, 21, 30, 100, 365 days.

        ✅ USE THIS TOOL when the user asks about:
        - "anak saya udah dapat badge apa?"
        - "achievement apa yang udah unlock?"
        - "badge apa yang masih bisa didapat?"
        - "anak saya udah pencapaian apa?"
        - "tinggal berapa hari lagi anak dapat badge berikutnya?"

        ❌ DO NOT use this tool when:
        - User asks about general brushing progress / streak → use get_brushing_stats
        - User asks about brushing history per day → use get_brushing_history
        - User asks how to earn badges → use search_app_faq

        Returns a dict with keys:
        - has_data: bool
        - child_name: str
        - current_streak: int — current consecutive days
        - total_unlocked: int — count of unlocked achievements
        - total_available: int — total possible achievements (always 8)
        - next_target: dict | null with milestone_days, label, days_remaining
        - achievements: list of {milestone_days, label, is_unlocked, achieved_at}
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_brushing_achievements",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
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

            data = await call_internal_get(
                f"/api/v1/internal/agent/brushing-achievements/{user_id}"
            )

            if span:
                next_target = data.get("next_target") or {}
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "current_streak": data.get("current_streak"),
                    "total_unlocked": data.get("total_unlocked"),
                    "total_available": data.get("total_available"),
                    "next_target_label": next_target.get("label"),
                    "next_target_days_remaining": next_target.get("days_remaining"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_brushing_achievements


# =============================================================================
# ToolSpec registrations — Bagian C: registry pattern
# =============================================================================
from app.agents.tools.registry import ToolSpec, register_tool, BridgeContext


# ── get_brushing_stats ──────────────────────────────────────────────────────

def _bridge_brushing_stats(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: stats result → agent_results['rapot_peri']."""
    agent_results["rapot_peri"] = result


def _inject_brushing_stats(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject brushing stats (streak, best_streak, achievements) to system prompt.
    
    CRITICAL — sebelumnya tool result HILANG (di-bridge tapi tidak di-consume).
    Fix Bagian C: inject ke system prompt supaya LLM bisa jawab dengan data tepat.
    
    Phase 2 enrichment:
    - last_complete_date — kapan terakhir sikat LENGKAP (pagi+malam)
    - last_brushing_date — kapan terakhir ada catatan sikat (slot apapun) [Phase 2.1]
    - next_milestone, days_to_next_milestone — untuk "berapa lagi sampai badge"
    - today_status — pagi/malam udah sikat hari ini, plus is_window_open
    
    Phase 2.1 hotfix: bedakan "terakhir sikat lengkap" (last_complete_date) vs
    "terakhir ada catatan sikat slot apapun" (last_brushing_date). Kalau user
    nanya "kapan terakhir sikat?" tanpa qualifier, ini ambigu — display kedua
    field supaya LLM bisa pilih yang relevan dengan context user.
    """
    if not data or not data.get("has_data"):
        return ""
    
    streak = data.get("streak") or {}
    current_streak = streak.get("current_streak", 0)
    best_streak = streak.get("best_streak", 0)
    last_complete_date = streak.get("last_complete_date")
    last_brushing_date = streak.get("last_brushing_date")  # Phase 2.1
    last_brushing_slots = streak.get("last_brushing_slots")  # Phase 2.1
    next_milestone = streak.get("next_milestone")
    days_to_next_milestone = streak.get("days_to_next_milestone")
    
    weekly = data.get("weekly_summary") or {}
    achievements = data.get("achievements") or []
    today_status = data.get("today_status")
    
    text = (
        f"\n\nData rapot sikat gigi {child_name} (LATEST dari tool call):"
        f"\n- Streak saat ini: {current_streak} hari"
        f"\n- Rekor terbaik: {best_streak} hari"
    )
    
    # Phase 2.1: Bedakan complete vs any-slot
    # Definisi:
    #   - last_complete_date: tanggal terakhir aaa sikat LENGKAP (pagi+malam dua-duanya)
    #   - last_brushing_date: tanggal terakhir aaa ada catatan sikat (slot apapun)
    
    # Display last_brushing_date dulu (any-slot, lebih general)
    if last_brushing_date:
        slots_text = ""
        if last_brushing_slots:
            slot_id_to_label = {"morning": "pagi", "evening": "malam"}
            slot_labels = [slot_id_to_label.get(s, s) for s in last_brushing_slots]
            if len(slot_labels) == 2:
                slots_text = " (sikat pagi + malam — lengkap)"
            elif len(slot_labels) == 1:
                slots_text = f" (sikat {slot_labels[0]} saja)"
        text += f"\n- Terakhir ada catatan sikat: {last_brushing_date}{slots_text}"
    else:
        text += f"\n- Belum pernah ada catatan sikat sama sekali"
    
    # Display last_complete_date — explicit info untuk LLM
    if last_complete_date:
        text += f"\n- Terakhir sikat lengkap (pagi+malam dalam 1 hari): {last_complete_date}"
    elif last_brushing_date:
        # Ada catatan sikat tapi belum pernah complete day
        text += f"\n- Belum pernah ada hari yang sikat lengkap (pagi+malam) di 1 hari yang sama"
    # Kalau dua-duanya null, sudah ke-handle di "Belum pernah ada catatan sikat" di atas
    
    # Phase 2: berapa lagi menuju milestone berikutnya
    if next_milestone is not None and days_to_next_milestone is not None:
        text += (
            f"\n- Menuju target {next_milestone} hari berturut-turut: "
            f"butuh {days_to_next_milestone} hari lagi"
        )
    elif next_milestone is None and current_streak > 0:
        text += "\n- Sudah capai semua target streak (max milestone tercapai)"
    
    # Phase 2: status hari ini (pagi vs malam, dengan window awareness)
    if today_status:
        morning = today_status.get("morning") or {}
        evening = today_status.get("evening") or {}
        m_check = "✓ sudah" if morning.get("is_checked") else "✗ belum"
        e_check = "✓ sudah" if evening.get("is_checked") else "✗ belum"
        m_window = morning.get("is_window_open")
        e_window = evening.get("is_window_open")
        
        text += f"\n- Hari ini ({today_status.get('date', 'today')}):"
        text += f"\n  • Sikat pagi: {m_check}"
        if not morning.get("is_checked") and not m_window:
            text += " (slot pagi sudah lewat — tidak bisa centang lagi)"
        elif not morning.get("is_checked") and m_window:
            text += " (slot pagi masih aktif, bisa centang sekarang)"
        text += f"\n  • Sikat malam: {e_check}"
        if not evening.get("is_checked") and not e_window:
            text += " (slot malam belum buka)"
        elif not evening.get("is_checked") and e_window:
            text += " (slot malam aktif sekarang)"
    
    if weekly:
        text += (
            f"\n- Minggu ini: {weekly.get('completed_sessions', 0)} dari "
            f"{weekly.get('total_sessions', 14)} sesi sikat gigi tercatat"
        )
    if achievements:
        unlocked = [a for a in achievements if a.get("is_unlocked")]
        text += f"\n- Total badge unlocked: {len(unlocked)} dari {len(achievements)}"
    
    return text


register_tool(ToolSpec(
    tool_name="get_brushing_stats",
    agent_key="rapot_peri",
    required_agent="rapot_peri",
    bridge_handler=_bridge_brushing_stats,
    prompt_injector=_inject_brushing_stats,
    thinking_label="Mengecek rapot sikat gigi...",
))


# ── get_brushing_history ────────────────────────────────────────────────────

def _bridge_brushing_history(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: history result → agent_results['rapot_peri_history'].
    
    Pakai key terpisah (bukan 'rapot_peri') supaya stats + history bisa coexist.
    """
    agent_results["rapot_peri_history"] = result


def _inject_brushing_history(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject brushing history (calendar + weekly + slot_compliance) to system prompt.
    
    Phase 2: tambah slot_compliance — pagi vs malam mana yang lebih sering kelewat.
    """
    if not data or not data.get("has_data"):
        # Loud signal ke LLM bahwa tool returned empty/error
        if data and data.get("reason"):
            return (
                f"\n\nData history sikat gigi {child_name}: TIDAK TERSEDIA "
                f"(reason: {data.get('reason')}). "
                f"JANGAN karang data — kasih tahu user dengan jujur."
            )
        return ""
    
    month = data.get("month", "-")
    calendar = data.get("calendar") or {}
    days = calendar.get("days") or []
    this_week = data.get("this_week") or {}
    slot_compliance = data.get("slot_compliance") or {}  # Phase 2
    
    # Hitung total hari dengan brushing tercatat di bulan tersebut
    days_with_brushing = sum(
        1 for d in days if d.get("morning") or d.get("evening")
    )
    
    text = (
        f"\n\nData history sikat gigi {child_name} (bulan {month}, dari tool call):"
        f"\n- Total hari dengan sikat gigi tercatat: {days_with_brushing} dari {len(days)} hari"
        f"\n- Minggu ini: {this_week.get('completed_sessions', 0)} sesi tercatat "
        f"(streak: {this_week.get('current_streak', 0)} hari)"
    )
    
    # Phase 2: Slot compliance — pagi vs malam comparison
    if slot_compliance and slot_compliance.get("total_days", 0) > 0:
        m_pct = slot_compliance.get("morning_percentage")
        e_pct = slot_compliance.get("evening_percentage")
        m_count = slot_compliance.get("morning_count", 0)
        e_count = slot_compliance.get("evening_count", 0)
        total = slot_compliance.get("total_days", 0)
        
        text += (
            f"\n- Konsistensi pagi: {m_count}/{total} hari "
            f"({m_pct}%)" if m_pct is not None else f"\n- Konsistensi pagi: {m_count}/{total} hari"
        )
        text += (
            f"\n- Konsistensi malam: {e_count}/{total} hari "
            f"({e_pct}%)" if e_pct is not None else f"\n- Konsistensi malam: {e_count}/{total} hari"
        )
        # Hint untuk LLM: pagi atau malam yang lebih sering kelewat?
        if m_pct is not None and e_pct is not None:
            if m_pct < e_pct - 5:
                text += "\n  (Slot pagi lebih sering kelewat dibanding malam)"
            elif e_pct < m_pct - 5:
                text += "\n  (Slot malam lebih sering kelewat dibanding pagi)"
            # else: kira-kira sama, no hint
    
    # Detail per-hari (cap di 10 hari terakhir untuk avoid prompt bloat)
    if days:
        recent_days = days[-10:]
        detail_lines = []
        for d in recent_days:
            date = d.get("date")
            morning = "✓" if d.get("morning") else "✗"
            evening = "✓" if d.get("evening") else "✗"
            detail_lines.append(f"  {date}: pagi={morning} malam={evening}")
        text += f"\n- Detail 10 hari terakhir:\n" + "\n".join(detail_lines)
    
    text += (
        f"\n\nPENTING: Pakai data ini untuk jawab pertanyaan tanggal spesifik. "
        f"Kalau tanggal tidak ada di list di atas, artinya BELUM ADA DATA, "
        f"BUKAN belum sikat. Jawab dengan jujur."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_brushing_history",
    agent_key="rapot_peri_history",
    required_agent="rapot_peri",
    bridge_handler=_bridge_brushing_history,
    prompt_injector=_inject_brushing_history,
    thinking_label="Mengecek riwayat sikat gigi...",
))


# ── get_brushing_achievements ───────────────────────────────────────────────

def _bridge_brushing_achievements(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: achievements result → agent_results['rapot_peri_achievements']."""
    agent_results["rapot_peri_achievements"] = result


def _inject_brushing_achievements(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject achievements (badges unlocked + next target) to system prompt."""
    if not data or not data.get("has_data"):
        return ""
    
    current_streak = data.get("current_streak", 0)
    total_unlocked = data.get("total_unlocked", 0)
    total_available = data.get("total_available", 0)
    next_target = data.get("next_target") or {}
    achievements = data.get("achievements") or []
    
    text = (
        f"\n\nData badge/achievement {child_name} (dari tool call):"
        f"\n- Streak saat ini: {current_streak} hari"
        f"\n- Total badge unlocked: {total_unlocked} dari {total_available}"
    )
    
    # List badges yang unlocked
    unlocked = [a for a in achievements if a.get("is_unlocked")]
    if unlocked:
        names = [a.get("label") for a in unlocked if a.get("label")]
        text += f"\n- Badge yang sudah diraih: {', '.join(names)}"
    else:
        text += f"\n- BELUM ADA BADGE yang diraih (streak masih {current_streak} hari)"
    
    if next_target:
        text += (
            f"\n- Target berikutnya: {next_target.get('label')} "
            f"({next_target.get('days_remaining', 0)} hari lagi)"
        )
    
    return text


register_tool(ToolSpec(
    tool_name="get_brushing_achievements",
    agent_key="rapot_peri_achievements",
    required_agent="rapot_peri",
    bridge_handler=_bridge_brushing_achievements,
    prompt_injector=_inject_brushing_achievements,
    thinking_label="Mengecek badge sikat gigi...",
))
