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
        """Get the child's tooth brushing statistics, including current streak, best streak, achievements, and weekly summary.

        Use this tool when the user asks about:
        - Brushing progress or streak ("streak anak saya berapa hari?")
        - Achievements or milestones ("rapot anak saya apa aja?")
        - Whether the child has been brushing consistently
        - Weekly or monthly brushing patterns

        Returns a dict with keys:
        - has_data: bool — true if brushing data available
        - streak: dict with current_streak, best_streak, etc
        - child_name: name of the child for personalized response
        - source: "api" | "user_context" | "user_context_fallback"
        - error: str (only if has_data=False)

        Do NOT use this tool for:
        - Questions about WHY brushing matters (use search_dental_knowledge)
        - Setting up reminders (not supported in chat — direct user to app)
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
        """Get the child's BRUSHING HISTORY — monthly calendar + this week's stats.

        ✅ USE THIS TOOL when the user asks about:
        - "anak kemarin sikat ga?" (yesterday)
        - "minggu ini anak sikat berapa hari?" (this week)
        - "tanggal 15 anak sikat?" (specific date)
        - "bulan ini progressnya gimana?" (monthly overview)
        - "anak sikat di hari Senin Selasa Rabu kemarin?"

        ❌ DO NOT use this tool when:
        - User only asks about today's status / current streak → use get_brushing_stats (lighter)
        - User asks about achievements/badges → use get_brushing_achievements
        - User asks how to log brushing → use search_app_faq

        Args:
            month: format "YYYY-MM" (optional, default current month)

        Returns a dict with keys:
        - has_data: bool
        - child_name: str
        - month: "YYYY-MM" string
        - calendar: dict with year, month, days (list of {date, morning, evening, complete})
        - this_week: dict with week_start, week_end, completed_sessions, total_sessions,
                     compliance_percentage, current_streak
        - reason / fallback_message: only when has_data=False
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
    """
    if not data or not data.get("has_data"):
        return ""
    
    streak = data.get("streak") or {}
    current_streak = streak.get("current_streak", 0)
    best_streak = streak.get("best_streak", 0)
    weekly = data.get("weekly_summary") or {}
    achievements = data.get("achievements") or []
    
    text = (
        f"\n\nData rapot sikat gigi {child_name} (LATEST dari tool call):"
        f"\n- Streak saat ini: {current_streak} hari"
        f"\n- Rekor terbaik: {best_streak} hari"
    )
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
    """Inject brushing history (calendar + weekly) to system prompt."""
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
