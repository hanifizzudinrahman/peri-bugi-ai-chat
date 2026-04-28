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
