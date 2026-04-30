"""
Tool: get_parenting_tip_today

Fetches today's parenting tip for the current user.
Wraps GET /api/v1/internal/agent/tip-today/{user_id}.

NOTE: Tips by design random per call (consistent dengan public /tips/today + mobile).
Tool follows same behavior — same call may return different tip on refresh.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get

logger = logging.getLogger(__name__)


def make_get_parenting_tip_today_tool(user_id: Optional[str]):
    """
    Factory: build get_parenting_tip_today tool with user_id closure.

    Args:
        user_id: Current user ID (from state.user_context.user.id)

    Returns:
        @tool decorated async function.
    """

    @tool
    async def get_parenting_tip_today() -> dict[str, Any]:
        """Get today's parenting tip from Peri Bugi (tip yang muncul di Beranda).

        Tip is personalized by child's age range when available.

        ✅ USE THIS TOOL when the user asks about:
        - "ada tips hari ini?"
        - "tips parenting buat anak saya apa?"
        - "tip dari Peri Bugi hari ini apa?"
        - "saran parenting dong"

        ❌ DO NOT use this tool when:
        - User asks about specific situation handling → use search_dental_knowledge
        - User asks about app features → use search_app_faq
        - User asks about brushing/scan/cerita → use respective tools

        Returns a dict with keys:
        - has_data: bool
        - tip: dict with id, title, content, category
          - category enum: general | brushing | nutrition | development | care | dental_visit
        - reason / fallback_message: only when has_data=False
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency
        # dengan existing tools pattern.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_parenting_tip_today",
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
                f"/api/v1/internal/agent/tip-today/{user_id}"
            )

            if span:
                tip = data.get("tip") or {}
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "tip_id": tip.get("id"),
                    "tip_category": tip.get("category"),
                    "tip_title": tip.get("title"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_parenting_tip_today
