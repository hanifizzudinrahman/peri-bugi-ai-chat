"""
Tool: get_user_profile

Reads user info from user_context injected pre-graph by peri-bugi-api.
NO API call needed — data already in state.

Note (Phase 2 design choice — Q2 from Hanif):
We chose AUTO-INJECT user data into system prompt (Q2 = Auto-kasih).
This means generate.py already has user_context in system prompt.
HOWEVER, get_user_profile is still useful for cases like:
- LLM wants to confirm child name spelling
- LLM doing reasoning that needs explicit profile data
- Fallback when system prompt context is unclear

Replaces: app/agents/sub_agents/__init__.py::user_profile_agent
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def make_get_user_profile_tool(user_context: dict):
    """
    Factory: build get_user_profile tool with user_context closure.

    Args:
        user_context: Snapshot of state.user_context.model_dump() at turn start.
                      Contains user, child, brushing, mata_peri_last_result.

    Returns:
        @tool decorated async function ready for LLM bind_tools.
    """
    user = user_context.get("user") or {}
    child = user_context.get("child") or {}
    brushing = user_context.get("brushing")
    mata_peri = user_context.get("mata_peri_last_result")

    @tool
    async def get_user_profile() -> dict[str, Any]:
        """Get the parent and child profile information for the current chat user.

        Use this when you need explicit access to:
        - Parent's name, gender (for personalized greetings)
        - Child's name, age, gender (for age-appropriate dental advice)
        - Whether the user has brushing data or scan history available

        Returns a dict with keys:
        - profile: parent info (name, gender)
        - child: child info (name, age_years, gender) or null if no child
        - brushing: latest brushing snapshot or null
        - mata_peri_last: latest scan result snapshot or null
        - has_brushing_data: bool
        - has_scan_data: bool

        Note: User profile is also auto-injected into your system prompt,
        so most of the time you don't need to call this. Use it only when
        you need to confirm a specific field or the system prompt context
        is unclear.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_user_profile",
            state=None,  # state context not needed for this read-only tool
            input_data={
                "has_user": bool(user),
                "has_child": bool(child),
                "has_brushing": brushing is not None,
                "has_scan": mata_peri is not None,
            },
        ) as span:
            result = {
                "profile": {
                    "name": user.get("nickname") or user.get("full_name", "User"),
                    "gender": user.get("gender"),
                },
                "child": {
                    "name": child.get("nickname") or child.get("full_name"),
                    "age_years": child.get("age_years"),
                    "gender": child.get("gender"),
                } if child else None,
                "brushing": brushing,
                "mata_peri_last": mata_peri,
                "has_brushing_data": brushing is not None,
                "has_scan_data": mata_peri is not None,
            }

            if span:
                span.update(output={
                    "child_name": result["child"]["name"] if result["child"] else None,
                    "child_age": result["child"]["age_years"] if result["child"] else None,
                    "has_brushing_data": result["has_brushing_data"],
                    "has_scan_data": result["has_scan_data"],
                    "data": _safe_dict_for_trace(result),
                })

            return result

    return get_user_profile
