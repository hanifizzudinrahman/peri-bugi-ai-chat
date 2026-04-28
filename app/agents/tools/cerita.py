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

        Use this tool when the user asks about:
        - Cerita Peri progress ("anak saya udah belajar apa aja di Cerita Peri?")
        - Completed modules or stars earned
        - Which module to do next ("anak saya harus lanjut modul apa?")
        - Whether the child has finished a specific topic

        Returns a dict with keys:
        - has_data: bool
        - total_modules: int — total modules available
        - completed_count: int — modules completed
        - current_module_id: str — next module to do
        - total_stars: int — stars earned across all modules
        - modules: list — per-module progress detail
        - error: str (only if has_data=False)

        Do NOT use this tool for:
        - General education content questions (use search_dental_knowledge instead)
        - Asking the user to start a module (just suggest in your response)
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
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "completed_count": data.get("completed_count", 0),
                    "total_stars": data.get("total_stars", 0),
                    "current_module_id": data.get("current_module_id"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_cerita_progress
