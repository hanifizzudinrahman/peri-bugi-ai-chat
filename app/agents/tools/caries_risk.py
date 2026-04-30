"""
Tool: get_caries_risk_latest

Fetches latest caries risk assessment for the current child user.
Wraps GET /api/v1/internal/agent/caries-risk/{user_id}.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get

logger = logging.getLogger(__name__)


def make_get_caries_risk_tool(user_id: Optional[str]):
    """
    Factory: build get_caries_risk_latest tool with user_id closure.

    Security: user_id is closure-captured (NOT a tool param), so LLM
    cannot spoof another user's data.

    Args:
        user_id: Current user ID (from state.user_context.user.id)

    Returns:
        @tool decorated async function.
    """

    @tool
    async def get_caries_risk_latest() -> dict[str, Any]:
        """Get the child's LATEST caries risk assessment result (hasil kuesioner risiko karies).

        ✅ USE THIS TOOL when the user asks about:
        - "anak saya hasil kuesionernya apa?"
        - "berapa skor risiko karies anak saya?"
        - "anak saya beresiko tinggi atau rendah?"
        - "rekomendasi dari hasil kuesioner kemarin gimana?"
        - "kenapa anak saya beresiko tinggi?"

        ❌ DO NOT use this tool when:
        - User asks how to fill the questionnaire → use search_app_faq
        - User asks about caries in general (not their specific child) → use search_dental_knowledge
        - User asks about brushing → use get_brushing_stats or get_brushing_history
        - User has never filled questionnaire → tool returns has_data=False with fallback message

        Returns a dict with keys:
        - has_data: bool — true if user has filled questionnaire at least once
        - child_name: str — child's nickname for personalized response
        - child_age_years: int | null
        - total_score: int — final score from questionnaire
        - result_level: str — "low" | "moderate" | "high" (English enum)
        - result_label: str — human-friendly Indonesian label (e.g. "Risiko Tinggi")
        - recommendation: str — single text recommendation (NOT a list)
        - submitted_at: ISO datetime string
        - reason / fallback_message: only when has_data=False
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency
        # dengan existing tools pattern (tool:get_brushing_stats, tool:get_user_profile, dll).
        # Sebelumnya: tool execution invisible di trace tree, HTTP child span nempel ke agent_node.
        # Sekarang: span "tool:get_caries_risk_latest" muncul as proper child di tools span.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_caries_risk_latest",
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
                f"/api/v1/internal/agent/caries-risk/{user_id}"
            )

            if span:
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "result_level": data.get("result_level"),
                    "result_label": data.get("result_label"),
                    "total_score": data.get("total_score"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_caries_risk_latest
