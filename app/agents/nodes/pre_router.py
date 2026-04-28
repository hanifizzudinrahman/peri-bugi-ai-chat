"""
Pre-router node — Phase 2 Step 2a

Runs FIRST before agent_node. Detects special cases that require deterministic
tool routing (no LLM decision needed):

1. Image upload + no view_hint → force `analyze_chat_image` tool
2. Clarification reply (user just answered ClarificationCard) → force `analyze_chat_image` again
   (with their selected view as input)

For all other cases (text-only chat), pre_router passes through with
no forced routing. agent_node will decide tools via LLM.

Why a deterministic pre-router instead of letting LLM decide:
- Image flow has been tuned + tested in legacy mata_peri_agent. We preserve
  that UX exactly, no LLM-based routing risk.
- Gemini bind_tools sometimes ignores image context — explicit forcing
  guarantees correct behavior.
- Faster: skip 1 LLM call when intent is clear from state.image being set.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.state import AgentState, ThinkingStep

logger = logging.getLogger(__name__)


async def pre_router_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: detect forced routing scenarios.

    Returns partial state update dict. Does NOT mutate input state.

    State updates:
    - thinking_steps: appended (step 1 = "Memahami pertanyaanmu...")
    - forced_tool_calls (NEW field, see state.py): list of dict
        Format: [{"name": "tool_name", "args": {}}]
        Empty list = no forced routing, agent_node free to decide.
    """
    new_thinking = ThinkingStep(step=1, label="Memahami pertanyaanmu...", done=True)
    thinking_update = [*state.thinking_steps, new_thinking]

    forced: list[dict] = []

    # Image upload detection — force analyze_chat_image
    # Note: tool itself reads image_url, user_id, clarification_selected, dll
    # via factory closure. We just signal "must call this tool".
    if state.image is not None and state.image.image_url:
        if "mata_peri" in (state.control.allowed_agents or []):
            forced.append({
                "name": "analyze_chat_image",
                "args": {},  # no LLM-controlled args; tool reads from closure
            })
            logger.info(
                f"[pre_router] Image detected, force analyze_chat_image. "
                f"clarification_selected={state.image.clarification_selected}"
            )
        else:
            logger.warning(
                "[pre_router] Image detected but mata_peri not in allowed_agents — "
                "agent_node will decide (image likely ignored)."
            )

    return {
        "thinking_steps": thinking_update,
        "forced_tool_calls": forced,
    }
