"""
Tool Registry — Phase 2 Step 1

Factory `make_tools(state)` builds the per-turn tool list with auth/turn
context bound via closures. The LLM only sees the @tool decorators and
their docstrings; it cannot supply user_id, image_url, or other sensitive
context (factory injects from state).

USAGE (Step 2 will wire this into graph):
    from app.agents.tools import make_tools

    tools = make_tools(state)            # build per-turn, allowed_agents-filtered
    llm = get_llm().bind_tools(tools)    # LLM sees tool schemas
    ...

ALLOWED_AGENTS FILTERING:
The state.control.allowed_agents list (from peri-bugi-api per-user permissions)
gates which tools are available. If a user lacks "kb_dental" permission, the
search_dental_knowledge tool simply isn't included in the returned list.

TOOL ↔ AGENT_KEY MAPPING (for allowed_agents filtering):
    kb_dental    → search_dental_knowledge
    app_faq      → search_app_faq
    user_profile → get_user_profile
    rapot_peri   → get_brushing_stats
    cerita_peri  → get_cerita_progress
    mata_peri    → get_scan_history + analyze_chat_image (both gated together)
    janji_peri   → (Phase 6 — not implemented)
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.state import AgentState

from app.agents.tools.brushing import make_get_brushing_stats_tool
from app.agents.tools.cerita import make_get_cerita_progress_tool
from app.agents.tools.knowledge import (
    make_search_app_faq_tool,
    make_search_dental_knowledge_tool,
)
from app.agents.tools.profile import make_get_user_profile_tool
from app.agents.tools.scan import (
    make_analyze_chat_image_tool,
    make_get_scan_history_tool,
)

logger = logging.getLogger(__name__)


def _extract_user_text(state: AgentState) -> str:
    """Extract the most recent user message text from state.messages."""
    for msg in reversed(state.messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multi-modal message — extract text part(s)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def make_tools(state: AgentState) -> list[Any]:
    """
    Build per-turn tool list with auth/turn context bound via closures.

    Filters by state.control.allowed_agents. Tools the user lacks permission
    for are omitted (not just disabled — completely absent from LLM's view).

    Args:
        state: Current AgentState (Pydantic). Tool factories pull
               user_context, image, control.agent_configs, prompts, rnd, etc.

    Returns:
        list of @tool-decorated async functions, ready for llm.bind_tools().
    """
    user_ctx_dump = state.user_context.model_dump() if state.user_context else {}
    user_obj = (user_ctx_dump.get("user") or {}) if user_ctx_dump else {}
    user_id = user_obj.get("id")

    image_url = state.image.image_url if state.image else None
    clarification_selected = state.image.clarification_selected if state.image else None

    chat_message_id = state.session.chat_message_id
    trace_id = state.session.trace_id

    user_text = _extract_user_text(state)

    agent_configs = state.control.agent_configs or {}
    prompts = state.prompts or {}

    rnd_llm_provider = state.rnd.llm_provider
    rnd_llm_model = state.rnd.llm_model
    rnd_emb_provider = state.rnd.embedding_provider
    rnd_emb_model = state.rnd.embedding_model

    allowed = set(state.control.allowed_agents or [])

    tools: list[Any] = []

    # ── Knowledge / FAQ tools (gated by kb_dental / app_faq) ─────────────────
    if "kb_dental" in allowed:
        tools.append(make_search_dental_knowledge_tool(
            embedding_provider_override=rnd_emb_provider,
            embedding_model_override=rnd_emb_model,
        ))

    if "app_faq" in allowed:
        tools.append(make_search_app_faq_tool(
            embedding_provider_override=rnd_emb_provider,
            embedding_model_override=rnd_emb_model,
        ))

    # ── User profile tool (gated by user_profile) ────────────────────────────
    if "user_profile" in allowed:
        tools.append(make_get_user_profile_tool(user_context=user_ctx_dump))

    # ── Brushing / Rapot tool (gated by rapot_peri) ──────────────────────────
    if "rapot_peri" in allowed:
        tools.append(make_get_brushing_stats_tool(
            user_id=user_id,
            user_context=user_ctx_dump,
        ))

    # ── Cerita Peri progress tool (gated by cerita_peri) ─────────────────────
    if "cerita_peri" in allowed:
        tools.append(make_get_cerita_progress_tool(user_id=user_id))

    # ── Mata Peri tools (gated by mata_peri — both history + analyze) ────────
    if "mata_peri" in allowed:
        tools.append(make_get_scan_history_tool(
            user_id=user_id,
            user_context=user_ctx_dump,
        ))
        tools.append(make_analyze_chat_image_tool(
            user_id=user_id,
            image_url=image_url,
            chat_message_id=chat_message_id,
            trace_id=trace_id,
            user_text=user_text,
            clarification_selected=clarification_selected,
            agent_configs=agent_configs,
            prompts=prompts,
            rnd_llm_provider=rnd_llm_provider,
            rnd_llm_model=rnd_llm_model,
        ))

    # ── janji_peri (Phase 6 — placeholder) ───────────────────────────────────
    # if "janji_peri" in allowed:
    #     tools.append(...)

    logger.info(
        f"[tools.make_tools] built {len(tools)} tools for "
        f"allowed_agents={list(allowed)}"
    )
    return tools


# Re-export for convenience
__all__ = ["make_tools"]
