"""
Node: agent_dispatcher

Phase 1 — runs selected agents (parallel or sequential) based on execution_plan.mode.
Replaces `_run_agents_parallel` + `_run_agents_sequential` from old graph.py.

PHASE 1 SHIM: existing legacy agents (kb_dental_agent, rapot_peri_agent, etc)
masih pakai dict-mutation interface. Wrap Pydantic AgentState ke legacy dict
via .model_dump() + reverse-merge update ke partial state update dict yang
di-return.

Phase 2 nanti: agents jadi @tool dengan signature `async def tool(...) -> dict`,
ToolNode (LangGraph built-in) handle execution. Shim ini akan dihapus.

Heartbeat events (saat node lama running) di-handle di stream_adapter
(app/agents/streaming.py) — bukan di node ini.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.state import AgentState, ToolCallRecord
from app.agents.sub_agents import (
    kb_dental_agent,
    user_profile_agent,
    app_faq_agent,
)
from app.agents.sub_agents.phase2_agents import (
    rapot_peri_agent,
    cerita_peri_agent,
    mata_peri_agent,
)

logger = logging.getLogger(__name__)


# Registry agent_key → async function
# janji_peri commented out — disabled (is_globally_active=false di DB).
# Phase 4 akan re-enable dengan tool-based implementation.
_AGENT_REGISTRY = {
    "kb_dental": kb_dental_agent,
    "user_profile": user_profile_agent,
    "app_faq": app_faq_agent,
    "rapot_peri": rapot_peri_agent,
    "cerita_peri": cerita_peri_agent,
    "mata_peri": mata_peri_agent,
    # "janji_peri": janji_peri_agent,  # Phase 4
}


def _build_legacy_dict_state(state: AgentState) -> dict:
    """
    Build legacy dict-format state untuk backward compat dengan existing agents.

    Existing agents (sub_agents/__init__.py + phase2_agents.py) akses state via
    state["key"] dan mutate via state["key"] = value. Mereka pakai 60+ flat keys.

    Pattern: build comprehensive dict yang cover semua keys yang agents butuh.
    Agents akan mutate dict ini in-place; kita extract perubahan setelahnya.
    """
    # Convert messages ke legacy dict format
    #
    # PHASE 2A FIX: Skip empty AIMessages dari agent_node.
    # Di Phase 2, agent_node return AIMessage(content="", tool_calls=[...])
    # untuk tool selection (atau AIMessage("") kalau smalltalk no tools).
    # LangGraph add_messages reducer APPEND ini ke state.messages.
    # Kalau kita keep empty AIMessage di history → generate.py LLM lihat
    # "assistant already responded with nothing" → respond empty (0 token).
    # Filter rules:
    #   - HumanMessage: ALWAYS keep (even empty content — image-only upload)
    #   - AIMessage with non-empty content: keep (real assistant turn)
    #   - AIMessage with empty content: SKIP (Phase 2 agent_node artifact)
    #   - AIMessage with tool_calls only (no content): SKIP (internal routing)
    #   - ToolMessage: keep with role "tool" (generate._build_messages
    #     handles by skipping non-user/assistant roles)
    msgs_legacy = []
    for m in state.messages:
        if not hasattr(m, "type"):
            continue
        msg_type = m.type
        content = m.content if m.content is not None else ""

        if msg_type == "ai":
            # Skip empty AIMessages (Phase 2 agent_node artifacts)
            has_content = bool(content and str(content).strip())
            has_tool_calls = bool(getattr(m, "tool_calls", None))
            if not has_content:
                # No content → skip (regardless of tool_calls).
                # tool_calls audit captured separately via state.tool_calls (ToolCallRecord).
                continue
            msgs_legacy.append({"role": "assistant", "content": content})
        elif msg_type == "human":
            msgs_legacy.append({"role": "user", "content": content})
        elif msg_type == "tool":
            # Tool result — generate.py doesn't use these directly (uses agent_results).
            # Keep with role "tool" so future generate refactor can read if needed.
            msgs_legacy.append({"role": "tool", "content": content})
        else:
            # System/other — pass through with original type as role
            msgs_legacy.append({"role": msg_type, "content": content})

    return {
        # Static context
        "session_id": state.session.session_id,
        "user_context": state.user_context.model_dump(),
        "messages": msgs_legacy,
        "prompts": state.prompts,
        "image_url": state.image.image_url if state.image else None,
        "image_url_public": state.image.image_url_public if state.image else None,
        "clarification_selected": state.image.clarification_selected if state.image else None,
        "quick_reply_option_id": state.image.quick_reply_option_id if state.image else None,
        "chat_message_id": state.session.chat_message_id,
        "trace_id": state.session.trace_id,
        "source": state.session.source,
        # Control
        "allowed_agents": state.control.allowed_agents,
        "agent_configs": state.control.agent_configs,
        "response_mode": state.session.response_mode,
        "memory_context": state.memory.model_dump(),
        # Routing decisions
        "agents_selected": state.agents_selected,
        "execution_plan": state.execution_plan,
        # Agent results (input untuk berikutnya kalau sequential)
        "agent_results": dict(state.agent_results),
        "retrieved_docs": list(state.retrieved_docs),
        "image_analysis": state.image_analysis,
        "scan_session_id": state.scan_session_id,
        # Clarification/quick reply
        "needs_clarification": state.needs_clarification,
        "clarification_data": state.clarification_data,
        "quick_reply_data": state.quick_reply_data,
        "suggestion_chips": state.suggestion_chips,
        # Audit
        "thinking_steps": [t.model_dump() for t in state.thinking_steps],
        "tool_calls": [],  # accumulated by agents — extract after
        "llm_call_logs": [l.model_dump() for l in state.llm_call_logs],
        "final_response": state.final_response,
        "llm_metadata": dict(state.llm_metadata),
        # RnD overrides
        "llm_provider_override": state.rnd.llm_provider,
        "llm_model_override": state.rnd.llm_model,
        "llm_temperature_override": state.rnd.llm_temperature,
        "llm_max_tokens_override": state.rnd.llm_max_tokens,
        "embedding_provider_override": state.rnd.embedding_provider,
        "embedding_model_override": state.rnd.embedding_model,
        "top_k_docs": state.rnd.top_k_docs,
        "force_intent": state.rnd.force_intent,
        "include_prompt_debug": state.rnd.include_prompt_debug,
        "prompt_debug": state.prompt_debug,
    }


def _extract_state_update_from_legacy(legacy_state: dict, base_state: AgentState) -> dict:
    """
    Extract state update dict dari mutated legacy_state.

    Legacy agents mutate beberapa fields:
    - tool_calls (appended)
    - retrieved_docs (set by kb_dental_agent)
    - image_analysis, scan_session_id (set by mata_peri_agent)
    - needs_clarification, clarification_data (set by mata_peri_agent saat ambiguous)
    - quick_reply_data (set by some agents — preserve kalau di-set)
    """
    update = {}

    # tool_calls — append new ones
    new_tool_calls = legacy_state.get("tool_calls", [])
    if new_tool_calls:
        update["tool_calls"] = [
            *base_state.tool_calls,
            *[ToolCallRecord(**tc) for tc in new_tool_calls],
        ]

    # retrieved_docs — kb_dental_agent set this
    new_docs = legacy_state.get("retrieved_docs", [])
    if new_docs and new_docs != base_state.retrieved_docs:
        update["retrieved_docs"] = new_docs

    # image_analysis — mata_peri_agent (Mode A success)
    new_image_analysis = legacy_state.get("image_analysis")
    if new_image_analysis is not None and new_image_analysis != base_state.image_analysis:
        update["image_analysis"] = new_image_analysis

    # scan_session_id — mata_peri_agent (Mode A success)
    new_scan_id = legacy_state.get("scan_session_id")
    if new_scan_id is not None and new_scan_id != base_state.scan_session_id:
        update["scan_session_id"] = new_scan_id

    # Clarification — mata_peri_agent (view_hint ambiguous)
    if legacy_state.get("needs_clarification") and not base_state.needs_clarification:
        update["needs_clarification"] = True
        update["clarification_data"] = legacy_state.get("clarification_data")

    # Quick reply (defensive — preserve kalau di-set agent)
    if legacy_state.get("quick_reply_data") and not base_state.quick_reply_data:
        update["quick_reply_data"] = legacy_state["quick_reply_data"]

    return update


async def _run_parallel(state: AgentState, keys: list[str]) -> dict:
    """Run agents in parallel via asyncio.gather."""
    legacy_state = _build_legacy_dict_state(state)

    tasks = []
    valid_keys = []
    for key in keys:
        fn = _AGENT_REGISTRY.get(key)
        if fn:
            tasks.append(fn(legacy_state))
            valid_keys.append(key)
        else:
            logger.warning(f"[agent_dispatcher] Agent '{key}' tidak ada di registry — skip")

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks, return_exceptions=True)

    agent_results = dict(state.agent_results)
    for key, res in zip(valid_keys, results):
        if isinstance(res, Exception):
            logger.error(f"[agent_dispatcher] Agent '{key}' failed: {res}", exc_info=res)
            agent_results[key] = {"error": str(res)}
        else:
            agent_results[key] = res

    update = _extract_state_update_from_legacy(legacy_state, state)
    update["agent_results"] = agent_results
    return update


async def _run_sequential(state: AgentState, keys: list[str]) -> dict:
    """Run agents sequentially. Each agent sees mutations from previous one."""
    legacy_state = _build_legacy_dict_state(state)

    agent_results = dict(state.agent_results)
    for key in keys:
        fn = _AGENT_REGISTRY.get(key)
        if not fn:
            logger.warning(f"[agent_dispatcher] Agent '{key}' tidak ada di registry — skip")
            continue
        try:
            res = await fn(legacy_state)
            agent_results[key] = res
            # Update legacy_state.agent_results untuk agent berikutnya bisa baca
            legacy_state["agent_results"][key] = res
        except Exception as e:
            logger.error(f"[agent_dispatcher] Agent '{key}' failed: {e}", exc_info=True)
            agent_results[key] = {"error": str(e)}

    update = _extract_state_update_from_legacy(legacy_state, state)
    update["agent_results"] = agent_results
    return update


# =============================================================================
# LangGraph node
# =============================================================================


async def agent_dispatcher_node(state: AgentState) -> dict:
    """
    LangGraph node: run agents according to execution_plan.mode.

    No-op kalau agents_selected kosong (smalltalk / clarification answer case).
    Returns: state update dict.
    """
    agents_selected = state.agents_selected
    if not agents_selected:
        return {}  # no-op

    mode = state.execution_plan.get("mode", "sequential")

    if mode == "parallel" and len(agents_selected) > 1:
        return await _run_parallel(state, agents_selected)
    else:
        return await _run_sequential(state, agents_selected)
