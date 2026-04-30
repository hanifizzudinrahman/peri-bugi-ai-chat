"""
Tool bridge node — Phase 2 Step 2a

Runs AFTER tools_node (LangGraph prebuilt ToolNode). Maps tool outputs
(stored as ToolMessage in state.messages) to LEGACY state fields that
generate_node still reads:

    state.agent_results       (keyed by legacy agent_key, not tool name)
    state.retrieved_docs       (kb_dental docs, separate field)
    state.image_analysis       (mata_peri scan result)
    state.scan_session_id      (mata_peri scan session id)
    state.needs_clarification  (mata_peri ambiguous view_hint)
    state.clarification_data   (ClarificationCard payload)

This bridge keeps generate_node 100% unchanged in Step 2a. Step 6 (refactor
generate.py) will eliminate this bridge by making generate read from
state.messages directly.

TOOL → LEGACY KEY MAPPING:
    search_dental_knowledge → agent_results["kb_dental"] + retrieved_docs
    search_app_faq          → agent_results["app_faq"]
    get_brushing_stats      → agent_results["rapot_peri"]
    get_cerita_progress     → agent_results["cerita_peri"]
    get_scan_history        → agent_results["mata_peri"]
    analyze_chat_image      → agent_results["mata_peri"] + image_analysis +
                              scan_session_id + (clarification fields if needed)
    get_user_profile        → agent_results["user_profile"]
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.agents.state import AgentState, ToolCallRecord

logger = logging.getLogger(__name__)


# Tool name → legacy agent_key mapping
_TOOL_TO_AGENT_KEY = {
    "search_dental_knowledge": "kb_dental",
    "search_app_faq": "app_faq",
    "get_brushing_stats": "rapot_peri",
    "get_cerita_progress": "cerita_peri",
    "get_scan_history": "mata_peri",
    "analyze_chat_image": "mata_peri",
    "get_user_profile": "user_profile",
}


def _parse_tool_content(content: Any) -> dict:
    """
    Tool returns dict, but ToolNode wraps as ToolMessage.content which is
    typically str (JSON-serialized). Parse defensively.
    """
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": content}
    return {"_raw": str(content)}


async def tool_bridge_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: bridge tool outputs → legacy state fields.

    Reads recent ToolMessage entries from state.messages (those added by
    tools_node), converts them to legacy state fields, returns partial
    state update.
    """
    # Find ToolMessages from this turn — they are the most recent ones
    # appended after the AIMessage with tool_calls.
    # We walk state.messages from the end until we hit the AIMessage.
    tool_messages: list[ToolMessage] = []
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            tool_messages.append(msg)
        else:
            # Stop at the first non-ToolMessage (typically the AIMessage that
            # triggered the tool calls, or older history)
            break
    tool_messages.reverse()  # restore chronological order

    if not tool_messages:
        # No tools were executed (LLM decided no tools, or smalltalk).
        # Pass through with no state changes.
        logger.debug("[tool_bridge] No ToolMessages found — pass through.")
        return {}

    # FIX (Langfuse audit Bagian B2): Wrap tool_bridge with trace_node for visibility.
    # Sebelumnya: bridging logic INVISIBLE — sulit debug "kenapa agent_results gak terisi?"
    # Sekarang: span "node:tool_bridge" muncul + capture mapping decisions.
    from app.config.observability import trace_node

    legacy_state_for_trace = {
        "session_id": state.session.session_id,
        "user_context": state.user_context.model_dump() if state.user_context else {},
        "response_mode": state.session.response_mode,
    }

    async with trace_node(
        name="node:tool_bridge",
        state=legacy_state_for_trace,
        input_data={
            "tool_messages_count": len(tool_messages),
            "tool_names": [tm.name for tm in tool_messages],
        },
    ) as span:
        # Build state update incrementally
        agent_results: dict[str, Any] = dict(state.agent_results or {})
        retrieved_docs: list = list(state.retrieved_docs or [])
        image_analysis = state.image_analysis
        scan_session_id = state.scan_session_id
        needs_clarification = state.needs_clarification
        clarification_data = state.clarification_data
        new_tool_call_records: list[ToolCallRecord] = []
        agents_selected_set: list[str] = list(state.agents_selected or [])

        for tm in tool_messages:
            tool_name = tm.name or ""
            result = _parse_tool_content(tm.content)
            agent_key = _TOOL_TO_AGENT_KEY.get(tool_name)

            if not agent_key:
                logger.warning(f"[tool_bridge] Unknown tool name: {tool_name!r} — skip")
                continue

            # Audit record
            new_tool_call_records.append(ToolCallRecord(
                tool=tool_name,
                agent=agent_key,
                input={},  # ToolNode doesn't preserve input; agent_node trace has it
                result={
                    "has_data": result.get("has_data"),
                    "needs_clarification": result.get("needs_clarification"),
                    "error": result.get("error"),
                },
            ))

            # Update agents_selected (legacy field — generate.py uses for "agents_used" metadata)
            if agent_key not in agents_selected_set:
                agents_selected_set.append(agent_key)

            # Tool-specific bridging
            if tool_name == "search_dental_knowledge":
                docs = result.get("docs", []) or []
                agent_results["kb_dental"] = {
                    "docs": docs,
                    "source_count": result.get("source_count", len(docs)),
                }
                if docs:
                    retrieved_docs = docs

            elif tool_name == "search_app_faq":
                agent_results["app_faq"] = {
                    "docs": result.get("docs", []),
                    "source_count": result.get("source_count", 0),
                }

            elif tool_name == "get_brushing_stats":
                agent_results["rapot_peri"] = result

            elif tool_name == "get_cerita_progress":
                agent_results["cerita_peri"] = result

            elif tool_name == "get_scan_history":
                agent_results["mata_peri"] = result

            elif tool_name == "analyze_chat_image":
                # Special — clarification flow OR success OR failure
                if result.get("needs_clarification"):
                    needs_clarification = True
                    clarification_data = result.get("clarification_data")
                    # Also record in agent_results so generate has visibility
                    agent_results["mata_peri"] = result
                    logger.info(
                        "[tool_bridge] analyze_chat_image needs_clarification — "
                        "set state.needs_clarification=True"
                    )
                elif result.get("has_data"):
                    # Success — extract image_analysis + scan_session_id
                    image_analysis = result.get("image_analysis")
                    scan_session_id = result.get("scan_session_id")
                    agent_results["mata_peri"] = result
                else:
                    # Failure (mode=new_scan_failed) — preserve fallback_text in agent_results
                    agent_results["mata_peri"] = result

            elif tool_name == "get_user_profile":
                agent_results["user_profile"] = result

        # Build state update — only include fields that changed
        update: dict[str, Any] = {
            "agent_results": agent_results,
            "agents_selected": agents_selected_set,
        }
        if retrieved_docs != list(state.retrieved_docs or []):
            update["retrieved_docs"] = retrieved_docs
        if image_analysis is not None and image_analysis != state.image_analysis:
            update["image_analysis"] = image_analysis
        if scan_session_id and scan_session_id != state.scan_session_id:
            update["scan_session_id"] = scan_session_id
        if needs_clarification != state.needs_clarification:
            update["needs_clarification"] = needs_clarification
        if clarification_data is not None and clarification_data != state.clarification_data:
            update["clarification_data"] = clarification_data
        if new_tool_call_records:
            update["tool_calls"] = [*state.tool_calls, *new_tool_call_records]

        logger.info(
            f"[tool_bridge] Bridged {len(tool_messages)} tool result(s) "
            f"→ agent_results keys: {list(agent_results.keys())}, "
            f"needs_clarification={needs_clarification}"
        )
        # FIX (Langfuse audit Bagian B2): Capture bridging decisions for diagnostic.
        if span:
            span.update(output={
                "agent_results_keys": list(agent_results.keys()),
                "retrieved_docs_count": len(retrieved_docs),
                "needs_clarification": needs_clarification,
                "image_analysis_present": image_analysis is not None,
                "scan_session_id": scan_session_id,
                "tool_call_records_added": len(new_tool_call_records),
            })

        return update
