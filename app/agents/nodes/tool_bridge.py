"""
Tool Bridge — Phase 2 Step 2a (Bagian C: now uses registry pattern)

Purpose:
    Convert ToolMessage outputs from tools_node into legacy state fields
    yang dipakai oleh generate_node (agent_results, retrieved_docs, etc.).

Why a separate node:
    LangGraph add_messages reducer hanya append ToolMessage ke state.messages.
    Tapi generate.py butuh data ter-struktur di agent_results, image_analysis,
    retrieved_docs, dll. Tool_bridge translate ToolMessage → legacy state.

Bagian C refactor (architectural):
    Sebelum: hardcode _TOOL_TO_AGENT_KEY dict + if/elif per tool.
    Sesudah: lookup via tool registry. Setiap ToolSpec sudah declare
    bridge_handler-nya sendiri di tool file. Bridge cuma orchestrate.

    Benefit: tambah tool baru = update 1 file (tool's own file), tidak
    perlu sentuh tool_bridge.py lagi.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.agents.state import AgentState, ToolCallRecord
# Bagian C: import registry
from app.agents.tools.registry import get_tool_spec, BridgeContext

logger = logging.getLogger(__name__)


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

    Bagian C: Setiap tool's bridge logic di-define di file tool itu sendiri
    (via ToolSpec.bridge_handler). Node ini cuma:
    1. Find recent ToolMessages
    2. Lookup ToolSpec dari registry
    3. Call spec.bridge_handler(parsed, agent_results, ctx)
    4. Build state update dict
    """
    # FIX (Langfuse audit Bagian B2): Wrap tool_bridge with trace_node for visibility.
    from app.config.observability import trace_node

    # Find ToolMessages from this turn — they are the most recent ones
    # appended after the AIMessage with tool_calls.
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
        new_tool_call_records: list[ToolCallRecord] = []
        agents_selected_set: list[str] = list(state.agents_selected or [])

        # Bagian C: BridgeContext untuk capture non-agent_results state updates
        # (retrieved_docs, image_analysis, scan_session_id, clarification_data).
        # Initialize with current state values supaya tools yang tidak update
        # field tertentu tetap preserve existing value.
        ctx = BridgeContext(
            retrieved_docs=list(state.retrieved_docs or []),
            image_analysis=state.image_analysis,
            scan_session_id=state.scan_session_id,
            needs_clarification=state.needs_clarification,
            clarification_data=state.clarification_data,
            # Image-Failure-Guard: preserve existing flag (typically False at start
            # of turn, but defensive — agent could re-process state)
            image_analysis_failed=state.image_analysis_failed,
            image_analysis_fallback_text=state.image_analysis_fallback_text,
        )

        unknown_tools: list[str] = []  # collect for end-of-loop loud error

        for tm in tool_messages:
            tool_name = tm.name or ""
            result = _parse_tool_content(tm.content)

            # Bagian C: lookup ToolSpec dari registry
            spec = get_tool_spec(tool_name)
            if spec is None:
                # Loud signal — tool executed but tidak ada ToolSpec.
                # Ini bug — developer harus tambah ToolSpec di tool file.
                logger.error(
                    f"[tool_bridge] Tool '{tool_name}' executed tapi tidak terdaftar "
                    f"di registry. Tambah ToolSpec di tools/<file>.py untuk fix. "
                    f"Tool result akan di-DROP — LLM akan halusinasi tanpa data!"
                )
                unknown_tools.append(tool_name)
                continue

            # Bagian C v2: Layer 3 defense — detect "tool unavailable" result.
            # LLM kadang halusinasi panggil tool yang tidak di-bind (gated off via
            # allowed_agents). Tools_node return synthetic error msg untuk ini.
            # Kita detect, mark di ctx.unavailable_tools, dan SKIP populate
            # agent_results (supaya LLM tidak kira tool jalan dengan empty data).
            from app.agents.tools.registry import is_tool_unavailable_result

            if is_tool_unavailable_result(result):
                logger.warning(
                    f"[tool_bridge] LLM called unavailable tool '{tool_name}' — "
                    f"feature gated off via allowed_agents. Marking di "
                    f"ctx.unavailable_tools, NOT populating agent_results."
                )
                ctx.unavailable_tools.append(tool_name)

                # Audit record (still log it untuk diagnostic)
                new_tool_call_records.append(ToolCallRecord(
                    tool=tool_name,
                    agent=spec.agent_key,
                    input={},
                    result={
                        "has_data": False,
                        "error": "tool_unavailable",
                        "needs_clarification": False,
                    },
                ))
                continue  # SKIP bridge_handler — don't pollute agent_results

            # Audit record (normal path)
            new_tool_call_records.append(ToolCallRecord(
                tool=tool_name,
                agent=spec.agent_key,
                input={},  # ToolNode doesn't preserve input; agent_node trace has it
                result={
                    "has_data": result.get("has_data"),
                    "needs_clarification": result.get("needs_clarification"),
                    "error": result.get("error"),
                },
            ))

            # Update agents_selected (legacy field — generate.py uses for "agents_used" metadata)
            if spec.agent_key not in agents_selected_set:
                agents_selected_set.append(spec.agent_key)

            # Bagian C: delegate ke spec.bridge_handler
            try:
                spec.bridge_handler(result, agent_results, ctx)
            except Exception as e:
                logger.error(
                    f"[tool_bridge] Bridge handler for '{tool_name}' raised: {e}",
                    exc_info=True,
                )
                # Don't fail the entire turn — log and continue with empty result
                # Tool result effectively dropped, but at least main flow doesn't crash.

        # Build state update — only include fields that changed
        update: dict[str, Any] = {
            "agent_results": agent_results,
            "agents_selected": agents_selected_set,
        }
        if ctx.retrieved_docs != list(state.retrieved_docs or []):
            update["retrieved_docs"] = ctx.retrieved_docs
        if ctx.image_analysis is not None and ctx.image_analysis != state.image_analysis:
            update["image_analysis"] = ctx.image_analysis
        if ctx.scan_session_id and ctx.scan_session_id != state.scan_session_id:
            update["scan_session_id"] = ctx.scan_session_id
        if ctx.needs_clarification != state.needs_clarification:
            update["needs_clarification"] = ctx.needs_clarification
        if ctx.clarification_data is not None and ctx.clarification_data != state.clarification_data:
            update["clarification_data"] = ctx.clarification_data
        if new_tool_call_records:
            update["tool_calls"] = [*state.tool_calls, *new_tool_call_records]

        # Bagian C v2: Layer 3 — propagate unavailable_tools ke state supaya
        # generate.py bisa inject warning ke system prompt.
        if ctx.unavailable_tools:
            update["unavailable_tools"] = ctx.unavailable_tools

        # Image-Failure-Guard: propagate failure flag + fallback text supaya
        # generate.py bisa SKIP cached scan/brushing data dan FORCE fallback
        # answer. Critical untuk medical safety (lihat scan.py docstring
        # _bridge_analyze_chat_image).
        if ctx.image_analysis_failed:
            update["image_analysis_failed"] = True
            update["image_analysis_fallback_text"] = ctx.image_analysis_fallback_text

        logger.info(
            f"[tool_bridge] Bridged {len(tool_messages)} tool result(s) "
            f"→ agent_results keys: {list(agent_results.keys())}, "
            f"needs_clarification={ctx.needs_clarification}"
            + (f", UNKNOWN_TOOLS={unknown_tools}" if unknown_tools else "")
            + (f", UNAVAILABLE_TOOLS={ctx.unavailable_tools}" if ctx.unavailable_tools else "")
        )

        # Capture span output for diagnostic (B2 fix preserved)
        if span:
            span.update(output={
                "agent_results_keys": list(agent_results.keys()),
                "retrieved_docs_count": len(ctx.retrieved_docs),
                "needs_clarification": ctx.needs_clarification,
                "image_analysis_present": ctx.image_analysis is not None,
                "scan_session_id": ctx.scan_session_id,
                "tool_call_records_added": len(new_tool_call_records),
                "unknown_tools": unknown_tools,
                # Bagian C v2: surface unavailable tools untuk debug
                "unavailable_tools": ctx.unavailable_tools,
                # Image-Failure-Guard: surface image failure untuk Langfuse trace
                "image_analysis_failed": ctx.image_analysis_failed,
            })

        return update
