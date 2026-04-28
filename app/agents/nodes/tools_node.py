"""
Tools node — Phase 2 Step 2a (custom replacement for langgraph.prebuilt.ToolNode)

Why custom (not prebuilt ToolNode):
    Our tools are built per-turn via make_tools(state) factory. The factory
    captures user_id, image_url, agent_configs, etc. into closures.
    LangGraph prebuilt ToolNode requires a FIXED tool list at graph build time,
    incompatible with per-turn closures.

This node:
    1. Reads the most recent AIMessage from state.messages
    2. For each tool_call in that AIMessage, finds the matching tool (per-turn)
    3. Invokes the tool's coroutine with tool_call args
    4. Wraps result as ToolMessage and appends to state.messages
    5. Runs all tool calls in PARALLEL via asyncio.gather

Error handling:
    - Tool not found → ToolMessage with error content (LLM/generate sees the error)
    - Tool raises exception → ToolMessage with error content (no graph crash)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import AgentState
from app.agents.tools import make_tools

logger = logging.getLogger(__name__)


def _find_last_ai_message(messages: list) -> AIMessage | None:
    """Walk messages from end, return first AIMessage encountered."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _serialize_tool_result(result: Any) -> str:
    """Convert tool result (typically dict) to string for ToolMessage.content."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"[tools_node] Could not JSON-serialize tool result: {e}")
        return str(result)


async def _invoke_tool(tool_obj, args: dict, tool_call_id: str, tool_name: str) -> ToolMessage:
    """Invoke single tool, wrap result/error as ToolMessage."""
    try:
        # All our tools are async (decorated with @tool, defined with `async def`)
        # langchain @tool decorator exposes both .invoke() and .ainvoke().
        result = await tool_obj.ainvoke(args)
        content = _serialize_tool_result(result)
        return ToolMessage(
            content=content,
            name=tool_name,
            tool_call_id=tool_call_id,
        )
    except Exception as e:
        logger.error(
            f"[tools_node] Tool '{tool_name}' raised exception: {e}",
            exc_info=True,
        )
        error_payload = {
            "error": f"Tool execution failed: {str(e)[:200]}",
            "has_data": False,
        }
        return ToolMessage(
            content=json.dumps(error_payload),
            name=tool_name,
            tool_call_id=tool_call_id,
        )


async def tools_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: execute pending tool_calls from the last AIMessage in parallel.

    Returns partial state update:
    - messages: list of new ToolMessage (LangGraph's add_messages reducer
      will append, not replace)
    """
    last_ai = _find_last_ai_message(state.messages)
    if last_ai is None or not last_ai.tool_calls:
        logger.debug("[tools_node] No tool_calls to execute — pass through.")
        return {}

    # Build per-turn tool list and index by name
    tools_list = make_tools(state)
    tools_by_name = {getattr(t, "name", "") or "": t for t in tools_list}

    # Prepare invocation tasks
    tasks = []
    for tc in last_ai.tool_calls:
        tool_name = tc.get("name") or ""
        tool_call_id = tc.get("id") or f"call_{tool_name}"
        args = tc.get("args") or {}

        tool_obj = tools_by_name.get(tool_name)
        if tool_obj is None:
            logger.warning(
                f"[tools_node] LLM called unknown tool: {tool_name!r}. "
                f"Available: {list(tools_by_name.keys())}"
            )
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # placeholder

            # We'll create a synthetic ToolMessage for unknown-tool error below.
            # For now, create a "future" wrapper that returns the error message.
            async def _missing(name=tool_name, tcid=tool_call_id):
                return ToolMessage(
                    content=json.dumps({
                        "error": f"Tool '{name}' not available for this user",
                        "has_data": False,
                    }),
                    name=name,
                    tool_call_id=tcid,
                )
            # Replace placeholder with the real coroutine
            tasks[-1].cancel()
            tasks[-1] = asyncio.create_task(_missing())
            continue

        tasks.append(asyncio.create_task(
            _invoke_tool(tool_obj, args, tool_call_id, tool_name)
        ))

    # Run all tools in parallel
    results = await asyncio.gather(*tasks, return_exceptions=False)

    tool_messages = [r for r in results if isinstance(r, ToolMessage)]

    logger.info(
        f"[tools_node] Executed {len(tool_messages)} tool(s): "
        f"{[tm.name for tm in tool_messages]}"
    )

    return {"messages": tool_messages}
