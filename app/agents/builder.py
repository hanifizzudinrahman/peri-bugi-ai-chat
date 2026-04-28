"""
StateGraph builder untuk Tanya Peri (Phase 2 Step 2a — Single-Pass Tools).

PHASE 2 GRAPH STRUCTURE:

    START → pre_router → agent → tools → tool_bridge → END

Flow:
1. pre_router: detect image / clarification → maybe set state.forced_tool_calls
2. agent_node: LLM with bind_tools, single invocation
   - If forced_tool_calls set → emit AIMessage with those tool_calls (no LLM)
   - Else → LLM picks tools (or none for smalltalk)
3. tools_node: execute tool_calls in parallel (custom node, not prebuilt ToolNode)
4. tool_bridge: convert ToolMessages → legacy state fields (agent_results,
   retrieved_docs, image_analysis, needs_clarification, ...)
5. END

generate_node (per-token streaming) di-invoke OUTSIDE graph oleh streaming.py
sebagai async generator. Hybrid pattern dipertahankan dari Phase 1.

If agent_node decides NO tools (smalltalk), tools_node + tool_bridge are still
visited but become no-ops. State passes through unchanged. generate_node will
respond from system prompt + history only.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes.pre_router import pre_router_node
from app.agents.nodes.agent import agent_node
from app.agents.nodes.tools_node import tools_node
from app.agents.nodes.tool_bridge import tool_bridge_node

logger = logging.getLogger(__name__)


# Singleton compiled graph
_compiled_graph = None


async def get_compiled_graph():
    """
    Return compiled StateGraph singleton.

    PHASE 2 graph: START → pre_router → agent → tools → tool_bridge → END
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    from app.agents.memory.checkpointer import get_checkpointer
    cp = await get_checkpointer()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("pre_router", pre_router_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("tool_bridge", tool_bridge_node)

    # Edges (linear single-pass — no loop back to agent)
    builder.add_edge(START, "pre_router")
    builder.add_edge("pre_router", "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", "tool_bridge")
    builder.add_edge("tool_bridge", END)

    if cp is not None:
        _compiled_graph = builder.compile(checkpointer=cp)
        logger.info(
            "[builder] StateGraph compiled (Phase 2 ReAct) WITH PostgresSaver checkpointer"
        )
    else:
        _compiled_graph = builder.compile()
        logger.warning(
            "[builder] StateGraph compiled WITHOUT checkpointer (in-memory only). "
            "Per-session state tidak akan persist across restart."
        )

    return _compiled_graph


async def render_graph_mermaid() -> str:
    """For debugging — output graph as Mermaid diagram."""
    graph = await get_compiled_graph()
    return graph.get_graph().draw_mermaid()


async def reset_compiled_graph():
    """For testing — force re-compile on next get_compiled_graph() call."""
    global _compiled_graph
    _compiled_graph = None
