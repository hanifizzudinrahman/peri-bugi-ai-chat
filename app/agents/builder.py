"""
StateGraph builder untuk Tanya Peri (Phase 1).

Phase 1 graph structure (preserves existing flow 1:1):

    START → supervisor → agent_dispatcher → generate → END

Flow ini IDENTIK dengan custom orchestrator sebelumnya — purposely simple
supaya migration ke real LangGraph tidak ubah behavior.

Phase 2 akan tambah:
- ReAct loop (agent ↔ tools cycle)
- Conditional edges berbasis tool_calls
- ToolNode (LangGraph built-in)
- Pre-router untuk forced image flow
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.nodes.agent_dispatcher import agent_dispatcher_node
from app.agents.nodes.generate import generate_node

logger = logging.getLogger(__name__)


# Singleton compiled graph
_compiled_graph = None


async def get_compiled_graph():
    """
    Return compiled StateGraph singleton.

    First call: build + compile dengan checkpointer (kalau available).
    Subsequent calls: return cached instance.

    Behavior:
    - Checkpointer available → compile dengan persistence
    - Checkpointer unavailable → compile in-memory (graceful degradation)
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    from app.agents.memory.checkpointer import get_checkpointer
    cp = await get_checkpointer()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("agent_dispatcher", agent_dispatcher_node)
    builder.add_node("generate", generate_node)

    # Edges (linear flow)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "agent_dispatcher")
    builder.add_edge("agent_dispatcher", "generate")
    builder.add_edge("generate", END)

    if cp is not None:
        _compiled_graph = builder.compile(checkpointer=cp)
        logger.info("[builder] StateGraph compiled WITH PostgresSaver checkpointer")
    else:
        _compiled_graph = builder.compile()
        logger.warning(
            "[builder] StateGraph compiled WITHOUT checkpointer (in-memory only). "
            "Per-session state tidak akan persist across restart."
        )

    return _compiled_graph


async def render_graph_mermaid() -> str:
    """For debugging — output graph as Mermaid diagram.

    Run via:
        docker compose exec ai-chat python -c "
        import asyncio
        from app.agents.builder import render_graph_mermaid
        print(asyncio.run(render_graph_mermaid()))
        "
    """
    graph = await get_compiled_graph()
    return graph.get_graph().draw_mermaid()


async def reset_compiled_graph():
    """For testing — force re-compile on next get_compiled_graph() call."""
    global _compiled_graph
    _compiled_graph = None
