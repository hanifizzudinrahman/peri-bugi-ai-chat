"""
StateGraph builder untuk Tanya Peri (Phase 1 — Hybrid).

PHASE 1 GRAPH STRUCTURE (post-fix):

    START → supervisor → agent_dispatcher → END

generate_node TIDAK masuk graph — di-invoke langsung sebagai async generator
oleh streaming.py setelah graph selesai. Alasan: LangGraph stream API tidak
reliable emit per-token chunk untuk Gemini provider via langchain_google_genai.
Dengan keep generate sebagai async generator, kita preserve:
- Per-token streaming (FE terima word-by-word) ✅
- Pydantic AgentState ✅ (state shared via dict-compat shim)
- Checkpointer untuk supervisor + dispatcher state ✅

Phase 2 nanti akan refactor ulang setelah ReAct loop in place — pakai LangGraph
ToolNode + dedicated streaming dengan custom stream writer pattern.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.nodes.agent_dispatcher import agent_dispatcher_node

logger = logging.getLogger(__name__)


# Singleton compiled graph
_compiled_graph = None


async def get_compiled_graph():
    """
    Return compiled StateGraph singleton.

    PHASE 1 graph: START → supervisor → agent_dispatcher → END
    (generate node di-invoke terpisah oleh streaming.py untuk preserve
    per-token streaming.)
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

    # Edges (linear flow — generate handled separately)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "agent_dispatcher")
    builder.add_edge("agent_dispatcher", END)

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
    """For debugging — output graph as Mermaid diagram."""
    graph = await get_compiled_graph()
    return graph.get_graph().draw_mermaid()


async def reset_compiled_graph():
    """For testing — force re-compile on next get_compiled_graph() call."""
    global _compiled_graph
    _compiled_graph = None
