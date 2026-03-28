"""
Peri Agent — LangGraph graph utama Tanya Peri.

Graph flow:
    START → router → [branch by intent] → [optional tools] → generate → END

Branch:
    dental_qa           → retrieve → check_clarification → generate
    context_query       → check_clarification → generate
    image               → image_analysis → generate
    clarification_answer→ generate
    smalltalk           → generate
"""
from typing import AsyncIterator

from app.agents.nodes.generate import check_clarification_node, generate_node
from app.agents.nodes.router import router_node
from app.agents.tools.image import image_node
from app.agents.tools.retrieve import retrieve_node
from app.schemas.chat import (
    AgentState,
    make_done_event,
    make_error_event,
)


async def run_agent(initial_state: AgentState) -> AsyncIterator[str]:
    """
    Entry point utama — jalankan seluruh graph dan yield SSE events.

    Ini adalah simplified graph runner yang mengeksekusi node secara sequential
    berdasarkan intent. LangGraph StateGraph dipakai untuk state management,
    tapi streaming events dikirim langsung dari sini agar lebih kontrol.
    """
    state = initial_state

    # Init state fields yang mungkin belum ada
    state.setdefault("thinking_steps", [])
    state.setdefault("tool_calls", [])
    state.setdefault("retrieved_docs", [])
    state.setdefault("image_analysis", None)
    state.setdefault("needs_clarification", False)
    state.setdefault("clarification_data", None)
    state.setdefault("final_response", "")
    state.setdefault("llm_metadata", {})
    state.setdefault("intent", "")

    try:
        # ── Node 1: Router ────────────────────────────────────────────────
        async for event in router_node(state):
            yield event

        intent = state.get("intent", "dental_qa")

        # ── Node 2: Tool nodes berdasarkan intent ─────────────────────────
        if intent == "dental_qa":
            async for event in retrieve_node(state):
                yield event
            async for event in check_clarification_node(state):
                yield event

        elif intent == "context_query":
            async for event in check_clarification_node(state):
                yield event

        elif intent == "image":
            async for event in image_node(state):
                yield event

        # clarification_answer dan smalltalk → langsung ke generate

        # ── Node 3: Generate ──────────────────────────────────────────────
        async for event in generate_node(state):
            yield event

        # ── Done ──────────────────────────────────────────────────────────
        # Kalau ada clarification, final_response kosong — tidak emit done
        if state.get("needs_clarification") and not state.get("final_response"):
            return

        yield make_done_event(
            content=state["final_response"],
            metadata=state.get("llm_metadata", {}),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield make_error_event(f"Maaf, Tanya Peri sedang mengalami gangguan: {str(e)}")


def _build_initial_state(request) -> AgentState:
    """
    Build AgentState dari ChatRequest.
    Dipanggil oleh endpoint sebelum run_agent.
    """
    return AgentState(
        session_id=request.session_id,
        user_context=request.user_context,
        messages=request.messages,
        prompts=request.prompts,
        image_url=request.image_url,
        clarification_selected=request.clarification_selected,
        intent="",
        retrieved_docs=[],
        image_analysis=None,
        needs_clarification=False,
        clarification_data=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
    )
