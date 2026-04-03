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
    ChatRequest,
    RnDChatRequest,
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
    state.setdefault("llm_call_logs", [])
    state.setdefault("intent", "")
    state.setdefault("top_k_docs", 3)
    state.setdefault("force_intent", None)
    state.setdefault("llm_provider_override", None)
    state.setdefault("llm_model_override", None)
    state.setdefault("llm_temperature_override", None)
    state.setdefault("llm_max_tokens_override", None)
    state.setdefault("embedding_provider_override", None)
    state.setdefault("embedding_model_override", None)
    state.setdefault("include_prompt_debug", False)
    state.setdefault("prompt_debug", None)

    try:
        # ── Node 1: Router ────────────────────────────────────────────────
        async for event in router_node(state):
            yield event

        intent = state.get("intent", "dental_qa")

        # ── Node 2: Tool nodes berdasarkan intent ─────────────────────────
        if intent == "dental_qa":
            async for event in retrieve_node(state):
                yield event
            await check_clarification_node(state)

        elif intent == "context_query":
            await check_clarification_node(state)

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

        # Kumpulkan semua llm_call_logs untuk dikirim ke api
        llm_call_logs = state.get("llm_call_logs", [])
        # Inject session_id ke setiap log
        session_id = state.get("session_id", "")
        if session_id:
            for log in llm_call_logs:
                if not log.get("session_id"):
                    log["session_id"] = session_id

        yield make_done_event(
            content=state["final_response"],
            metadata={
                **state.get("llm_metadata", {}),
                "llm_call_logs": llm_call_logs,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield make_error_event(f"Maaf, Tanya Peri sedang mengalami gangguan: {str(e)}")


def _build_initial_state(request: ChatRequest) -> AgentState:
    """
    Build AgentState dari ChatRequest (production/integration mode).
    Dipanggil oleh endpoint /chat/stream sebelum run_agent.
    """
    return AgentState(
        session_id=request.session_id,
        user_context=request.user_context,
        messages=request.messages,
        prompts=request.prompts,
        image_url=request.image_url,
        clarification_selected=request.clarification_selected,
        # Fields dengan nilai default (tidak ada di ChatRequest)
        llm_provider_override=None,
        llm_model_override=None,
        llm_temperature_override=None,
        llm_max_tokens_override=None,
        embedding_provider_override=None,
        embedding_model_override=None,
        top_k_docs=3,
        force_intent=None,
        include_prompt_debug=False,
        # Output fields
        intent="",
        retrieved_docs=[],
        image_analysis=None,
        needs_clarification=False,
        clarification_data=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )


def _build_rnd_state(request: RnDChatRequest) -> AgentState:
    """
    Build AgentState dari RnDChatRequest (RnD/standalone mode).
    Dipanggil oleh endpoint /chat/rnd sebelum run_agent.
    """
    # Build messages: gabung history + pesan baru
    messages = list(request.conversation_history)
    messages.append({"role": "user", "content": request.message})

    # Build prompts: gabung default dari DB (kosong karena standalone) + custom override
    prompts = dict(request.custom_prompts)

    # Inject system prompt override jika ada
    if request.system_prompt:
        prompts["_override_system"] = request.system_prompt

    return AgentState(
        session_id=f"rnd-{request.experiment_id or 'test'}",
        user_context=request.user_context,
        messages=messages,
        prompts=prompts,
        image_url=None,
        clarification_selected=None,
        # Override dari RnD request
        llm_provider_override=request.provider,
        llm_model_override=request.model,
        llm_temperature_override=request.temperature,
        llm_max_tokens_override=request.max_tokens,
        embedding_provider_override=request.embedding_provider,
        embedding_model_override=request.embedding_model,
        top_k_docs=request.top_k_docs,
        force_intent=request.force_intent,
        include_prompt_debug=request.include_prompt_in_response,
        # Output fields
        intent="",
        retrieved_docs=[],
        image_analysis=None,
        needs_clarification=False,
        clarification_data=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )
