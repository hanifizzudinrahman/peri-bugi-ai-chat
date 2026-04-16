"""
graph.py — LangGraph StateGraph untuk Tanya Peri v2.

Phase 1: kb_dental, user_profile, app_faq
Phase 2: rapot_peri, cerita_peri, mata_peri  ← aktif sekarang
Phase 4: janji_peri
"""
import asyncio
from typing import AsyncIterator

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.nodes.generate import generate_node
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
from app.schemas.chat import (
    ChatRequest,
    RnDChatRequest,
    make_done_event,
    make_error_event,
    make_thinking_event,
)

# Registry agent → function
_AGENT_REGISTRY = {
    # Phase 1
    "kb_dental": kb_dental_agent,
    "user_profile": user_profile_agent,
    "app_faq": app_faq_agent,
    # Phase 2
    "rapot_peri": rapot_peri_agent,
    "cerita_peri": cerita_peri_agent,
    "mata_peri": mata_peri_agent,
    # Phase 4 — uncomment saat siap:
    # "janji_peri": janji_peri_agent,
}


async def run_agent(initial_state: AgentState) -> AsyncIterator[str]:
    """Entry point utama — jalankan seluruh graph dan yield SSE events."""
    state = initial_state

    state.setdefault("thinking_steps", [])
    state.setdefault("tool_calls", [])
    state.setdefault("retrieved_docs", [])
    state.setdefault("agent_results", {})
    state.setdefault("image_analysis", None)
    state.setdefault("needs_clarification", False)
    state.setdefault("clarification_data", None)
    state.setdefault("quick_reply_data", None)
    state.setdefault("suggestion_chips", None)
    state.setdefault("final_response", "")
    state.setdefault("llm_metadata", {})
    state.setdefault("llm_call_logs", [])
    state.setdefault("agents_selected", [])
    state.setdefault("execution_plan", {})
    state.setdefault("allowed_agents", [])
    state.setdefault("agent_configs", {})
    state.setdefault("response_mode", "simple")
    state.setdefault("memory_context", {})
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
    state.setdefault("quick_reply_option_id", None)

    try:
        # Step 1: Supervisor
        async for event in supervisor_node(state):
            yield event

        agents_selected = state.get("agents_selected", [])
        execution_plan = state.get("execution_plan", {})

        # Step 2: Run selected agents
        if agents_selected:
            step_num = len(state.get("thinking_steps", [])) + 1
            label = _thinking_label(agents_selected)

            yield make_thinking_event(step=step_num, label=label, done=False)

            mode = execution_plan.get("mode", "sequential")
            if mode == "parallel" and len(agents_selected) > 1:
                await _run_agents_parallel(state, agents_selected)
            else:
                await _run_agents_sequential(state, agents_selected)

            yield make_thinking_event(step=step_num, label=label, done=True)
            state["thinking_steps"].append({"step": step_num, "label": label, "done": True})

        # Step 3: Generate
        async for event in generate_node(state):
            yield event

        # Step 4: Emit done
        if state.get("needs_clarification") and not state.get("final_response"):
            return
        if state.get("quick_reply_data") and not state.get("final_response"):
            return

        agents_used = list(state.get("agent_results", {}).keys())
        llm_call_logs = state.get("llm_call_logs", [])
        session_id = state.get("session_id", "")
        for log in llm_call_logs:
            if not log.get("session_id"):
                log["session_id"] = session_id

        yield make_done_event(
            content=state["final_response"],
            metadata={
                **state.get("llm_metadata", {}),
                "agents_used": agents_used,
                "llm_call_logs": llm_call_logs,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield make_error_event(f"Maaf, Tanya Peri sedang mengalami gangguan: {str(e)}")


def _thinking_label(agents_selected: list[str]) -> str:
    if len(agents_selected) > 1:
        return f"Mengumpulkan info dari {len(agents_selected)} sumber..."
    labels = {
        "kb_dental": "Mencari referensi kesehatan gigi...",
        "user_profile": "Membaca data profil...",
        "app_faq": "Mencari info aplikasi...",
        "rapot_peri": "Mengecek rapot sikat gigi...",
        "cerita_peri": "Mengecek progress Cerita Peri...",
        "mata_peri": "Menganalisis data gigi...",
        "janji_peri": "Mencari informasi dokter...",
    }
    return labels.get(agents_selected[0], "Mengumpulkan informasi...")


async def _run_agents_parallel(state: AgentState, agent_keys: list[str]) -> None:
    tasks = []
    valid_keys = []
    for key in agent_keys:
        fn = _AGENT_REGISTRY.get(key)
        if fn:
            tasks.append(fn(state))
            valid_keys.append(key)
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for key, result in zip(valid_keys, results):
        state["agent_results"][key] = {"error": str(result)} if isinstance(result, Exception) else result


async def _run_agents_sequential(state: AgentState, agent_keys: list[str]) -> None:
    for key in agent_keys:
        fn = _AGENT_REGISTRY.get(key)
        if fn:
            try:
                result = await fn(state)
                state["agent_results"][key] = result
            except Exception as e:
                state["agent_results"][key] = {"error": str(e)}


# =============================================================================
# State builders
# =============================================================================

def build_initial_state(request: ChatRequest) -> AgentState:
    return AgentState(
        session_id=request.session_id,
        user_context=request.user_context,
        messages=request.messages,
        prompts=request.prompts,
        image_url=request.image_url,
        clarification_selected=request.clarification_selected,
        quick_reply_option_id=request.quick_reply_option_id,
        allowed_agents=request.allowed_agents,
        agent_configs=request.agent_configs,
        response_mode=request.response_mode,
        memory_context=request.memory_context,
        llm_provider_override=None,
        llm_model_override=None,
        llm_temperature_override=None,
        llm_max_tokens_override=None,
        embedding_provider_override=None,
        embedding_model_override=None,
        top_k_docs=3,
        force_intent=None,
        include_prompt_debug=False,
        agents_selected=[],
        execution_plan={},
        agent_results={},
        retrieved_docs=[],
        image_analysis=None,
        needs_clarification=False,
        clarification_data=None,
        quick_reply_data=None,
        suggestion_chips=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )


def build_rnd_state(request: RnDChatRequest) -> AgentState:
    messages = list(request.conversation_history)
    messages.append({"role": "user", "content": request.message})

    prompts = dict(request.custom_prompts)
    if request.system_prompt:
        prompts["_override_system"] = request.system_prompt

    return AgentState(
        session_id=f"rnd-{request.experiment_id or 'test'}",
        user_context=request.user_context,
        messages=messages,
        prompts=prompts,
        image_url=None,
        clarification_selected=None,
        quick_reply_option_id=None,
        allowed_agents=request.allowed_agents or list(_AGENT_REGISTRY.keys()),
        agent_configs={},
        response_mode=request.response_mode or "simple",
        memory_context={},
        llm_provider_override=request.provider,
        llm_model_override=request.model,
        llm_temperature_override=request.temperature,
        llm_max_tokens_override=request.max_tokens,
        embedding_provider_override=request.embedding_provider,
        embedding_model_override=request.embedding_model,
        top_k_docs=request.top_k_docs,
        force_intent=request.force_intent,
        include_prompt_debug=request.include_prompt_in_response,
        agents_selected=[],
        execution_plan={},
        agent_results={},
        retrieved_docs=[],
        image_analysis=None,
        needs_clarification=False,
        clarification_data=None,
        quick_reply_data=None,
        suggestion_chips=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )
