"""
graph.py — entry points untuk LangGraph orchestration (Phase 1 refactor).

Phase 1: file ini SLIM — hanya expose:
- run_agent(state)        : main streaming entry, delegates ke streaming.langgraph_to_sse_stream
- build_initial_state()   : convert ChatRequest → Pydantic AgentState
- build_rnd_state()       : convert RnDChatRequest → Pydantic AgentState

Logic kompleks (orchestration, supervisor, generate, agent dispatch) sekarang
ada di:
- app/agents/builder.py            : StateGraph compile
- app/agents/supervisor.py         : supervisor node (refactored, return dict)
- app/agents/nodes/agent_dispatcher.py : agent runner (parallel/sequential)
- app/agents/nodes/generate.py     : generate node (refactored, return dict)
- app/agents/streaming.py          : LangGraph events → SSE adapter

Backward compat: signature run_agent(initial_state) → AsyncIterator[str] PRESERVED.
ChatRequest → AgentState mapping PRESERVED — fields yang kompat di-map ke
Pydantic sub-states.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.state import (
    AgentState,
    AgentControl,
    ImageInput,
    MemorySnapshot,
    ModeBehavior,
    RnDOverrides,
    SessionMeta,
    UserContextData,
)
from app.schemas.chat import ChatRequest, RnDChatRequest

logger = logging.getLogger(__name__)


# Default agent registry (untuk RnD mode default)
_DEFAULT_RND_AGENTS = [
    "kb_dental",
    "user_profile",
    "app_faq",
    "rapot_peri",
    "cerita_peri",
    "mata_peri",
]


# =============================================================================
# Public entry — run_agent
# =============================================================================


async def run_agent(initial_state: AgentState) -> AsyncIterator[str]:
    """
    Main streaming entry point — invoked dari main.py SSE handler.

    Phase 1: delegates ke streaming.langgraph_to_sse_stream yang
    invoke compiled LangGraph + map astream_events() → SSE format.
    """
    from app.agents.streaming import langgraph_to_sse_stream

    async for event in langgraph_to_sse_stream(initial_state):
        yield event


# =============================================================================
# State builders — convert HTTP request → Pydantic AgentState
# =============================================================================


def _convert_messages_to_basemessage(msgs: list) -> list:
    """Convert legacy dict messages [{role, content}] → LangChain BaseMessage objects."""
    result = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        # Skip system/other roles — system prompt built fresh di generate_node
    return result


def build_initial_state(request: ChatRequest) -> AgentState:
    """Convert ChatRequest → Pydantic AgentState untuk graph entry."""
    # Build sub-states
    session = SessionMeta(
        session_id=request.session_id,
        response_mode=request.response_mode or "simple",
        source=request.source or "web",
        timezone=getattr(request, "timezone", None) or "Asia/Jakarta",
        trace_id=request.trace_id,
        chat_message_id=request.chat_message_id,
    )

    user_context = UserContextData(**(request.user_context or {}))

    image = None
    if request.image_url:
        image = ImageInput(
            image_url=request.image_url,
            image_url_public=request.image_url_public,
            clarification_selected=request.clarification_selected,
            quick_reply_option_id=request.quick_reply_option_id,
        )

    control = AgentControl(
        allowed_agents=request.allowed_agents or [],
        agent_configs=request.agent_configs or {},
    )

    memory = MemorySnapshot(**(request.memory_context or {}))

    mode_behavior = ModeBehavior.from_response_mode(session.response_mode)

    return AgentState(
        messages=_convert_messages_to_basemessage(request.messages),
        user_context=user_context,
        session=session,
        image=image,
        control=control,
        memory=memory,
        prompts=dict(request.prompts or {}),
        mode_behavior=mode_behavior,
        # rnd, agents_selected, agent_results, etc gunakan default factory
    )


def build_rnd_state(request: RnDChatRequest) -> AgentState:
    """Convert RnDChatRequest → Pydantic AgentState untuk sandbox."""
    # Combine conversation history + current message
    messages_dicts = list(request.conversation_history or [])
    messages_dicts.append({"role": "user", "content": request.message})

    prompts = dict(request.custom_prompts or {})
    if request.system_prompt:
        prompts["_override_system"] = request.system_prompt

    session = SessionMeta(
        session_id=f"rnd-{request.experiment_id or 'test'}",
        response_mode=request.response_mode or "simple",
        source="sandbox",
    )

    user_context = UserContextData(**(request.user_context or {}))

    control = AgentControl(
        allowed_agents=request.allowed_agents or list(_DEFAULT_RND_AGENTS),
        agent_configs={},
    )

    rnd = RnDOverrides(
        llm_provider=request.provider,
        llm_model=request.model,
        llm_temperature=request.temperature,
        llm_max_tokens=request.max_tokens,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        top_k_docs=request.top_k_docs or 3,
        force_intent=request.force_intent,
        include_prompt_debug=request.include_prompt_in_response or False,
    )

    mode_behavior = ModeBehavior.from_response_mode(session.response_mode)

    return AgentState(
        messages=_convert_messages_to_basemessage(messages_dicts),
        user_context=user_context,
        session=session,
        image=None,
        control=control,
        memory=MemorySnapshot(),
        prompts=prompts,
        mode_behavior=mode_behavior,
        rnd=rnd,
    )
