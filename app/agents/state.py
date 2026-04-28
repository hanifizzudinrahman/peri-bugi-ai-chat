"""
AgentState — Pydantic state untuk LangGraph StateGraph (Phase 1 redesign).

Migrated dari TypedDict (68 flat fields) → Pydantic dengan namespaced sub-states.
LangGraph 0.2.60+ supports Pydantic state via .compile().

Backward compat: existing nodes yang akses state["key"] dibuat compatible via
StateAccessor wrapper (transitional, akan di-refactor di Phase 2).

Design principle (Phase 1):
- Static context (user_context, session, image, control, memory, prompts) — set once
- Mutable per node (agents_selected, agent_results, thinking_steps, etc) — appended/updated
- Transient (clarification_data, quick_reply_data) — emitted as SSE event then cleared

Phase 2 nanti akan:
- Replace agents_selected + agent_results dengan messages-based ReAct pattern
- Hapus shim StateAccessor — agents jadi pure functions
- Add ModeBehavior enforcement (sub-task 1.6 lay groundwork)
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Sub-state models — namespaced for clarity
# =============================================================================


class UserContextData(BaseModel):
    """Static user info — di-inject sekali per turn dari peri-bugi-api."""
    model_config = ConfigDict(extra="allow")  # forgiving: allow extra fields

    user: dict = Field(default_factory=dict)
    child: Optional[dict] = None
    brushing: Optional[dict] = None
    mata_peri_last_result: Optional[dict] = None


class SessionMeta(BaseModel):
    """Session-level metadata."""
    session_id: str
    response_mode: str = "simple"  # simple | medium | detailed
    source: str = "web"
    timezone: str = "Asia/Jakarta"
    trace_id: Optional[str] = None
    chat_message_id: Optional[str] = None


class ImageInput(BaseModel):
    """Set kalau user kirim image di turn ini."""
    image_url: str
    image_url_public: Optional[str] = None
    clarification_selected: Optional[list[str]] = None
    quick_reply_option_id: Optional[str] = None


class AgentControl(BaseModel):
    """Permission & config (di-inject dari api)."""
    allowed_agents: list[str] = Field(default_factory=list)
    agent_configs: dict[str, dict] = Field(default_factory=dict)


class MemorySnapshot(BaseModel):
    """L2/L3 memory snapshot — di-inject pre-graph dari api."""
    session_summaries: list[str] = Field(default_factory=list)
    user_facts: list[dict] = Field(default_factory=list)


class RnDOverrides(BaseModel):
    """RnD/sandbox overrides — separated for clarity."""
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k_docs: int = 3
    force_intent: Optional[str] = None
    include_prompt_debug: bool = False


class ModeBehavior(BaseModel):
    """
    Mode-aware behavior matrix (sub-task 1.6 groundwork).

    NOT ACTIVE in Phase 1 — this class is constructed and stored in state,
    but no node enforces these constraints yet. Phase 2 ReAct loop akan
    pakai ini untuk constrain LLM tool calls + recursion depth.

    Mapping dari response_mode:
      simple   → max_llm_calls=2, max_tool_calls=1, latency target 2-3s
      medium   → max_llm_calls=3, max_tool_calls=2, latency target 3-5s
      detailed → max_llm_calls=5, max_tool_calls=5, latency target 6-10s
    """
    max_llm_calls: int = 2
    max_tool_calls: int = 1
    enable_memory_recall: bool = False
    enable_cross_feature: bool = False
    output_format: str = "concise"  # concise | structured | comprehensive

    @classmethod
    def from_response_mode(cls, mode: str) -> "ModeBehavior":
        """Factory dari response_mode string."""
        if mode == "simple":
            return cls(
                max_llm_calls=2, max_tool_calls=1,
                enable_memory_recall=False, enable_cross_feature=False,
                output_format="concise",
            )
        elif mode == "medium":
            return cls(
                max_llm_calls=3, max_tool_calls=2,
                enable_memory_recall=True, enable_cross_feature=False,
                output_format="structured",
            )
        elif mode == "detailed":
            return cls(
                max_llm_calls=5, max_tool_calls=5,
                enable_memory_recall=True, enable_cross_feature=True,
                output_format="comprehensive",
            )
        # Fallback ke simple
        return cls.from_response_mode("simple")


class BudgetGuard(BaseModel):
    """
    Hard ceiling untuk prevent runaway agent (Phase 2 enforcement).

    Phase 1: di-track tapi tidak di-enforce.
    Phase 2: nodes akan check can_continue() sebelum LLM call.
    """
    llm_calls_used: int = 0
    tool_calls_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    abort_reason: Optional[str] = None

    def can_continue(self, mode: ModeBehavior) -> bool:
        if self.abort_reason is not None:
            return False
        if self.llm_calls_used >= mode.max_llm_calls:
            return False
        if self.tool_calls_used >= mode.max_tool_calls:
            return False
        return True


class ToolCallRecord(BaseModel):
    """Per-tool invocation audit (Phase 1: tracking only)."""
    tool: str
    agent: Optional[str] = None
    input: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)


class ThinkingStep(BaseModel):
    """SSE thinking event payload."""
    step: int
    label: str
    done: bool = False


class LLMCallLog(BaseModel):
    """Equivalent to LLMCallLogPayload from schemas/chat.py — moved here for clarity."""
    session_id: Optional[str] = None
    prompt_key: Optional[str] = None
    prompt_version: Optional[int] = None
    model: str
    provider: str
    node: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[dict] = None


# =============================================================================
# Top-level AgentState
# =============================================================================


class AgentState(BaseModel):
    """
    Single source of truth selama graph execution.

    Pydantic replaces TypedDict — type safety, validation, debug-ability.
    LangGraph compile(checkpointer=...) supports Pydantic state.

    PHASE 1 NOTES:
    - Existing legacy fields (agents_selected, agent_results, etc) PRESERVED
      untuk backward compat dengan supervisor_node + legacy agents.
    - ModeBehavior + BudgetGuard tracked tapi NOT enforced.
    - Phase 2 akan tambah `tool_history` dan refactor pattern jadi
      messages-based ReAct.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # untuk BaseMessage
    )

    # ── LangGraph-managed messages (special: append via add_messages reducer) ──
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # ── Static context (set once, immutable selama 1 turn) ────────────────────
    user_context: UserContextData = Field(default_factory=UserContextData)
    session: SessionMeta
    image: Optional[ImageInput] = None
    control: AgentControl = Field(default_factory=AgentControl)
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    prompts: dict[str, str] = Field(default_factory=dict)

    # ── Mode-aware behavior (Phase 1: tracked, Phase 2: enforced) ─────────────
    mode_behavior: ModeBehavior = Field(default_factory=ModeBehavior)
    budget_guard: BudgetGuard = Field(default_factory=BudgetGuard)

    # ── Routing decisions (Phase 1 supervisor; Phase 2 will use messages-based) ──
    agents_selected: list[str] = Field(default_factory=list)
    execution_plan: dict = Field(default_factory=dict)

    # ── Per-agent results (Phase 1: dict keyed by agent_key) ──────────────────
    agent_results: dict[str, Any] = Field(default_factory=dict)
    retrieved_docs: list[str] = Field(default_factory=list)
    image_analysis: Optional[dict] = None
    scan_session_id: Optional[str] = None

    # ── Clarification / quick reply (transient SSE event triggers) ────────────
    needs_clarification: bool = False
    clarification_data: Optional[dict] = None
    quick_reply_data: Optional[dict] = None
    suggestion_chips: Optional[list[str]] = None

    # ── Audit & metrics ───────────────────────────────────────────────────────
    thinking_steps: list[ThinkingStep] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_call_logs: list[LLMCallLog] = Field(default_factory=list)
    final_response: str = ""
    llm_metadata: dict = Field(default_factory=dict)

    # ── RnD / debug ───────────────────────────────────────────────────────────
    rnd: RnDOverrides = Field(default_factory=RnDOverrides)
    prompt_debug: Optional[dict] = None

    # =========================================================================
    # Backward-compat dict-style access (PHASE 1 SHIM)
    # =========================================================================
    # Existing code (main.py _stream_with_logging, _rnd_collect_response) akses
    # state via state.get("key") dan state["key"]. Untuk avoid touching banyak
    # call-sites, kita support dict-like access via methods berikut.
    # Phase 2 nanti akan refactor call-sites dan hapus methods ini.

    def get(self, key: str, default=None):
        """Dict-like .get() access untuk backward compat."""
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str):
        """Dict-like state["key"] access untuk backward compat."""
        # Map flat keys → Pydantic hierarchy
        if key == "session_id":
            return self.session.session_id
        elif key == "response_mode":
            return self.session.response_mode
        elif key == "source":
            return self.session.source
        elif key == "trace_id":
            return self.session.trace_id
        elif key == "chat_message_id":
            return self.session.chat_message_id
        elif key == "user_context":
            return self.user_context.model_dump()
        elif key == "messages":
            return [_msg_to_dict(m) for m in self.messages]
        elif key == "image_url":
            return self.image.image_url if self.image else None
        elif key == "image_url_public":
            return self.image.image_url_public if self.image else None
        elif key == "clarification_selected":
            return self.image.clarification_selected if self.image else None
        elif key == "quick_reply_option_id":
            return self.image.quick_reply_option_id if self.image else None
        elif key == "allowed_agents":
            return self.control.allowed_agents
        elif key == "agent_configs":
            return self.control.agent_configs
        elif key == "memory_context":
            return self.memory.model_dump()
        elif key == "agents_selected":
            return self.agents_selected
        elif key == "execution_plan":
            return self.execution_plan
        elif key == "agent_results":
            return self.agent_results
        elif key == "retrieved_docs":
            return self.retrieved_docs
        elif key == "image_analysis":
            return self.image_analysis
        elif key == "scan_session_id":
            return self.scan_session_id
        elif key == "needs_clarification":
            return self.needs_clarification
        elif key == "clarification_data":
            return self.clarification_data
        elif key == "quick_reply_data":
            return self.quick_reply_data
        elif key == "suggestion_chips":
            return self.suggestion_chips
        elif key == "thinking_steps":
            return [t.model_dump() for t in self.thinking_steps]
        elif key == "tool_calls":
            return [t.model_dump() for t in self.tool_calls]
        elif key == "llm_call_logs":
            return [l.model_dump() for l in self.llm_call_logs]
        elif key == "final_response":
            return self.final_response
        elif key == "llm_metadata":
            return self.llm_metadata
        elif key == "prompt_debug":
            return self.prompt_debug
        elif key == "prompts":
            return self.prompts
        elif key == "force_intent":
            return self.rnd.force_intent
        elif key == "include_prompt_debug":
            return self.rnd.include_prompt_debug
        # Fallback: direct attribute (untuk extensions di Phase 2)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


# =============================================================================
# StateAccessor — backward-compat shim for legacy dict-style access
# =============================================================================


class StateAccessor:
    """
    Wrapper supaya legacy code yang akses state["key"] tetap work selama
    transisi Phase 1 → 2. Setelah Phase 2 (agents jadi tools, return dict
    proper), accessor ini bisa dihapus.

    Map flat keys ke Pydantic hierarchy:
      state["session_id"]       → state.session.session_id
      state["user_context"]     → state.user_context.model_dump()
      state["image_url"]        → state.image.image_url (or None)
      state["allowed_agents"]   → state.control.allowed_agents
      state["agent_configs"]    → state.control.agent_configs
      state["memory_context"]   → state.memory.model_dump()
      state["response_mode"]    → state.session.response_mode
      ... etc
    """

    def __init__(self, state: AgentState):
        self._state = state

    # Mapping from legacy keys → (getter_fn, setter_fn)
    # Setter returns AgentState (immutable mutation pattern via .model_copy)
    _LEGACY_GETTERS = {
        "session_id": lambda s: s.session.session_id,
        "response_mode": lambda s: s.session.response_mode,
        "source": lambda s: s.session.source,
        "timezone": lambda s: s.session.timezone,
        "trace_id": lambda s: s.session.trace_id,
        "chat_message_id": lambda s: s.session.chat_message_id,
        "user_context": lambda s: s.user_context.model_dump(),
        "messages": lambda s: [_msg_to_dict(m) for m in s.messages],
        "prompts": lambda s: s.prompts,
        "image_url": lambda s: s.image.image_url if s.image else None,
        "image_url_public": lambda s: s.image.image_url_public if s.image else None,
        "clarification_selected": lambda s: s.image.clarification_selected if s.image else None,
        "quick_reply_option_id": lambda s: s.image.quick_reply_option_id if s.image else None,
        "allowed_agents": lambda s: s.control.allowed_agents,
        "agent_configs": lambda s: s.control.agent_configs,
        "memory_context": lambda s: s.memory.model_dump(),
        "agents_selected": lambda s: s.agents_selected,
        "execution_plan": lambda s: s.execution_plan,
        "agent_results": lambda s: s.agent_results,
        "retrieved_docs": lambda s: s.retrieved_docs,
        "image_analysis": lambda s: s.image_analysis,
        "scan_session_id": lambda s: s.scan_session_id,
        "needs_clarification": lambda s: s.needs_clarification,
        "clarification_data": lambda s: s.clarification_data,
        "quick_reply_data": lambda s: s.quick_reply_data,
        "suggestion_chips": lambda s: s.suggestion_chips,
        "thinking_steps": lambda s: [t.model_dump() for t in s.thinking_steps],
        "tool_calls": lambda s: [t.model_dump() for t in s.tool_calls],
        "llm_call_logs": lambda s: [l.model_dump() for l in s.llm_call_logs],
        "final_response": lambda s: s.final_response,
        "llm_metadata": lambda s: s.llm_metadata,
        "llm_provider_override": lambda s: s.rnd.llm_provider,
        "llm_model_override": lambda s: s.rnd.llm_model,
        "llm_temperature_override": lambda s: s.rnd.llm_temperature,
        "llm_max_tokens_override": lambda s: s.rnd.llm_max_tokens,
        "embedding_provider_override": lambda s: s.rnd.embedding_provider,
        "embedding_model_override": lambda s: s.rnd.embedding_model,
        "top_k_docs": lambda s: s.rnd.top_k_docs,
        "force_intent": lambda s: s.rnd.force_intent,
        "include_prompt_debug": lambda s: s.rnd.include_prompt_debug,
        "prompt_debug": lambda s: s.prompt_debug,
    }

    def get(self, key: str, default=None):
        getter = self._LEGACY_GETTERS.get(key)
        if getter is None:
            return default
        try:
            val = getter(self._state)
            return val if val is not None else default
        except (AttributeError, KeyError):
            return default

    def __getitem__(self, key: str):
        getter = self._LEGACY_GETTERS.get(key)
        if getter is None:
            raise KeyError(key)
        return getter(self._state)


def _msg_to_dict(msg: BaseMessage) -> dict:
    """Convert LangChain BaseMessage → legacy dict format."""
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    role = role_map.get(msg.type, msg.type)
    return {"role": role, "content": msg.content}
