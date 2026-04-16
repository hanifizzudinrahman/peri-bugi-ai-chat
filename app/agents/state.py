"""
AgentState — state yang dibawa antar node di LangGraph graph v2.

Dipisah dari schemas/chat.py agar lebih clean dan tidak circular import.
"""
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input dari request ────────────────────────────────────────────────────
    session_id: str
    user_context: dict
    messages: list[dict]                # conversation history
    prompts: dict[str, str]             # prompt templates dari DB
    image_url: Optional[str]
    clarification_selected: Optional[list[str]]
    quick_reply_option_id: Optional[str]

    # ── Agent routing ─────────────────────────────────────────────────────────
    allowed_agents: list[str]           # agents yang boleh dipakai user ini
    agent_configs: dict[str, dict]      # LLM override per agent dari DB
    response_mode: str                  # simple | medium | detailed

    # ── Memory context (injected dari api) ────────────────────────────────────
    memory_context: dict                # {session_summaries, user_facts}

    # ── Supervisor output ─────────────────────────────────────────────────────
    agents_selected: list[str]          # agent mana yang Supervisor pilih
    execution_plan: dict                # {mode: parallel|sequential, reasoning}

    # ── Per-agent results ─────────────────────────────────────────────────────
    agent_results: dict[str, Any]       # {agent_key: result_dict}

    # ── Tool + retrieval ──────────────────────────────────────────────────────
    retrieved_docs: list[str]           # dokumen dari RAG (kb_dental)
    image_analysis: Optional[dict]

    # ── Clarification / quick reply ────────────────────────────────────────────
    needs_clarification: bool
    clarification_data: Optional[dict]
    quick_reply_data: Optional[dict]    # untuk quick_reply event
    suggestion_chips: Optional[list[str]]

    # ── Akumulasi output ──────────────────────────────────────────────────────
    thinking_steps: list[dict]
    tool_calls: list[dict]
    final_response: str
    llm_metadata: dict
    llm_call_logs: list[dict]

    # ── RnD / override ────────────────────────────────────────────────────────
    llm_provider_override: Optional[str]
    llm_model_override: Optional[str]
    llm_temperature_override: Optional[float]
    llm_max_tokens_override: Optional[int]
    embedding_provider_override: Optional[str]
    embedding_model_override: Optional[str]
    top_k_docs: int
    force_intent: Optional[str]
    include_prompt_debug: bool
    prompt_debug: Optional[dict]
