"""
Schemas untuk ai-chat service v2.

ChatRequest       : payload dari peri-bugi-api (production)
RnDChatRequest    : payload untuk endpoint /chat/rnd (standalone research)
LLMCallLogPayload : metadata yang dikirim ke peri-bugi-api untuk logging
SSE Events        : event yang di-stream balik ke client
"""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Request dari peri-bugi-api (production)
# =============================================================================

class ChatRequest(BaseModel):
    """Payload lengkap dari peri-bugi-api ke ai-chat."""
    session_id: str
    user_context: dict = Field(default_factory=dict)
    messages: list[dict] = Field(default_factory=list)
    prompts: dict[str, str] = Field(default_factory=dict)
    # v2 additions
    allowed_agents: list[str] = Field(
        default_factory=list,
        description="Agent yang boleh dipakai user ini (sudah difilter oleh api)",
    )
    agent_configs: dict[str, dict] = Field(
        default_factory=dict,
        description="Config per agent: {agent_key: {llm_provider, llm_model, ...}}",
    )
    response_mode: str = Field(
        default="simple",
        description="Mode jawaban: simple | medium | detailed",
    )
    memory_context: dict = Field(
        default_factory=dict,
        description="{session_summaries: [...], user_facts: [...]}",
    )
    timezone: str = Field(default="Asia/Jakarta")
    source: str = Field(default="web")
    image_url: Optional[str] = None
    clarification_selected: Optional[list[str]] = None
    quick_reply_option_id: Optional[str] = None


# =============================================================================
# RnD / Standalone Request
# =============================================================================

class RnDChatRequest(BaseModel):
    """Request untuk endpoint /chat/rnd — riset dan eksperimen."""

    message: str
    conversation_history: list[dict] = Field(default_factory=list)

    # Override LLM
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    # Override Embedding & RAG
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k_docs: int = Field(default=3, ge=0, le=10)

    # Override Prompt
    system_prompt: Optional[str] = None
    custom_prompts: dict[str, str] = Field(default_factory=dict)

    # Intent / agent override
    force_intent: Optional[str] = None
    allowed_agents: Optional[list[str]] = None

    # v2 additions
    response_mode: Optional[str] = Field(default="simple")

    # Simulasi context
    user_context: dict = Field(
        default_factory=lambda: {
            "user": {"full_name": "RnD User", "nickname": "Peneliti"},
            "child": {"full_name": "Anak Test", "age_years": 6},
            "brushing": None,
            "mata_peri_last_result": None,
        }
    )

    # Metadata eksperimen
    experiment_id: Optional[str] = None
    experiment_tags: list[str] = Field(default_factory=list)
    expected_intent: Optional[str] = None
    expected_keywords: list[str] = Field(default_factory=list)

    # Output
    stream: bool = True
    include_prompt_in_response: bool = False


# =============================================================================
# LLM Call Log Payload
# =============================================================================

class LLMCallLogPayload(BaseModel):
    """Metadata satu LLM call — dikirim ke peri-bugi-api untuk disimpan."""
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
# SSE Event helpers
# =============================================================================

def _sse(event: str, data: Any) -> str:
    import json
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


def make_thinking_event(step: int, label: str, done: bool = False) -> str:
    return _sse("thinking", {"step": step, "label": label, "done": done})


def make_token_event(token: str) -> str:
    return _sse("token", token)


def make_clarify_event(question: str, options: list[dict], allow_multiple: bool = False) -> str:
    return _sse("clarify", {
        "question": question,
        "options": options,
        "allow_multiple": allow_multiple,
    })


def make_quick_reply_event(
    qr_type: str = "single_select",
    question: Optional[str] = None,
    options: list[dict] = None,
    allow_multiple: bool = False,
    dismissible: bool = True,
) -> str:
    return _sse("quick_reply", {
        "type": qr_type,
        "question": question,
        "options": options or [],
        "allow_multiple": allow_multiple,
        "dismissible": dismissible,
    })


def make_suggestions_event(chips: list[str]) -> str:
    return _sse("suggestions", {"chips": chips})


def make_tool_event(tool: str, input: dict, result: Any) -> str:
    return _sse("tool", {"tool": tool, "input": input, "result": result})


def make_done_event(content: str, metadata: dict | None = None) -> str:
    return _sse("done", {"content": content, "metadata": metadata or {}})


def make_error_event(message: str) -> str:
    return _sse("error", message)


def make_metrics_event(metrics: dict) -> str:
    """RnD only — kirim metrics setelah done."""
    return _sse("metrics", metrics)
