"""
Schemas untuk ai-chat service.

ChatRequest  : payload yang diterima dari peri-bugi-api
SSE Events   : event yang di-stream balik ke peri-bugi-api
AgentState   : state internal LangGraph yang dibawa antar node
"""
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# Request dari peri-bugi-api
# =============================================================================

class ChatRequest(BaseModel):
    """
    Payload lengkap yang dikirim peri-bugi-api ke ai-chat.
    Semua konteks user sudah di-inject — ai-chat tidak akses DB.
    """
    session_id: str = Field(..., description="UUID chat session")
    user_context: dict = Field(
        ...,
        description=(
            "Konteks user dari DB: "
            "{user: {id, full_name, nickname, gender}, "
            "child: {id, full_name, age_years, gender}, "
            "brushing: {current_streak, best_streak, last_complete_date}, "
            "mata_peri_last_result: {scan_date, summary_status, summary_text, ...}}"
        ),
    )
    messages: list[dict] = Field(
        ...,
        description="History percakapan: [{role: user|assistant, content: str}]",
    )
    prompts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Prompt templates dari DB, key → content. "
            "Contoh: {persona_system: '...', router_classify: '...'}"
        ),
    )
    timezone: str = Field(default="Asia/Jakarta")
    source: str = Field(default="web")
    image_url: Optional[str] = Field(
        default=None,
        description="URL gambar gigi dari user (opsional)",
    )
    clarification_selected: Optional[list[str]] = Field(
        default=None,
        description="Jawaban user untuk clarification checkpoint sebelumnya",
    )


# =============================================================================
# SSE Events — dikirim balik ke peri-bugi-api sebagai stream
# =============================================================================

class SSEEvent(BaseModel):
    event: str
    data: Any


def make_thinking_event(step: int, label: str, done: bool = False) -> str:
    import json
    return f"data: {json.dumps({'event': 'thinking', 'data': {'step': step, 'label': label, 'done': done}})}\n\n"


def make_token_event(token: str) -> str:
    import json
    return f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"


def make_clarify_event(question: str, options: list[dict], allow_multiple: bool = False) -> str:
    import json
    return f"data: {json.dumps({'event': 'clarify', 'data': {'question': question, 'options': options, 'allow_multiple': allow_multiple}})}\n\n"


def make_tool_event(tool: str, input: dict, result: Any) -> str:
    import json
    return f"data: {json.dumps({'event': 'tool', 'data': {'tool': tool, 'input': input, 'result': result}})}\n\n"


def make_done_event(content: str, metadata: dict | None = None) -> str:
    import json
    return f"data: {json.dumps({'event': 'done', 'data': {'content': content, 'metadata': metadata or {}}})}\n\n"


def make_error_event(message: str) -> str:
    import json
    return f"data: {json.dumps({'event': 'error', 'data': message})}\n\n"


# =============================================================================
# LangGraph Agent State
# =============================================================================

class AgentState(TypedDict):
    """
    State yang dibawa antar node di LangGraph graph.
    Setiap node bisa read dan update state ini.
    """
    # Input dari request
    session_id: str
    user_context: dict
    messages: list[dict]            # conversation history
    prompts: dict[str, str]         # prompt templates dari DB
    image_url: Optional[str]
    clarification_selected: Optional[list[str]]

    # Hasil routing
    intent: str                     # dental_qa | context_query | image | clarification_answer | smalltalk

    # Hasil per node
    retrieved_docs: list[str]       # dokumen dari RAG
    image_analysis: Optional[dict]  # hasil inference dari ai-cv
    needs_clarification: bool
    clarification_data: Optional[dict]  # question + options

    # Akumulasi response
    thinking_steps: list[dict]      # dikumpulkan dari semua node
    tool_calls: list[dict]          # log tool calls
    final_response: str             # response lengkap setelah streaming

    # Metadata untuk logging
    llm_metadata: dict              # token count, latency, model, dll
