"""
Schemas untuk ai-chat service.

ChatRequest      : payload dari peri-bugi-api (production/integration)
RnDChatRequest   : payload untuk endpoint /chat/rnd (standalone research)
LLMCallLogPayload: metadata yang dikirim ke peri-bugi-api untuk disimpan
SSE Events       : event yang di-stream balik
AgentState       : state internal LangGraph yang dibawa antar node
"""
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# Request dari peri-bugi-api (production)
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
# RnD / Standalone Request — untuk riset tanpa perlu integrasi ke api
# =============================================================================

class RnDChatRequest(BaseModel):
    """
    Request untuk endpoint /chat/rnd — dipakai untuk riset dan eksperimen.

    Tidak perlu session_id atau user_context lengkap.
    Semua parameter LLM bisa di-override langsung dari sini.

    Cocok untuk:
    - Test berbagai model (ganti provider/model tanpa restart)
    - Bandingkan akurasi antar model dengan pertanyaan yang sama
    - Experiment dengan berbagai parameter (temperature, max_tokens)
    - Test custom system prompt tanpa ubah DB
    - Collect metrics untuk paper (latency, TTFT, token count)
    """

    # ── Pesan ──────────────────────────────────────────────────────────────
    message: str = Field(..., description="Pertanyaan / pesan user")
    conversation_history: list[dict] = Field(
        default_factory=list,
        description=(
            "History percakapan sebelumnya (opsional). "
            "Format: [{role: 'user'|'assistant', content: '...'}]"
        ),
    )

    # ── Override LLM ───────────────────────────────────────────────────────
    provider: Optional[str] = Field(
        default=None,
        description="Override provider: ollama | gemini | openai. Default: dari .env",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Override model name. Contoh: 'gemma2:2b', 'qwen3.5', "
            "'gemini-1.5-flash', 'gpt-4o-mini'. Default: dari .env"
        ),
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override temperature (0.0–2.0). Default: dari .env",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Override max output tokens. Default: dari .env",
    )

    # ── Override Embedding & RAG ───────────────────────────────────────────
    embedding_provider: Optional[str] = Field(
        default=None,
        description="Override embedding provider: local | gemini | openai",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Override embedding model. "
            "Contoh: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', "
            "'models/text-embedding-004'"
        ),
    )
    top_k_docs: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Jumlah dokumen RAG yang diambil dari Qdrant. 0 = skip RAG",
    )

    # ── Override Prompt ────────────────────────────────────────────────────
    system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Override system prompt sepenuhnya. "
            "Jika diisi, persona_system dari DB diabaikan."
        ),
    )
    custom_prompts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Override prompt templates tertentu saja. "
            "Key yang tidak ada di sini tetap pakai default dari DB/hardcoded. "
            "Contoh: {'router_classify': '...custom prompt...'}"
        ),
    )

    # ── Intent override (skip router) ─────────────────────────────────────
    force_intent: Optional[str] = Field(
        default=None,
        description=(
            "Paksa intent tertentu, skip router node. "
            "Berguna untuk test generate node secara isolated. "
            "Nilai: dental_qa | context_query | image | clarification_answer | smalltalk"
        ),
    )

    # ── Context simulasi ───────────────────────────────────────────────────
    user_context: dict = Field(
        default_factory=lambda: {
            "user": {"full_name": "RnD User", "nickname": "Peneliti"},
            "child": {"full_name": "Anak Test", "age_years": 6},
            "brushing": None,
            "mata_peri_last_result": None,
        },
        description=(
            "Konteks user simulasi untuk test. "
            "Boleh dikosongkan atau diisi sesuai skenario yang ingin ditest."
        ),
    )

    # ── Metadata eksperimen ────────────────────────────────────────────────
    experiment_id: Optional[str] = Field(
        default=None,
        description=(
            "ID eksperimen untuk tracking. Disimpan di metadata response. "
            "Berguna saat run batch experiment untuk filter hasil."
        ),
    )
    experiment_tags: list[str] = Field(
        default_factory=list,
        description="Tags eksperimen. Contoh: ['baseline', 'v2-prompt', 'gemini-comparison']",
    )
    expected_intent: Optional[str] = Field(
        default=None,
        description=(
            "Intent yang diharapkan (ground truth). "
            "Jika diisi, response akan include flag 'intent_correct' untuk evaluasi router."
        ),
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Kata kunci yang diharapkan ada di response (soft evaluation). "
            "Response akan include 'keyword_hit_rate' untuk evaluasi kualitas jawaban."
        ),
    )

    # ── Opsi output ────────────────────────────────────────────────────────
    stream: bool = Field(
        default=True,
        description="True = SSE streaming (real-time). False = tunggu response penuh lalu return JSON.",
    )
    include_prompt_in_response: bool = Field(
        default=False,
        description="Jika True, response include full prompt yang dikirim ke LLM (untuk debug).",
    )


# =============================================================================
# LLM Call Log Payload — dikirim ke peri-bugi-api
# =============================================================================

class LLMCallLogPayload(BaseModel):
    """
    Metadata satu LLM call yang dikirim ke peri-bugi-api untuk disimpan
    di tabel llm_call_logs.

    Dikirim sebagai fire-and-forget HTTP POST (tidak blocking stream).
    """
    session_id: Optional[str] = Field(default=None)
    prompt_key: Optional[str] = Field(default=None)
    prompt_version: Optional[int] = Field(default=None)
    model: str
    provider: str
    node: Optional[str] = Field(default=None)
    input_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None)
    ttft_ms: Optional[int] = Field(default=None)
    success: bool = True
    error_message: Optional[str] = Field(default=None)
    metadata: Optional[dict] = Field(default=None)


# =============================================================================
# SSE Events — dikirim balik ke peri-bugi-api / client sebagai stream
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


def make_metrics_event(metrics: dict) -> str:
    """Event khusus RnD — kirim metrics setelah done untuk evaluasi."""
    import json
    return f"data: {json.dumps({'event': 'metrics', 'data': metrics})}\n\n"


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

    # Override per-request (untuk RnD)
    llm_provider_override: Optional[str]
    llm_model_override: Optional[str]
    llm_temperature_override: Optional[float]
    llm_max_tokens_override: Optional[int]
    embedding_provider_override: Optional[str]
    embedding_model_override: Optional[str]
    top_k_docs: int
    force_intent: Optional[str]
    include_prompt_debug: bool

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
    llm_call_logs: list[dict]       # list semua LLM call dalam satu request (untuk logging ke api)
    prompt_debug: Optional[dict]    # full prompt yang dikirim ke LLM (hanya untuk RnD)
