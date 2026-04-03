"""
ai-chat — Tanya Peri AI Service
FastAPI app + endpoints:
  POST /chat/stream   — production endpoint (internal, perlu X-Internal-Secret)
  POST /chat/rnd      — RnD standalone endpoint (perlu RND_MODE=True)
  GET  /health        — health check
  GET  /health/gpu    — GPU/device info untuk debugging
"""
import json
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents.peri_agent import _build_initial_state, _build_rnd_state, run_agent
from app.config.settings import settings
from app.schemas.chat import (
    ChatRequest,
    RnDChatRequest,
    make_done_event,
    make_error_event,
    make_metrics_event,
)
from app.services.llm_logger import send_llm_call_logs

app = FastAPI(
    title="Tanya Peri — AI Chat Service",
    description=(
        "Internal AI service untuk chatbot Tanya Peri. "
        "Tidak diakses langsung oleh frontend — selalu melalui peri-bugi-api. "
        "Endpoint /chat/rnd tersedia untuk RnD dan eksperimen (jika RND_MODE=True)."
    ),
    version="1.1.0",
    # Docs hanya aktif di development
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # private service, tidak perlu restrict
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Security: validasi internal secret
# =============================================================================

def _verify_internal_secret(x_internal_secret: str | None) -> None:
    """Validasi bahwa request berasal dari peri-bugi-api."""
    if not settings.INTERNAL_SECRET:
        return  # Secret belum diset — skip validasi (dev mode)
    if x_internal_secret != settings.INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid internal secret")


# =============================================================================
# Stream wrapper: intercept done event untuk kirim LLM logs ke api
# =============================================================================

async def _stream_with_logging(state) -> AsyncIterator[str]:
    """
    Wrapper di atas run_agent yang:
    1. Proxy semua event ke client
    2. Setelah event 'done', fire-and-forget kirim LLM logs ke peri-bugi-api

    Kenapa di sini bukan di dalam run_agent:
    - run_agent tidak tahu soal HTTP/logging — murni agent logic
    - Separation of concerns: transport layer yang handle logging
    """
    llm_call_logs: list[dict] = []
    session_id: str = state.get("session_id", "")

    async for event_str in run_agent(state):
        yield event_str

        # Parse event untuk cek apakah ini done event
        try:
            if event_str.startswith("data: "):
                parsed = json.loads(event_str[6:])
                if parsed.get("event") == "done":
                    # Ambil logs dari metadata done event
                    data = parsed.get("data", {})
                    if isinstance(data, dict):
                        metadata = data.get("metadata", {})
                        if isinstance(metadata, dict):
                            llm_call_logs = metadata.pop("llm_call_logs", [])
        except (json.JSONDecodeError, Exception):
            pass

    # Fire-and-forget kirim logs ke api setelah streaming selesai
    if llm_call_logs:
        import asyncio
        asyncio.create_task(send_llm_call_logs(llm_call_logs, session_id))


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check untuk Cloud Run dan monitoring."""
    return {
        "status": "ok",
        "service": "tanya-peri-ai-chat",
        "provider": settings.LLM_PROVIDER,
        "model": settings.llm_model_name,
        "rnd_mode": settings.RND_MODE,
    }


@app.get("/health/gpu")
async def health_gpu():
    """
    Info GPU/device — berguna saat debugging performa embedding.
    Cek apakah embedding berjalan di GPU atau CPU.
    """
    info: dict = {
        "embedding_device_setting": settings.EMBEDDING_DEVICE,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1
            )
            info["gpu_memory_reserved_mb"] = round(
                torch.cuda.memory_reserved(0) / 1024 / 1024, 1
            )
        else:
            info["cuda_available"] = False
            info["note"] = "CUDA tidak tersedia — embedding akan berjalan di CPU"
    except ImportError:
        info["torch_installed"] = False
        info["note"] = "torch tidak terinstall — embedding device tidak bisa dideteksi"

    return info


@app.post(
    "/chat/stream",
    summary="Stream response Tanya Peri (SSE) — internal",
    description=(
        "Endpoint internal — hanya dipanggil oleh peri-bugi-api. "
        "Terima ChatRequest lengkap dengan user context + prompts, "
        "return SSE stream: thinking | token | clarify | tool | done | error"
    ),
)
async def chat_stream(
    request: ChatRequest,
    x_internal_secret: str | None = Header(default=None),
):
    _verify_internal_secret(x_internal_secret)

    initial_state = _build_initial_state(request)

    return StreamingResponse(
        _stream_with_logging(initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/chat/rnd",
    summary="RnD endpoint — standalone testing & research",
    description=(
        "Endpoint untuk RnD dan eksperimen. Tidak perlu internal secret. "
        "Support override semua parameter LLM, embedding, dan prompt. "
        "Aktif hanya jika RND_MODE=True di .env. "
        "MATIKAN di production! "
        "\n\nContoh use case:\n"
        "- Test berbagai model (gemma2:2b vs qwen3.5 vs gemini-flash)\n"
        "- Experiment dengan temperature dan max_tokens\n"
        "- Test custom system prompt tanpa ubah DB\n"
        "- Collect metrics TTFT, latency, token count untuk paper\n"
        "- Evaluate router accuracy dengan expected_intent\n"
        "- Evaluate response quality dengan expected_keywords"
    ),
)
async def chat_rnd(request: RnDChatRequest):
    if not settings.RND_MODE:
        raise HTTPException(
            status_code=403,
            detail="RnD mode tidak aktif. Set RND_MODE=True di .env untuk mengaktifkan.",
        )

    initial_state = _build_rnd_state(request)

    if request.stream:
        # SSE streaming mode
        return StreamingResponse(
            _rnd_stream_with_metrics(initial_state, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: tunggu selesai, return JSON
        return await _rnd_collect_response(initial_state, request)


async def _rnd_stream_with_metrics(state, request: RnDChatRequest) -> AsyncIterator[str]:
    """
    Stream untuk RnD mode. Setelah done event, emit metrics event tambahan
    dengan hasil evaluasi (intent_correct, keyword_hit_rate, dll).
    """
    final_metadata: dict = {}

    async for event_str in run_agent(state):
        yield event_str

        try:
            if event_str.startswith("data: "):
                parsed = json.loads(event_str[6:])
                if parsed.get("event") == "done":
                    data = parsed.get("data", {})
                    if isinstance(data, dict):
                        final_metadata = data.get("metadata", {})
        except (json.JSONDecodeError, Exception):
            pass

    # Emit metrics event setelah done
    metrics = _build_rnd_metrics(state, request, final_metadata)
    yield make_metrics_event(metrics)


async def _rnd_collect_response(state, request: RnDChatRequest) -> JSONResponse:
    """
    Non-streaming mode untuk RnD: collect semua events, return JSON.
    Berguna untuk batch experiment (loop banyak pertanyaan, collect semua results).
    """
    events = []
    full_content = ""
    final_metadata = {}

    async for event_str in run_agent(state):
        if event_str.startswith("data: "):
            try:
                parsed = json.loads(event_str[6:])
                events.append(parsed)
                if parsed.get("event") == "token":
                    full_content += parsed.get("data", "")
                elif parsed.get("event") == "done":
                    data = parsed.get("data", {})
                    if isinstance(data, dict):
                        full_content = data.get("content", full_content)
                        final_metadata = data.get("metadata", {})
            except json.JSONDecodeError:
                pass

    metrics = _build_rnd_metrics(state, request, final_metadata)

    response_data = {
        "response": full_content,
        "intent": state.get("intent"),
        "thinking_steps": state.get("thinking_steps", []),
        "tool_calls": state.get("tool_calls", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "metadata": final_metadata,
        "metrics": metrics,
    }

    if request.include_prompt_in_response and state.get("prompt_debug"):
        response_data["prompt_debug"] = state["prompt_debug"]

    return JSONResponse(content=response_data)


def _build_rnd_metrics(state, request: RnDChatRequest, metadata: dict) -> dict:
    """
    Build metrics dict untuk evaluasi RnD.

    Metrics yang tersedia:
    - latency_ms         : total waktu dari request sampai last token (ms)
    - ttft_ms            : time-to-first-token (ms) — penting untuk UX streaming
    - output_tokens_approx: estimasi jumlah output tokens
    - intent_detected    : intent yang dideteksi router
    - intent_correct     : apakah intent sesuai expected (jika diisi)
    - keyword_hit_rate   : % expected keywords yang ada di response (0.0-1.0)
    - keywords_found     : list keywords yang ketemu di response
    - keywords_missing   : list keywords yang tidak ada di response
    - model              : model yang dipakai
    - provider           : provider yang dipakai
    - experiment_id      : dari request
    - experiment_tags    : dari request
    - llm_call_count     : jumlah LLM call dalam satu request
    """
    metrics: dict = {
        "latency_ms": metadata.get("latency_ms"),
        "ttft_ms": metadata.get("ttft_ms"),
        "output_tokens_approx": metadata.get("output_tokens_approx"),
        "intent_detected": state.get("intent"),
        "model": metadata.get("model"),
        "provider": metadata.get("provider"),
        "experiment_id": request.experiment_id,
        "experiment_tags": request.experiment_tags,
        "llm_call_count": len(state.get("llm_call_logs", [])),
        "rag_docs_retrieved": len(state.get("retrieved_docs", [])),
    }

    # Evaluasi intent accuracy
    if request.expected_intent:
        metrics["intent_correct"] = (state.get("intent") == request.expected_intent)
        metrics["expected_intent"] = request.expected_intent

    # Evaluasi keyword coverage
    if request.expected_keywords:
        response_text = (state.get("final_response") or "").lower()
        found = [kw for kw in request.expected_keywords if kw.lower() in response_text]
        missing = [kw for kw in request.expected_keywords if kw.lower() not in response_text]
        metrics["keyword_hit_rate"] = len(found) / len(request.expected_keywords) if request.expected_keywords else 1.0
        metrics["keywords_found"] = found
        metrics["keywords_missing"] = missing

    # LLM call breakdown (per node)
    call_logs = state.get("llm_call_logs", [])
    if call_logs:
        metrics["llm_calls"] = [
            {
                "node": log.get("node"),
                "latency_ms": log.get("latency_ms"),
                "ttft_ms": log.get("ttft_ms"),
                "output_tokens": log.get("output_tokens"),
                "success": log.get("success"),
            }
            for log in call_logs
        ]

    return metrics
