"""
ai-chat — Tanya Peri AI Service v2
"""
import json
import logging
import time
from typing import AsyncIterator, Optional

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import build_initial_state, build_rnd_state, run_agent
from app.config.settings import settings
from app.middleware.rate_limit import check_benchmark_rate_limit, check_rnd_rate_limit
from app.schemas.chat import ChatRequest, RnDChatRequest, make_metrics_event
from app.services.llm_logger import send_llm_call_logs

logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if not settings.is_production else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

log = structlog.get_logger()

app = FastAPI(
    title="Tanya Peri — AI Chat Service v2",
    description="Internal AI service untuk chatbot Tanya Peri (multi-agent).",
    version="2.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path in ("/health", "/health/gpu", "/health/llm", "/health/agents"):
        return await call_next(request)
    start = time.monotonic()
    response = await call_next(request)
    log.info("http_request", method=request.method, path=request.url.path,
             status=response.status_code, duration_ms=int((time.monotonic()-start)*1000))
    return response


def _verify_internal_secret(x_internal_secret: str | None) -> None:
    if not settings.INTERNAL_SECRET:
        return
    if x_internal_secret != settings.INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid internal secret")


def _require_rnd_mode() -> None:
    if not settings.RND_MODE:
        raise HTTPException(status_code=403, detail="RnD mode tidak aktif. Set RND_MODE=True di .env.")


async def _stream_with_logging(state) -> AsyncIterator[str]:
    llm_call_logs: list[dict] = []
    session_id: str = state.get("session_id", "")
    async for event_str in run_agent(state):
        yield event_str
        try:
            if event_str.startswith("data: "):
                parsed = json.loads(event_str[6:])
                if parsed.get("event") == "done":
                    data = parsed.get("data", {})
                    if isinstance(data, dict):
                        metadata = data.get("metadata", {})
                        if isinstance(metadata, dict):
                            llm_call_logs = metadata.pop("llm_call_logs", [])
        except Exception:
            pass
    if llm_call_logs:
        import asyncio
        asyncio.create_task(send_llm_call_logs(llm_call_logs, session_id))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tanya-peri-ai-chat", "version": "2.0.0",
            "provider": settings.LLM_PROVIDER, "model": settings.llm_model_name, "rnd_mode": settings.RND_MODE}


@app.get("/health/agents")
async def health_agents():
    from app.agents.graph import _AGENT_REGISTRY
    return {"status": "ok", "registered_agents": list(_AGENT_REGISTRY.keys()), "agent_count": len(_AGENT_REGISTRY)}


@app.get("/health/gpu")
async def health_gpu():
    info: dict = {"embedding_device_setting": settings.EMBEDDING_DEVICE,
                  "embedding_provider": settings.EMBEDDING_PROVIDER, "embedding_model": settings.EMBEDDING_MODEL}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_installed"] = False
    return info


@app.get("/health/llm")
async def health_llm():
    provider = settings.LLM_PROVIDER
    result: dict = {"provider": provider, "model": settings.llm_model_name}
    if provider == "ollama":
        import urllib.request, json as _j
        try:
            with urllib.request.urlopen(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
                data = _j.loads(r.read())
                models = [m["name"] for m in data.get("models", [])]
                result["ollama_status"] = "ok"
                result["available_models"] = models
                result["configured_model_available"] = any(
                    settings.OLLAMA_MODEL in m or m.startswith(settings.OLLAMA_MODEL.split(":")[0]) for m in models)
        except Exception as e:
            result["ollama_status"] = "error"
            result["error"] = str(e)
    elif provider == "gemini":
        result["status"] = "ok" if settings.GEMINI_API_KEY else "error"
        if not settings.GEMINI_API_KEY:
            result["error"] = "GEMINI_API_KEY belum diset"
    elif provider == "openai":
        result["status"] = "ok" if settings.OPENAI_API_KEY else "error"
        if not settings.OPENAI_API_KEY:
            result["error"] = "OPENAI_API_KEY belum diset"
    result["overall"] = "error" if result.get("status") == "error" or result.get("ollama_status") == "error" else "ok"
    return result


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, x_internal_secret: str | None = Header(default=None)):
    _verify_internal_secret(x_internal_secret)
    initial_state = build_initial_state(request)
    return StreamingResponse(_stream_with_logging(initial_state),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/chat/rnd")
async def chat_rnd(request: RnDChatRequest, _rl: None = Depends(check_rnd_rate_limit)):
    _require_rnd_mode()
    initial_state = build_rnd_state(request)
    log.info("rnd_request", model=request.model or settings.llm_model_name,
             response_mode=request.response_mode, allowed_agents=request.allowed_agents,
             experiment_id=request.experiment_id)
    if request.stream:
        return StreamingResponse(_rnd_stream_with_metrics(initial_state, request),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await _rnd_collect_response(initial_state, request)


class BenchmarkRequest(BaseModel):
    message: str
    n: int = Field(default=3, ge=1, le=10)
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=0.1)
    max_tokens: Optional[int] = Field(default=200)
    response_mode: Optional[str] = Field(default="simple")


@app.post("/chat/rnd/benchmark")
async def benchmark(request: BenchmarkRequest, _rl: None = Depends(check_benchmark_rate_limit)):
    _require_rnd_mode()
    results = []
    for i in range(request.n):
        rnd_req = RnDChatRequest(message=request.message, provider=request.provider,
                                  model=request.model, temperature=request.temperature,
                                  max_tokens=request.max_tokens, stream=False,
                                  response_mode=request.response_mode,
                                  experiment_id=f"benchmark-run-{i+1}")
        state = build_rnd_state(rnd_req)
        result = await _rnd_collect_response_raw(state, rnd_req)
        result["run"] = i + 1
        result["is_cold"] = (i == 0)
        results.append(result)
    latencies = [r["metrics"]["latency_ms"] for r in results if r["metrics"].get("latency_ms")]
    ttfts = [r["metrics"]["ttft_ms"] for r in results if r["metrics"].get("ttft_ms")]
    tps_list = [r["metrics"]["tokens_per_second"] for r in results if r["metrics"].get("tokens_per_second")]
    return {
        "summary": {
            "model": request.model or settings.llm_model_name,
            "provider": request.provider or settings.LLM_PROVIDER,
            "n_runs": request.n,
            "cold_latency_ms": latencies[0] if latencies else None,
            "warm_avg_latency_ms": int(sum(latencies[1:])/len(latencies[1:])) if len(latencies) > 1 else None,
            "ttft_cold_ms": ttfts[0] if ttfts else None,
            "ttft_warm_avg_ms": int(sum(ttfts[1:])/len(ttfts[1:])) if len(ttfts) > 1 else None,
            "avg_tokens_per_second": round(sum(tps_list)/len(tps_list), 1) if tps_list else None,
        },
        "runs": results,
    }


async def _rnd_stream_with_metrics(state, request: RnDChatRequest) -> AsyncIterator[str]:
    final_metadata: dict = {}
    async for event_str in run_agent(state):
        yield event_str
        try:
            if event_str.startswith("data: "):
                parsed = json.loads(event_str[6:])
                if parsed.get("event") == "done":
                    final_metadata = parsed.get("data", {}).get("metadata", {})
        except Exception:
            pass
    yield make_metrics_event(_build_rnd_metrics(state, request, final_metadata))


async def _rnd_collect_response(state, request: RnDChatRequest) -> JSONResponse:
    return JSONResponse(content=await _rnd_collect_response_raw(state, request))


async def _rnd_collect_response_raw(state, request: RnDChatRequest) -> dict:
    full_content = ""
    final_metadata = {}
    async for event_str in run_agent(state):
        if event_str.startswith("data: "):
            try:
                parsed = json.loads(event_str[6:])
                if parsed.get("event") == "token":
                    full_content += parsed.get("data", "")
                elif parsed.get("event") == "done":
                    data = parsed.get("data", {})
                    if isinstance(data, dict):
                        full_content = data.get("content", full_content)
                        final_metadata = data.get("metadata", {})
            except json.JSONDecodeError:
                pass
    result = {
        "response": full_content,
        "agents_selected": state.get("agents_selected", []),
        "agent_results": {k: v for k, v in state.get("agent_results", {}).items() if not isinstance(v, Exception)},
        "thinking_steps": state.get("thinking_steps", []),
        "tool_calls": state.get("tool_calls", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "metadata": final_metadata,
        "metrics": _build_rnd_metrics(state, request, final_metadata),
    }
    if request.include_prompt_in_response and state.get("prompt_debug"):
        result["prompt_debug"] = state["prompt_debug"]
    return result


def _build_rnd_metrics(state, request: RnDChatRequest, metadata: dict) -> dict:
    metrics: dict = {
        "latency_ms": metadata.get("latency_ms"), "ttft_ms": metadata.get("ttft_ms"),
        "generation_ms": metadata.get("generation_ms"), "tokens_per_second": metadata.get("tokens_per_second"),
        "output_tokens_approx": metadata.get("output_tokens_approx"),
        "agents_used": metadata.get("agents_used", []),
        "response_mode": state.get("response_mode", "simple"),
        "model": metadata.get("model"), "provider": metadata.get("provider"),
        "experiment_id": request.experiment_id, "experiment_tags": request.experiment_tags,
        "llm_call_count": len(state.get("llm_call_logs", [])),
        "rag_docs_retrieved": len(state.get("retrieved_docs", [])),
    }
    if request.expected_keywords:
        response_text = (state.get("final_response") or "").lower()
        found = [kw for kw in request.expected_keywords if kw.lower() in response_text]
        missing = [kw for kw in request.expected_keywords if kw.lower() not in response_text]
        metrics["keyword_hit_rate"] = round(len(found)/len(request.expected_keywords), 3)
        metrics["keywords_found"] = found
        metrics["keywords_missing"] = missing
    return metrics


# =============================================================================
# Memory endpoints — Phase 2
# =============================================================================

class SummarizeRequest(BaseModel):
    session_id: str = Field(..., description="UUID session yang mau di-summarize")
    messages: list[dict] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)


@app.post(
    "/memory/summarize",
    summary="Generate dan simpan L2 session summary",
    description="Dipanggil background task dari peri-bugi-api. Tidak blocking.",
)
async def memory_summarize(
    request: SummarizeRequest,
    x_internal_secret: str | None = Header(default=None),
):
    _verify_internal_secret(x_internal_secret)

    from app.agents.memory_job import (
        generate_session_summary,
        extract_topics_from_messages,
        send_summary_to_api,
    )

    summary_text = await generate_session_summary(
        session_id=request.session_id,
        user_id="",
        messages=request.messages,
        agents_used=request.agents_used,
    )

    if not summary_text:
        return {"status": "skipped", "reason": "no_summary_generated"}

    topics = extract_topics_from_messages(request.messages)
    sent = await send_summary_to_api(
        session_id=request.session_id,
        summary_text=summary_text,
        key_topics=topics,
        agents_used=request.agents_used,
    )

    return {
        "status": "ok" if sent else "generated_not_saved",
        "summary_length": len(summary_text),
        "topics": topics,
    }
