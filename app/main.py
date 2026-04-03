"""
ai-chat — Tanya Peri AI Service
"""
import json
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.peri_agent import _build_initial_state, _build_rnd_state, run_agent
from app.config.settings import settings
from app.schemas.chat import (
    ChatRequest,
    RnDChatRequest,
    make_metrics_event,
)
from app.services.llm_logger import send_llm_call_logs

app = FastAPI(
    title="Tanya Peri — AI Chat Service",
    description=(
        "Internal AI service untuk chatbot Tanya Peri. "
        "Endpoint /chat/rnd tersedia untuk RnD jika RND_MODE=True."
    ),
    version="1.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Security
# =============================================================================

def _verify_internal_secret(x_internal_secret: str | None) -> None:
    if not settings.INTERNAL_SECRET:
        return
    if x_internal_secret != settings.INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid internal secret")


def _require_rnd_mode() -> None:
    if not settings.RND_MODE:
        raise HTTPException(
            status_code=403,
            detail="RnD mode tidak aktif. Set RND_MODE=True di .env.",
        )


# =============================================================================
# Stream wrapper
# =============================================================================

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
        except (json.JSONDecodeError, Exception):
            pass

    if llm_call_logs:
        import asyncio
        asyncio.create_task(send_llm_call_logs(llm_call_logs, session_id))


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "tanya-peri-ai-chat",
        "provider": settings.LLM_PROVIDER,
        "model": settings.llm_model_name,
        "rnd_mode": settings.RND_MODE,
    }


@app.get("/health/gpu")
async def health_gpu():
    """Info GPU/device — untuk debugging performa embedding."""
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
            info["note"] = "CUDA tidak tersedia — embedding berjalan di CPU"
    except ImportError:
        info["torch_installed"] = False
    return info


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_internal_secret: str | None = Header(default=None),
):
    _verify_internal_secret(x_internal_secret)
    initial_state = _build_initial_state(request)
    return StreamingResponse(
        _stream_with_logging(initial_state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/chat/rnd",
    summary="RnD endpoint — standalone testing & research",
    description=(
        "Endpoint untuk RnD dan eksperimen. Tidak perlu internal secret.\n\n"
        "**stream: false** → return JSON (cocok untuk Swagger & batch experiment)\n"
        "**stream: true** → SSE streaming"
    ),
)
async def chat_rnd(request: RnDChatRequest):
    _require_rnd_mode()
    initial_state = _build_rnd_state(request)

    if request.stream:
        return StreamingResponse(
            _rnd_stream_with_metrics(initial_state, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        return await _rnd_collect_response(initial_state, request)


# =============================================================================
# Benchmark endpoint
# =============================================================================

class BenchmarkRequest(BaseModel):
    """
    Kirim pertanyaan yang sama N kali untuk melihat latency pattern.

    Cara baca hasil:
    - Kalau cold TTFT jauh lebih tinggi dari warm → GPU aktif, ada cold start
    - Kalau semua TTFT hampir sama dan rendah → model sudah warm sebelum benchmark
      (ini BAGUS, artinya keep_alive aktif atau model baru saja dipakai)
    - Kalau semua TTFT tinggi dan konsisten → kemungkinan CPU, tidak ada GPU warmup
    """
    message: str = Field(..., description="Pertanyaan yang akan dikirim N kali")
    n: int = Field(default=3, ge=1, le=10, description="Jumlah run (default: 3)")
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=0.1)
    max_tokens: Optional[int] = Field(default=200)


@app.post(
    "/chat/rnd/benchmark",
    summary="Benchmark latency — analisa warm/cold pattern",
    description=(
        "Kirim pertanyaan yang sama N kali dan bandingkan latency setiap run.\n\n"
        "**Cara baca:**\n"
        "- `ttft_cold_ms` tinggi, `ttft_warm_avg_ms` rendah → GPU aktif, cold start normal\n"
        "- Semua TTFT rendah dan konsisten → model sudah warm (keep_alive aktif) — ini normal & bagus\n"
        "- Semua TTFT tinggi dan konsisten → kemungkinan CPU\n\n"
        "Aktif hanya jika RND_MODE=True."
    ),
)
async def benchmark(request: BenchmarkRequest):
    _require_rnd_mode()

    results = []
    for i in range(request.n):
        rnd_request = RnDChatRequest(
            message=request.message,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            experiment_id=f"benchmark-run-{i+1}",
            experiment_tags=[f"run-{i+1}", "benchmark"],
        )
        state = _build_rnd_state(rnd_request)
        result = await _rnd_collect_response_raw(state, rnd_request)
        result["run"] = i + 1
        result["is_cold"] = (i == 0)
        results.append(result)

    latencies = [r["metrics"]["latency_ms"] for r in results if r["metrics"].get("latency_ms")]
    ttfts = [r["metrics"]["ttft_ms"] for r in results if r["metrics"].get("ttft_ms")]
    tps_list = [r["metrics"]["tokens_per_second"] for r in results if r["metrics"].get("tokens_per_second")]

    cold_ttft = ttfts[0] if ttfts else None
    warm_ttfts = ttfts[1:] if len(ttfts) > 1 else []
    warm_ttft_avg = int(sum(warm_ttfts) / len(warm_ttfts)) if warm_ttfts else None

    cold_latency = latencies[0] if latencies else None
    warm_latencies = latencies[1:] if len(latencies) > 1 else []
    warm_avg = int(sum(warm_latencies) / len(warm_latencies)) if warm_latencies else None

    summary = {
        "model": request.model or settings.llm_model_name,
        "provider": request.provider or settings.LLM_PROVIDER,
        "n_runs": request.n,
        "cold_latency_ms": cold_latency,
        "warm_avg_latency_ms": warm_avg,
        "speedup_warm_vs_cold": round(cold_latency / warm_avg, 2) if cold_latency and warm_avg else None,
        "ttft_cold_ms": cold_ttft,
        "ttft_warm_avg_ms": warm_ttft_avg,
        "avg_tokens_per_second": round(sum(tps_list) / len(tps_list), 1) if tps_list else None,
        "interpretation": _interpret_benchmark(cold_ttft, warm_ttft_avg, tps_list),
    }

    return {"summary": summary, "runs": results}


def _interpret_benchmark(cold_ttft_ms, warm_ttft_avg_ms, tps_list) -> str:
    """
    Interpretasi hasil benchmark berdasarkan TTFT pattern, bukan latency ratio.

    TTFT adalah indikator paling reliable:
    - TTFT tinggi di cold run = model di-load ke VRAM (cold start)
    - TTFT rendah di semua run = model sudah warm (keep_alive aktif / baru dipakai)
    - TTFT tinggi di semua run = kemungkinan CPU (tidak ada GPU warmup effect)
    """
    if cold_ttft_ms is None:
        return "Data tidak cukup untuk interpretasi."

    avg_tps = round(sum(tps_list) / len(tps_list), 1) if tps_list else 0

    # Analisa throughput dulu — ini indikator GPU paling kuat
    if avg_tps >= 80:
        gpu_verdict = "✅ GPU aktif"
        gpu_detail = f"throughput {avg_tps} tok/s (typical GPU range untuk model kecil)"
    elif avg_tps >= 30:
        gpu_verdict = "⚠️ Mungkin GPU"
        gpu_detail = f"throughput {avg_tps} tok/s (bisa GPU low-end atau CPU modern)"
    elif avg_tps > 0:
        gpu_verdict = "❌ Kemungkinan CPU"
        gpu_detail = f"throughput {avg_tps} tok/s (terlalu lambat untuk GPU)"
    else:
        gpu_verdict = "❓ Tidak dapat dideteksi"
        gpu_detail = "data throughput tidak tersedia"

    # Analisa TTFT pattern
    if warm_ttft_avg_ms is not None and cold_ttft_ms > warm_ttft_avg_ms * 3:
        warm_status = f"Cold start terdeteksi (TTFT cold {cold_ttft_ms}ms vs warm {warm_ttft_avg_ms}ms)"
    elif cold_ttft_ms < 300:
        warm_status = f"Model sudah warm dari awal (TTFT cold {cold_ttft_ms}ms) — keep_alive aktif atau baru dipakai"
    else:
        warm_status = f"TTFT {cold_ttft_ms}ms"

    # Speed verdict
    effective_latency = warm_ttft_avg_ms or cold_ttft_ms or 9999
    if effective_latency < 300:
        speed = "🚀 Sangat responsif untuk UX streaming"
    elif effective_latency < 800:
        speed = "✅ Responsif"
    elif effective_latency < 2000:
        speed = "⚠️ Agak lambat untuk streaming UX"
    else:
        speed = "❌ Terlalu lambat untuk streaming UX"

    return f"{gpu_verdict} — {gpu_detail}. {warm_status}. {speed}."


# =============================================================================
# RnD helpers
# =============================================================================

async def _rnd_stream_with_metrics(state, request: RnDChatRequest) -> AsyncIterator[str]:
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
    metrics = _build_rnd_metrics(state, request, final_metadata)
    yield make_metrics_event(metrics)


async def _rnd_collect_response(state, request: RnDChatRequest) -> JSONResponse:
    data = await _rnd_collect_response_raw(state, request)
    return JSONResponse(content=data)


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

    metrics = _build_rnd_metrics(state, request, final_metadata)
    result = {
        "response": full_content,
        "intent": state.get("intent"),
        "thinking_steps": state.get("thinking_steps", []),
        "tool_calls": state.get("tool_calls", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "metadata": final_metadata,
        "metrics": metrics,
    }
    if request.include_prompt_in_response and state.get("prompt_debug"):
        result["prompt_debug"] = state["prompt_debug"]
    return result


def _build_rnd_metrics(state, request: RnDChatRequest, metadata: dict) -> dict:
    metrics: dict = {
        "latency_ms": metadata.get("latency_ms"),
        "ttft_ms": metadata.get("ttft_ms"),
        "generation_ms": metadata.get("generation_ms"),
        "tokens_per_second": metadata.get("tokens_per_second"),
        "output_tokens_approx": metadata.get("output_tokens_approx"),
        "intent_detected": state.get("intent"),
        "model": metadata.get("model"),
        "provider": metadata.get("provider"),
        "experiment_id": request.experiment_id,
        "experiment_tags": request.experiment_tags,
        "llm_call_count": len(state.get("llm_call_logs", [])),
        "rag_docs_retrieved": len(state.get("retrieved_docs", [])),
    }
    if request.expected_intent:
        metrics["intent_correct"] = (state.get("intent") == request.expected_intent)
        metrics["expected_intent"] = request.expected_intent
    if request.expected_keywords:
        response_text = (state.get("final_response") or "").lower()
        found = [kw for kw in request.expected_keywords if kw.lower() in response_text]
        missing = [kw for kw in request.expected_keywords if kw.lower() not in response_text]
        metrics["keyword_hit_rate"] = round(len(found) / len(request.expected_keywords), 3)
        metrics["keywords_found"] = found
        metrics["keywords_missing"] = missing
    call_logs = state.get("llm_call_logs", [])
    if call_logs:
        metrics["llm_calls"] = [
            {
                "node": log.get("node"),
                "latency_ms": log.get("latency_ms"),
                "ttft_ms": log.get("ttft_ms"),
                "generation_ms": (log.get("metadata") or {}).get("generation_ms"),
                "tokens_per_second": (log.get("metadata") or {}).get("tokens_per_second"),
                "output_tokens": log.get("output_tokens"),
                "success": log.get("success"),
            }
            for log in call_logs
        ]
    return metrics
