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
    # Phase 1: _AGENT_REGISTRY moved from graph.py to nodes/agent_dispatcher.py
    from app.agents.nodes.agent_dispatcher import _AGENT_REGISTRY
    return {"status": "ok", "registered_agents": list(_AGENT_REGISTRY.keys()), "agent_count": len(_AGENT_REGISTRY)}


# =============================================================================
# Phase 1 — LangGraph checkpointer lifecycle hooks
# =============================================================================


@app.on_event("startup")
async def startup_checkpointer():
    """Warm up LangGraph PostgresSaver checkpointer pool + ensure tables exist."""
    try:
        from app.agents.memory.checkpointer import get_checkpointer
        cp = await get_checkpointer()
        if cp is not None:
            log.info("checkpointer_ready", status="ok")
        else:
            log.warning("checkpointer_unavailable",
                       status="degraded",
                       msg="Tanya Peri akan jalan tanpa state persistence")
    except Exception as e:
        log.error("checkpointer_init_failed", error=str(e))


@app.on_event("shutdown")
async def shutdown_checkpointer_hook():
    """Cleanly close checkpointer pool."""
    try:
        from app.agents.memory.checkpointer import shutdown_checkpointer
        await shutdown_checkpointer()
        log.info("checkpointer_shutdown", status="ok")
    except Exception as e:
        log.error("checkpointer_shutdown_failed", error=str(e))


@app.get("/health/checkpointer")
async def health_checkpointer():
    """Health check khusus untuk LangGraph checkpointer (Phase 1)."""
    from app.agents.memory.checkpointer import checkpointer_healthy
    healthy = await checkpointer_healthy()
    return {
        "status": "ok" if healthy else "degraded",
        "checkpointer_healthy": healthy,
        "note": (
            "Checkpointer healthy = state persistence aktif."
            if healthy
            else "Checkpointer not available — chat masih jalan tanpa persistence."
        ),
    }


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
    # FIX (Langfuse audit Bagian B4): Wrap memory job dengan parent span.
    # Sebelumnya: memory_summarize jalan sebagai background task tanpa parent span.
    # Trace muncul as "Unnamed trace" dengan ChatGoogleGenerativeAI orphan + http
    # call orphan, TIDAK linked ke main session via session_id.
    # Setelah fix: span "job:memory-summarize" muncul as proper parent + linked ke
    # session via update_trace, supaya bisa di-search "tampilkan semua trace untuk
    # session X" dan dapat memory summarize trace yang related.
    from app.config.observability import get_langfuse_client

    langfuse = get_langfuse_client()

    if langfuse is None:
        # Langfuse disabled — execute without trace wrapping
        return await _do_memory_summarize(
            request,
            generate_session_summary,
            extract_topics_from_messages,
            send_summary_to_api,
        )

    # Langfuse enabled — wrap with parent span
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="job:memory-summarize",
            input={
                "session_id": request.session_id,
                "messages_count": len(request.messages),
                "agents_used": request.agents_used,
            },
        ) as span:
            # CRITICAL: Link orphan trace ke main session_id supaya bisa di-search
            # bersama main `tanya-peri-message` trace di Langfuse UI.
            try:
                span.update_trace(
                    session_id=request.session_id,
                    tags=["memory-summary", "background-job"],
                    metadata={
                        "agents_used": request.agents_used,
                        "messages_count": len(request.messages),
                    },
                )
            except Exception:
                pass  # defensive

            result = await _do_memory_summarize(
                request,
                generate_session_summary,
                extract_topics_from_messages,
                send_summary_to_api,
            )

            # Capture output for span
            try:
                span.update(output={
                    "status": result.get("status"),
                    "summary_length": result.get("summary_length", 0),
                    "topics_count": len(result.get("topics", []) or []),
                    "topics": result.get("topics", [])[:10],  # cap di 10 topics
                })
            except Exception:
                pass

            return result
    except Exception as e:
        # Trace setup failed — fallback to non-traced execution
        # (jangan break flow karena tracing error)
        return await _do_memory_summarize(
            request,
            generate_session_summary,
            extract_topics_from_messages,
            send_summary_to_api,
        )


async def _do_memory_summarize(
    request,
    generate_session_summary,
    extract_topics_from_messages,
    send_summary_to_api,
):
    """Pure logic — no trace setup. Extracted untuk B4 fix supaya bisa
    di-call dengan/tanpa parent span wrapping."""
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


# =============================================================================
# Knowledge Base Management (Qdrant)
# =============================================================================
# Endpoint ini dipanggil dari peri-bugi-api (via internal secret) atau
# langsung dari super admin panel (via proxy di api).
#
# GET  /knowledge/collections        → info semua collection (doc count)
# GET  /knowledge/documents          → list dokumen per collection (dengan pagination)
# POST /knowledge/documents          → ingest teks/PDF chunk manual
# POST /knowledge/documents/upload   → upload & ingest PDF file
# DELETE /knowledge/documents/{id}   → hapus satu dokumen by point ID
# DELETE /knowledge/documents        → hapus semua dokumen di collection (clear)
# =============================================================================


def _get_kb_qdrant_client():
    """Get Qdrant client untuk knowledge base operations."""
    import warnings
    from qdrant_client import QdrantClient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )


def _get_kb_embeddings():
    """Get embedding model untuk knowledge base operations."""
    from app.agents.sub_agents import _get_embeddings
    return _get_embeddings()


class CollectionInfoResponse(BaseModel):
    collection: str
    display_name: str
    doc_count: int
    exists: bool


class DocumentItem(BaseModel):
    id: str
    content: str
    source: str
    metadata: dict


class IngestTextRequest(BaseModel):
    content: str = Field(..., description="Teks yang akan di-index")
    source: str = Field(..., description="Nama sumber, misal: 'WHO_2024' atau 'manual_input'")
    collection: str = Field(default="dental", description="'dental' atau 'faq'")
    doc_type: Optional[str] = Field(default=None, description="Tipe dokumen: guideline | article | faq | manual")
    metadata: Optional[dict] = Field(default=None, description="Metadata tambahan")


class DeleteDocumentsRequest(BaseModel):
    collection: str = Field(default="dental", description="'dental' atau 'faq'")


@app.get(
    "/knowledge/collections",
    summary="Info semua Qdrant collections (jumlah dokumen)",
)
async def get_collections(x_internal_secret: str | None = Header(default=None)):
    _verify_internal_secret(x_internal_secret)

    collections_info = []
    display_map = {
        settings.QDRANT_COLLECTION: "Knowledge Base Dental",
        settings.QDRANT_FAQ_COLLECTION: "FAQ Aplikasi",
    }

    try:
        client = _get_kb_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}

        for col_name, display_name in display_map.items():
            if col_name in existing:
                info = client.get_collection(col_name)
                doc_count = info.points_count or 0
            else:
                doc_count = 0

            collections_info.append(CollectionInfoResponse(
                collection=col_name,
                display_name=display_name,
                doc_count=doc_count,
                exists=col_name in existing,
            ))
    except Exception as e:
        log.error("get_collections error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")

    return {"collections": [c.model_dump() for c in collections_info]}


@app.get(
    "/knowledge/documents",
    summary="List dokumen di collection (pagination + optional metadata filter)",
)
async def list_documents(
    collection: str = "dental",
    limit: int = 20,
    offset: int = 0,
    filter_key: Optional[str] = None,
    filter_value: Optional[str] = None,
    is_active: Optional[bool] = None,
    x_internal_secret: str | None = Header(default=None),
):
    _verify_internal_secret(x_internal_secret)

    # Resolve collection name
    col_name = settings.QDRANT_COLLECTION if collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        client = _get_kb_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}

        if col_name not in existing:
            return {"documents": [], "total": 0, "collection": col_name, "offset": offset, "limit": limit}

        # Phase 4.1: build optional Qdrant filter
        # filter_key + filter_value → metadata.{key} = value
        # is_active → metadata.is_active = bool
        scroll_filter = None
        if filter_key and filter_value is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            must_conditions = [
                FieldCondition(key=f"metadata.{filter_key}", match=MatchValue(value=filter_value))
            ]
            if is_active is not None:
                must_conditions.append(
                    FieldCondition(key="metadata.is_active", match=MatchValue(value=is_active))
                )
            scroll_filter = Filter(must=must_conditions)
        elif is_active is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            scroll_filter = Filter(
                must=[FieldCondition(key="metadata.is_active", match=MatchValue(value=is_active))]
            )

        # Qdrant scroll untuk list semua points dengan pagination
        scroll_kwargs = {
            "collection_name": col_name,
            "limit": limit,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
        }
        if scroll_filter is not None:
            scroll_kwargs["scroll_filter"] = scroll_filter

        scroll_result = client.scroll(**scroll_kwargs)
        points = scroll_result[0]

        # Count total — kalau ada filter, count masih global (Qdrant scroll tidak return total).
        # Untuk simplicity tetap pakai global count + kasih flag filter applied.
        col_info = client.get_collection(col_name)
        total = col_info.points_count or 0

        documents = []
        for pt in points:
            payload = pt.payload or {}
            metadata_dict = payload.get("metadata", {}) or {}
            documents.append({
                "id": str(pt.id),
                "content": payload.get("page_content", payload.get("content", ""))[:500],
                "content_full": payload.get("page_content", payload.get("content", "")),
                "source": metadata_dict.get("source", payload.get("source", "unknown")),
                "doc_type": metadata_dict.get("doc_type", "-"),
                "page": metadata_dict.get("page"),
                "chunk_idx": metadata_dict.get("chunk_idx"),
                "is_active": metadata_dict.get("is_active", True),  # Default true (legacy chunks)
                "feature": metadata_dict.get("feature"),
                "category": metadata_dict.get("category"),
                "metadata": metadata_dict,
            })

        return {
            "documents": documents,
            "total": total,
            "collection": col_name,
            "offset": offset,
            "limit": limit,
            "filter_applied": {
                "filter_key": filter_key,
                "filter_value": filter_value,
                "is_active": is_active,
            } if (filter_key or is_active is not None) else None,
        }

    except Exception as e:
        log.error("list_documents error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")


@app.post(
    "/knowledge/documents",
    summary="Ingest teks manual ke knowledge base",
)
async def ingest_text(
    request: IngestTextRequest,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Ingest satu potongan teks ke Qdrant.
    Berguna untuk input manual atau konten yang bukan PDF.
    Source wajib diisi untuk traceability.
    """
    _verify_internal_secret(x_internal_secret)

    import warnings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from langchain_core.documents import Document

    col_name = settings.QDRANT_COLLECTION if request.collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        embeddings = _get_kb_embeddings()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = _get_kb_qdrant_client()

        # Pastikan collection ada
        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            test_vec = embeddings.embed_query("test")
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=len(test_vec), distance=Distance.COSINE),
            )

        # Buat Document dengan metadata lengkap
        metadata = {
            "source": request.source,
            "doc_type": request.doc_type or "manual",
            "ingested_via": "manual_text",
            **(request.metadata or {}),
        }
        doc = Document(page_content=request.content, metadata=metadata)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=col_name,
            embedding=embeddings,
        )
        ids = vector_store.add_documents([doc])

        return {
            "status": "ok",
            "message": f"Teks berhasil di-ingest ke collection '{col_name}'.",
            "doc_id": ids[0] if ids else None,
            "collection": col_name,
            "source": request.source,
        }

    except Exception as e:
        log.error("ingest_text error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


@app.post(
    "/knowledge/documents/upload",
    summary="Upload & ingest PDF file ke knowledge base",
)
async def upload_pdf(
    x_internal_secret: str | None = Header(default=None),
    collection: str = "dental",
    chunk_size: int = 400,
    chunk_overlap: int = 40,
):
    """
    Endpoint ini menerima multipart/form-data dengan field 'file' (PDF).
    Proses: save temp → load → split → embed → ingest ke Qdrant.
    Metadata source otomatis diisi dengan nama file.
    """
    _verify_internal_secret(x_internal_secret)
    # Import di sini karena UploadFile dari fastapi
    import tempfile
    import os
    import warnings
    from fastapi import UploadFile, File, Form
    raise HTTPException(
        status_code=501,
        detail="Endpoint ini belum diimplementasi. Gunakan /knowledge/documents untuk ingest teks, atau jalankan scripts/ingest_pdf.py dari container."
    )


@app.delete(
    "/knowledge/documents/{point_id}",
    summary="Hapus satu dokumen dari collection by ID",
)
async def delete_document(
    point_id: str,
    collection: str = "dental",
    x_internal_secret: str | None = Header(default=None),
):
    """
    Hapus satu dokumen dari Qdrant by point ID.
    ID bisa didapat dari GET /knowledge/documents.
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        client = _get_kb_qdrant_client()

        # Coba parse sebagai integer (Qdrant bisa pakai integer atau UUID)
        try:
            parsed_id = int(point_id)
        except ValueError:
            parsed_id = point_id

        from qdrant_client.models import PointIdsList
        client.delete(
            collection_name=col_name,
            points_selector=PointIdsList(points=[parsed_id]),
        )

        return {
            "status": "ok",
            "message": f"Dokumen '{point_id}' berhasil dihapus dari '{col_name}'.",
        }

    except Exception as e:
        log.error("delete_document error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.delete(
    "/knowledge/documents",
    summary="Hapus semua dokumen di collection (clear)",
)
async def clear_collection(
    request: DeleteDocumentsRequest,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Hapus semua dokumen di collection. Collection-nya sendiri tidak dihapus.
    DESTRUCTIVE — tidak bisa di-undo. Pastikan sebelum eksekusi.
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if request.collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        client = _get_kb_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}

        if col_name not in existing:
            return {"status": "ok", "message": f"Collection '{col_name}' tidak ada, tidak ada yang dihapus."}

        before_count = client.get_collection(col_name).points_count or 0

        from qdrant_client.models import Filter
        client.delete(collection_name=col_name, points_selector=Filter())

        return {
            "status": "ok",
            "message": f"Collection '{col_name}' berhasil dikosongkan.",
            "deleted_count": before_count,
        }

    except Exception as e:
        log.error("clear_collection error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


# =============================================================================
# Knowledge Base — Upload PDF (real implementation, replace 501 stub)
# =============================================================================

@app.post(
    "/knowledge/upload",
    summary="Upload & ingest PDF file (multipart/form-data) — endpoint aktif",
)
async def upload_pdf_real(
    request: Request,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Terima PDF via multipart/form-data, split jadi chunks, embed, simpan ke Qdrant.

    Form fields:
      file         — binary PDF content (required)
      filename     — nama file asli (required)
      source       — nama sumber untuk referensi (required)
      collection   — 'dental' atau 'faq' (default: dental)
      doc_type     — guideline | article | faq | manual (default: guideline)
      chunk_size   — integer (default: 400)
      chunk_overlap— integer (default: 40)
    """
    _verify_internal_secret(x_internal_secret)

    import os
    import tempfile
    import warnings
    import time
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    # Parse multipart body
    form = await request.form()
    file_field = form.get("file")
    filename = form.get("filename", "upload.pdf")
    source = form.get("source", "")
    collection_key = form.get("collection", "dental")
    doc_type = form.get("doc_type", "guideline")
    chunk_size = int(form.get("chunk_size", 400))
    chunk_overlap = int(form.get("chunk_overlap", 40))

    if not file_field:
        raise HTTPException(status_code=400, detail="Field 'file' wajib diisi.")
    if not source:
        raise HTTPException(status_code=400, detail="Field 'source' wajib diisi.")

    col_name = settings.QDRANT_COLLECTION if collection_key == "dental" else settings.QDRANT_FAQ_COLLECTION

    # Baca bytes dari UploadFile atau bytes langsung
    if hasattr(file_field, "read"):
        file_bytes = await file_field.read()
    else:
        file_bytes = file_field

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File kosong.")

    # Simpan ke temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        if not pages:
            raise HTTPException(status_code=422, detail="PDF tidak bisa dibaca atau kosong.")

        # Split chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)

        # Tambah metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = source
            chunk.metadata["filename"] = str(filename)
            chunk.metadata["doc_type"] = doc_type
            chunk.metadata["chunk_idx"] = i
            chunk.metadata["ingested_via"] = "ui_upload"

        # Embed + ingest
        embeddings = _get_kb_embeddings()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = _get_kb_qdrant_client()

        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            test_vec = embeddings.embed_query("test")
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=len(test_vec), distance=Distance.COSINE),
            )

        # Ingest in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=col_name,
                embedding=embeddings,
            )
            vector_store.add_documents(batch)

        return {
            "status": "ok",
            "message": f"PDF '{filename}' berhasil di-ingest ke collection '{col_name}'.",
            "chunk_count": len(chunks),
            "page_count": len(pages),
            "collection": col_name,
            "source": source,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("upload_pdf_real error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get(
    "/knowledge/embedding-info",
    summary="Info embedding model yang sedang aktif",
)
async def get_embedding_info(x_internal_secret: str | None = Header(default=None)):
    """Kembalikan embedding provider, model, dan device yang sedang aktif."""
    _verify_internal_secret(x_internal_secret)
    return {
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_device": settings.EMBEDDING_DEVICE,
    }


@app.get(
    "/knowledge/sources",
    summary="List unique source names di collection",
)
async def get_sources(
    collection: str = "dental",
    x_internal_secret: str | None = Header(default=None),
):
    """
    Return daftar unique source names di collection.
    Dipakai FE untuk filter bulk delete by source.
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        client = _get_kb_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            return {"sources": [], "collection": col_name}

        # Scroll semua points untuk kumpulkan unique sources
        # Untuk collection besar, gunakan batch scroll
        all_sources: dict[str, int] = {}  # source -> count
        offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=col_name,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = scroll_result
            for pt in points:
                payload = pt.payload or {}
                source = payload.get("metadata", {}).get("source") or payload.get("source", "unknown")
                all_sources[source] = all_sources.get(source, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        sources_list = [
            {"source": s, "chunk_count": c}
            for s, c in sorted(all_sources.items(), key=lambda x: x[0])
        ]
        return {"sources": sources_list, "collection": col_name}

    except Exception as e:
        log.error("get_sources error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")


class DeleteBySourceRequest(BaseModel):
    sources: list[str] = Field(..., description="List nama source yang akan dihapus")
    collection: str = Field(default="dental", description="'dental' atau 'faq'")


@app.delete(
    "/knowledge/documents/by-source",
    summary="Hapus semua chunk dari satu atau beberapa source",
)
async def delete_by_source(
    request: DeleteBySourceRequest,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Hapus semua chunk yang metadata.source-nya cocok dengan list yang diberikan.
    Jauh lebih efisien daripada hapus satu per satu untuk PDF besar.
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if request.collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        client = _get_kb_qdrant_client()

        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            return {"status": "ok", "message": "Collection tidak ada.", "deleted_sources": []}

        before_count = client.get_collection(col_name).points_count or 0

        # Hapus dengan filter metadata.source
        # Qdrant payload filter: key "metadata.source" untuk nested, atau "source" untuk flat
        deleted_count = 0
        for source in request.sources:
            # Coba filter nested metadata.source dulu
            try:
                client.delete(
                    collection_name=col_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="metadata.source", match=MatchAny(any=[source]))]
                    ),
                )
            except Exception:
                pass
            # Juga hapus dengan flat source (untuk dokumen lama)
            try:
                client.delete(
                    collection_name=col_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="source", match=MatchAny(any=[source]))]
                    ),
                )
            except Exception:
                pass
            deleted_count += 1

        after_count = client.get_collection(col_name).points_count or 0
        removed = before_count - after_count

        return {
            "status": "ok",
            "message": f"{deleted_count} source dihapus, ~{removed} chunk diremove dari '{col_name}'.",
            "deleted_sources": request.sources,
            "chunks_before": before_count,
            "chunks_after": after_count,
        }

    except Exception as e:
        log.error("delete_by_source error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


# =============================================================================
# Phase 4.1 — Knowledge Base: Toggle is_active per chunk (or bulk per feature)
# =============================================================================

class UpdateMetadataRequest(BaseModel):
    """Request body untuk PATCH /knowledge/documents/{point_id}."""
    collection: str = Field(default="dental", description="'dental' atau 'faq'")
    metadata_updates: dict = Field(..., description="Field metadata yang mau di-update, e.g. {'is_active': False}")


class BulkToggleRequest(BaseModel):
    """Request body untuk PATCH /knowledge/documents/bulk-toggle."""
    collection: str = Field(default="faq", description="'dental' atau 'faq'")
    filter_key: str = Field(..., description="Metadata key untuk filter, e.g. 'feature' atau 'category'")
    filter_value: str = Field(..., description="Value untuk match, e.g. 'mata_peri' atau 'prevention'")
    is_active: bool = Field(..., description="Set semua matching chunks ke active=true atau false")


# NOTE Phase 4.1: bulk-toggle endpoint MUST be defined BEFORE /{point_id}
# Otherwise FastAPI matches "bulk-toggle" as point_id (anything matches dynamic param).
# Route registration order matters in FastAPI — first match wins.
@app.patch(
    "/knowledge/documents/bulk-toggle",
    summary="Bulk toggle is_active untuk semua chunks dengan metadata key/value tertentu",
)
async def bulk_toggle_active(
    request: BulkToggleRequest,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Bulk update field is_active untuk semua chunks yang match.

    Contoh use case:
    - Disable semua FAQ Janji Peri sekaligus:
        {"collection": "faq", "filter_key": "feature", "filter_value": "janji_peri", "is_active": false}
    - Enable semua dental KB kategori prevention:
        {"collection": "dental", "filter_key": "category", "filter_value": "prevention", "is_active": true}

    Note: filter_key adalah nama field di metadata (tanpa prefix "metadata.").
    Internal akan compose ke "metadata.{filter_key}".
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if request.collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = _get_kb_qdrant_client()

        existing = {c.name for c in client.get_collections().collections}
        if col_name not in existing:
            return {
                "status": "ok",
                "message": f"Collection '{col_name}' tidak ada.",
                "updated_count": 0,
            }

        # Scroll semua matching points untuk hitung dulu + dapat ID-nya
        # Pakai filter metadata.{key} = value
        metadata_key = f"metadata.{request.filter_key}"
        match_filter = Filter(
            must=[
                FieldCondition(key=metadata_key, match=MatchValue(value=request.filter_value))
            ]
        )

        # Scroll dengan pagination — ambil semua IDs
        all_point_ids: list = []
        next_offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=col_name,
                scroll_filter=match_filter,
                limit=200,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = scroll_result
            for pt in points:
                # Update metadata: merge is_active ke metadata
                payload = pt.payload or {}
                existing_meta = payload.get("metadata", {}) or {}
                new_meta = {**existing_meta, "is_active": request.is_active}
                client.set_payload(
                    collection_name=col_name,
                    payload={"metadata": new_meta},
                    points=[pt.id],
                )
                all_point_ids.append(pt.id)

            if next_offset is None:
                break

        return {
            "status": "ok",
            "message": (
                f"{len(all_point_ids)} chunks di '{col_name}' ter-update "
                f"{request.filter_key}={request.filter_value} → is_active={request.is_active}."
            ),
            "updated_count": len(all_point_ids),
            "collection": col_name,
            "filter": f"{request.filter_key}={request.filter_value}",
            "is_active": request.is_active,
        }

    except Exception as e:
        log.error("bulk_toggle_active error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Bulk toggle error: {str(e)}")


@app.patch(
    "/knowledge/documents/{point_id}",
    summary="Update metadata satu dokumen (Phase 4.1 — toggle is_active per chunk)",
)
async def update_document_metadata(
    point_id: str,
    request: UpdateMetadataRequest,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Update field di metadata dokumen tertentu.
    Tipikal use case: toggle is_active=true/false untuk enable/disable
    chunk dari retrieval AI.

    metadata_updates di-merge ke metadata existing, tidak replace.
    Field yang tidak ada di request tetap utuh.
    """
    _verify_internal_secret(x_internal_secret)

    col_name = settings.QDRANT_COLLECTION if request.collection == "dental" else settings.QDRANT_FAQ_COLLECTION

    try:
        client = _get_kb_qdrant_client()

        # Parse point_id (Qdrant supports int or UUID)
        try:
            parsed_id = int(point_id)
        except ValueError:
            parsed_id = point_id

        # Get current point untuk verifikasi exists
        existing_points = client.retrieve(
            collection_name=col_name,
            ids=[parsed_id],
            with_payload=True,
        )
        if not existing_points:
            raise HTTPException(status_code=404, detail=f"Document '{point_id}' not found in '{col_name}'.")

        existing_payload = existing_points[0].payload or {}
        existing_metadata = existing_payload.get("metadata", {}) or {}

        # Merge updates ke metadata existing
        new_metadata = {**existing_metadata, **request.metadata_updates}

        # set_payload merge update — pakai set_payload dengan key path "metadata"
        client.set_payload(
            collection_name=col_name,
            payload={"metadata": new_metadata},
            points=[parsed_id],
        )

        return {
            "status": "ok",
            "message": f"Metadata dokumen '{point_id}' berhasil di-update.",
            "point_id": point_id,
            "collection": col_name,
            "updated_metadata": new_metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("update_document_metadata error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")
