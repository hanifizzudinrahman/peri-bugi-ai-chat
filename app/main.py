"""
ai-chat — Tanya Peri AI Service
FastAPI app + endpoint /chat/stream
"""
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agents.peri_agent import _build_initial_state, run_agent
from app.config.settings import settings
from app.schemas.chat import ChatRequest

app = FastAPI(
    title="Tanya Peri — AI Chat Service",
    description=(
        "Internal AI service untuk chatbot Tanya Peri. "
        "Tidak diakses langsung oleh frontend — selalu melalui peri-bugi-api."
    ),
    version="1.0.0",
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
    }


@app.post(
    "/chat/stream",
    summary="Stream response Tanya Peri (SSE)",
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
        run_agent(initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
