"""
LLM Call Logger Service

Kirim metadata LLM call ke peri-bugi-api untuk disimpan di tabel llm_call_logs.
Berjalan sebagai fire-and-forget (tidak blocking stream response ke user).

Kenapa fire-and-forget:
- Logging tidak boleh memperlambat response ke user
- Jika api tidak tersedia, chat tetap berjalan normal
- Retry bisa ditambahkan di background tanpa impact user
"""
import asyncio
import logging
from typing import Optional

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


async def send_llm_call_logs(
    logs: list[dict],
    session_id: Optional[str] = None,
) -> None:
    """
    Kirim list LLM call logs ke peri-bugi-api.
    Fire-and-forget — tidak raise exception ke caller.

    Dipanggil setelah streaming selesai (dari done event handler di main.py).
    """
    if not logs:
        return

    if not settings.PERI_API_URL:
        # Logging dinonaktifkan (misal: RnD standalone tanpa api)
        return

    # Inject session_id ke setiap log yang belum punya
    if session_id:
        for log in logs:
            if not log.get("session_id"):
                log["session_id"] = session_id

    url = f"{settings.PERI_API_URL}/api/v1/internal/llm-call-logs"

    # Phase 3: trace HTTP call as child observation.
    # NOTE: send_llm_call_logs dipanggil sebagai background task (asyncio.create_task)
    # SETELAH "done" event di-yield. Parent span (tanya-peri-message) sudah closed —
    # span ini akan jadi standalone trace.
    #
    # FIX (Langfuse audit): Link standalone trace ke main session via update_trace.
    # Sebelumnya: trace ini orphan, tidak filterable by session_id di Langfuse UI.
    # Setelah fix: walaupun trace terpisah, session_id di trace property level
    # supaya bisa di-search "tampilkan semua trace untuk session X".
    # Phase 4.5: pass body summary so log content visible per fire-and-forget trace
    from app.config.observability import trace_http_call

    # Build sanitized body — kirim summary, bukan full logs (bisa panjang)
    body_summary = {
        "log_count": len(logs),
        "first_log_node": logs[0].get("node") if logs else None,
        "first_log_model": logs[0].get("model") if logs else None,
    }

    async with trace_http_call(
        name="http-internal-post-llm-call-logs",
        method="POST",
        url=url,
        body=body_summary,  # Phase 4.5: body summary (not full logs to avoid bloat)
        body_keys=["logs"],
        metadata={"log_count": len(logs)},
    ) as span:
        # FIX (Langfuse audit): Link orphan trace ke main session_id.
        # Tanpa ini, trace ini tidak bisa di-search by session di Langfuse UI.
        if span and session_id:
            try:
                span.update_trace(
                    session_id=session_id,
                    tags=["fire-and-forget", "llm-logging"],
                )
            except Exception:
                # Defensive — jangan block flow kalau update_trace gagal
                pass

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={"logs": logs},
                    headers={
                        "Content-Type": "application/json",
                        "X-Internal-Secret": settings.INTERNAL_SECRET,
                    },
                )
                ok = resp.status_code in (200, 201)
                if span:
                    span.update(output={"status_code": resp.status_code, "ok": ok})
                if not ok:
                    logger.warning(
                        "llm_logger: api returned %s — logs tidak tersimpan",
                        resp.status_code,
                    )
        except httpx.RequestError as e:
            # Silence — jangan block stream karena logging gagal
            logger.warning("llm_logger: gagal kirim ke api — %s", str(e))
            if span:
                span.update(
                    output={"error": str(e)[:200]},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
        except Exception as e:
            logger.warning("llm_logger: unexpected error — %s", str(e))
            if span:
                span.update(
                    output={"error": str(e)[:200]},
                    level="ERROR",
                    status_message=str(e)[:200],
                )


def fire_and_forget_logs(logs: list[dict], session_id: Optional[str] = None) -> None:
    """
    Buat asyncio task untuk kirim logs tanpa await.
    Dipanggil dari context yang tidak bisa async (misal: generator callback).
    """
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(send_llm_call_logs(logs, session_id))
    except RuntimeError:
        # Tidak ada event loop — skip silently
        pass
