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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.PERI_API_URL}/api/v1/internal/llm-call-logs",
                json={"logs": logs},
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Secret": settings.INTERNAL_SECRET,
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "llm_logger: api returned %s — logs tidak tersimpan",
                    resp.status_code,
                )
    except httpx.RequestError as e:
        # Silence — jangan block stream karena logging gagal
        logger.warning("llm_logger: gagal kirim ke api — %s", str(e))
    except Exception as e:
        logger.warning("llm_logger: unexpected error — %s", str(e))


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
