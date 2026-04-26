"""
L2 Memory Background Job — Session Summary Generator.

Dipanggil sebagai background task setelah session idle > 1 jam.
Generate ringkasan 3-5 kalimat dari percakapan, simpan ke chat_session_summaries.

Untuk Phase 2: LLM-generated summary (lebih akurat dari rule-based Phase 1).
"""
import logging
import time
from typing import Optional

from app.agents.state import AgentState
from app.config.llm import get_llm, get_model_name, get_provider_name
from app.config.settings import settings
from app.schemas.chat import LLMCallLogPayload

logger = logging.getLogger(__name__)


async def generate_session_summary(
    session_id: str,
    user_id: str,
    messages: list[dict],
    agents_used: list[str],
) -> Optional[str]:
    """
    Generate summary dari session menggunakan LLM kecil.
    Return summary text atau None jika gagal.

    Dipanggil sebagai background task dari peri-bugi-api setelah session idle.
    """
    if not messages:
        return None

    # Ambil hanya pesan user + assistant (skip system)
    conversation = [
        m for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if len(conversation) < 2:
        return None  # Terlalu pendek, tidak perlu summary

    # Build conversation text (max 3000 chars untuk hemat token)
    conv_text = ""
    for m in conversation[-10:]:  # Ambil 10 pesan terakhir
        role = "User" if m["role"] == "user" else "Tanya Peri"
        content = (m.get("content") or "")[:300]  # Truncate per pesan
        conv_text += f"{role}: {content}\n"
        if len(conv_text) > 3000:
            break

    prompt = (
        "Buat ringkasan singkat (3-4 kalimat) dari percakapan ini dalam Bahasa Indonesia. "
        "Fokus pada: topik utama yang dibahas, pertanyaan user, dan saran yang diberikan. "
        "Jangan tambahkan penilaian atau komentar. Langsung tulis ringkasannya.\n\n"
        f"Percakapan:\n{conv_text}\n\nRingkasan:"
    )

    start = time.monotonic()
    try:
        from langchain_core.messages import HumanMessage
        from app.config.observability import build_trace_config

        llm = get_llm(temperature=0.3, max_tokens=200, streaming=False)
        # Build pseudo-state untuk trace metadata (memory_job tidak punya AgentState)
        trace_config = build_trace_config(
            state={"session_id": session_id, "user_context": {"user_id": user_id}},
            agent_name="memory_summary",
        )
        result = await llm.ainvoke([HumanMessage(content=prompt)], config=trace_config)
        summary = result.content.strip()
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(f"Session summary generated: session={session_id} latency={latency_ms}ms")
        return summary if len(summary) > 20 else None
    except Exception as e:
        logger.warning(f"Failed to generate session summary: {e}")
        return None


async def send_summary_to_api(
    session_id: str,
    summary_text: str,
    key_topics: list[str],
    agents_used: list[str],
) -> bool:
    """
    Kirim summary ke peri-bugi-api internal endpoint untuk disimpan.
    Return True jika berhasil.
    """
    if not settings.PERI_API_URL:
        return False

    import httpx
    url = f"{settings.PERI_API_URL.rstrip('/')}/api/v1/internal/memory/summary"
    body = {
        "session_id": session_id,
        "summary_text": summary_text,
        "key_topics": key_topics,
        "agents_used": agents_used,
    }

    # Phase 3: trace HTTP call as child observation
    # Phase 4.5: pass full body for diagnostic visibility
    from app.config.observability import trace_http_call

    async with trace_http_call(
        name="http-internal-post-memory-summary",
        method="POST",
        url=url,
        body=body,  # Phase 4.5: full body (auto-redacted)
        body_keys=list(body.keys()),
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json=body,
                    headers={"X-Internal-Secret": settings.INTERNAL_SECRET},
                )
                ok = resp.status_code == 200
                if span:
                    span.update(output={"status_code": resp.status_code, "ok": ok})
                return ok
        except Exception as e:
            logger.warning(f"Failed to send summary to api: {e}")
            if span:
                span.update(
                    output={"error": str(e)[:200]},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
            return False


# Topic keywords untuk extract dari conversation (sama dengan memory_service.py di api)
_TOPIC_KEYWORDS = {
    "karies": ["karies", "berlubang", "lubang gigi"],
    "sikat gigi": ["sikat gigi", "gosok gigi", "pasta gigi"],
    "gigi susu": ["gigi susu", "gigi pertama", "gigi bayi"],
    "dokter gigi": ["dokter gigi", "drg", "klinik", "puskesmas"],
    "plak": ["plak", "karang gigi"],
    "gusi": ["gusi", "bengkak"],
    "fluoride": ["fluoride", "pasta gigi"],
    "cerita peri": ["cerita peri", "modul", "literatur"],
    "rapot peri": ["rapot", "streak", "sikat gigi anak"],
    "mata peri": ["scan gigi", "foto gigi", "hasil scan"],
}


def extract_topics_from_messages(messages: list[dict]) -> list[str]:
    """Extract topik dari pesan user."""
    all_text = " ".join(
        (m.get("content") or "").lower()
        for m in messages
        if m.get("role") == "user"
    )
    found = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in all_text for kw in keywords):
            found.append(topic)
    return found[:5]
