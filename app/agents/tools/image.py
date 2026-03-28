"""
Tool: call_mata_peri
Forward gambar ke peri-bugi-ai-cv untuk inference.
Dipanggil untuk intent image.
"""
from typing import AsyncIterator

import httpx

from app.config.settings import settings
from app.schemas.chat import AgentState, make_thinking_event


async def image_node(state: AgentState) -> AsyncIterator[str]:
    """
    LangGraph node: call_mata_peri
    Kirim image_url ke ai-cv, ambil hasil inference.
    """
    thinking_step = len(state.get("thinking_steps", [])) + 1
    yield make_thinking_event(
        step=thinking_step,
        label="Menganalisis gambar gigi...",
        done=False,
    )

    image_url = state.get("image_url")
    if not image_url or not settings.AI_CV_URL:
        # Tidak ada gambar atau AI CV tidak tersedia
        state["image_analysis"] = None
        state["tool_calls"].append({
            "tool": "call_mata_peri",
            "input": {"image_url": image_url},
            "result": {"error": "image_url atau AI_CV_URL tidak tersedia"},
        })
        yield make_thinking_event(
            step=thinking_step,
            label="Menganalisis gambar gigi...",
            done=True,
        )
        state["thinking_steps"].append({
            "step": thinking_step,
            "label": "Menganalisis gambar gigi...",
            "done": True,
        })
        return

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{settings.AI_CV_URL}/analyze",
                json={"image_url": image_url},
                headers={"X-Internal-Secret": settings.INTERNAL_SECRET},
            )
            response.raise_for_status()
            result = response.json()

        state["image_analysis"] = result
        state["tool_calls"].append({
            "tool": "call_mata_peri",
            "input": {"image_url": image_url[:80]},
            "result": {"status": "success", "summary_status": result.get("summary_status")},
        })

    except httpx.HTTPError as e:
        state["image_analysis"] = None
        state["tool_calls"].append({
            "tool": "call_mata_peri",
            "input": {"image_url": image_url[:80]},
            "result": {"error": str(e)},
        })

    yield make_thinking_event(
        step=thinking_step,
        label="Menganalisis gambar gigi...",
        done=True,
    )
    state["thinking_steps"].append({
        "step": thinking_step,
        "label": "Menganalisis gambar gigi...",
        "done": True,
    })
