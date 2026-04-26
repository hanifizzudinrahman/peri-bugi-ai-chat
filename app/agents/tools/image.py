"""
Tool: call_mata_peri (DEPRECATED — keep import-compatible only)

Modul ini DEPRECATED. Production flow image analysis sekarang lewat:

    graph.py → _AGENT_REGISTRY['mata_peri'] → sub_agents/phase2_agents.mata_peri_agent
             → _analyze_chat_image() → POST /api/v1/internal/agent/tanya-peri-analyze-image

Versi lama image_node() di file ini dulu langsung POST ke ai-cv /analyze
dengan payload {"image_url": ...}, tapi schema ai-cv require session_id,
child_id, analysis_job_id, views — jadi kalau ke-trigger PASTI 422.

Yang panggil image_node sekarang cuma peri_agent.py (sendiri-nya juga
deprecated — main.py udah migrate ke graph.py:run_agent). Karena import
masih ada di peri_agent.py, kita simpan stub aman supaya ImportError
tidak terjadi kalau peri_agent kebetulan di-import (e.g. test lama).

JANGAN delete file ini sebelum semua reference di peri_agent.py dihapus.
JANGAN tambah logic baru di sini — gunakan sub_agents/phase2_agents.

Roadmap cleanup:
1. (this PR) Stub: emit warning + skip, jangan crash, jangan call ai-cv.
2. (next PR) Delete peri_agent.py (verified main.py tidak impor).
3. (after that) Delete tools/image.py.
"""
import logging
from typing import AsyncIterator

from app.schemas.chat import AgentState, make_thinking_event

logger = logging.getLogger(__name__)


async def image_node(state: AgentState) -> AsyncIterator[str]:
    """
    DEPRECATED stub. NO-OP yang aman.

    Kalau ke-trigger, log warning supaya kita tahu ada code path yang
    masih reach sini, tapi TIDAK call ai-cv (karena payload schema legacy
    tidak match endpoint /analyze yang require session_id/child_id/etc).

    Tidak emit error event ke user — silently skip dan biarkan generate_node
    tetap jalan dengan state.image_analysis = None.
    """
    logger.warning(
        "[DEPRECATED] tools.image.image_node() ke-trigger. "
        "Production flow harus pakai sub_agents.phase2_agents.mata_peri_agent. "
        "Skip — set image_analysis=None dan continue."
    )

    # Set state seperti dulu (None = no analysis available) supaya
    # downstream node (generate_node) tidak crash kalau cek state.
    state["image_analysis"] = None
    state["tool_calls"].append({
        "tool": "call_mata_peri",
        "input": {"deprecated": True},
        "result": {"status": "skipped", "reason": "deprecated_path"},
    })

    # Yield satu thinking event supaya FE tidak hang menunggu (defensive
    # — semestinya path ini tidak pernah ke-reach di production).
    thinking_step = len(state.get("thinking_steps", [])) + 1
    yield make_thinking_event(
        step=thinking_step,
        label="Memproses gambar...",
        done=True,
    )
    state["thinking_steps"].append({
        "step": thinking_step,
        "label": "Memproses gambar...",
        "done": True,
    })
