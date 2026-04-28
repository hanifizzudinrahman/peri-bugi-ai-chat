"""
Pre-router node — Phase 2 Step 2a + Step 2b

Runs FIRST before agent_node. Detects special cases that require deterministic
routing (no LLM decision needed):

1. Image upload + no view_hint → force `analyze_chat_image` tool
2. Clarification reply (user just answered ClarificationCard) → force `analyze_chat_image` again
   (with their selected view as input)
3. **(Step 2b NEW)** Smalltalk detection → set `state.is_smalltalk=True` so
   generate_node uses lean prompt path (skip scan/streak/memory injection).
   Detection uses STRICT criteria (Q4A decision):
     - regex match (pattern dari legacy Phase 1 router, sudah teruji)
     - word count <= 5
     - NO dental keyword (gigi, karies, plak, scan, streak, etc)

For all other cases (text-only chat with substantive question), pre_router
passes through with no forced routing and is_smalltalk=False. agent_node
will decide tools via LLM.

Why a deterministic pre-router instead of letting LLM decide:
- Image flow has been tuned + tested in legacy mata_peri_agent. We preserve
  that UX exactly, no LLM-based routing risk.
- Gemini bind_tools sometimes ignores image context — explicit forcing
  guarantees correct behavior.
- Faster: skip 1 LLM call when intent is clear from state.image being set.
- Smalltalk detection at pre_router level = no extra LLM call, deterministic.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agents.state import AgentState, ThinkingStep

logger = logging.getLogger(__name__)


# =============================================================================
# Smalltalk detection (Step 2b)
# =============================================================================
# Pattern dari legacy app/agents/nodes/router.py (Phase 1 router, sudah teruji).
# Anchored dengan ^ supaya match dari awal pesan, bukan di tengah.

_SMALLTALK_PATTERNS = [
    re.compile(r"^(halo|hai|hi|hey|assalam|pagi|siang|sore|malam)\b", re.IGNORECASE),
    re.compile(r"^apa kabar", re.IGNORECASE),
    re.compile(r"^siapa kamu", re.IGNORECASE),
    re.compile(r"^kamu (siapa|apa|bisa)", re.IGNORECASE),
    re.compile(r"^terima kasih", re.IGNORECASE),
    re.compile(r"^makasih", re.IGNORECASE),
    re.compile(r"^thanks", re.IGNORECASE),
    re.compile(r"^oke?$", re.IGNORECASE),
]

# Defensive: jika user mention salah satu kata ini, JANGAN klasifikasi sbg
# smalltalk meskipun match regex dan pendek. Lebih baik miss smalltalk
# (response masih valid) daripada false positive (dental question dapat
# response generic = bug visible).
_DENTAL_KEYWORDS = {
    "gigi", "karies", "berlubang", "plak", "karang",
    "sikat", "fluoride", "dokter", "behel", "gusi",
    "mulut", "nafas", "scan", "streak", "rapot",
    "rapor", "foto", "gambar", "rontgen", "kawat",
    "ortodonsi", "ortodontik",
}

_MAX_WORD_COUNT = 5  # threshold konservatif (Q4A decision)


def _detect_smalltalk(user_message: str) -> bool:
    """
    Strict smalltalk detection (Q4A decision):
      1. Regex match (minimal 1 dari _SMALLTALK_PATTERNS)
      2. Word count <= 5
      3. NO dental keyword

    Returns True only if ALL 3 criteria met.

    Examples:
      "halo peri"          → True (2 kata, regex match, no dental)
      "kenapa gigi sakit?" → False (no regex match + ada "gigi")
      "halo gigi anak"     → False (ada "gigi")
      "halo peri, gigi anak ada berapa?" → False (>5 kata + ada "gigi")
      "makasih ya"         → True
      "kamu siapa?"        → True
      ""                   → False (empty, image-only handled separately)
    """
    if not user_message or not user_message.strip():
        return False

    msg = user_message.strip()

    # Criterion 2: word count <= 5
    word_count = len(msg.split())
    if word_count > _MAX_WORD_COUNT:
        return False

    # Criterion 1: regex match minimal 1 pattern
    if not any(p.search(msg) for p in _SMALLTALK_PATTERNS):
        return False

    # Criterion 3: no dental keyword
    msg_lower = msg.lower()
    if any(kw in msg_lower for kw in _DENTAL_KEYWORDS):
        return False

    return True


def _extract_user_message(state: AgentState) -> str:
    """Extract last user message text from state.messages."""
    for msg in reversed(state.messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multimodal — find text part
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


# =============================================================================
# LangGraph node
# =============================================================================


async def pre_router_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: detect forced routing + smalltalk scenarios.

    Returns partial state update dict. Does NOT mutate input state.

    State updates:
    - thinking_steps: appended (step 1 = "Memahami pertanyaanmu...")
    - forced_tool_calls (NEW field, see state.py): list of dict
        Format: [{"name": "tool_name", "args": {}}]
        Empty list = no forced routing, agent_node free to decide.
    - is_smalltalk (Step 2b NEW): bool
        True = match smalltalk criteria → generate_node uses lean prompt path.
        False = normal flow.
    """
    new_thinking = ThinkingStep(step=1, label="Memahami pertanyaanmu...", done=True)
    thinking_update = [*state.thinking_steps, new_thinking]

    forced: list[dict] = []
    is_smalltalk: bool = False

    # ────────────────────────────────────────────────────────────────────────
    # Path 1: Image upload — force analyze_chat_image
    # Image flow takes precedence over smalltalk (image always means analysis).
    # ────────────────────────────────────────────────────────────────────────
    if state.image is not None and state.image.image_url:
        if "mata_peri" in (state.control.allowed_agents or []):
            forced.append({
                "name": "analyze_chat_image",
                "args": {},  # no LLM-controlled args; tool reads from closure
            })
            logger.info(
                f"[pre_router] Image detected, force analyze_chat_image. "
                f"clarification_selected={state.image.clarification_selected}"
            )
        else:
            logger.warning(
                "[pre_router] Image detected but mata_peri not in allowed_agents — "
                "agent_node will decide (image likely ignored)."
            )

    # ────────────────────────────────────────────────────────────────────────
    # Path 2: Smalltalk detection (Step 2b)
    # ONLY check if no image (image flow always wins).
    # ────────────────────────────────────────────────────────────────────────
    elif not forced:
        user_message = _extract_user_message(state)
        if _detect_smalltalk(user_message):
            is_smalltalk = True
            logger.info(
                f"[pre_router] Smalltalk detected: '{user_message[:50]}' "
                f"(words={len(user_message.split())}). "
                f"generate_node will use lean prompt path."
            )

    return {
        "thinking_steps": thinking_update,
        "forced_tool_calls": forced,
        "is_smalltalk": is_smalltalk,
    }
