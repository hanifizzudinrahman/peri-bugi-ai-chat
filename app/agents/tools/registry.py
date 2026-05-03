"""
Tool Registry — Single Source of Truth for all Tanya Peri tools.

Problem solved:
    Sebelumnya, tambah tool baru butuh update 3 file scattered:
    - tools/<name>.py (factory)
    - tool_bridge.py (mapping + bridging logic)
    - generate.py (system prompt consumer)
    Lupa salah satu = silent halusinasi (LLM karang jawaban tanpa data).

Solution:
    Setiap tool DEFINE 1 ToolSpec di file-nya sendiri yang nyatain SEMUA
    behavior (factory + bridge + prompt injection). Tool_bridge dan generate
    pakai registry ini, sehingga:
    - Tambah tool baru = update 1 file (self-contained)
    - Lupa register? → Loud RuntimeError saat startup (fail fast)
    - Pattern enforced by structure (impossible untuk skip)

Pattern reference: Plugin architecture (Django apps, FastAPI routers, etc).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Type aliases for clarity
# =============================================================================

# Bridge handler — converts ToolMessage result → updates agent_results dict.
# Signature: (parsed_result: dict, agent_results: dict, ctx: BridgeContext) -> None
# - parsed_result: parsed dict from tool's ToolMessage.content
# - agent_results: mutate in-place to add data
# - ctx: extra context (retrieved_docs, image_analysis, etc) for tools that
#        need to update fields beyond agent_results
BridgeHandler = Callable[[dict, dict, "BridgeContext"], None]

# Prompt injector — generates text to APPEND to system_prompt based on tool data.
# Signature: (agent_data: dict, child_name: str, prompts: dict, response_mode: str) -> str
# - agent_data: data yang udah ada di agent_results[<agent_key>]
# - child_name: anak's nickname (ready to embed)
# - prompts: prompts dict (untuk tool yang butuh DB-stored template)
# - response_mode: "simple"|"medium"|"detailed"
# - returns: text to append (empty str = nothing to inject)
PromptInjector = Callable[[dict, str, dict, str], str]


# =============================================================================
# Bridge context — for tools that need to update non-agent_results state
# =============================================================================

@dataclass
class BridgeContext:
    """Mutable context passed to bridge handlers.
    
    Untuk tools yang perlu update field di luar agent_results
    (e.g., retrieved_docs, image_analysis, scan_session_id, needs_clarification).
    
    Bagian C v2: Added unavailable_tools — list tools yang LLM panggil
    tapi tidak available (gated off via allowed_agents). Generate.py akan
    inject warning ke system prompt supaya LLM kasih honest answer.

    Image-Failure-Guard (medical safety, kritis):
    Saat user kirim foto + analyze_chat_image gagal (mode='new_scan_failed'),
    state["image_analysis"] tidak di-set (tetap None). Tanpa field tambahan
    di sini, generate.py SKIP image branch — TAPI tetap inject
    user_context.mata_peri_last_result (data scan kemarin) → LLM jawab seolah
    foto sukses dan giginya bersih. Itu BAHAYA untuk medical context.

    Solusi: bridge handler set image_analysis_failed=True kalau gagal, dan
    simpan fallback_text-nya. Generate.py pakai flag ini untuk:
    1. SKIP injection mata_peri_last_result + brushing (data lain tidak relevan)
    2. INJECT hard guard: 'JANGAN bilang gigi bersih — kamu tidak punya datanya'
    3. FORCE jawaban pakai fallback_text apa adanya
    """
    retrieved_docs: list = field(default_factory=list)
    image_analysis: Optional[dict] = None
    scan_session_id: Optional[str] = None
    needs_clarification: bool = False
    clarification_data: Optional[dict] = None
    # Bagian C v2: track tools yang dipanggil tapi tidak available
    unavailable_tools: list = field(default_factory=list)
    # Image-Failure-Guard: set saat analyze_chat_image return mode='new_scan_failed'
    image_analysis_failed: bool = False
    image_analysis_fallback_text: Optional[str] = None


# =============================================================================
# Tool unavailability detection — shared logic
# =============================================================================

# Pattern yang ditandai tools_node saat tool tidak ditemukan in tools_by_name.
# See nodes/tools_node.py — `_missing` synthetic ToolMessage.
_UNAVAILABLE_ERROR_MARKER = "not available for this user"


def is_tool_unavailable_result(result: dict) -> bool:
    """Check apakah tool result adalah error 'tool unavailable'.
    
    Tools_node generates synthetic error result saat LLM halusinasi panggil
    tool yang tidak di-bind (karena gated off via allowed_agents). Format-nya:
        {"error": "Tool 'X' not available for this user", "has_data": False}
    
    Bridge handler harus detect ini dan SKIP populate agent_results — supaya
    LLM tidak kira tool berhasil tapi data kosong.
    
    Args:
        result: Parsed dict dari ToolMessage.content
    
    Returns:
        True kalau result adalah unavailable error.
    """
    if not isinstance(result, dict):
        return False
    error_msg = result.get("error", "") or ""
    return _UNAVAILABLE_ERROR_MARKER in str(error_msg)


# =============================================================================
# Friendly feature names for user-facing messages
# =============================================================================

# Map agent_key (from allowed_agents) → friendly feature name (Indonesian).
# Used by generate.py to compose "feature X tidak tersedia" messages.
AGENT_KEY_TO_FEATURE_NAME = {
    "kb_dental": "Konsultasi Kesehatan Gigi",
    "app_faq": "Bantuan Aplikasi",
    "user_profile": "Profil Pengguna",
    "rapot_peri": "Rapot Sikat Gigi",
    "rapot_peri_history": "Riwayat Sikat Gigi",
    "rapot_peri_achievements": "Badge Sikat Gigi",
    "caries_risk": "Kuesioner Risiko Karies",
    "cerita_peri": "Cerita Peri",
    "cerita_module_detail": "Modul Cerita Peri",
    "mata_peri": "Mata Peri (Scan Gigi)",
    "mata_peri_scan_detail": "Detail Hasil Scan",
    "tips": "Tips Parenting Harian",
    # Phase 3 — new tools
    "brushing_settings": "Pengaturan Reminder Sikat Gigi",
    "brushing_trend": "Tren Sikat Gigi (Multi Bulan)",
    "caries_questionnaire_preview": "Preview Kuesioner Karies",
    "cerita_modules_summary": "Rekap Modul Cerita Peri",
}


def get_friendly_feature_name(tool_name: str) -> str:
    """Get user-facing feature name for a tool.
    
    Lookup via ToolSpec.agent_key → AGENT_KEY_TO_FEATURE_NAME.
    Falls back to humanizing tool_name kalau tidak ada mapping.
    """
    spec = _REGISTRY.get(tool_name)
    if spec is None:
        # Tool tidak di-register — humanize tool_name
        return tool_name.replace("_", " ").replace("get ", "").title()
    
    return AGENT_KEY_TO_FEATURE_NAME.get(
        spec.agent_key,
        spec.agent_key.replace("_", " ").title(),
    )


# =============================================================================
# ToolSpec — single source of truth per tool
# =============================================================================

@dataclass
class ToolSpec:
    """Single source of truth for 1 tool.
    
    Setiap tool wajib declare ToolSpec di akhir file-nya.
    Registry mengumpulkan semua spec saat import time.
    """
    
    # ── Identity ──────────────────────────────────────────────────────────────
    tool_name: str
    """Name yang LLM lihat & panggil. Must match @tool function name."""
    
    agent_key: str
    """Key di agent_results dict. Multiple tools boleh share agent_key
    (e.g., get_brushing_stats + get_brushing_history → both 'rapot_peri')."""
    
    required_agent: str
    """Permission gate — must be in state.control.allowed_agents.
    Usually = agent_key, but boleh beda (e.g., user_profile tool punya
    required_agent='user_profile' tapi agent_key='user_profile')."""
    
    # ── Bridging ──────────────────────────────────────────────────────────────
    bridge_handler: BridgeHandler
    """Function yang convert tool result → update agent_results.
    Wajib ada — even kalau cuma assign result ke agent_results[agent_key]."""
    
    # ── Generate consumer ─────────────────────────────────────────────────────
    prompt_injector: Optional[PromptInjector] = None
    """Function yang convert agent_data → text untuk system_prompt.
    Optional — kalau None, data ada di agent_results tapi tidak di-inject
    ke prompt (e.g., metadata-only tools).
    
    KRITIKAL: Kalau tool punya data yang LLM butuh untuk jawab, prompt_injector
    WAJIB ada. Kalau None, LLM akan halusinasi (tidak punya data)."""
    
    # ── Optional metadata ─────────────────────────────────────────────────────
    thinking_label: Optional[str] = None
    """Label untuk thinking_steps display di FE.
    Falls back to default 'Memproses...' jika None."""


# =============================================================================
# Registry storage
# =============================================================================

# Module-level dict: tool_name → ToolSpec
_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    """Register a tool spec. Called from each tool file at import time.
    
    Raises:
        ValueError: If tool_name already registered (duplicate detection).
    """
    if spec.tool_name in _REGISTRY:
        existing = _REGISTRY[spec.tool_name]
        raise ValueError(
            f"Duplicate ToolSpec registration for '{spec.tool_name}': "
            f"already registered with agent_key='{existing.agent_key}'. "
            f"Check tools/<file>.py for accidental double-registration."
        )
    _REGISTRY[spec.tool_name] = spec
    logger.debug(
        f"[registry] Registered tool '{spec.tool_name}' "
        f"(agent_key={spec.agent_key}, required={spec.required_agent})"
    )


def get_tool_spec(tool_name: str) -> Optional[ToolSpec]:
    """Lookup ToolSpec by tool name. Returns None if not found."""
    return _REGISTRY.get(tool_name)


def iter_tool_specs():
    """Iterate all registered ToolSpec instances."""
    return list(_REGISTRY.values())


def get_registered_tool_names() -> list[str]:
    """List all registered tool names (for diagnostics)."""
    return sorted(_REGISTRY.keys())


def get_specs_by_agent_key(agent_key: str) -> list[ToolSpec]:
    """Get all ToolSpec instances yang share agent_key.
    
    Useful kalau multiple tools nge-update same key (e.g., rapot_peri).
    """
    return [s for s in _REGISTRY.values() if s.agent_key == agent_key]


# =============================================================================
# Validation — fail fast at startup
# =============================================================================

def validate_registry(expected_tool_names: list[str]) -> None:
    """Verify all expected tools are registered.
    
    Called at startup setelah all tool files imported. Kalau ada tool yang
    bisa di-build via factory tapi tidak punya ToolSpec → loud error
    (instead of silent halusinasi at runtime).
    
    Args:
        expected_tool_names: list of tool names yang harus ter-register.
                             Diambil dari hasil make_tools() di test session.
    
    Raises:
        RuntimeError: Kalau ada tool yang missing ToolSpec.
    """
    registered = set(_REGISTRY.keys())
    expected = set(expected_tool_names)
    missing = expected - registered
    
    if missing:
        raise RuntimeError(
            f"Tools missing ToolSpec registration: {sorted(missing)}. "
            f"Tambah ToolSpec declaration di akhir tools/<file>.py untuk "
            f"setiap tool ini. Pattern: lihat tools/registry.py docstring."
        )
    
    extra = registered - expected
    if extra:
        # Extra registrations OK (mungkin tools yang feature-flagged off),
        # tapi log warning untuk awareness.
        logger.info(
            f"[registry] Tools registered but not in expected list: {sorted(extra)} "
            f"(OK — mungkin behind feature flag)"
        )
    
    logger.info(
        f"[registry] Validation OK — {len(registered)} tools registered: "
        f"{sorted(registered)}"
    )


# =============================================================================
# Diagnostic helpers
# =============================================================================

def diagnostic_summary() -> dict:
    """Return registry summary for debugging / health endpoint."""
    return {
        "total_registered": len(_REGISTRY),
        "tools": [
            {
                "tool_name": s.tool_name,
                "agent_key": s.agent_key,
                "required_agent": s.required_agent,
                "has_prompt_injector": s.prompt_injector is not None,
                "thinking_label": s.thinking_label,
            }
            for s in _REGISTRY.values()
        ],
    }
