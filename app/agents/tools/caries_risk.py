"""
Tool: get_caries_risk_latest

Fetches latest caries risk assessment for the current child user.
Wraps GET /api/v1/internal/agent/caries-risk/{user_id}.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.agents.tools._http import call_internal_get

logger = logging.getLogger(__name__)


def make_get_caries_risk_tool(user_id: Optional[str]):
    """
    Factory: build get_caries_risk_latest tool with user_id closure.

    Security: user_id is closure-captured (NOT a tool param), so LLM
    cannot spoof another user's data.

    Args:
        user_id: Current user ID (from state.user_context.user.id)

    Returns:
        @tool decorated async function.
    """

    @tool
    async def get_caries_risk_latest() -> dict[str, Any]:
        """Get the child's LATEST caries risk assessment result with detailed answers.

        ✅ USE THIS TOOL when the user asks about:
        - "anak saya hasil kuesionernya apa?" → check result_label + recommendation
        - "berapa skor risiko karies anak saya?" → check total_score + result_level
        - "anak saya beresiko tinggi atau rendah?" → check result_level/result_label
        - "rekomendasi dari hasil kuesioner kemarin gimana?" → check recommendation
        - "kenapa anak saya beresiko tinggi?" → check recent_answers untuk yang impact='menambah_risiko'
        - "saya jawab apa di kuesioner kemarin?" → check recent_answers list (yang isi orang tua, bukan anak)
        - "pertanyaan apa aja yang bikin skor naik?" → filter recent_answers by impact='menambah_risiko'

        ❌ DO NOT use this tool when:
        - User asks how to fill the questionnaire → use search_app_faq
        - User asks about caries in general (not their specific child) → use search_dental_knowledge
        - User asks about brushing → use get_brushing_stats or get_brushing_history
        - User has never filled questionnaire → tool returns has_data=False with fallback message
        - User minta history multi-time (kuesioner ini cuma diisi 1x, tidak ada history)

        Returns a dict with keys:
        - has_data: bool — true kalau user sudah pernah isi kuesioner
        - child_name: str
        - child_age_years: int | null
        - total_score: int — skor akhir (-4 sampai 24+)
        - result_level: str — "low" | "moderate" | "high" (English enum)
        - result_label: str — label Indonesia (e.g. "Risiko Tinggi")
        - recommendation: str — rekomendasi text panjang
        - submitted_at: ISO datetime string
        - recent_answers: list[{question_text, option_text, score, impact}]
                          impact: "menambah_risiko" | "melindungi" | "netral"
        - reason / fallback_message: kalau has_data=False

        ANTI-HALU:
        - YANG ISI KUESIONER ADALAH ORANG TUA (BUNDA), BUKAN ANAK. Pakai bahasa
          "Bunda jawab Ya/Tidak untuk pertanyaan...", BUKAN "anak jawab".
        - Kalau user minta jawaban kuesioner detail, JANGAN sebut score angka spesifik
          ke ibu (terlalu teknis). Cukup bilang "pertanyaan tentang X bikin nilai naik /
          melindungi". Pakai bahasa sederhana.
        - Kalau recent_answers null/empty, JANGAN halu tentang jawaban — bilang
          datanya tidak tersedia.
        """
        # FIX (Langfuse audit Bagian B1): Add trace_node wrapper for consistency
        # dengan existing tools pattern (tool:get_brushing_stats, tool:get_user_profile, dll).
        # Sebelumnya: tool execution invisible di trace tree, HTTP child span nempel ke agent_node.
        # Sekarang: span "tool:get_caries_risk_latest" muncul as proper child di tools span.
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_caries_risk_latest",
            state=None,
            input_data={
                "has_user_id": bool(user_id),
            },
        ) as span:
            if not user_id:
                result = {
                    "has_data": False,
                    "reason": "no_user_id",
                    "fallback_message": "User ID tidak tersedia.",
                }
                if span:
                    span.update(output={
                        "has_data": False,
                        "reason": "no_user_id",
                    })
                return result

            data = await call_internal_get(
                f"/api/v1/internal/agent/caries-risk/{user_id}"
            )

            if span:
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "result_level": data.get("result_level"),
                    "result_label": data.get("result_label"),
                    "total_score": data.get("total_score"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_caries_risk_latest


# =============================================================================
# ToolSpec registration — Bagian C: registry pattern
# =============================================================================
from app.agents.tools.registry import ToolSpec, register_tool, BridgeContext


def _bridge_caries_risk(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: caries risk → agent_results['caries_risk']."""
    agent_results["caries_risk"] = result


def _inject_caries_risk(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject caries risk questionnaire result + recent_answers to system prompt.
    
    CRITICAL: Sebelum fix, tool ini di-call tapi data dibuang — LLM halusinasi
    jawaban tentang scan padahal user tanya kuesioner. Sekarang inject data REAL.
    
    Phase 2: tambah recent_answers detail per pertanyaan dengan impact label.
    """
    if not data or not data.get("has_data"):
        if data and data.get("reason"):
            return (
                f"\n\nHasil kuesioner risiko karies {child_name}: BELUM PERNAH DIISI "
                f"(reason: {data.get('reason')}). "
                f"INSTRUKSI: Beri tahu user dengan jujur bahwa kuesioner belum diisi, "
                f"sarankan mengisi via Beranda → Kuesioner Karies."
            )
        return ""
    
    total_score = data.get("total_score")
    result_level = data.get("result_level")
    result_label = data.get("result_label")
    recommendation = data.get("recommendation")
    submitted_at = data.get("submitted_at")
    recent_answers = data.get("recent_answers") or []  # Phase 2
    
    text = (
        f"\n\nHasil kuesioner risiko karies {child_name} (dari tool call — DATA TEPAT):"
        f"\n- Skor total: {total_score}"
        f"\n- Level risiko: {result_label} ({result_level})"
        f"\n- Tanggal isi kuesioner: {submitted_at}"
    )
    
    if recommendation:
        text += f"\n- Rekomendasi: {recommendation}"
    
    # Phase 2: detail jawaban orang tua per pertanyaan
    if recent_answers:
        # Group by impact for clearer LLM understanding
        risk_increasers = [a for a in recent_answers if a.get("impact") == "menambah_risiko"]
        protectives = [a for a in recent_answers if a.get("impact") == "melindungi"]
        
        text += "\n\nDETAIL jawaban orang tua di kuesioner (untuk reference, JANGAN dump semua ke user):"
        
        if risk_increasers:
            text += f"\n- Pertanyaan yang dijawab YA → menambah risiko ({len(risk_increasers)} pertanyaan):"
            for ans in risk_increasers[:5]:  # cap di 5 supaya tidak boros token
                qt = ans.get("question_text", "")[:120]  # potong panjang
                text += f"\n  • {qt} → jawab '{ans.get('option_text', '')}'"
            if len(risk_increasers) > 5:
                text += f"\n  • ...dan {len(risk_increasers) - 5} pertanyaan lain"
        
        if protectives:
            text += f"\n- Kebiasaan baik yang sudah dilakukan ({len(protectives)} hal):"
            for ans in protectives[:5]:
                qt = ans.get("question_text", "")[:120]
                text += f"\n  • {qt} → jawab '{ans.get('option_text', '')}'"
    
    text += (
        f"\n\nINSTRUKSI: Pakai data ini untuk jawab user. JANGAN bingungkan dengan "
        f"hasil scan Mata Peri (itu beda hal). Kuesioner karies = self-assessment, "
        f"scan Mata Peri = analisis foto gigi. "
        f"Yang isi kuesioner adalah ORANG TUA, BUKAN ANAK — pakai bahasa "
        f"\"Bunda jawab...\", bukan \"anak jawab...\". "
        f"Hindari sebut score angka spesifik ke ibu, gunakan bahasa kualitatif "
        f"(\"menambah risiko\", \"melindungi\", \"sudah baik\")."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_caries_risk_latest",
    agent_key="caries_risk",
    required_agent="rapot_peri",  # gated under rapot_peri block in tools/__init__.py
    bridge_handler=_bridge_caries_risk,
    prompt_injector=_inject_caries_risk,
    thinking_label="Mengecek hasil kuesioner risiko karies...",
))
