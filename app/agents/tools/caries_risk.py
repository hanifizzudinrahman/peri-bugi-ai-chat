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


# =============================================================================
# Phase 3 — Tool: get_caries_questionnaire_preview
# =============================================================================

def make_get_caries_questionnaire_preview_tool():
    """Factory: build get_caries_questionnaire_preview tool.
    
    Note: TIDAK butuh user_id — questionnaire active sama untuk semua user.
    """

    @tool
    async def get_caries_questionnaire_preview(mode: str = "summary") -> dict[str, Any]:
        """Preview pertanyaan kuesioner risiko karies (untuk user yang mau tau dulu sebelum isi).

        ✅ USE THIS TOOL when user asks about:
        - "kuesionernya tanya apa aja?"
        - "pertanyaannya gimana sih kuesionernya?"
        - "mau tau dulu sebelum isi"
        - "kategori pertanyaannya apa?"
        - "berapa pertanyaan total kuesionernya?"

        ❌ DO NOT use this tool when:
        - User minta hasil kuesioner mereka → use get_caries_risk_latest
        - User minta general info caries → use search_dental_knowledge
        - User minta cara isi kuesioner → use search_app_faq

        Args:
            mode: "summary" (default) or "full".
                - summary: 4 kategori dengan 2 contoh pertanyaan per kategori
                          Cocok untuk user general yang minta gambaran singkat.
                - full: semua pertanyaan + opsi jawaban.
                       Pakai HANYA kalau user explicit minta detail / list semua.

        Returns dict with keys:
        - has_data: bool
        - mode: str — echo of input
        - questionnaire_title: str
        - total_questions: int — total active questions
        - total_categories: int
        - categories: list of {code, label, description, question_count, ...}
          - summary mode: each category has 'example_questions' (max 2)
          - full mode: each category has 'questions' list with options

        ANTI-HALU:
        - Default to "summary" mode untuk hindari overwhelm user.
        - Kalau user minta detail / "lengkap" / "semua" → mode="full".
        - JANGAN halu pertanyaan yang tidak ada di list — pakai data tool.
        - Note: kuesioner ini diisi oleh ORANG TUA tentang anak, bukan anak.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        async with trace_node(
            name="tool:get_caries_questionnaire_preview",
            state=None,
            input_data={"mode": mode},
        ) as span:
            data = await call_internal_get(
                f"/api/v1/internal/agent/caries-questionnaire-preview?mode={mode}"
            )

            if span:
                span.update(output={
                    "has_data": data.get("has_data", False),
                    "mode": data.get("mode"),
                    "total_questions": data.get("total_questions"),
                    "total_categories": data.get("total_categories"),
                    "data": _safe_dict_for_trace(data),
                })

            return data

    return get_caries_questionnaire_preview


def _bridge_caries_questionnaire_preview(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: preview → agent_results['caries_questionnaire_preview']."""
    agent_results["caries_questionnaire_preview"] = result


def _inject_caries_questionnaire_preview(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject caries questionnaire preview to system prompt."""
    if not data or not data.get("has_data"):
        if data and data.get("reason"):
            return (
                f"\n\nPreview kuesioner karies: BELUM TERSEDIA "
                f"(reason: {data.get('reason')}). Bilang user dengan jujur."
            )
        return ""
    
    mode = data.get("mode", "summary")
    title = data.get("questionnaire_title", "Kuesioner Risiko Karies")
    total_questions = data.get("total_questions", 0)
    total_categories = data.get("total_categories", 0)
    categories = data.get("categories") or []
    
    text = (
        f"\n\nPreview Kuesioner Risiko Karies (mode={mode}, dari tool call):"
        f"\n- Judul: {title}"
        f"\n- Total: {total_questions} pertanyaan dalam {total_categories} kategori"
    )
    
    for cat in categories:
        label = cat.get("label", "-")
        count = cat.get("question_count", 0)
        text += f"\n\n**{label}** ({count} pertanyaan):"
        
        if cat.get("description"):
            text += f"\n  _{cat['description']}_"
        
        if mode == "summary":
            examples = cat.get("example_questions") or []
            if examples:
                text += f"\n  Contoh pertanyaan:"
                for q in examples:
                    text += f"\n  • {q}"
        else:  # full
            questions = cat.get("questions") or []
            for i, q in enumerate(questions, 1):
                text += f"\n  {i}. {q.get('question_text', '-')}"
                if q.get("help_text"):
                    text += f"\n     _Bantuan: {q['help_text']}_"
                opts = q.get("options") or []
                if opts:
                    opt_labels = [o.get("option_text", "?") for o in opts]
                    text += f"\n     Opsi: {' / '.join(opt_labels)}"
    
    text += (
        f"\n\nINSTRUKSI: "
        f"Kalau user minta lebih detail dari summary, panggil tool lagi dengan mode='full'. "
        f"Inget: kuesioner ini diisi oleh ORANG TUA tentang kondisi anak. "
        f"Bahasa ibu-friendly, JANGAN dump semua pertanyaan kalau user cuma minta gambaran."
    )
    
    return text


register_tool(ToolSpec(
    tool_name="get_caries_questionnaire_preview",
    agent_key="caries_questionnaire_preview",
    required_agent="rapot_peri",
    bridge_handler=_bridge_caries_questionnaire_preview,
    prompt_injector=_inject_caries_questionnaire_preview,
    thinking_label="Mengambil preview kuesioner...",
))
