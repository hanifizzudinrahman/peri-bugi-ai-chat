"""
Node: generate
Generate response dari LLM dengan streaming token per token.

Update v2:
- Support response_mode (simple/medium/detailed) → pilih prompt template berbeda
- Build context dari agent_results (multi-agent)
- Inject memory context (L2+L3) ke system prompt
- Support quick_reply event

Update v3 (Step 2b):
- Smalltalk path: jika state.is_smalltalk=True (set by pre_router), pakai
  lean prompt minimal (Q2B: keep nama anak doang). SKIP injection of
  brushing, mata_peri_last_result, memory_context, agent_results.
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.config.llm import get_llm, get_model_name, get_provider_name
from app.config.observability import build_trace_config
from app.schemas.chat import (
    LLMCallLogPayload,
    make_clarify_event,
    make_quick_reply_event,
    make_thinking_event,
    make_token_event,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Bagian C v6 — Phase 1: Foundation Blocks
# ═════════════════════════════════════════════════════════════════════════════
# Pattern: Foundation blocks di-inject paling ATAS system prompt (sebelum
# persona) supaya LLM lihat aturan dasar duluan. Pattern ini mirror best practice
# OpenAI Assistants & Anthropic Claude system prompts: time context + behavioral
# rules → persona → data → instructions.
#
# Foundation 1 (TIME): KONTEKS WAKTU — generated per-turn, supaya LLM tau
#   "hari ini", "kemarin", "besok" tanggal berapa. Anti-halu untuk pertanyaan
#   tentang masa depan / past dates.
#
# Foundation 2 (ANTI-HALU): Aturan keras tentang JANGAN karang data. Hardcoded
#   karena ini technical guardrail — kalau di-edit di admin prompt accidentally,
#   LLM mulai halu.
#
# Foundation 3 (BAHASA): Bahasa ibu-friendly — anak SD/SMP/SMA. Hardcoded juga
#   karena ini target audience character, bukan persona detail.
# ═════════════════════════════════════════════════════════════════════════════

# Bahasa Indonesia day & month names — for natural readable date format
_HARI_INDO = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}

_BULAN_INDO = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


def _build_time_context_block(timezone_str: str = "Asia/Jakarta") -> str:
    """Build KONTEKS WAKTU block — Foundation 1.
    
    Generated PER-TURN, supaya kalau conversation lintas hari (misal mulai jam
    23:55 Selasa, lanjut jam 00:30 Rabu), "hari ini" tetap akurat.
    
    Resilience:
    - Try ZoneInfo first (proper DST handling kalau timezone non-Indonesia).
    - Kalau ZoneInfo fail (tzdata missing, etc.) → fallback ke fixed offset
      (Indonesia tidak ada DST, jadi offset tetap aman).
    - Kalau semua fail → final fallback pakai datetime.now() naive +
      asumsi WIB (UTC+7).
    Bulletproof: function ini TIDAK BOLEH raise exception — dipanggil setiap
    request, kalau error = service down.
    
    Args:
        timezone_str: Timezone IANA string. Default Asia/Jakarta (WIB).
                     Future: bisa pass dari user_context kalau Hanif tambah
                     field timezone di profile.
    
    Returns:
        Multi-line string block ready to append to system prompt.
    """
    # Indonesia timezone offsets (Indonesia tidak punya DST, fixed forever)
    _ID_OFFSETS = {
        "Asia/Jakarta": (timedelta(hours=7), "WIB"),
        "Asia/Pontianak": (timedelta(hours=7), "WIB"),
        "Asia/Makassar": (timedelta(hours=8), "WITA"),
        "Asia/Jayapura": (timedelta(hours=9), "WIT"),
    }
    
    try:
        now = None
        tz_label = "WIB"
        
        # Layer 1: Try ZoneInfo (proper DST aware untuk multi-region future)
        try:
            tz = ZoneInfo(timezone_str)
            now = datetime.now(tz)
            # Determine label
            if timezone_str in _ID_OFFSETS:
                tz_label = _ID_OFFSETS[timezone_str][1]
            else:
                # Non-Indonesia timezone — fallback label dari abbreviation
                tz_label = now.tzname() or "LOCAL"
        except Exception:
            # Layer 2: ZoneInfo failed (e.g., tzdata missing) — fallback fixed offset
            try:
                offset, label = _ID_OFFSETS.get(
                    timezone_str, _ID_OFFSETS["Asia/Jakarta"]
                )
                tz = timezone(offset)
                now = datetime.now(tz)
                tz_label = label
            except Exception:
                # Layer 3: Even timezone() failed (very unlikely) — final fallback
                # Pakai naive UTC + manual offset, tag sebagai WIB.
                now = datetime.utcnow() + timedelta(hours=7)
                tz_label = "WIB"
        
        today = now.date()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        
        # Compute Senin (start) & Minggu (end) of current week (ISO week, Senin=0)
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Last month
        if today.month == 1:
            last_month_year = today.year - 1
            last_month_num = 12
        else:
            last_month_year = today.year
            last_month_num = today.month - 1
        
        block = (
            f"\n\nKONTEKS WAKTU SEKARANG:\n"
            f"- Hari ini: {_HARI_INDO[today.weekday()]}, "
            f"{today.day} {_BULAN_INDO[today.month]} {today.year}, "
            f"jam {now.hour:02d}:{now.minute:02d} {tz_label}\n"
            f"- Kemarin: {_HARI_INDO[yesterday.weekday()]}, "
            f"{yesterday.day} {_BULAN_INDO[yesterday.month]} {yesterday.year}\n"
            f"- Besok: {_HARI_INDO[tomorrow.weekday()]}, "
            f"{tomorrow.day} {_BULAN_INDO[tomorrow.month]} {tomorrow.year}\n"
            f"- Minggu ini: {week_start.day} {_BULAN_INDO[week_start.month]} – "
            f"{week_end.day} {_BULAN_INDO[week_end.month]} {week_end.year}\n"
            f"- Bulan ini: {_BULAN_INDO[today.month]} {today.year}\n"
            f"- Bulan kemarin: {_BULAN_INDO[last_month_num]} {last_month_year}\n"
            f"- Tahun ini: {today.year}. Tahun lalu: {today.year - 1}.\n"
        )
        return block
    except Exception as e:
        # Outer safety net: kalau ada bug di logic dalam, log + return empty.
        # Service tetap jalan, cuma kehilangan TIME context untuk turn ini.
        # Better than crash.
        logger.error(
            f"[_build_time_context_block] Unexpected failure: {e}. "
            f"Returning empty block — service stays up."
        )
        return ""


# Foundation 2 — Anti-halu rules (hardcoded, tidak editable di admin)
_ANTI_HALU_RULES = (
    "\n\nATURAN PENTING — JANGAN HALUSINASI:\n"
    "1. JANGAN mengarang angka spesifik (skor, durasi, tanggal, jumlah hari, "
    "persentase) yang tidak ada di tool result atau konteks. Kalau tidak punya "
    "data, bilang \"saya belum punya datanya\" atau \"perlu cek lebih lanjut\".\n"
    "2. Kalau user tanya tentang MASA DEPAN (besok, minggu depan, tahun depan), "
    "bilang datanya belum ada — kejadian belum terjadi. JANGAN mengira-ira.\n"
    "3. Kalau tool kasih `has_data: false` atau error, JANGAN guess — bilang "
    "ke user bahwa datanya belum tersedia.\n"
    "4. Kalau tool result punya data partial, jawab pakai yang ADA, dan secara "
    "ramah acknowledge bagian yang TIDAK ADA. Contoh: \"Kalau soal sikat pagi "
    "kemarin sudah Bunda centang, tapi soal malamnya belum ada catatan.\"\n"
    "5. Memory dari ringkasan percakapan sebelumnya itu HANYA konteks topik. "
    "JANGAN treat angka/status/data spesifik dari memory sebagai data current.\n"
    "6. Kalau tidak yakin antara dua kemungkinan, lebih baik bilang \"saya kurang "
    "yakin\" atau tanya balik ke user untuk konfirmasi — JANGAN asal pilih.\n"
)


# Foundation 3 — Bahasa & tone (hardcoded)
_BAHASA_RULES = (
    "\n\nATURAN BAHASA & TONE:\n"
    "- User adalah orang tua Indonesia, kebanyakan ibu lulusan SD/SMP/SMA. "
    "Pakai bahasa sehari-hari, sederhana, tidak formal kaku.\n"
    "- HINDARI istilah teknis: \"score\", \"compliance percentage\", \"milestone\", "
    "\"streak\" (boleh kalau user yang sebut duluan), \"cleanliness ratio\", \"risk weight\". "
    "Ganti ke bahasa percakapan: \"nilai\", \"konsistensi\", \"target hari berikutnya\", "
    "\"hari berturut-turut\", \"kebersihan\", \"tingkat risiko\".\n"
    "- Sapaan: pakai \"Bunda\" atau \"Ayah\" (ikuti gender user kalau tahu) atau nama panggilan.\n"
    "- Anak: sebut dengan nama panggilan anak. Ingat: yang menggunakan aplikasi "
    "adalah ORANG TUA, bukan anak. Jadi \"Bunda yang ngerjain modul Cerita Peri\", "
    "bukan \"anak ngerjain modul\".\n"
    "- Emoji boleh, tapi seperlunya saja (1-2 per response). Hindari spam emoji.\n"
    "- Panjang: 3-5 kalimat default. Kalau user minta detail baru lebih panjang.\n"
)


def _should_skip_foundation_blocks(state) -> bool:
    """Apakah skip foundation blocks (TIME/ANTI-HALU/BAHASA) untuk turn ini?
    
    Skip kalau:
    - is_smalltalk path (smalltalk pakai lean prompt sendiri, tidak butuh)
    - _override_system aktif (RnD mode, full custom prompt)
    
    Returns:
        True kalau harus skip, False kalau harus inject.
    """
    if state.get("is_smalltalk"):
        return True
    prompts = state.get("prompts", {}) or {}
    if "_override_system" in prompts:
        return True
    return False


def _build_smalltalk_system_prompt(state: AgentState, prompts: dict) -> str:
    """
    Lean smalltalk system prompt (Step 2b — Q2B decision).

    Triggered when state.is_smalltalk=True (set by pre_router_node via strict
    regex+length+keyword check).

    SKIP injection: brushing, mata_peri_last_result, memory_context, agent_results.
    KEEP: persona + nama orang tua + nama anak.

    Q3A graceful fallback: kalau 'generate_smalltalk' tidak ada di state.prompts
    (DB seeder belum jalan / key inactive), pakai inline default + warning log.
    """
    SMALLTALK_FALLBACK = (
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak yang ramah dan hangat.\n\n"
        "User baru kasih sapaan singkat. Bales dengan hangat dan natural — JANGAN dump info detail.\n\n"
        "ATURAN:\n"
        "- Sapa user dengan nama orang tua-nya\n"
        "- Sebut nama anak (kalau diketahui) untuk personalisasi\n"
        "- Tanyakan ada yang bisa kamu bantu hari ini\n"
        "- Maksimal 1-2 kalimat\n"
        "- 1-2 emoji wajar (🧚 ✨ 👋 💙)\n"
        "- JANGAN sebutkan hasil scan, streak, atau detail teknis lain\n"
        "- JANGAN tanya pertanyaan medis spesifik\n\n"
        "KONTEKS USER:\n"
        "- Orang tua: {user_name}\n"
        "- Anak: {child_name}\n"
    )

    template = prompts.get("generate_smalltalk")
    if not template:
        logger.warning(
            "[generate] 'generate_smalltalk' prompt not in state.prompts — "
            "using inline fallback. Run scripts/seed_prompts.py to populate DB."
        )
        template = SMALLTALK_FALLBACK

    # Inject {user_name} + {child_name}
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"

    rendered = template.replace("{user_name}", user_name)
    rendered = rendered.replace("{child_name}", child_name)

    return rendered


def _get_allowed_agents_from_state(state) -> set:
    """Extract allowed_agents from state — single source of truth lookup.
    
    Bagian C v5: Centralize lookup logic supaya semua callers (sanitize,
    build prompt, selective skip user_context) pakai path yang sama.
    
    Lookup order (paling reliable dulu):
    1. state["allowed_agents"]              ← legacy dict (production path)
    2. state.control.allowed_agents         ← Pydantic AgentState (direct invoke)
    3. state["user_context"]["allowed_agents"] ← legacy fallback (nested)
    
    Args:
        state: AgentState (Pydantic) atau legacy dict dari
               _build_legacy_dict_state.
    
    Returns:
        set of agent_key strings. Empty set kalau tidak ditemukan.
    """
    # Path 1: legacy dict top-level
    if hasattr(state, 'get'):
        top_level = state.get("allowed_agents")
        if top_level:
            return set(top_level)
    
    # Path 2: Pydantic AgentState
    if hasattr(state, 'control'):
        ctrl_allowed = getattr(state.control, 'allowed_agents', None)
        if ctrl_allowed:
            return set(ctrl_allowed)
    
    # Path 3: legacy nested fallback
    if hasattr(state, 'get'):
        user_ctx = state.get("user_context", {}) or {}
        if isinstance(user_ctx, dict):
            nested = user_ctx.get("allowed_agents")
            if nested:
                return set(nested)
    
    return set()


def _get_unavailable_features_for_state(state) -> list[str]:
    """Return list of friendly feature names yang OFF di akun user.
    
    Bagian C v5: Replaces keyword-based detection with simple
    "feature in registry but agent not in allowed_agents" lookup.
    
    Args:
        state: AgentState (Pydantic) atau legacy dict.
    
    Returns:
        Sorted list of friendly feature names (e.g., ["Cerita Peri",
        "Rapot Sikat Gigi"]). Empty list kalau semua agent ON atau
        registry tidak loaded.
    """
    try:
        from app.agents.tools.registry import iter_tool_specs, AGENT_KEY_TO_FEATURE_NAME
        all_specs = iter_tool_specs()
        allowed_agents = _get_allowed_agents_from_state(state)
        
        unavailable_features: set = set()
        for spec in all_specs:
            if spec.required_agent and spec.required_agent not in allowed_agents:
                feature = AGENT_KEY_TO_FEATURE_NAME.get(
                    spec.agent_key, spec.agent_key.replace("_", " ").title()
                )
                if feature:
                    unavailable_features.add(feature)
        
        return sorted(unavailable_features)
    except Exception as e:
        logger.warning(f"[_get_unavailable_features_for_state] Failed: {e}")
        return []


def _get_available_features_for_state(state) -> list[str]:
    """Return list of friendly feature names yang AKTIF di akun user.
    
    Bagian C v5: Counterpart to _get_unavailable_features_for_state.
    Used by Robust System Prompt block to inform LLM "tools available".
    
    Args:
        state: AgentState (Pydantic) atau legacy dict.
    
    Returns:
        Sorted list of friendly feature names yang ON.
    """
    try:
        from app.agents.tools.registry import iter_tool_specs, AGENT_KEY_TO_FEATURE_NAME
        all_specs = iter_tool_specs()
        allowed_agents = _get_allowed_agents_from_state(state)
        
        available_features: set = set()
        for spec in all_specs:
            if spec.required_agent and spec.required_agent in allowed_agents:
                feature = AGENT_KEY_TO_FEATURE_NAME.get(
                    spec.agent_key, spec.agent_key.replace("_", " ").title()
                )
                if feature:
                    available_features.add(feature)
        
        return sorted(available_features)
    except Exception as e:
        logger.warning(f"[_get_available_features_for_state] Failed: {e}")
        return []


def _sanitize_session_summaries(
    summaries: list,
    state: AgentState,
) -> tuple[list[str], bool]:
    """Flag session summaries kalau ada agent OFF di akun.
    
    Bagian C v5 SIMPLIFIED: Drop keyword match (terlalu brittle — banyak
    miss case). New behavior: kalau ada minimal 1 agent OFF, flag has_stale=True
    supaya disclaimer di-inject. Generate.py akan kasih instruksi LLM untuk
    treat summary sebagai konteks topik aja, bukan data current.
    
    Trade-off: lebih konservatif (disclaimer muncul untuk SEMUA summary kalau
    ada agent OFF, walau summary itu mungkin tidak reference agent OFF).
    Tapi disclaimer-nya soft, jadi tidak harm summary yang valid. Lebih
    important: tidak ada miss case.
    
    Args:
        summaries: List of summary strings dari memory.
        state: AgentState atau legacy dict.
    
    Returns:
        Tuple (summaries_unchanged, has_stale):
        - summaries_unchanged: Same list (no filtering — pure pass-through)
        - has_stale: True kalau ada minimal 1 agent OFF
    """
    if not summaries:
        return [], False
    
    # Bagian C v5: simplest possible logic — kalau ada agent OFF, flag stale.
    # Ini lebih konservatif tapi NO MISS CASE (vs keyword match yang miss
    # banyak query like "streak").
    unavailable_features = _get_unavailable_features_for_state(state)
    has_stale = len(unavailable_features) > 0
    
    if has_stale:
        logger.debug(
            f"[_sanitize_session_summaries] {len(summaries)} summaries flagged "
            f"as potentially stale (unavailable features: {unavailable_features})"
        )
    
    return list(summaries), has_stale


def _build_system_prompt(state: AgentState) -> str:
    """
    Build system prompt dengan:
    1. Persona Tanya Peri (dari DB atau default)
    2. Data user & anak dari user_context
    3. Memory context (L2 summary + L3 facts)
    4. Hasil agent (docs, profile data, dll)
    5. Response mode instructions

    Step 2b: Early branch ke lean smalltalk prompt jika state.is_smalltalk=True
    (set by pre_router via strict regex check). Skip injection brushing/scan/memory.
    """
    prompts = state.get("prompts", {})

    # Override system prompt penuh (RnD mode) — UNCHANGED
    if "_override_system" in prompts:
        return prompts["_override_system"]

    # ─────────────────────────────────────────────────────────────────────────
    # SMALLTALK PATH (Step 2b — Q2B): early return with lean prompt
    # ─────────────────────────────────────────────────────────────────────────
    if state.get("is_smalltalk"):
        return _build_smalltalk_system_prompt(state, prompts)

    # ─────────────────────────────────────────────────────────────────────────
    # NORMAL PATH (existing logic — UNCHANGED below)
    # ─────────────────────────────────────────────────────────────────────────
    
    # ═════════════════════════════════════════════════════════════════════════
    # Bagian C v6 — Phase 1: Foundation Blocks (TIME, ANTI-HALU, BAHASA)
    # ═════════════════════════════════════════════════════════════════════════
    # Inject di paling ATAS prompt (sebelum persona). Pattern industri besar:
    # behavioral rules + time context come FIRST, then persona/data follows.
    # 
    # Skip blocks ini untuk smalltalk path (sudah early-returned) dan RnD
    # override mode. Untuk normal path, foundation blocks selalu di-inject.
    # 
    # Build foundation prefix (3 blocks combined):
    foundation_prefix = ""
    if not _should_skip_foundation_blocks(state):
        # Try to extract user timezone from user_context (fallback Asia/Jakarta).
        # Future enhancement: store user.timezone di profile, pass via state.
        user_ctx_for_tz = state.get("user_context", {}) or {}
        user_for_tz = user_ctx_for_tz.get("user") or {}
        timezone_str = user_for_tz.get("timezone") or "Asia/Jakarta"
        
        foundation_prefix = (
            _build_time_context_block(timezone_str)
            + _ANTI_HALU_RULES
            + _BAHASA_RULES
        )
    
    # Persona base
    persona = prompts.get(
        "persona_system",
        "Kamu adalah Tanya Peri 🧚, asisten kesehatan gigi anak dari aplikasi Peri Bugi. "
        "Kamu ramah, sabar, dan berbicara dengan bahasa yang mudah dipahami orang tua Indonesia. "
        "Jangan pernah memberikan diagnosis medis langsung. "
        "Selalu sarankan konsultasi dokter gigi untuk masalah serius.",
    )

    # Inject user & child data
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}

    user_name = user.get("nickname") or user.get("full_name") or "Bunda/Ayah"
    child_name = child.get("nickname") or child.get("full_name") or "si kecil"
    child_age = f"{child.get('age_years')} tahun" if child.get("age_years") else "?"

    # Bagian C v6 — Phase 1: Prepend foundation prefix (TIME + ANTI-HALU + BAHASA)
    # Foundation blocks come BEFORE persona, supaya LLM pahami aturan dasar
    # dulu sebelum mencerna persona/data. Empty kalau smalltalk/override mode.
    # Separator: foundation_prefix sudah ada trailing \n, persona inline — ok.
    if foundation_prefix:
        system = foundation_prefix + "\n" + persona.replace("{user_name}", user_name)
    else:
        system = persona.replace("{user_name}", user_name)
    system = system.replace("{child_name}", child_name)
    system = system.replace("{child_age}", child_age)

    # Bagian C v5 — Fix 5: Selective skip static context based on allowed_agents.
    # Static context (brushing, mata_peri_last_result) di-pass dari peri-bugi-api
    # via user_context payload. Tapi kalau agent terkait OFF, inject data-nya
    # ke prompt = inkonsistensi: tool tidak bisa di-call (gated), tapi data
    # static masih ada → LLM bisa pakai data itu sebagai "fakta current" dan
    # halusinasi tentang hal lain dari fitur OFF. Solusinya: skip injection
    # kalau agent OFF.
    #
    # Note: ini fix di ai-chat side (sederhana). Proper fix sebenarnya di
    # peri-bugi-api side (jangan kirim brushing field kalau rapot_peri OFF).
    # Tapi karena api side touch repo lain, kita defense-in-depth di sini.
    _allowed_agents_set = _get_allowed_agents_from_state(state)

    # Brushing data dari user_context (juga bisa dari rapot_peri agent di Phase 2).
    # Skip kalau rapot_peri OFF — konsisten dengan tool gating.
    brushing = ctx.get("brushing")
    if brushing and "rapot_peri" in _allowed_agents_set:
        system += (
            f"\n\nData sikat gigi {child_name}: "
            f"streak {brushing.get('current_streak', 0)} hari, "
            f"rekor terbaik {brushing.get('best_streak', 0)} hari."
        )
    elif brushing:
        logger.debug(
            f"[generate] Skip brushing inject — rapot_peri not in allowed_agents "
            f"(brushing field present di user_context tapi agent OFF)"
        )

    # Mata Peri scan result. Skip kalau mata_peri OFF.
    mata_peri = ctx.get("mata_peri_last_result")
    if mata_peri and mata_peri.get("summary_text") and "mata_peri" in _allowed_agents_set:
        system += (
            f"\n\nHasil scan gigi terakhir {child_name} "
            f"({mata_peri.get('scan_date', 'tidak diketahui')}): "
            f"{mata_peri.get('summary_text')}. "
            f"Status: {mata_peri.get('summary_status', 'tidak diketahui')}."
        )
    elif mata_peri and mata_peri.get("summary_text"):
        logger.debug(
            f"[generate] Skip mata_peri scan inject — mata_peri not in allowed_agents"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Bagian C v5 — Robust List-Based Tools Awareness Block ─────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Drop keyword-based detection (terlalu brittle — banyak miss case like
    # "streak" not matching "Rapot Sikat Gigi"). New approach: kalau ada
    # minimal 1 agent OFF di akun user, inject explicit list ke system prompt:
    #   - Fitur yang AKTIF (tools available)
    #   - Fitur yang TIDAK AKTIF (tools NOT available)
    # Plus aturan: "kalau user nanya feature OFF, jawab honest. Kalau gak,
    # jawab normal." LLM smart enough untuk match user query ke list via
    # natural language understanding.
    #
    # Why this is better:
    # - No keyword maintenance — fitur baru cuma update AGENT_KEY_TO_FEATURE_NAME
    # - Robust — LLM bisa match "streak" → Rapot Peri tanpa hard-code
    # - Lower latency — no extra LLM call for classification
    # - Self-documenting — system prompt jelas about what's available
    #
    # Conditional inject: kalau SEMUA agent ON, block tidak muncul (clean prompt
    # untuk normal user, tidak overhead token).
    #
    # Position: di ATAS memory context supaya primacy effect — LLM lebih taat
    # instruksi awal daripada data spesifik di summary lama.
    #
    # Layer 3 (Bagian C v2) — unavailable_tools dari tool_bridge masih dipakai
    # sebagai SECONDARY signal: kalau LLM halusinasi panggil tool yang OFF,
    # tool_bridge mark di state. Kita combine dengan list dari allowed_agents
    # untuk lengkapi context.
    unavailable_tools = state.get("unavailable_tools", []) or []
    no_tools_reason = state.get("no_tools_reason")

    available_features = _get_available_features_for_state(state)
    unavailable_features = _get_unavailable_features_for_state(state)

    # Optionally augment unavailable_features dengan tool yang LLM coba panggil
    # (Layer 3 catch). Convert tool_name → friendly feature name.
    if unavailable_tools:
        from app.agents.tools.registry import get_friendly_feature_name
        for tool_name in unavailable_tools:
            feature = get_friendly_feature_name(tool_name)
            if feature and feature not in unavailable_features:
                unavailable_features = sorted(unavailable_features + [feature])

    if unavailable_features:
        # Build list strings
        available_text = (
            "\n".join(f"- {f}" for f in available_features)
            if available_features
            else "- (tidak ada — semua fitur sedang tidak aktif)"
        )
        unavailable_text = "\n".join(f"- {f}" for f in unavailable_features)

        system += (
            f"\n\n⚠️ INFO PENTING — TOOLS YANG TERSEDIA UNTUK USER INI:\n"
            f"\n"
            f"Fitur yang AKTIF (kamu bisa panggil tool / jawab pakai data):\n"
            f"{available_text}\n"
            f"\n"
            f"Fitur yang TIDAK AKTIF (BELUM tersedia untuk akun ini):\n"
            f"{unavailable_text}\n"
            f"\n"
            f"ATURAN MUTLAK:\n"
            f"1. Kalau user nanya tentang fitur yang TIDAK AKTIF (atau aspek apapun "
            f"dari fitur itu — streak, modul, riwayat, scan, badge, kuesioner, dll), "
            f"JAWAB HONEST: \"Maaf {user_name}, saat ini fitur [X] belum tersedia "
            f"di akun {child_name}.\" Sarankan alternative dari fitur yang AKTIF.\n"
            f"2. JANGAN karang/halusinasi data fitur yang TIDAK AKTIF (streak X hari, "
            f"modul Y selesai, scan Z, badge yang dapat, dll). Kamu TIDAK PUNYA data "
            f"current untuk fitur OFF.\n"
            f"3. JANGAN sebut alasan teknis (admin, allowed_agents, gated, off, dll) "
            f"— cukup bilang 'belum tersedia'.\n"
            f"4. Kalau user nanya tentang fitur AKTIF, jawab normal pakai data dari "
            f"tool call atau konteks yang tersedia.\n"
            f"5. PENTING: Walaupun di 'Ringkasan percakapan sebelumnya' (DI BAWAH) "
            f"ada info tentang fitur OFF dari session lama (e.g., 'Modul 4 terkunci', "
            f"'streak X hari'), JANGAN gunakan sebagai DATA CURRENT. Itu cuma "
            f"context historis dari session lama; fitur-nya sekarang OFF.\n"
            f"\n"
            f"Contoh respons baik (kalau user nanya feature OFF):\n"
            f"\"Maaf {user_name}, saat ini fitur [nama fitur] belum tersedia di akun "
            f"{child_name}. Tapi Peri masih bisa bantu untuk fitur lain yang aktif "
            f"ya! 🧚✨\"\n"
        )

    # ── Memory context (L2 + L3) ──────────────────────────────────────────────
    memory = state.get("memory_context", {})

    summaries = memory.get("session_summaries", [])
    if summaries:
        # Bagian C v3 — Part C2: Sanitize summaries.
        # Memory summaries dari background job. Kalau session lama bahas feature
        # yang sekarang OFF, summary bisa leak info "modul terkunci" yang bikin
        # LLM halusinasi data pas user tanya feature OFF.
        # Strategy: detect summaries yang reference unavailable features, lalu
        # PREPEND disclaimer ("info historis, fitur sekarang OFF").
        sanitized_summaries, has_stale = _sanitize_session_summaries(
            summaries, state
        )

        if sanitized_summaries:
            summaries_text = "\n".join(f"- {s}" for s in sanitized_summaries)
            system += f"\n\nRingkasan percakapan sebelumnya:\n{summaries_text}"

            if has_stale:
                # Loud disclaimer ke LLM bahwa summaries punya info dari feature OFF
                system += (
                    f"\n\n⚠️ DISCLAIMER untuk Ringkasan di atas:\n"
                    f"Beberapa entry di ringkasan membahas feature yang SAAT INI TIDAK AKTIF "
                    f"di akun user. Detail data spesifik (status modul, badge, scan, dll) "
                    f"di ringkasan itu adalah info HISTORIS dari session lama.\n"
                    f"\n"
                    f"PERATURAN:\n"
                    f"- Pakai ringkasan untuk konteks topik percakapan saja.\n"
                    f"- JANGAN treat angka/status/data spesifik di ringkasan sebagai "
                    f"data CURRENT user.\n"
                    f"- Untuk data current, HANYA gunakan hasil tool call di session ini.\n"
                    f"- Kalau tidak ada tool call, JANGAN karang data."
                )

    facts = memory.get("user_facts", [])
    if facts:
        facts_text = "\n".join(
            f"- {f.get('value', '')}" for f in facts if f.get("value")
        )
        system += f"\n\nYang kamu ketahui tentang user ini:\n{facts_text}"

    # ── Agent results (Bagian C: registry-driven prompt injection) ────────────
    agent_results = state.get("agent_results", {})

    # FIX (Bagian C): Loop semua ToolSpec yang punya prompt_injector dan inject
    # data dari agent_results ke system prompt. Ini fix CRITICAL BUG dimana
    # tool result sebelumnya HILANG (di-bridge tapi tidak di-consume), bikin
    # LLM halusinasi. Sekarang setiap tool inject data-nya sendiri.
    #
    # KENAPA pakai loop instead of inline if/elif:
    # - Tambah tool baru = update 1 file (tool itu sendiri), tidak perlu sentuh generate.py
    # - Impossible to forget — registry validate at startup
    # - Pattern consistent untuk all tools
    from app.agents.tools.registry import iter_tool_specs

    response_mode = state.get("response_mode", "simple")
    
    # SAFETY: ensure registry populated. iter_tool_specs() returns empty list
    # kalau tools/__init__.py belum di-import (edge case in test/standalone). 
    # Force-import to trigger registration.
    _all_specs = iter_tool_specs()
    if not _all_specs:
        try:
            import app.agents.tools  # noqa: F401 — trigger __init__ side effects
            _all_specs = iter_tool_specs()
            if not _all_specs:
                logger.error(
                    "[generate] Tool registry KOSONG setelah force-import. "
                    "Tool injection skipped — LLM akan halusinasi tanpa data!"
                )
        except Exception as e:
            logger.error(f"[generate] Failed to force-load tools registry: {e}")
            _all_specs = []
    
    for spec in _all_specs:
        if spec.prompt_injector is None:
            # Tool tidak butuh prompt injection (e.g., analyze_chat_image
            # pakai template terpisah di bawah)
            continue

        agent_data = agent_results.get(spec.agent_key)
        if not agent_data:
            # Tool tidak di-call this turn, atau result-nya empty
            continue

        try:
            injection_text = spec.prompt_injector(
                agent_data, child_name, prompts, response_mode
            )
        except Exception as e:
            logger.error(
                f"[generate] prompt_injector for '{spec.tool_name}' raised: {e}",
                exc_info=True,
            )
            injection_text = ""

        if injection_text:
            system += injection_text

    # NOTE: KB dental & FAQ docs inject lewat registry sekarang. Behavior 1:1
    # preserved — _inject_dental_kb dan _inject_app_faq di knowledge.py mirror
    # exact format string sebelumnya ("Referensi dari knowledge base:\n...").
    # Backward compat untuk legacy retrieved_docs (pre-Phase 2) handled below.

    # Backward compat — kalau ada retrieved_docs di state tapi tidak ke-inject
    # via kb_dental (e.g., dari old retrieve_node yang masih ada), inject manual.
    if not agent_results.get("kb_dental") and state.get("retrieved_docs"):
        legacy_docs = state.get("retrieved_docs", [])[:3]
        if legacy_docs:
            docs_text = "\n\n".join(legacy_docs)
            system += f"\n\nReferensi dari knowledge base:\n{docs_text}"

    # Image analysis (Mata Peri agent - Phase 2 / Tanya Peri image - Phase 4 Batch B)
    # Phase 6: prompts pindah ke DB — dipilih berdasarkan response_mode.
    #
    # Phase 4.1.2 Cluster 2 — Halusinasi guard (defense in depth):
    # Layer 1a — ai-cv produces gagal-safe text (orchestrator.py:600-614):
    #            saat foto invalid, summary_text & rec_text TIDAK mention gigi,
    #            cuma instruksi ambil ulang foto. Aman untuk di-inject langsung.
    # Layer 1b — Prompt branching: kalau sum_status='gagal', baik DB template
    #            (seed_prompts.py simple/medium/detailed v2) maupun fallback
    #            builder (_build_image_analysis_fallback_prompt) punya cabang
    #            khusus yang melarang LLM bahas kondisi gigi.
    # Catatan: DB template render pakai .format() tetap inject {summary_text} +
    # {recommendation_text} dengan teks ai-cv yang sudah safe. LLM tinggal
    # rephrase, bukan invent dari nol.
    image_analysis = state.get("image_analysis")
    if image_analysis:
        # AnalyzeResponse dari ai-cv punya struktur:
        # { session_summary: { summary_status, summary_text, recommendation_text, ... }, results: [...] }
        # Backward compat: kalau dari _analyze_image lama (deprecated), fallback ke field 'summary'.
        session_summary = image_analysis.get("session_summary") if isinstance(image_analysis, dict) else None

        if session_summary:
            sum_status = session_summary.get("summary_status") or "tidak diketahui"
            sum_text = session_summary.get("summary_text") or ""
            rec_text = session_summary.get("recommendation_text") or ""
            requires_review = session_summary.get("requires_dentist_review", False)

            # Status emoji untuk visual reinforcement
            status_emoji = {
                "ok": "✅",
                "perlu_perhatian": "⚠️",
                "perlu_dokter": "🚨",
                "gagal": "❌",
            }.get(sum_status, "📋")

            # Pull prompt template dari DB berdasarkan response_mode aktif.
            # Key: tanya_peri_image_response_{simple|medium|detailed}
            response_mode = state.get("response_mode", "simple")
            image_prompt_key = f"tanya_peri_image_response_{response_mode}"
            image_prompt_template = prompts.get(image_prompt_key)

            if image_prompt_template:
                # Render template dengan placeholder
                try:
                    rendered = image_prompt_template.format(
                        summary_status=sum_status,
                        summary_text=sum_text,
                        recommendation_text=rec_text,
                        child_name=child_name,
                        requires_dentist_review=str(requires_review).lower(),
                        status_emoji=status_emoji,
                    )
                    system += "\n\n" + rendered
                except KeyError as e:
                    # Prompt ada placeholder yang tidak kita support — log + fallback
                    logger.warning(
                        f"[generate] Prompt {image_prompt_key} pakai placeholder "
                        f"yang tidak dikenal: {e}. Fallback ke hardcoded format."
                    )
                    system += _build_image_analysis_fallback_prompt(
                        sum_status, sum_text, rec_text, child_name,
                        requires_review, status_emoji,
                    )
            else:
                # DB tidak punya prompt (seeder belum jalan / disabled) → hardcoded fallback
                system += _build_image_analysis_fallback_prompt(
                    sum_status, sum_text, rec_text, child_name,
                    requires_review, status_emoji,
                )
        else:
            # Backward compat — old _analyze_image format (Phase 2)
            system += (
                f"\n\nHasil analisis gambar gigi: "
                f"{image_analysis.get('summary', 'tidak tersedia')}."
            )

    # ── Response mode instructions ────────────────────────────────────────────
    response_mode = state.get("response_mode", "simple")
    mode_key = f"generate_{response_mode}"
    mode_instruction = prompts.get(mode_key)

    if not mode_instruction:
        # Default instructions per mode
        if response_mode == "simple":
            mode_instruction = (
                "Jawab dengan singkat, ramah, dan mudah dipahami. "
                "Gunakan bahasa sehari-hari. Boleh pakai emoji yang relevan. "
                "Maksimal 3-5 kalimat."
            )
        elif response_mode == "medium":
            mode_instruction = (
                "Jawab dengan cukup detail dan informatif. "
                "Gunakan 1-2 paragraf yang jelas. "
                "Tidak perlu cantumkan sumber atau referensi."
            )
        elif response_mode == "detailed":
            mode_instruction = (
                "Jawab secara lengkap dan ilmiah. Sertakan penjelasan mekanisme, "
                "pencegahan, dan penanganan. Cantumkan referensi atau sumber "
                "jika relevan (WHO, jurnal gigi, dll). "
                "Gunakan struktur yang jelas."
            )

    if mode_instruction:
        system += f"\n\nCara menjawab: {mode_instruction}"

    return system


def _build_messages(state: AgentState, system_prompt: str) -> list:
    """Build list LangChain messages dari state.

    PHASE 2A DEFENSIVE FIX: Skip empty assistant messages.
    Background: Sebelum Phase 2a fix1, agent_node mengappend empty AIMessage
    ke state.messages, yang ke-save ke DB via chat_messages.content="".
    Saat user kirim turn berikutnya, history dari DB include empty assistant
    rows. LLM dapat empty AIMessage → respond empty.
    Defensive filter ini protect future turns kalau ada empty assistant
    rows di DB (artifact dari turn yang gagal sebelum fix).
    """
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in state.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            # Skip empty assistant messages (artifact dari Phase 2a bug atau
            # turn yang gagal sebelum fix shipped).
            if not content or not str(content).strip():
                continue
            lc_messages.append(AIMessage(content=content))
        elif role == "user":
            # Always keep user messages (even empty — image-only upload valid)
            lc_messages.append(HumanMessage(content=content))
        # Other roles (system, tool) skipped — system prompt built separately,
        # tool results consumed via agent_results.
    return lc_messages


def _get_llm_for_state(state: AgentState):
    """Dapatkan LLM — bisa pakai override per-agent atau default global."""
    # Cek apakah ada agent-specific config
    agents_selected = state.get("agents_selected", [])
    agent_configs = state.get("agent_configs", {})

    provider_override = state.get("llm_provider_override")
    model_override = state.get("llm_model_override")

    # Jika ada satu agent dengan config khusus dan tidak ada multi-agent
    if len(agents_selected) == 1 and not provider_override:
        agent_key = agents_selected[0]
        agent_conf = agent_configs.get(agent_key, {})
        if agent_conf.get("llm_provider"):
            provider_override = agent_conf["llm_provider"]
        if agent_conf.get("llm_model"):
            model_override = agent_conf["llm_model"]

    return get_llm(
        streaming=True,
        provider=provider_override,
        model=model_override,
        temperature=state.get("llm_temperature_override"),
        max_tokens=state.get("llm_max_tokens_override"),
    ), provider_override, model_override


async def generate_node(state: AgentState) -> AsyncIterator[str]:
    """Node: generate — stream response dari LLM."""
    thinking_step = len(state.get("thinking_steps", [])) + 1

    agents_selected = state.get("agents_selected", [])
    intent_label = {
        "kb_dental": "Menyiapkan jawaban kesehatan gigi...",
        "user_profile": f"Membaca data {_get_child_name(state)}...",
        "rapot_peri": f"Mengecek rapot sikat gigi {_get_child_name(state)}...",
        "mata_peri": "Menganalisis hasil scan gigi...",
        "cerita_peri": "Mengecek progress Cerita Peri...",
        "app_faq": "Mencari info aplikasi...",
        "janji_peri": "Mencari informasi dokter...",
    }

    if len(agents_selected) > 1:
        thinking_label = "Merangkum informasi untuk kamu..."
    elif agents_selected:
        thinking_label = intent_label.get(agents_selected[0], "Menyiapkan respons...")
    else:
        thinking_label = "Menyiapkan respons..."

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=False)

    # Handle clarification (sudah ada di state dari check sebelumnya)
    if state.get("needs_clarification") and state.get("clarification_data"):
        clarify = state["clarification_data"]
        yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
        yield make_clarify_event(
            question=clarify.get("question", ""),
            options=clarify.get("options", []),
            allow_multiple=clarify.get("allow_multiple", False),
        )
        state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})
        return

    # Handle quick reply
    if state.get("quick_reply_data"):
        qr = state["quick_reply_data"]
        yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
        yield make_quick_reply_event(
            qr_type=qr.get("type", "single_select"),
            question=qr.get("question"),
            options=qr.get("options", []),
            allow_multiple=qr.get("allow_multiple", False),
            dismissible=qr.get("dismissible", True),
        )
        state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})
        return

    system_prompt = _build_system_prompt(state)
    lc_messages = _build_messages(state, system_prompt)

    if state.get("include_prompt_debug"):
        state["prompt_debug"] = {
            "system": system_prompt,
            "messages": [
                {"role": type(m).__name__, "content": m.content}
                for m in lc_messages
            ],
        }

    llm, _provider, _model = _get_llm_for_state(state)

    t_start = time.monotonic()
    t_first_token: float | None = None
    full_response = ""
    output_tokens = 0
    success = True
    error_msg: str | None = None

    yield make_thinking_event(step=thinking_step, label=thinking_label, done=True)
    state["thinking_steps"].append({"step": thinking_step, "label": thinking_label, "done": True})

    # Per-call trace config (no-op kalau Langfuse disabled)
    trace_config = build_trace_config(state=state, agent_name="generate")

    # Phase 4: extract last user message untuk trace input
    user_msg_text = ""
    last_user_msg = next(
        (m for m in reversed(state.get("messages", []))
         if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if last_user_msg:
        content = last_user_msg.get("content", "")
        if isinstance(content, str):
            user_msg_text = content
        elif isinstance(content, list):
            # Multimodal — extract text part
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_msg_text = part.get("text", "")
                    break

    # Phase 4: generation span — capture full prompt + output untuk diagnostic
    # Phase 4.5: tambah agent_results context yang membentuk system_prompt
    from app.config.observability import trace_generation, _safe_dict_for_trace

    # Phase 4.5: extract agent_results context — yang ini di-inject ke system_prompt
    agent_results = state.get("agent_results", {})
    image_analysis = state.get("image_analysis")
    memory_context = state.get("memory_context", {})
    retrieved_docs = state.get("retrieved_docs", []) or []

    # FIX (Langfuse audit): Filter SystemMessage dari messages untuk trace
    # supaya tidak duplicate dengan system_prompt arg yang sudah di-pass.
    # Sebelumnya: system prompt muncul 2x di Langfuse UI (sebagai arg + sebagai messages[0]).
    messages_for_trace = [m for m in lc_messages if not isinstance(m, SystemMessage)]

    # FIX (Langfuse audit): Track actual input/output token counts dari LLM response.
    # Sebelumnya: usage_details cuma report `output: chunk_count` (bukan token count).
    # Akibat: trace report "0 prompt → 7 completion" padahal actual 539 prompt → 119 completion.
    # Gemini API expose usage_metadata di chunks (terutama last chunk).
    input_tokens_actual = 0
    output_tokens_actual = 0

    async with trace_generation(
        name="generate",
        model=get_model_name(provider=_provider, model=_model),
        system_prompt=system_prompt,
        messages=messages_for_trace,  # FIX: tanpa SystemMessage (avoid duplicate)
        user_message=user_msg_text,
        metadata={
            "agents_used": list(agent_results.keys()),
            "response_mode": state.get("response_mode", "simple"),
            "has_image_analysis": bool(image_analysis),
            "child_name": _get_child_name(state),
            "provider": get_provider_name(provider=_provider),
            # Phase 4.5: cross-reference info supaya Hanif bisa correlate
            # context yang masuk → system_prompt yang di-render
            "agent_results_keys": list(agent_results.keys()),
            "image_analysis_present": bool(image_analysis),
            "kb_docs_count": len(retrieved_docs),
            "memory_summaries_count": len(memory_context.get("session_summaries", [])),
            "memory_facts_count": len(memory_context.get("user_facts", [])),
        },
    ) as gen_span:
        try:
            async for chunk in llm.astream(lc_messages, config=trace_config):
                # FIX (Langfuse audit): Capture token counts dari API response.
                # Gemini chunks may carry usage_metadata (terutama last chunk).
                # Pakai max() supaya kalau muncul di multiple chunks, ambil yang terbesar.
                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta:
                    input_tokens_actual = max(
                        input_tokens_actual,
                        usage_meta.get("input_tokens", 0) or 0,
                    )
                    output_tokens_actual = max(
                        output_tokens_actual,
                        usage_meta.get("output_tokens", 0) or 0,
                    )

                token = chunk.content
                if not token:
                    continue
                if t_first_token is None:
                    t_first_token = time.monotonic()
                full_response += token
                output_tokens += 1
                yield make_token_event(token)

        except Exception as e:
            success = False
            error_msg = str(e)
            if gen_span:
                gen_span.update(
                    output=f"[ERROR] {str(e)[:500]}",
                    level="ERROR",
                    status_message=str(e)[:200],
                )

        # Phase 4: capture output ke generation span (sebelum exit context)
        # Phase 4.5: tambah agent_results di metadata untuk full diagnostic
        if gen_span and success:
            try:
                # FIX (Langfuse audit): Report actual token counts from API response.
                # Fallback ke chunk count kalau API tidak expose usage_metadata.
                # Naming follows Langfuse convention: "input"/"output" tokens.
                usage_payload: dict = {}
                if input_tokens_actual > 0:
                    usage_payload["input"] = input_tokens_actual
                if output_tokens_actual > 0:
                    usage_payload["output"] = output_tokens_actual
                else:
                    # Fallback: chunk count (better than nothing, kalau API tidak expose)
                    usage_payload["output"] = output_tokens

                gen_span.update(
                    output=full_response[:5000] if full_response else "(empty)",
                    usage_details=usage_payload,
                    # Phase 4.5: capture agent_results in update metadata —
                    # diakses via "Additional Input" panel di Langfuse UI
                    metadata={
                        "agent_results_summary": _safe_dict_for_trace(agent_results),
                        "image_analysis_summary": _safe_dict_for_trace(image_analysis) if image_analysis else None,
                        "memory_context_summary": _safe_dict_for_trace(memory_context) if memory_context else None,
                        # FIX (Langfuse audit): Track output_tokens accuracy untuk debug
                        "output_chunks_yielded": output_tokens,
                        "output_tokens_from_api": output_tokens_actual,
                        "input_tokens_from_api": input_tokens_actual,
                    },
                )
            except Exception:
                pass  # defensive — jangan break flow kalau Langfuse error

    t_end = time.monotonic()
    total_latency_ms = int((t_end - t_start) * 1000)
    ttft_ms = int((t_first_token - t_start) * 1000) if t_first_token else None
    generation_ms = int((t_end - t_first_token) * 1000) if t_first_token else None
    tps = round(output_tokens / (generation_ms / 1000), 1) if generation_ms and generation_ms > 0 else None

    state["final_response"] = full_response
    # Build llm_metadata for client. Tambahan field has_image_analysis +
    # scan_session_id untuk FE render compact card di chat bubble.
    metadata: dict[str, Any] = {
        "model": get_model_name(provider=_provider, model=_model),
        "provider": get_provider_name(provider=_provider),
        "agents_used": list(state.get("agent_results", {}).keys()),
        "response_mode": state.get("response_mode", "simple"),
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "tokens_per_second": tps,
        "output_tokens_approx": output_tokens,
    }
    # Tag has_image_analysis kalau ada session_summary structure
    # (Tanya Peri image analysis dari Phase 4 Batch B).
    image_analysis_state = state.get("image_analysis")
    if isinstance(image_analysis_state, dict) and image_analysis_state.get("session_summary"):
        metadata["has_image_analysis"] = True
        scan_session_id = state.get("scan_session_id")
        if scan_session_id:
            metadata["scan_session_id"] = scan_session_id
        # Include compact subset of session_summary buat FE render card tanpa fetch ulang
        ss = image_analysis_state.get("session_summary", {})
        metadata["image_analysis_summary"] = {
            "summary_status": ss.get("summary_status"),
            "summary_text": ss.get("summary_text"),
            "recommendation_text": ss.get("recommendation_text"),
            "requires_dentist_review": ss.get("requires_dentist_review", False),
        }
        # Phase 6 — Forward overlay artifact URLs (mask gigi hijau + karies merah).
        # Konsisten dengan Mata Peri 5-view: same pattern dari ai-cv pipeline.
        # FE detail screen pakai ini untuk render foto dengan annotation.
        # NOTE: URL ini signed dengan TTL 1 jam. Untuk akses ulang setelah expired,
        # FE harus fetch detail dari /mata-peri/sessions/{id} yang regenerate URL fresh.
        #
        # Phase 4.1.2 fix (Bug B2+B3): Skip emit image_artifacts kalau result tidak valid.
        # Reasoning:
        #   - is_valid=False: ai-cv tidak detect mulut sama sekali (mis. foto buah,
        #     dokumen, dll). Render overlay = misleading user.
        #   - teeth.detected_count=0: mulut detected tapi tidak ada gigi visible
        #     (mulut tertutup, atau model gagal segment gigi). Render overlay
        #     hijau di area kosong = confuse user.
        # Dalam kedua case, ai-cv sudah set summary_status='gagal' atau text
        # yang appropriate (lihat orchestrator.py:600-614 di ai-cv) — jadi
        # AI tetap kasih response empati ke user untuk ambil ulang foto.
        # Yang kita skip cuma overlay-nya saja.
        # Plus surface flag image_unreadable supaya FE bisa render hint card
        # "Foto belum bisa terbaca jelas" tanpa overlay yang menyesatkan.
        any_skipped = False
        skip_count = 0
        image_results = image_analysis_state.get("results", []) or []
        if image_results and isinstance(image_results, list):
            artifacts_list = []
            for ir in image_results:
                if not isinstance(ir, dict):
                    continue

                # Phase 4.1.2: skip rendering kalau foto bermasalah
                is_valid = ir.get("is_valid", True)
                teeth = ir.get("teeth_result") or {}
                teeth_count = (
                    teeth.get("detected_count", 0)
                    if isinstance(teeth, dict) else 0
                )
                if not is_valid or teeth_count == 0:
                    any_skipped = True
                    skip_count += 1
                    continue  # skip — jangan render overlay menyesatkan

                artifacts = ir.get("artifacts") or {}
                if not isinstance(artifacts, dict):
                    continue
                view_type = ir.get("view_type")
                crop_url = artifacts.get("crop_image_url")
                overlay_url = artifacts.get("overlay_image_url")
                if crop_url or overlay_url:
                    artifacts_list.append({
                        "view_type": view_type,
                        "crop_image_url": crop_url,
                        "overlay_image_url": overlay_url,
                    })
            if artifacts_list:
                metadata["image_artifacts"] = artifacts_list

        # Phase 4.1.2 (Bug B3): Surface skip flag — FE bisa render hint card
        # "Foto belum bisa terbaca jelas" tanpa overlay yang menyesatkan.
        # Kunci 'image_unreadable' = boolean True kalau ALL views skipped (tidak
        # ada artifacts list yang valid).
        if any_skipped and not metadata.get("image_artifacts"):
            metadata["image_unreadable"] = True
            metadata["image_unreadable_count"] = skip_count
    state["llm_metadata"] = metadata

    log = LLMCallLogPayload(
        prompt_key=f"generate_{state.get('response_mode', 'simple')}",
        model=get_model_name(provider=_provider, model=_model),
        provider=get_provider_name(provider=_provider),
        node="generate",
        output_tokens=output_tokens,
        latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        success=success,
        error_message=error_msg,
        metadata={
            "agents_used": list(state.get("agent_results", {}).keys()),
            "response_mode": state.get("response_mode", "simple"),
            "generation_ms": generation_ms,
            "tokens_per_second": tps,
        },
    )
    state["llm_call_logs"].append(log.model_dump())

    # Emit suggestion chips jika ada
    if state.get("suggestion_chips"):
        from app.schemas.chat import make_suggestions_event
        yield make_suggestions_event(state["suggestion_chips"])


def _get_child_name(state: AgentState) -> str:
    ctx = state.get("user_context", {})
    child = ctx.get("child") or {}
    return child.get("nickname") or child.get("full_name") or "si kecil"


def _build_image_analysis_fallback_prompt(
    sum_status: str,
    sum_text: str,
    rec_text: str,
    child_name: str,
    requires_review: bool,
    status_emoji: str,
) -> str:
    """
    Hardcoded fallback prompt untuk image analysis kalau DB belum punya
    prompt key `tanya_peri_image_response_{simple|medium|detailed}`.

    Pattern: konteks + instruction berbasis poin (bukan paragraph).
    Pakai bahasa explicit "JANGAN minta upload" karena small LLM mudah miss.

    Catatan: setelah seeder jalan, function ini tidak akan ke-trigger.
    Sengaja simpan untuk safety net + dev tanpa DB seed.

    Phase 4.1.2 Cluster 2 — halusinasi guard:
    Kalau sum_status == 'gagal', branch ke template khusus yang TIDAK include
    summary_text/rec_text ai-cv (defense layer 2: pre-LLM context cleaning),
    plus instruksi explicit DILARANG bahas kondisi gigi. Tujuan: hindari LLM
    halusinasi "gigi tampak sehat" saat foto invalid (mis. foto buah, dokumen,
    mulut tertutup, dll). Lihat orchestrator.py:600-614 untuk source 'gagal'.
    """
    # Cluster 2 — Halusinasi guard: foto invalid → branch khusus
    if sum_status == "gagal":
        return (
            f"\n\n=== KONTEKS PENTING — FOTO INVALID ===\n"
            f"USER SUDAH MENGIRIM FOTO, tapi sistem AI TIDAK BERHASIL menganalisa "
            f"foto tersebut (kemungkinan: pencahayaan kurang, foto blur, mulut "
            f"tidak terlihat, atau bukan foto gigi).\n\n"
            f"📸 Status Analisis: {status_emoji} GAGAL\n\n"
            f"=== INSTRUKSI WAJIB (FOTO GAGAL DIANALISA) ===\n"
            f"WAJIB respond dengan template ini (3-4 kalimat saja):\n"
            f"1. Sapa empati singkat — sebut nama '{child_name}'.\n"
            f"2. Jelaskan foto belum bisa terbaca jelas — JANGAN sebutkan kondisi "
            f"gigi apapun, karena kamu TIDAK PUNYA data valid.\n"
            f"3. Sarankan ambil ulang dengan 1-2 tips: pencahayaan terang, mulut "
            f"terbuka lebar, atau kamera lebih dekat ke mulut.\n"
            f"4. Tutup dengan ajakan positif untuk coba lagi.\n\n"
            f"DILARANG KERAS:\n"
            f"- DILARANG mention 'gigi tampak sehat' / 'ada karies' / 'gigi bersih' / "
            f"sejenisnya — kamu tidak tau kondisi gigi.\n"
            f"- DILARANG sarankan konsultasi dokter gigi (tidak ada data untuk basis).\n"
            f"- DILARANG fabrikasi atau tebak hasil analisis.\n"
            f"- DILARANG minta upload foto lagi dengan kalimat '/upload' — cukup "
            f"sarankan ambil ulang foto yang lebih jelas.\n"
            f"\nTone: hangat, sabar, tidak menyalahkan user. Pakai bahasa "
            f"orang tua ke orang tua.\n"
            f"=== AKHIR KONTEKS ===\n"
        )

    # Status valid (ok / perlu_perhatian / segera_ke_dokter) — path normal
    return (
        f"\n\n=== KONTEKS PENTING ===\n"
        f"USER SUDAH MENGIRIM FOTO GIGI ANAK dan SUDAH dianalisis oleh sistem AI. "
        f"JANGAN minta user upload foto lagi.\n\n"
        f"📸 Hasil Analisis Foto Gigi {child_name}:\n"
        f"{status_emoji} Status: {sum_status}\n"
        f"📝 Ringkasan: {sum_text}\n"
        f"💡 Saran: {rec_text}\n\n"
        f"INSTRUKSI MERESPON:\n"
        f"1. JANGAN minta user upload foto — foto sudah ada dan sudah dianalisis.\n"
        f"2. Acknowledge hasil analisis singkat (1-2 kalimat empati).\n"
        f"3. Jawab pertanyaan user kalau ada.\n"
        f"4. Hasil sudah ditampilkan di card terpisah — cukup komentar singkat.\n"
        f"5. {'Anjurkan konsultasi dokter gigi.' if requires_review else 'Tetap pantau dan jaga kebiasaan sikat gigi.'}\n"
        f"=== AKHIR KONTEKS ===\n"
    )
