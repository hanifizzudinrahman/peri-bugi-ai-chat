"""
View Hint Detector — Hybrid Tiered Classification.

Detect angle gigi yang dimaksud user dari pesan teks. Dipakai oleh mata_peri_agent
saat user kirim foto + pesan untuk decide ke `view_type` apa foto akan dianalisa.

Pattern: Hybrid Tiered (production-grade pattern dipakai di Anthropic/Google/etc):
  Layer 1: Rule-based regex (95% case, 0ms, free)
  Layer 2: LLM fallback (5% case, ~500ms, ~$0.0001)

Decision flow:
  text → Layer 1 (rule)
       → match? return (hint, 'rule')
       → no match → Layer 2 (LLM)
                  → confident? return (hint, 'llm')
                  → uncertain → return (None, 'ambiguous')

Returns: tuple (ViewHint | None, decision_source: str)

Output decision_source untuk:
- Logging/observability (track rule vs LLM hit rate, identify rule gaps)
- A/B testing (compare quality rule vs LLM)
- Future: improve rules dari LLM-decided cases
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Type alias untuk view hint
ViewHint = Literal["front", "upper", "lower", "left", "right"]

# =============================================================================
# Layer 1: Rule-based patterns (Indonesian)
# =============================================================================
#
# Patterns ditulis untuk capture cara ibu Indonesia mendeskripsikan angle gigi.
# Order matters: more specific patterns checked first.
#
# Maintenance principle:
# - Tambah pattern saat menemukan cara baru user describe angle (dari LLM hits)
# - Test patterns dengan variasi typo, spasi, capitalization
# - Hindari greedy match yang false-positive (mis. "atas" tanpa "rahang/gigi")
# =============================================================================

_VIEW_PATTERNS: dict[ViewHint, list[str]] = {
    "front": [
        # Eksplisit "tampak depan", "foto depan", "gigi depan"
        r"\btampak\s*depan\b",
        r"\bfoto\s*depan\b",
        r"\bgambar\s*depan\b",
        r"\bgigi\s*depan\b",
        r"\bbagian\s*depan\b",
        # Frontal/anterior (dental terminology)
        r"\bfrontal\b",
        r"\banterior\b",
        # Senyum / depan mulut (foto ekspresi senyum biasanya frontal)
        r"\bsenyum\b",
    ],
    "upper": [
        # Eksplisit "rahang atas", "gigi atas"
        r"\brahang\s*atas\b",
        r"\bgigi\s*atas\b",
        r"\bgigi\s*-?\s*gigi\s*atas\b",
        r"\bgeligi\s*atas\b",
        r"\btampak\s*atas\b",
        # Atas + konteks gigi/mulut
        r"\batas\s*(gigi|rahang|mulut)\b",
        # Dental terminology
        r"\bmaksila\b",
        r"\bmaxilla\b",
    ],
    "lower": [
        r"\brahang\s*bawah\b",
        r"\bgigi\s*bawah\b",
        r"\bgigi\s*-?\s*gigi\s*bawah\b",
        r"\bgeligi\s*bawah\b",
        r"\btampak\s*bawah\b",
        r"\bbawah\s*(gigi|rahang|mulut)\b",
        # Dental terminology
        r"\bmandibula\b",
        r"\bmandible\b",
    ],
    "left": [
        # Sisi kiri / pipi kiri / sebelah kiri
        r"\bsisi\s*kiri\b",
        r"\btampak\s*kiri\b",
        r"\bsebelah\s*kiri\b",
        r"\bpipi\s*kiri\b",
        r"\bbagian\s*kiri\b",
        r"\bgigi\s*kiri\b",
        # Dental terminology (rare di casual chat tapi worth)
        r"\bbukal\s*kiri\b",
    ],
    "right": [
        r"\bsisi\s*kanan\b",
        r"\btampak\s*kanan\b",
        r"\bsebelah\s*kanan\b",
        r"\bpipi\s*kanan\b",
        r"\bbagian\s*kanan\b",
        r"\bgigi\s*kanan\b",
        r"\bbukal\s*kanan\b",
    ],
}

# Compile patterns sekali saat module load (performance)
_COMPILED_PATTERNS: dict[ViewHint, list[re.Pattern]] = {
    view: [re.compile(p, re.IGNORECASE) for p in patterns]
    for view, patterns in _VIEW_PATTERNS.items()
}


# =============================================================================
# Layer 1: Rule-based detection
# =============================================================================

def _detect_by_rule(text: str) -> Optional[ViewHint]:
    """
    Layer 1 detection — match patterns, return first hit.

    Note: kalau text match multiple views (e.g. "gigi depan dan atas"),
    return view dengan match pertama (front/upper/lower/left/right order
    sesuai dict iteration). Untuk Phase 1 ini cukup; Phase 2 bisa scoring.

    Args:
        text: User message text

    Returns:
        ViewHint kalau match, None kalau tidak ada pattern match
    """
    if not text or not text.strip():
        return None

    text_normalized = text.strip()

    for view in ("front", "upper", "lower", "left", "right"):
        for pattern in _COMPILED_PATTERNS[view]:
            if pattern.search(text_normalized):
                logger.debug(
                    f"[view_hint] Rule matched: view={view} "
                    f"pattern={pattern.pattern[:50]} text={text_normalized[:60]}"
                )
                return view  # type: ignore[return-value]

    return None


# =============================================================================
# Layer 2: LLM fallback
# =============================================================================

# Prompt template untuk LLM classification.
# Design choices:
# - Single-task prompt (bukan multi-step) — fast, deterministic
# - Force single-word output dengan max_tokens=5
# - Include examples untuk few-shot learning
# - Temperature 0 untuk deterministic
_LLM_VIEW_CLASSIFICATION_PROMPT = """Klasifikasikan tampak gigi yang dimaksud dalam pesan ini.

Pesan user: "{text}"

Pilih SATU dari:
- front: tampak depan / frontal / foto senyum
- upper: rahang atas / gigi atas / maksila
- lower: rahang bawah / gigi bawah / mandibula
- left: sisi kiri / pipi kiri
- right: sisi kanan / pipi kanan
- unclear: pesan tidak menyebutkan angle spesifik

Contoh:
- "ini foto gigi depan anak saya" → front
- "tolong cek rahang atas" → upper
- "bisa lihat ini? saya khawatir" → unclear
- "geraham bawah kanan ada masalah" → right

Jawab DENGAN SATU KATA SAJA: front, upper, lower, left, right, atau unclear."""


_VALID_LLM_OUTPUTS = {"front", "upper", "lower", "left", "right"}


async def _detect_by_llm(
    text: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> Optional[ViewHint]:
    """
    Layer 2 detection — LLM fallback untuk text yang rule tidak handle.

    Lazy import langchain dependencies untuk avoid loading saat tidak dipakai.
    Gracefully handle failure — return None biar caller emit clarification.

    Args:
        text: User message text
        llm_provider: Override provider (default ke config)
        llm_model: Override model name
        prompt_template: Prompt dari DB (key='view_hint_classification').
                        Kalau None, fallback ke hardcoded constant.

    Returns:
        ViewHint kalau LLM confident, None kalau LLM tidak yakin / fail
    """
    if not text or not text.strip():
        return None

    try:
        # Lazy import — avoid load saat module init
        from langchain_core.messages import HumanMessage
        from app.config.llm import get_llm
    except ImportError as e:
        logger.warning(f"[view_hint] LLM imports failed, skip LLM layer: {e}")
        return None

    # Pakai prompt dari DB kalau ada, fallback ke hardcoded.
    # Ini memungkinkan admin update prompt tanpa redeploy.
    template = prompt_template or _LLM_VIEW_CLASSIFICATION_PROMPT
    prompt = template.format(text=text.strip()[:500])

    try:
        # Build LLM dengan low temp + tight token limit (cost efficient)
        llm = get_llm(
            temperature=0,
            max_tokens=5,
            streaming=False,
            provider=llm_provider,
            model=llm_model,
        )
        result = await llm.ainvoke([HumanMessage(content=prompt)])

        # Parse output — extract first word, lowercase
        raw_content = (result.content or "").strip().lower()
        first_word = raw_content.split()[0] if raw_content else ""
        # Strip punctuation that might come from LLM
        first_word = re.sub(r"[^\w]", "", first_word)

        if first_word in _VALID_LLM_OUTPUTS:
            logger.info(
                f"[view_hint] LLM matched: view={first_word} "
                f"text={text[:60]}"
            )
            return first_word  # type: ignore[return-value]

        logger.debug(
            f"[view_hint] LLM returned unclear: '{raw_content}' "
            f"text={text[:60]}"
        )
        return None

    except Exception as e:
        # Graceful failure — caller akan treat sebagai ambiguous
        logger.warning(f"[view_hint] LLM call failed: {e}")
        return None


# =============================================================================
# Public API: hybrid tiered detection
# =============================================================================

DecisionSource = Literal["rule", "llm", "ambiguous"]


async def detect_view_hint(
    text: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    enable_llm_fallback: bool = True,
    prompt_template: Optional[str] = None,
) -> tuple[Optional[ViewHint], DecisionSource]:
    """
    Hybrid Tiered View Hint Detection.

    Dispatches text to layered detection:
      Layer 1 (rule)   → fast, free, deterministic
      Layer 2 (LLM)    → slow, ~$0.0001, smart at handling natural language

    Args:
        text: User message text
        llm_provider: Override LLM provider for Layer 2 (default config)
        llm_model: Override LLM model for Layer 2 (default config)
        enable_llm_fallback: Set False untuk skip Layer 2 (e.g. test mode)
        prompt_template: Prompt dari DB key='view_hint_classification'
                        untuk Layer 2. Kalau None, fallback ke hardcoded.

    Returns:
        (ViewHint | None, decision_source)
        - ('front', 'rule')      → Layer 1 hit
        - ('upper', 'llm')       → Layer 2 hit (rule miss)
        - (None, 'ambiguous')    → Both layers can't decide → emit clarification

    Usage:
        hint, source = await detect_view_hint(user_text)
        if hint:
            # Proceed analyze with this hint
            ...
        else:
            # Emit ClarificationCard untuk minta user pilih
            ...
    """
    # Layer 1: Rule-based (fast path)
    rule_hint = _detect_by_rule(text)
    if rule_hint:
        return rule_hint, "rule"

    # Layer 2: LLM fallback (slow path)
    if enable_llm_fallback:
        llm_hint = await _detect_by_llm(
            text, llm_provider, llm_model, prompt_template=prompt_template
        )
        if llm_hint:
            return llm_hint, "llm"

    # Both layers can't decide
    return None, "ambiguous"
