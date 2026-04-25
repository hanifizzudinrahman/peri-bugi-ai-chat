"""
Unit tests untuk view_hint_detector (Hybrid Tiered classifier).

Coverage:
- Layer 1 (rule-based) detection untuk 5 view types
- Edge cases: empty, ambiguous, mixed
- Layer 2 (LLM) di-test dengan mock — tidak panggil real LLM

Cara run:
    docker compose exec ai-chat pytest tests/unit/test_view_hint_detector.py -v

NOTE: Jangan run test ini saat container ai-chat lagi sibuk — pakai env terpisah
kalau perlu (test environment), atau pakai `pytest -k "view_hint"` filter.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.agents.utils.view_hint_detector import (
    _detect_by_rule,
    detect_view_hint,
)


# =============================================================================
# Layer 1: Rule-based — happy path
# =============================================================================

class TestRuleDetectFront:
    """Pattern: depan, frontal, anterior, senyum."""

    @pytest.mark.parametrize("text", [
        "tampak depan",
        "ini tampak depan ya",
        "foto depan",
        "gambar depan anak saya",
        "gigi depan",
        "bagian depan mulut",
        "frontal",
        "ini foto anterior",
        "senyum anak",
        "TAMPAK DEPAN",  # case insensitive
    ])
    def test_rule_detects_front(self, text):
        assert _detect_by_rule(text) == "front"


class TestRuleDetectUpper:
    """Pattern: rahang atas, gigi atas, maksila."""

    @pytest.mark.parametrize("text", [
        "rahang atas",
        "gigi atas",
        "tampak atas",
        "geligi atas",
        "ini foto rahang atas anak",
        "atas gigi yang depan",
        "maksila",
        "maxilla",
    ])
    def test_rule_detects_upper(self, text):
        assert _detect_by_rule(text) == "upper"


class TestRuleDetectLower:
    """Pattern: rahang bawah, gigi bawah, mandibula."""

    @pytest.mark.parametrize("text", [
        "rahang bawah",
        "gigi bawah",
        "tampak bawah",
        "geligi bawah",
        "bawah gigi anak",
        "mandibula",
        "mandible",
    ])
    def test_rule_detects_lower(self, text):
        assert _detect_by_rule(text) == "lower"


class TestRuleDetectLeft:
    """Pattern: sisi kiri, pipi kiri, sebelah kiri."""

    @pytest.mark.parametrize("text", [
        "sisi kiri",
        "pipi kiri",
        "sebelah kiri",
        "tampak kiri",
        "bagian kiri mulut",
        "gigi kiri belakang",
    ])
    def test_rule_detects_left(self, text):
        assert _detect_by_rule(text) == "left"


class TestRuleDetectRight:
    """Pattern: sisi kanan, pipi kanan, sebelah kanan."""

    @pytest.mark.parametrize("text", [
        "sisi kanan",
        "pipi kanan",
        "sebelah kanan",
        "tampak kanan",
        "bagian kanan mulut",
        "gigi kanan",
    ])
    def test_rule_detects_right(self, text):
        assert _detect_by_rule(text) == "right"


# =============================================================================
# Layer 1: Rule-based — edge cases
# =============================================================================

class TestRuleEdgeCases:
    """Test edge cases yang mungkin terjadi di production."""

    def test_empty_string_returns_none(self):
        assert _detect_by_rule("") is None

    def test_whitespace_only_returns_none(self):
        assert _detect_by_rule("   \n\t  ") is None

    def test_no_match_returns_none(self):
        """Pesan generic tanpa hint angle."""
        assert _detect_by_rule("tolong cek") is None
        assert _detect_by_rule("ini bagaimana ya") is None
        assert _detect_by_rule("anak saya umur 5 tahun") is None

    def test_misspelled_or_unrelated_returns_none(self):
        """Pesan dengan typo / tidak relevan."""
        assert _detect_by_rule("dpan") is None  # typo "depan"
        assert _detect_by_rule("gigi") is None  # ambigu, tanpa context

    def test_multiple_views_returns_first_in_iteration_order(self):
        """
        Kalau text mention multiple views, function return view pertama
        yang match (sesuai iteration order: front → upper → lower → left → right).
        Phase 1 behavior — Phase 2 bisa tambah scoring.
        """
        # "depan dan atas" → front (front check first)
        result = _detect_by_rule("ini foto gigi depan dan atas")
        assert result == "front"

    def test_long_text_with_hint_buried_inside(self):
        """Text panjang, hint di tengah."""
        text = (
            "Halo Peri, saya khawatir nih sama gigi anak saya, "
            "ini foto gigi depan dia ya, kira-kira ada masalah ngga?"
        )
        assert _detect_by_rule(text) == "front"


# =============================================================================
# Layer 1: Rule-based — pattern boundary correctness
# =============================================================================

class TestRulePatternBoundaries:
    """
    Test bahwa pattern tidak false-positive untuk substring yang bukan target.

    Contoh:
    - "atas" bisa muncul di kata lain seperti "selesainya kapan", harus tidak match
    - Pattern \\b (word boundary) penting untuk avoid this
    """

    def test_atas_alone_not_matched_without_dental_context(self):
        """Word 'atas' tanpa konteks gigi/rahang/mulut — tidak match upper."""
        # "kemarin atas saran dokter" — no "gigi/rahang" context
        result = _detect_by_rule("kemarin atas saran dokter")
        assert result is None

    def test_kanan_alone_in_unrelated_context(self):
        """'kanan' di konteks selain anatomy — tidak match right."""
        # "belok kanan" — bukan hint angle
        result = _detect_by_rule("dia belok kanan tadi")
        assert result is None


# =============================================================================
# Layer 2 + Hybrid Flow (LLM mocked)
# =============================================================================

class TestHybridDetectViewHint:
    """
    Test full hybrid flow: rule first, LLM fallback.
    LLM dipancing pakai mock untuk avoid real API call.
    """

    @pytest.mark.asyncio
    async def test_rule_hit_skips_llm(self):
        """
        Kalau rule match, LLM tidak dipanggil (efficiency).
        Verify dengan check decision_source = 'rule'.
        """
        hint, source = await detect_view_hint(
            text="rahang atas",
            enable_llm_fallback=True,  # even dengan LLM enabled
        )
        assert hint == "upper"
        assert source == "rule"

    @pytest.mark.asyncio
    async def test_rule_miss_triggers_llm(self):
        """
        Kalau rule miss, LLM dipanggil. Mock LLM return 'front'.
        Verify decision_source = 'llm'.
        """
        # Mock _detect_by_llm to return "front"
        with patch(
            "app.agents.utils.view_hint_detector._detect_by_llm",
            new=AsyncMock(return_value="front"),
        ):
            hint, source = await detect_view_hint(
                text="ini gigi belakang anak yang sebelah",  # ambiguous
                enable_llm_fallback=True,
            )
            assert hint == "front"
            assert source == "llm"

    @pytest.mark.asyncio
    async def test_both_layers_fail_returns_ambiguous(self):
        """
        Kalau rule miss + LLM return None → (None, 'ambiguous').
        Caller harus emit clarification card.
        """
        with patch(
            "app.agents.utils.view_hint_detector._detect_by_llm",
            new=AsyncMock(return_value=None),
        ):
            hint, source = await detect_view_hint(
                text="tolong",
                enable_llm_fallback=True,
            )
            assert hint is None
            assert source == "ambiguous"

    @pytest.mark.asyncio
    async def test_llm_disabled_returns_ambiguous_on_rule_miss(self):
        """
        Disable LLM fallback (test mode) → kalau rule miss langsung ambiguous.
        """
        hint, source = await detect_view_hint(
            text="tolong cek",
            enable_llm_fallback=False,
        )
        assert hint is None
        assert source == "ambiguous"

    @pytest.mark.asyncio
    async def test_empty_text_returns_ambiguous(self):
        """Empty text → ambiguous (no LLM call needed)."""
        hint, source = await detect_view_hint(
            text="",
            enable_llm_fallback=False,
        )
        assert hint is None
        assert source == "ambiguous"
