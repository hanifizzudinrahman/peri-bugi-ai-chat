"""Galat dari dalam blok ter-trace harus lolos apa adanya.

Kenapa ada
----------
`trace_node` dan `trace_generation` dulu membungkus seluruh isinya dengan satu
`try/except Exception` yang diakhiri `yield None`. Bentuk itu benar untuk
menelan kegagalan instrumentasi, tapi ia juga menangkap galat dari **badan**
blok `async with` — dan karena generatornya lalu `yield` untuk kedua kalinya,
Python menggantinya dengan:

    RuntimeError: generator didn't stop after athrow()

Penyebab aslinya hilang total. Yang benar-benar terjadi saat bug ini ketahuan:
penyedia LLM menolak dengan `PermissionDenied: 403` karena penagihan project,
dan yang muncul di log adalah RuntimeError yang tidak menyebut LLM sama sekali.
Berlaku di semua node ter-trace, termasuk jalur chat orang tua yang sudah live.

Test ini mengunci pembagian tugasnya: kegagalan instrumentasi ditelan,
kegagalan badan blok diteruskan.

Cara run:
    docker compose exec ai-chat pytest tests/unit/test_observability_error_passthrough.py -v
"""
from unittest.mock import MagicMock, patch

import pytest

from app.config import observability as obs


class _FakeSpan:
    def __init__(self):
        self.updates = []
        self.exit_args = None

    def update(self, **kw):
        self.updates.append(kw)


class _FakeCM:
    """Meniru context manager yang dikembalikan Langfuse."""

    def __init__(self, span, *, raise_on_enter=False):
        self.span = span
        self.raise_on_enter = raise_on_enter
        self.exited_with = None

    def __enter__(self):
        if self.raise_on_enter:
            raise RuntimeError("instrumentasi gagal saat masuk")
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.exited_with = exc_type
        return False  # jangan pernah menelan


def _fake_client(cm):
    client = MagicMock()
    client.start_as_current_observation.return_value = cm
    return client


class TestGalatBadanDiteruskan:
    @pytest.mark.asyncio
    async def test_trace_node_meneruskan_galat_asli(self):
        span = _FakeSpan()
        cm = _FakeCM(span)
        with patch.object(obs, "get_langfuse_client", return_value=_fake_client(cm)):
            with pytest.raises(PermissionError, match="403 ditolak"):
                async with obs.trace_node(name="uji"):
                    raise PermissionError("403 ditolak")

        assert cm.exited_with is PermissionError, (
            "Span harus tetap ditutup dengan informasi galatnya."
        )

    @pytest.mark.asyncio
    async def test_trace_generation_meneruskan_galat_asli(self):
        span = _FakeSpan()
        cm = _FakeCM(span)
        with patch.object(obs, "get_langfuse_client", return_value=_fake_client(cm)):
            with pytest.raises(ValueError, match="model menolak"):
                async with obs.trace_generation(name="uji", model="m"):
                    raise ValueError("model menolak")

        assert cm.exited_with is ValueError

    @pytest.mark.asyncio
    async def test_bukan_runtime_error_topeng(self):
        """Penjaga khusus untuk gejala yang dulu muncul."""
        cm = _FakeCM(_FakeSpan())
        with patch.object(obs, "get_langfuse_client", return_value=_fake_client(cm)):
            try:
                async with obs.trace_node(name="uji"):
                    raise KeyError("penyebab_asli")
            except BaseException as e:
                assert not isinstance(e, RuntimeError) or "athrow" not in str(e), (
                    "Galat aslinya tertelan lagi dan diganti RuntimeError."
                )
                assert isinstance(e, KeyError)


class TestJalanNormal:
    @pytest.mark.asyncio
    async def test_span_diserahkan_dan_ditutup_bersih(self):
        span = _FakeSpan()
        cm = _FakeCM(span)
        with patch.object(obs, "get_langfuse_client", return_value=_fake_client(cm)):
            async with obs.trace_node(name="uji") as s:
                assert s is span
                s.update(output={"ok": True})

        assert cm.exited_with is None
        assert span.updates == [{"output": {"ok": True}}]


class TestKegagalanInstrumentasiTetapDitelan:
    """Yang ini TIDAK boleh berubah — node harus tetap jalan tanpa Langfuse."""

    @pytest.mark.asyncio
    async def test_gagal_saat_masuk_tidak_menjatuhkan_node(self):
        cm = _FakeCM(_FakeSpan(), raise_on_enter=True)
        with patch.object(obs, "get_langfuse_client", return_value=_fake_client(cm)):
            jalan = False
            async with obs.trace_node(name="uji") as s:
                assert s is None
                jalan = True
            assert jalan

    @pytest.mark.asyncio
    async def test_langfuse_mati_menghasilkan_span_none(self):
        with patch.object(obs, "get_langfuse_client", return_value=None):
            async with obs.trace_node(name="uji") as s:
                assert s is None
            async with obs.trace_generation(name="uji") as s:
                assert s is None
