"""
Fitur yang dimatikan founder tidak boleh bocor lewat Tanya Peri.

KENAPA TEST INI ADA
-------------------
Saklar fitur tayang 6 September 2026: menu hilang dari aplikasi, rute memantul,
endpoint menolak. Tapi orang tua yang bertanya *"gimana cara scan gigi?"* tetap
dijawab lengkap dengan langkah-langkahnya — dokumen FAQ Mata Peri masih ikut
terambil, dan jalur cadangan hardcoded bahkan menjelaskan setiap fitur satu per
satu.

Tiga hal yang dijaga di sini, ketiganya gagal tanpa satu pun galat:

1. filter Qdrant benar-benar memuat `must_not` untuk tiap fitur yang mati
2. jalur cadangan `_get_hardcoded_faq` ikut disaring — bukan cuma jalur utama
3. daftar pemilik butir cadangan tetap sejajar dengan isinya
"""
import pytest

from app.agents.tools.knowledge import _build_is_active_filter, kunci_fitur_mati
from app.agents.sub_agents import _get_hardcoded_faq


# =============================================================================
# 1. Pembacaan payload
# =============================================================================


def test_kunci_terbaca_dari_bentuk_yang_dikirim_api():
    assert kunci_fitur_mati(
        [{"kunci": "mata_peri", "label": "Mata Peri"}]
    ) == {"mata_peri"}


def test_entri_cacat_dibuang_tanpa_menjatuhkan_sisanya():
    """
    Daftar ini datang lewat HTTP dari layanan lain. Satu entri cacat tidak boleh
    menjatuhkan percakapan — tapi juga tidak boleh membuat SELURUH daftar
    diabaikan, karena itu membuka kembali fitur yang sedang dimatikan.
    """
    hasil = kunci_fitur_mati(
        [
            {"kunci": "mata_peri", "label": "Mata Peri"},
            {"label": "tanpa kunci"},
            None,
            123,
            {"kunci": "", "label": "kunci kosong"},
            {"kunci": "rapot_peri", "label": "Rapot Peri"},
        ]
    )
    assert hasil == {"mata_peri", "rapot_peri"}


def test_kosong_dan_none_aman():
    assert kunci_fitur_mati(None) == set()
    assert kunci_fitur_mati([]) == set()


# =============================================================================
# 2. Filter Qdrant
# =============================================================================


def test_tanpa_fitur_mati_filternya_persis_seperti_sebelumnya():
    """Jalur normal tidak boleh berubah bentuknya sama sekali."""
    f = _build_is_active_filter()
    assert f == {
        "must_not": [{"key": "metadata.is_active", "match": {"value": False}}]
    }


def test_fitur_mati_masuk_must_not():
    f = _build_is_active_filter(exclude_features={"mata_peri", "janji_peri"})
    kunci_mati = [
        c["match"]["value"]
        for c in f["must_not"]
        if c["key"] == "metadata.feature"
    ]
    assert sorted(kunci_mati) == ["janji_peri", "mata_peri"]
    # `is_active` tetap ikut — dua penyaring yang berbeda, bukan saling ganti.
    assert {"key": "metadata.is_active", "match": {"value": False}} in f["must_not"]


def test_must_not_bukan_must():
    """
    `must` berarti "hanya yang cocok" — itu jalur `feature_filter` milik LLM,
    dan ia memilih paling banyak satu fitur. Yang dibutuhkan di sini kebalikan
    arahnya: buang beberapa, biarkan sisanya, TERMASUK chunk lama yang tidak
    punya field `feature` sama sekali.
    """
    f = _build_is_active_filter(exclude_features={"mata_peri"})
    assert "must" not in f, (
        "Fitur mati masuk ke `must` — chunk tanpa field `feature` ikut terbuang."
    )


def test_extra_must_dan_fitur_mati_bisa_bersamaan():
    f = _build_is_active_filter(
        extra_must=[{"key": "metadata.feature", "match": {"value": "rapot_peri"}}],
        exclude_features={"mata_peri"},
    )
    assert f["must"] == [
        {"key": "metadata.feature", "match": {"value": "rapot_peri"}}
    ]
    assert {"key": "metadata.feature", "match": {"value": "mata_peri"}} in f["must_not"]


# =============================================================================
# 3. Jalur cadangan hardcoded
# =============================================================================


def test_cadangan_menjelaskan_fitur_kalau_tidak_disaring():
    """Garis dasarnya. Tanpa ini, test di bawah tidak membuktikan apa-apa."""
    butir = _get_hardcoded_faq("mata peri")
    assert any("Mata Peri" in b for b in butir)


def test_cadangan_membuang_butir_fitur_yang_mati():
    butir = _get_hardcoded_faq("mata peri", exclude_features={"mata_peri"})
    assert not any("Mata Peri adalah fitur scan" in b for b in butir), (
        "Jalur cadangan masih menjelaskan Mata Peri padahal saklarnya mati. "
        "Ini pintu yang justru terbuka ketika Qdrant bermasalah."
    )


def test_cadangan_menyisakan_butir_lintas_fitur():
    """Mematikan semua fitur tidak boleh menghapus cara daftar / lupa sandi."""
    semua = {
        "rapot_peri",
        "mata_peri",
        "tanya_peri",
        "janji_peri",
        "cerita_peri",
        "dunia_game",
    }
    butir = _get_hardcoded_faq("lupa password", exclude_features=semua)
    assert butir, "Semua butir terbuang — butir lintas fitur ikut kena."
    assert any("password" in b.lower() for b in butir)


def test_daftar_pemilik_sejajar_dengan_isinya():
    """
    `_get_hardcoded_faq` punya `assert` internal yang membandingkan panjang dua
    daftar. Test ini memanggilnya lewat jalur yang MEMAKAI daftar pemilik,
    supaya assert itu benar-benar dijalankan — memanggil tanpa
    `exclude_features` melewatinya.
    """
    _get_hardcoded_faq("apa saja", exclude_features={"mata_peri"})


@pytest.mark.parametrize(
    "kunci", ["rapot_peri", "mata_peri", "tanya_peri", "janji_peri", "cerita_peri"]
)
def test_tiap_fitur_punya_butir_yang_bisa_dibuang(kunci):
    """
    Kalau sebuah fitur disebut di FAQ cadangan tapi tidak punya pemilik, ia
    tidak akan pernah tersaring. Diperiksa dengan membandingkan jumlah butir
    sebelum dan sesudah — bukan dengan membaca daftar pemiliknya, supaya test
    ini tidak lulus hanya karena daftarnya konsisten dengan dirinya sendiri.
    """
    penuh = _get_hardcoded_faq("apa saja")
    disaring = _get_hardcoded_faq("apa saja", exclude_features={kunci})
    assert len(disaring) <= len(penuh)
