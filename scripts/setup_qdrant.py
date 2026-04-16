"""
Setup Qdrant collections untuk Tanya Peri.

Yang dilakukan:
1. Buat collection peri_bugi_dental (jika belum ada)
2. Buat collection peri_bugi_faq (jika belum ada)
3. Ingest FAQ placeholder ke peri_bugi_faq
4. Info cara ingest PDF knowledge base dental

Cara pakai:
    docker compose exec ai-chat python scripts/setup_qdrant.py

Untuk ingest PDF dental knowledge base:
    docker compose exec ai-chat python scripts/ingest_pdf.py --pdf path/to/dental.pdf
"""
import asyncio
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FAQ_DOCUMENTS = [
    # Fitur Peri Bugi
    "Rapot Peri adalah fitur untuk memantau kebiasaan sikat gigi anak setiap hari. Orang tua bisa checklist pagi dan malam untuk melacak streak sikat gigi anak. Streak akan bertambah setiap hari jika checklist lengkap pagi dan malam.",
    "Mata Peri adalah fitur scan gigi anak menggunakan kamera HP. Foto gigi anak dan AI YOLO akan menganalisis kondisi gigi secara otomatis. Hasil scan menunjukkan kondisi gigi dan rekomendasi tindakan.",
    "Tanya Peri adalah asisten chatbot kesehatan gigi anak. Bisa menjawab pertanyaan seputar kesehatan gigi, memberikan tips, dan informasi edukatif tentang perawatan gigi anak.",
    "Janji Peri adalah fitur untuk booking jadwal ke dokter gigi di puskesmas terdekat. Pilih dokter, pilih jadwal yang tersedia, dan konfirmasi janji tanpa perlu telepon.",
    "Cerita Peri adalah konten edukasi dalam bentuk cerita bergambar tentang kesehatan gigi. Ada 6 modul dengan quiz di setiap modul. Selesaikan quiz untuk mendapat bintang.",
    "Dunia Game adalah fitur game edukatif tentang kesehatan gigi. Main game untuk kumpulkan XP dan badge. Semakin sering main, semakin banyak pencapaian.",
    # Cara pakai
    "Untuk mendaftar aplikasi Peri Bugi, klik tombol Daftar di halaman utama. Masukkan nomor HP untuk menerima kode OTP verifikasi. Setelah verifikasi, isi data profil orang tua dan data anak.",
    "Jika lupa password, klik Lupa Password di halaman login. Masukkan nomor HP yang terdaftar untuk menerima OTP reset password. Password baru bisa diset setelah verifikasi OTP.",
    "Aplikasi Peri Bugi bisa diakses di browser melalui app.peribugi.my.id. Bisa juga diinstall sebagai PWA (Progressive Web App) di HP untuk akses lebih mudah.",
    "Data profil anak bisa diubah di menu Profil kemudian pilih Data Anak. Masukkan nama lengkap, tanggal lahir, dan jenis kelamin anak.",
    # Streak dan rapot
    "Streak sikat gigi adalah jumlah hari berturut-turut anak sikat gigi lengkap pagi dan malam. Streak akan reset ke 0 jika melewatkan satu hari tanpa checklist. Target streak: 7 hari, 14 hari, 30 hari.",
    "Checklist sikat gigi bisa dilakukan di menu Rapot Peri. Centang slot pagi sebelum jam 12 siang dan slot malam sebelum tidur. Jika lupa checklist hari ini masih bisa dilakukan hari yang sama.",
    "Achievement atau pencapaian muncul saat anak mencapai milestone streak tertentu. Contoh: streak 7 hari, streak 14 hari, streak 30 hari. Achievement bisa dilihat di profil anak.",
    # FAQ teknis
    "Jika tidak bisa login, pastikan nomor HP yang dimasukkan benar dan sudah terdaftar. Coba kirim ulang OTP jika kode tidak diterima. Pastikan koneksi internet stabil.",
    "Foto gigi untuk Mata Peri sebaiknya diambil dengan cahaya yang cukup dan jarak sekitar 20-30 cm dari mulut. Pastikan semua gigi terlihat jelas di foto.",
    "Jika scan Mata Peri gagal, pastikan foto gigi jelas dan tidak blur. Coba ulangi foto dengan pencahayaan lebih baik. Pastikan koneksi internet stabil.",
    "Data sikat gigi dan progress tersimpan di server, jadi aman meski ganti HP atau browser. Login dengan nomor HP yang sama untuk mengakses data.",
]


def setup_qdrant():
    """Setup Qdrant collections dan ingest FAQ."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from app.config.settings import settings  # ai-chat pakai app.config, bukan app.core

    # Suppress warning untuk local dev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

    # Cek embedding dimensi
    embeddings = _get_embeddings(settings)
    test_vec = embeddings.embed_query("test")
    vec_size = len(test_vec)
    print(f"Embedding dimension: {vec_size}")

    # ── Collection peri_bugi_dental ───────────────────────────────────────────
    print(f"\nSetup collection '{settings.QDRANT_COLLECTION}'...")
    collections = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        print(f"  ✓ Collection '{settings.QDRANT_COLLECTION}' dibuat")
    else:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        print(f"  ~ Collection sudah ada ({info.points_count} dokumen)")

    # ── Collection peri_bugi_faq ──────────────────────────────────────────────
    faq_collection = getattr(settings, "QDRANT_FAQ_COLLECTION", "peri_bugi_faq")
    print(f"\nSetup collection '{faq_collection}'...")

    if faq_collection not in collections:
        client.create_collection(
            collection_name=faq_collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        print(f"  ✓ Collection '{faq_collection}' dibuat")
        need_ingest = True
    else:
        info = client.get_collection(faq_collection)
        existing_count = info.points_count
        print(f"  ~ Collection sudah ada ({existing_count} dokumen)")
        need_ingest = existing_count == 0

    # ── Ingest FAQ documents ──────────────────────────────────────────────────
    if need_ingest:
        print(f"\nIngesting {len(FAQ_DOCUMENTS)} FAQ dokumen...")
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.documents import Document

        docs = [
            Document(page_content=text, metadata={"source": "faq_placeholder", "idx": i})
            for i, text in enumerate(FAQ_DOCUMENTS)
        ]

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=faq_collection,
            embedding=embeddings,
        )
        vector_store.add_documents(docs)
        print(f"  ✓ {len(docs)} FAQ dokumen berhasil di-ingest")
    else:
        print("  ~ FAQ sudah ada, skip ingest")

    print("\n" + "=" * 50)
    print("Setup Qdrant selesai!")
    print("=" * 50)
    print(f"\n📚 Collection dental ({settings.QDRANT_COLLECTION}): siap (kosong)")
    print(f"   → Ingest PDF: python scripts/ingest_pdf.py --pdf <file.pdf>")
    print(f"\n❓ Collection FAQ ({faq_collection}): {len(FAQ_DOCUMENTS)} dokumen placeholder")
    print(f"   → Edit FAQ_DOCUMENTS di scripts/setup_qdrant.py untuk update konten")
    print("=" * 50)


def _get_embeddings(settings):
    """Get embedding model (sama dengan sub_agents/__init__.py)."""
    import warnings
    provider = settings.EMBEDDING_PROVIDER
    model = settings.EMBEDDING_MODEL

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from langchain_community.embeddings import HuggingFaceEmbeddings

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model or "models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY,
        )

    raise ValueError(f"EMBEDDING_PROVIDER tidak dikenal: {provider}")


if __name__ == "__main__":
    print("Setting up Qdrant collections untuk Tanya Peri...\n")
    setup_qdrant()
