"""
Script untuk ingest PDF knowledge base ke Qdrant.

Cara pakai (dari folder peri-bugi-ai-chat, conda env aktif):

    # Basic
    python scripts/ingest_pdf.py --pdf path/to/dental.pdf

    # Custom chunk size
    python scripts/ingest_pdf.py --pdf dental.pdf --chunk-size 400 --chunk-overlap 40

    # Ingest beberapa PDF sekaligus
    python scripts/ingest_pdf.py --pdf file1.pdf --pdf file2.pdf --pdf file3.pdf

    # Paksa CPU (kalau GPU ada masalah)
    python scripts/ingest_pdf.py --pdf dental.pdf --device cpu

    # Reset collection dulu sebelum ingest (hati-hati: hapus semua data lama)
    python scripts/ingest_pdf.py --pdf dental.pdf --reset-collection

    # Lihat isi collection yang sudah ada
    python scripts/ingest_pdf.py --info

Catatan:
    - Script ini dijalankan dari LUAR Docker (conda env), bukan dari dalam container.
    - QDRANT_URL di .env untuk script ini harus http://localhost:6333 (bukan qdrant:6333)
    - Container ai-chat pakai http://qdrant:6333 (nama service Docker internal)
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _print_step(icon: str, msg: str):
    print(f"{icon}  {msg}")


def _print_ok(msg: str):
    print(f"   ✅ {msg}")


def _print_info(msg: str):
    print(f"   ℹ️  {msg}")


def _print_warn(msg: str):
    print(f"   ⚠️  {msg}")


def _print_err(msg: str):
    print(f"   ❌ {msg}")


def get_embeddings(device_override: str | None = None):
    """
    Return embedding model dengan GPU support.
    Menggunakan langchain_huggingface (bukan langchain_community yang deprecated).
    """
    from app.config.settings import settings
    from app.agents.tools.retrieve import _detect_embedding_device

    provider = settings.EMBEDDING_PROVIDER
    model = settings.EMBEDDING_MODEL

    # Override device jika diset via CLI
    if device_override:
        device = device_override
    else:
        device = _detect_embedding_device()

    _print_info(f"Embedding provider: {provider}")
    _print_info(f"Embedding model: {model}")
    _print_info(f"Device: {device}")

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _print_info("Menggunakan langchain_huggingface (up-to-date)")
        except ImportError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from langchain_community.embeddings import HuggingFaceEmbeddings
            _print_warn("langchain_huggingface tidak terinstall, fallback ke langchain_community")
            _print_warn("Jalankan: pip install langchain-huggingface==0.1.2")

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    raise ValueError(f"EMBEDDING_PROVIDER tidak dikenal: {provider}")


def show_collection_info():
    """Tampilkan info collection yang sudah ada di Qdrant."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from qdrant_client import QdrantClient
    from app.config.settings import settings

    _print_step("🔍", f"Koneksi ke Qdrant: {settings.QDRANT_URL}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        collections = client.get_collections().collections

        if not collections:
            _print_info("Tidak ada collection di Qdrant")
            return

        print(f"\n{'Collection':<30} {'Vectors':>10} {'Status'}")
        print("-" * 55)
        for col in collections:
            info = client.get_collection(col.name)
            count = info.vectors_count or 0
            status = info.status
            marker = "← target" if col.name == settings.QDRANT_COLLECTION else ""
            print(f"{col.name:<30} {count:>10} vectors  {status}  {marker}")

        print()
        target = settings.QDRANT_COLLECTION
        try:
            info = client.get_collection(target)
            _print_ok(f"Collection '{target}': {info.vectors_count} vectors tersimpan")
        except Exception:
            _print_warn(f"Collection '{target}' belum ada — akan dibuat saat ingest pertama")

    except Exception as e:
        _print_err(f"Tidak bisa konek ke Qdrant: {e}")
        _print_warn("Pastikan Qdrant jalan: docker compose ps")
        _print_warn(f"QDRANT_URL di .env: {settings.QDRANT_URL}")


def ingest(
    pdf_paths: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 50,
    reset_collection: bool = False,
    device_override: str | None = None,
):
    """
    Ingest satu atau beberapa PDF ke Qdrant.

    Args:
        pdf_paths      : list path PDF yang akan diingest
        chunk_size     : ukuran chunk dalam karakter (default 500)
        chunk_overlap  : overlap antar chunk (default 50)
        batch_size     : jumlah chunks per batch upload ke Qdrant (default 50)
                         Kalau koneksi lambat atau ada timeout, kecilkan ke 20
        reset_collection: hapus collection lama sebelum ingest (default False)
        device_override: override device embedding ('cuda', 'cpu', atau None=auto)
    """
    import warnings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client.models import Distance, VectorParams
    from app.config.settings import settings

    # ── Validasi PDF ──────────────────────────────────────────────────────────
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            _print_err(f"File tidak ditemukan: {pdf_path}")
            sys.exit(1)

    # ── Load semua PDF ────────────────────────────────────────────────────────
    all_docs = []
    for pdf_path in pdf_paths:
        _print_step("📄", f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        _print_ok(f"{len(docs)} halaman")
        all_docs.extend(docs)

    total_pages = len(all_docs)
    _print_info(f"Total: {total_pages} halaman dari {len(pdf_paths)} file")

    # ── Split jadi chunks ─────────────────────────────────────────────────────
    _print_step("✂️ ", f"Splitting (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(all_docs)
    _print_ok(f"{len(chunks)} chunks siap diupload")

    # ── Koneksi Qdrant ────────────────────────────────────────────────────────
    _print_step("🔗", f"Koneksi ke Qdrant: {settings.QDRANT_URL}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        # Test koneksi
        client.get_collections()
        _print_ok("Terhubung ke Qdrant")
    except Exception as e:
        _print_err(f"Gagal konek ke Qdrant: {e}")
        _print_warn("Pastikan Qdrant jalan: docker compose ps")
        _print_warn("Untuk script ini (luar Docker): QDRANT_URL=http://localhost:6333")
        _print_warn("Untuk container ai-chat: QDRANT_URL=http://qdrant:6333")
        sys.exit(1)

    # ── Setup embedding ───────────────────────────────────────────────────────
    _print_step("🤖", "Loading embedding model...")
    _print_info("Model akan didownload otomatis jika belum ada (~500MB, hanya sekali)")
    embeddings = get_embeddings(device_override=device_override)

    # Test embedding untuk detect vector size
    _print_info("Menghitung vector size dari model...")
    sample = embeddings.embed_query("test dental health")
    vector_size = len(sample)
    _print_ok(f"Vector size: {vector_size} dimensions")

    # ── Setup collection ──────────────────────────────────────────────────────
    collections = [c.name for c in client.get_collections().collections]

    if reset_collection and settings.QDRANT_COLLECTION in collections:
        _print_step("🗑️ ", f"Reset collection '{settings.QDRANT_COLLECTION}'...")
        client.delete_collection(settings.QDRANT_COLLECTION)
        _print_ok("Collection dihapus")
        collections = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in collections:
        _print_step("📁", f"Membuat collection '{settings.QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        _print_ok(f"Collection dibuat (dim={vector_size}, metric=cosine)")
    else:
        existing = client.get_collection(settings.QDRANT_COLLECTION)
        existing_count = existing.vectors_count or 0
        _print_info(f"Collection '{settings.QDRANT_COLLECTION}' sudah ada ({existing_count} vectors)")
        _print_info("Chunks baru akan DITAMBAHKAN ke data yang sudah ada")
        _print_info("Gunakan --reset-collection untuk hapus data lama terlebih dahulu")

    # ── Upload dalam batch dengan progress ───────────────────────────────────
    _print_step("📤", f"Uploading {len(chunks)} chunks ke Qdrant (batch size: {batch_size})...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    uploaded = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        # Progress bar sederhana
        progress = int((batch_num / total_batches) * 30)
        bar = "█" * progress + "░" * (30 - progress)
        pct = int((batch_num / total_batches) * 100)
        print(f"\r   [{bar}] {pct}% ({batch_num}/{total_batches} batch)", end="", flush=True)

        try:
            vector_store.add_documents(batch)
            uploaded += len(batch)
        except Exception as e:
            print()  # newline setelah progress bar
            _print_err(f"Gagal upload batch {batch_num}: {e}")
            _print_warn(f"Berhasil upload {uploaded} chunks sebelum error")
            _print_warn("Jalankan ulang script — chunks yang sudah ada tidak akan duplikat")
            sys.exit(1)

    print()  # newline setelah progress bar selesai

    # ── Summary ───────────────────────────────────────────────────────────────
    final_info = client.get_collection(settings.QDRANT_COLLECTION)
    final_count = final_info.vectors_count or 0

    print()
    print("=" * 50)
    _print_ok(f"Ingest selesai!")
    _print_info(f"PDF diproses    : {len(pdf_paths)} file ({total_pages} halaman)")
    _print_info(f"Chunks diupload : {uploaded}")
    _print_info(f"Total vectors   : {final_count}")
    _print_info(f"Collection      : {settings.QDRANT_COLLECTION}")
    _print_info(f"Qdrant URL      : {settings.QDRANT_URL}")
    print("=" * 50)
    print()
    print("Sekarang RAG aktif! Test dengan:")
    print(f'  curl -X POST http://localhost:8003/chat/rnd \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"message": "apa itu karies?", "stream": false}}\'')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF knowledge base ke Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python scripts/ingest_pdf.py --pdf dental_guide.pdf
  python scripts/ingest_pdf.py --pdf file1.pdf --pdf file2.pdf
  python scripts/ingest_pdf.py --pdf dental.pdf --reset-collection
  python scripts/ingest_pdf.py --info
        """,
    )
    parser.add_argument(
        "--pdf",
        action="append",
        dest="pdf_paths",
        metavar="PATH",
        help="Path ke file PDF (bisa diulang untuk beberapa file)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Ukuran chunk dalam karakter (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap antar chunk (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Jumlah chunks per batch upload (default: 50, kecilkan jika timeout)",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Hapus collection lama sebelum ingest (HATI-HATI: hapus semua data)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override device embedding (default: dari EMBEDDING_DEVICE di .env)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Tampilkan info collection yang sudah ada, lalu keluar",
    )
    args = parser.parse_args()

    if args.info:
        show_collection_info()
        sys.exit(0)

    if not args.pdf_paths:
        parser.error("Harus ada minimal satu --pdf, atau gunakan --info untuk lihat collection")

    ingest(
        pdf_paths=args.pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset_collection=args.reset_collection,
        device_override=args.device,
    )
