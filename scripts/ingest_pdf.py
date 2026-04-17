"""
Ingest PDF ke Qdrant vector store untuk knowledge base dental Tanya Peri.

Cara pakai:
    docker compose exec ai-chat python scripts/ingest_pdf.py --pdf /path/to/dental.pdf

Dengan custom chunk size:
    docker compose exec ai-chat python scripts/ingest_pdf.py --pdf dental.pdf --chunk-size 400 --overlap 40

Ingest ke collection berbeda:
    docker compose exec ai-chat python scripts/ingest_pdf.py --pdf dental.pdf --collection peri_bugi_faq

Cek jumlah dokumen di collection:
    docker compose exec ai-chat python scripts/ingest_pdf.py --count

Hapus semua dokumen di collection (untuk re-ingest):
    docker compose exec ai-chat python scripts/ingest_pdf.py --clear

TIPS:
- Chunk size 300-500 = lebih spesifik, cocok untuk Q&A faktual
- Chunk size 600-800 = lebih kontekstual, cocok untuk penjelasan panjang
- Overlap 10-20% dari chunk size = hindari topik terpotong di boundary
- Untuk dental knowledge base, 400 chunk / 40 overlap sudah bagus
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_settings():
    from app.config.settings import settings
    return settings


def get_embeddings(settings):
    """Ambil embedding model sesuai config."""
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

        if settings.EMBEDDING_DEVICE.lower() == "cpu":
            device = "cpu"
        elif settings.EMBEDDING_DEVICE.lower() == "cuda":
            device = "cuda"

        print(f"  Loading embedding model: {model} (device: {device})")
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model or "models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )

    raise ValueError(f"EMBEDDING_PROVIDER tidak dikenal: {provider}")


def get_qdrant_client(settings):
    from qdrant_client import QdrantClient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )


def count_docs(collection: str, settings) -> int:
    client = get_qdrant_client(settings)
    try:
        info = client.get_collection(collection)
        return info.points_count
    except Exception:
        return 0


def clear_collection(collection: str, settings) -> None:
    """Hapus semua dokumen di collection (bukan hapus collection-nya)."""
    from qdrant_client.models import Filter
    client = get_qdrant_client(settings)
    client.delete(collection_name=collection, points_selector=Filter())
    print(f"  ✓ Collection '{collection}' dikosongkan")


def ingest_pdf(
    pdf_path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    settings,
) -> None:
    """Load PDF, split jadi chunks, embed, simpan ke Qdrant."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    if not os.path.exists(pdf_path):
        print(f"ERROR: File tidak ditemukan: {pdf_path}")
        sys.exit(1)

    print(f"\nMemuat PDF: {os.path.basename(pdf_path)}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  {len(pages)} halaman dimuat")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"  {len(chunks)} chunks dibuat (size={chunk_size}, overlap={chunk_overlap})")

    # Tambah metadata sumber
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = os.path.basename(pdf_path)
        chunk.metadata["chunk_idx"] = i

    print(f"\nMembuat embedding dan menyimpan ke Qdrant collection '{collection}'...")

    embeddings = get_embeddings(settings)

    # Cek apakah collection sudah ada, kalau belum buat dulu
    client = get_qdrant_client(settings)
    collections = [c.name for c in client.get_collections().collections]

    if collection not in collections:
        test_vec = embeddings.embed_query("test")
        vec_size = len(test_vec)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        print(f"  Collection '{collection}' dibuat (dim={vec_size})")

    # Ingest dalam batch kecil supaya ada progress feedback
    batch_size = 50
    t_start = time.time()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )
        vector_store.add_documents(batch)

        done = min(i + batch_size, len(chunks))
        pct = int(done / len(chunks) * 100)
        elapsed = time.time() - t_start
        print(f"  [{pct:3d}%] {done}/{len(chunks)} chunks — {elapsed:.1f}s")

    total_time = time.time() - t_start
    final_count = count_docs(collection, settings)

    print(f"\n{'='*50}")
    print(f"✓ Ingest selesai!")
    print(f"  File     : {os.path.basename(pdf_path)}")
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Collection: {collection}")
    print(f"  Total docs: {final_count}")
    print(f"  Waktu    : {total_time:.1f}s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF ke Qdrant vector store untuk Tanya Peri"
    )
    parser.add_argument("--pdf", type=str, help="Path ke file PDF")
    parser.add_argument("--collection", type=str, default=None,
                        help="Nama Qdrant collection (default: dari .env QDRANT_COLLECTION)")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Ukuran chunk dalam karakter (default: 400)")
    parser.add_argument("--overlap", type=int, default=40,
                        help="Overlap antar chunk dalam karakter (default: 40)")
    parser.add_argument("--count", action="store_true",
                        help="Hitung jumlah dokumen di collection")
    parser.add_argument("--clear", action="store_true",
                        help="Hapus semua dokumen di collection sebelum ingest")

    args = parser.parse_args()
    settings = get_settings()
    collection = args.collection or settings.QDRANT_COLLECTION

    if args.count:
        n = count_docs(collection, settings)
        print(f"Collection '{collection}': {n} dokumen")
        return

    if not args.pdf and not args.clear:
        parser.print_help()
        print("\nContoh:")
        print("  # Ingest PDF:")
        print("  python scripts/ingest_pdf.py --pdf dental_guide.pdf")
        print("\n  # Cek jumlah docs:")
        print("  python scripts/ingest_pdf.py --count")
        print("\n  # Hapus semua dan ingest ulang:")
        print("  python scripts/ingest_pdf.py --clear --pdf dental_guide.pdf")
        return

    if args.clear:
        print(f"Menghapus semua dokumen di collection '{collection}'...")
        clear_collection(collection, settings)

    if args.pdf:
        ingest_pdf(
            pdf_path=args.pdf,
            collection=collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            settings=settings,
        )


if __name__ == "__main__":
    main()
