"""
Script untuk ingest PDF knowledge base ke Qdrant.

Cara pakai:
    python scripts/ingest_pdf.py --pdf path/to/dental_knowledge.pdf

Proses:
1. Load PDF → split jadi chunks
2. Generate embeddings
3. Upsert ke Qdrant collection
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def ingest(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from app.agents.tools.retrieve import _get_embeddings
    from app.config.settings import settings

    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"   {len(docs)} halaman ditemukan")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(chunks)} chunks setelah split")

    print(f"🔗 Connecting ke Qdrant: {settings.QDRANT_URL}")
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )

    embeddings = _get_embeddings()

    # Buat collection jika belum ada
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        # Detect vector size dari embedding
        sample_embedding = embeddings.embed_query("test")
        vector_size = len(sample_embedding)
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"   ✅ Collection '{settings.QDRANT_COLLECTION}' dibuat (dim={vector_size})")
    else:
        print(f"   ℹ️  Collection '{settings.QDRANT_COLLECTION}' sudah ada")

    print(f"📤 Uploading {len(chunks)} chunks ke Qdrant...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    print(f"   ✅ Selesai! {len(chunks)} chunks berhasil diupload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF ke Qdrant knowledge base")
    parser.add_argument("--pdf", required=True, help="Path ke file PDF")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()
    ingest(args.pdf, args.chunk_size, args.chunk_overlap)
