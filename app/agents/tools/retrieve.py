"""
Tool: retrieve_knowledge
Ambil dokumen relevan dari Qdrant vector store (RAG).
Dipanggil untuk intent dental_qa.

Perubahan dari versi sebelumnya:
- GPU support untuk embedding lokal (CUDA auto-detect)
- top_k bisa di-override dari state (untuk RnD experiment)
- embedding_provider dan embedding_model bisa di-override dari state
"""
from typing import AsyncIterator

from app.config.settings import settings
from app.schemas.chat import AgentState, make_thinking_event


def _detect_embedding_device() -> str:
    """
    Deteksi device terbaik untuk embedding lokal.
    EMBEDDING_DEVICE di .env bisa: auto | cuda | cpu

    auto = coba CUDA dulu, fallback ke CPU jika tidak ada.
    Ini penting untuk performa: embedding di CPU bisa 10-50x lebih lambat.
    """
    device_setting = settings.EMBEDDING_DEVICE.lower()

    if device_setting == "cpu":
        return "cpu"

    if device_setting == "cuda":
        # Paksa CUDA — error jika tidak tersedia
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "EMBEDDING_DEVICE=cuda tapi CUDA tidak tersedia di sistem ini. "
                    "Ganti ke EMBEDDING_DEVICE=auto atau EMBEDDING_DEVICE=cpu"
                )
            return "cuda"
        except ImportError:
            raise RuntimeError("torch tidak terinstall. pip install torch")

    # auto mode (default)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def _get_embeddings(
    provider_override: str | None = None,
    model_override: str | None = None,
):
    """
    Return embedding model berdasarkan EMBEDDING_PROVIDER.
    Support override untuk RnD experiment.
    """
    provider = provider_override or settings.EMBEDDING_PROVIDER
    model = model_override or settings.EMBEDDING_MODEL

    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        device = _detect_embedding_device()
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            # Cache model di memori supaya tidak reload setiap request
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


def _get_qdrant_retriever(
    top_k: int = 3,
    provider_override: str | None = None,
    model_override: str | None = None,
):
    """Buat retriever dari Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )
    embeddings = _get_embeddings(
        provider_override=provider_override,
        model_override=model_override,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


async def retrieve_node(state: AgentState) -> AsyncIterator[str]:
    """
    LangGraph node: retrieve_knowledge
    Cari dokumen relevan dari knowledge base.

    top_k diambil dari state (bisa di-override per-request untuk RnD).
    """
    thinking_step = len(state.get("thinking_steps", [])) + 1
    yield make_thinking_event(
        step=thinking_step,
        label="Mencari referensi dari knowledge base...",
        done=False,
    )

    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    query = user_messages[-1].get("content", "") if user_messages else ""

    # Ambil top_k dari state (support override untuk RnD)
    top_k = state.get("top_k_docs", 3)

    # Kalau top_k = 0, skip RAG
    if top_k == 0:
        state["retrieved_docs"] = []
        state["tool_calls"].append({
            "tool": "retrieve_knowledge",
            "input": {"query": query[:100]},
            "result": {"doc_count": 0, "note": "top_k=0, RAG skipped"},
        })
        yield make_thinking_event(
            step=thinking_step,
            label="Mencari referensi dari knowledge base...",
            done=True,
        )
        state["thinking_steps"].append({
            "step": thinking_step,
            "label": "Mencari referensi dari knowledge base...",
            "done": True,
        })
        return

    retrieved = []
    try:
        retriever = _get_qdrant_retriever(
            top_k=top_k,
            provider_override=state.get("embedding_provider_override"),
            model_override=state.get("embedding_model_override"),
        )
        docs = await retriever.ainvoke(query)
        retrieved = [doc.page_content for doc in docs]
        state["tool_calls"].append({
            "tool": "retrieve_knowledge",
            "input": {"query": query[:100], "top_k": top_k},
            "result": {"doc_count": len(retrieved)},
        })
    except Exception as e:
        # Qdrant belum ada atau error — lanjut tanpa RAG
        state["tool_calls"].append({
            "tool": "retrieve_knowledge",
            "input": {"query": query[:100]},
            "result": {"error": str(e), "fallback": "no_rag"},
        })

    state["retrieved_docs"] = retrieved
    yield make_thinking_event(
        step=thinking_step,
        label="Mencari referensi dari knowledge base...",
        done=True,
    )
    state["thinking_steps"].append({
        "step": thinking_step,
        "label": "Mencari referensi dari knowledge base...",
        "done": True,
    })
