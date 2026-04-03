"""
Tool: retrieve_knowledge
Ambil dokumen relevan dari Qdrant vector store (RAG).
"""
import warnings
from typing import AsyncIterator

from app.config.settings import settings
from app.schemas.chat import AgentState, make_thinking_event


def _detect_embedding_device() -> str:
    """
    Deteksi device terbaik untuk embedding lokal.
    EMBEDDING_DEVICE di .env: auto | cuda | cpu
    """
    device_setting = settings.EMBEDDING_DEVICE.lower()

    if device_setting == "cpu":
        return "cpu"

    if device_setting == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "EMBEDDING_DEVICE=cuda tapi CUDA tidak tersedia. "
                    "Ganti ke EMBEDDING_DEVICE=auto atau cpu"
                )
            return "cuda"
        except ImportError:
            raise RuntimeError("torch tidak terinstall. pip install torch")

    # auto mode (default)
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_embeddings(
    provider_override: str | None = None,
    model_override: str | None = None,
):
    """
    Return embedding model berdasarkan EMBEDDING_PROVIDER.
    Support override untuk RnD experiment.

    Menggunakan langchain_huggingface (bukan langchain_community yang deprecated).
    """
    provider = provider_override or settings.EMBEDDING_PROVIDER
    model = model_override or settings.EMBEDDING_MODEL

    if provider == "local":
        # Pakai langchain_huggingface, bukan langchain_community.HuggingFaceEmbeddings
        # yang sudah deprecated sejak LangChain 0.2.2
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            # Fallback ke yang lama kalau langchain_huggingface belum terinstall
            # Suppress deprecation warning supaya logs tidak berisik
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from langchain_community.embeddings import HuggingFaceEmbeddings

        device = _detect_embedding_device()
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


def _get_qdrant_retriever(
    top_k: int = 3,
    provider_override: str | None = None,
    model_override: str | None = None,
):
    """Buat retriever dari Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    # Suppress warning "Api key is used with an insecure connection"
    # Warning ini normal untuk local dev (Qdrant tanpa HTTPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    top_k = state.get("top_k_docs", 3)

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
