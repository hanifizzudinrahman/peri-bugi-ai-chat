"""
Tool: retrieve_knowledge
Ambil dokumen relevan dari Qdrant vector store (RAG).
Dipanggil untuk intent dental_qa.
"""
from typing import AsyncIterator

from app.config.settings import settings
from app.schemas.chat import AgentState, make_thinking_event


def _get_qdrant_retriever(top_k: int = 3):
    """Buat retriever dari Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )
    embeddings = _get_embeddings()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def _get_embeddings():
    """Return embedding model berdasarkan EMBEDDING_PROVIDER."""
    if settings.EMBEDDING_PROVIDER == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    if settings.EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )

    if settings.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    raise ValueError(f"EMBEDDING_PROVIDER tidak dikenal: {settings.EMBEDDING_PROVIDER}")


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

    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    query = user_messages[-1]["content"] if user_messages else ""

    retrieved = []
    try:
        retriever = _get_qdrant_retriever(top_k=3)
        docs = await retriever.ainvoke(query)
        retrieved = [doc.page_content for doc in docs]
        state["tool_calls"].append({
            "tool": "retrieve_knowledge",
            "input": {"query": query[:100]},
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
