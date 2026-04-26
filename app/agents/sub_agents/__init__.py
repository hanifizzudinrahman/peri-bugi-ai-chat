"""
Sub-agents Phase 1: kb_dental, user_profile, app_faq

Setiap agent adalah async function yang:
1. Terima AgentState
2. Jalankan tool(s) yang relevan
3. Return dict hasil yang akan di-merge ke agent_results
4. Log tool calls ke state["tool_calls"]

Phase 2 agents (rapot_peri, cerita_peri, mata_peri) ditambahkan nanti.
Phase 4 agents (janji_peri) ditambahkan nanti.
"""
import time
import warnings
from typing import Any

from app.agents.state import AgentState
from app.config.observability import build_trace_config
from app.config.settings import settings


# =============================================================================
# KB Dental Agent — RAG dari Qdrant collection peri_bugi_dental
# =============================================================================

async def kb_dental_agent(state: AgentState) -> dict[str, Any]:
    """
    Ambil dokumen relevan dari knowledge base dental.
    Return: {docs: list[str], source_count: int}
    """
    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    query = user_messages[-1].get("content", "") if user_messages else ""
    top_k = state.get("top_k_docs", 3)

    # Phase 4: wrap agent body with trace_node
    from app.config.observability import trace_node

    async with trace_node(
        name="kb_dental_agent",
        state=state,
        input_data={
            "query": query[:300] if isinstance(query, str) else "",
            "top_k": top_k,
            "collection": settings.QDRANT_COLLECTION,
        },
    ) as span:
        if top_k == 0:
            state["tool_calls"].append({
                "tool": "qdrant_retriever",
                "agent": "kb_dental",
                "input": {"query": query[:100]},
                "result": {"doc_count": 0, "note": "top_k=0, RAG skipped"},
            })
            if span:
                span.update(output={"doc_count": 0, "skipped": True})
            return {"docs": [], "source_count": 0}

        docs = []
        try:
            retriever = _get_qdrant_retriever(
                collection=settings.QDRANT_COLLECTION,
                top_k=top_k,
                provider_override=state.get("embedding_provider_override"),
                model_override=state.get("embedding_model_override"),
            )
            # Per-call trace metadata: agent="kb_dental_retriever", session_id, dll
            trace_config = build_trace_config(state=state, agent_name="kb_dental_retriever")
            results = await retriever.ainvoke(query, config=trace_config)
            docs = [doc.page_content for doc in results]
            state["tool_calls"].append({
                "tool": "qdrant_retriever",
                "agent": "kb_dental",
                "input": {"query": query[:100], "top_k": top_k},
                "result": {"doc_count": len(docs)},
            })
            if span:
                span.update(output={
                    "doc_count": len(docs),
                    "docs_preview": [d[:200] for d in docs[:3]],  # first 3 docs, 200 chars each
                })
        except Exception as e:
            state["tool_calls"].append({
                "tool": "qdrant_retriever",
                "agent": "kb_dental",
                "input": {"query": query[:100]},
                "result": {"error": str(e), "fallback": "no_rag"},
            })
            if span:
                span.update(
                    output={"doc_count": 0, "error": str(e)[:200], "fallback": "no_rag"},
                    level="ERROR",
                    status_message=str(e)[:200],
                )

        # Simpan juga ke state["retrieved_docs"] untuk backward compat
        state["retrieved_docs"] = docs
        return {"docs": docs, "source_count": len(docs)}


# =============================================================================
# User Profile Agent — baca dari user_context yang sudah di-inject
# =============================================================================

async def user_profile_agent(state: AgentState) -> dict[str, Any]:
    """
    Extract info dari user_context yang sudah di-inject dari api.
    Tidak perlu tool call — data sudah ada di state.
    Return: {profile: dict, child: dict, has_brushing_data: bool}
    """
    ctx = state.get("user_context", {})
    user = ctx.get("user") or {}
    child = ctx.get("child") or {}
    brushing = ctx.get("brushing")
    mata_peri = ctx.get("mata_peri_last_result")

    # Phase 4: wrap agent body with trace_node
    from app.config.observability import trace_node

    async with trace_node(
        name="user_profile_agent",
        state=state,
        input_data={
            "fields_requested": ["user", "child", "brushing", "mata_peri"],
            "has_user": bool(user),
            "has_child": bool(child),
            "has_brushing": brushing is not None,
            "has_scan": mata_peri is not None,
        },
    ) as span:
        state["tool_calls"].append({
            "tool": "user_context_read",
            "agent": "user_profile",
            "input": {"fields": ["user", "child", "brushing", "mata_peri"]},
            "result": {
                "has_child": child is not None,
                "has_brushing": brushing is not None,
                "has_scan": mata_peri is not None,
            },
        })

        result = {
            "profile": {
                "name": user.get("nickname") or user.get("full_name", "User"),
                "gender": user.get("gender"),
            },
            "child": {
                "name": child.get("nickname") or child.get("full_name"),
                "age_years": child.get("age_years"),
                "gender": child.get("gender"),
            } if child else None,
            "brushing": brushing,
            "mata_peri_last": mata_peri,
            "has_brushing_data": brushing is not None,
            "has_scan_data": mata_peri is not None,
        }

        if span:
            # Output: ringkasan saja, jangan kirim sensitive data verbatim
            span.update(output={
                "child_name": result["child"]["name"] if result["child"] else None,
                "child_age": result["child"]["age_years"] if result["child"] else None,
                "has_brushing_data": result["has_brushing_data"],
                "has_scan_data": result["has_scan_data"],
                "current_streak": brushing.get("current_streak") if brushing else None,
            })

        return result


# =============================================================================
# App FAQ Agent — RAG dari Qdrant collection peri_bugi_faq
# =============================================================================

async def app_faq_agent(state: AgentState) -> dict[str, Any]:
    """
    Ambil jawaban FAQ aplikasi Peri Bugi dari Qdrant collection khusus FAQ.
    Fallback ke hardcoded FAQ jika collection kosong.
    """
    user_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    query = user_messages[-1].get("content", "") if user_messages else ""

    # Phase 4: wrap agent body with trace_node
    from app.config.observability import trace_node

    async with trace_node(
        name="app_faq_agent",
        state=state,
        input_data={
            "query": query[:300] if isinstance(query, str) else "",
            "top_k": 3,
        },
    ) as span:
        docs = []
        used_fallback = False
        try:
            faq_collection = getattr(settings, "QDRANT_FAQ_COLLECTION", "peri_bugi_faq")
            retriever = _get_qdrant_retriever(
                collection=faq_collection,
                top_k=3,
                provider_override=state.get("embedding_provider_override"),
                model_override=state.get("embedding_model_override"),
            )
            # Per-call trace metadata: agent="app_faq_retriever", session_id, dll
            trace_config = build_trace_config(state=state, agent_name="app_faq_retriever")
            results = await retriever.ainvoke(query, config=trace_config)
            docs = [doc.page_content for doc in results]
            state["tool_calls"].append({
                "tool": "qdrant_retriever_faq",
                "agent": "app_faq",
                "input": {"query": query[:100]},
                "result": {"doc_count": len(docs)},
            })
        except Exception as e:
            # Fallback ke hardcoded FAQ dasar
            docs = _get_hardcoded_faq(query)
            used_fallback = True
            state["tool_calls"].append({
                "tool": "qdrant_retriever_faq",
                "agent": "app_faq",
                "input": {"query": query[:100]},
                "result": {"error": str(e), "fallback": "hardcoded_faq", "doc_count": len(docs)},
            })

        if span:
            span.update(output={
                "doc_count": len(docs),
                "used_fallback": used_fallback,
                "docs_preview": [d[:200] for d in docs[:3]],
            })

        return {"docs": docs, "source_count": len(docs)}


def _get_hardcoded_faq(query: str) -> list[str]:
    """FAQ hardcoded sebagai fallback saat Qdrant collection kosong."""
    faq_items = [
        "Rapot Peri adalah fitur untuk memantau kebiasaan sikat gigi anak setiap hari. "
        "Orang tua bisa checklist pagi dan malam untuk melacak streak sikat gigi.",

        "Mata Peri adalah fitur scan gigi anak menggunakan kamera. "
        "Foto gigi anak dan AI akan menganalisis kondisi gigi secara otomatis.",

        "Tanya Peri adalah asisten chatbot kesehatan gigi anak yang bisa menjawab "
        "pertanyaan seputar kesehatan gigi, memberikan tips, dan informasi edukatif.",

        "Janji Peri adalah fitur untuk booking jadwal ke dokter gigi di puskesmas terdekat. "
        "Pilih dokter, pilih jadwal, dan konfirmasi janji tanpa telepon.",

        "Cerita Peri adalah konten edukasi dalam bentuk cerita bergambar tentang "
        "kesehatan gigi. Ada 6 modul dengan quiz di setiap modul.",

        "Untuk mendaftar, klik 'Daftar' di halaman utama, masukkan nomor HP, "
        "verifikasi OTP, lalu isi data profil dan data anak.",

        "Jika lupa password, klik 'Lupa Password' di halaman login dan masukkan "
        "nomor HP terdaftar untuk menerima OTP reset password.",

        "Aplikasi Peri Bugi tersedia di browser (web app) di app.peribugi.my.id "
        "dan bisa diinstal sebagai PWA di HP.",

        "Data anak bisa diubah di menu Profil > Data Anak. "
        "Masukkan nama, tanggal lahir, dan jenis kelamin anak.",

        "Streak sikat gigi akan reset ke 0 jika melewatkan 1 hari tanpa checklist. "
        "Pastikan checklist pagi dan malam setiap hari.",
    ]

    query_lower = query.lower()
    relevant = []

    # Simple keyword matching untuk FAQ
    keyword_map = {
        "rapot": [faq_items[0]],
        "mata peri": [faq_items[1]],
        "tanya peri": [faq_items[2]],
        "janji": [faq_items[3]],
        "cerita": [faq_items[4]],
        "daftar": [faq_items[5]],
        "password": [faq_items[6]],
        "download": [faq_items[7]],
        "install": [faq_items[7]],
        "data anak": [faq_items[8]],
        "streak": [faq_items[9]],
    }

    for keyword, items in keyword_map.items():
        if keyword in query_lower:
            relevant.extend(items)

    # Kalau tidak ada yang match, return 3 FAQ pertama sebagai default
    return relevant[:3] if relevant else faq_items[:3]


# =============================================================================
# Shared helper: Qdrant retriever
# =============================================================================

def _get_qdrant_retriever(
    collection: str,
    top_k: int = 3,
    provider_override: str | None = None,
    model_override: str | None = None,
):
    """Buat retriever dari Qdrant collection tertentu."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

    embeddings = _get_embeddings(provider_override, model_override)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def _get_embeddings(provider_override: str | None = None, model_override: str | None = None):
    """Return embedding model."""
    provider = provider_override or settings.EMBEDDING_PROVIDER
    model = model_override or settings.EMBEDDING_MODEL

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
