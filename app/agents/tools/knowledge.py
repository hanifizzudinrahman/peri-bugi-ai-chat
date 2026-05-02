"""
Tools: search_dental_knowledge, search_app_faq

RAG retrieval tools — wraps Qdrant collections via reused helper functions
from app/agents/sub_agents/__init__.py (_get_qdrant_retriever).

Why reuse:
- _get_qdrant_retriever and _get_embeddings are already used by main.py:421
  (admin RnD endpoints for KB ingest). Duplicating them in tools/ would
  create drift. We import from sub_agents instead.

Replaces:
- app/agents/sub_agents/__init__.py::kb_dental_agent
- app/agents/sub_agents/__init__.py::app_faq_agent

Hardcoded FAQ fallback (preserved):
- _get_hardcoded_faq is in sub_agents/__init__.py (line 252) — reused.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from app.config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _get_qdrant_retriever_with_filter(
    collection: str,
    search_kwargs: dict,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Build retriever with custom search_kwargs (untuk Qdrant metadata filter).

    Why separate helper:
    - `_get_qdrant_retriever` (di sub_agents/__init__.py) hardcoded `{"k": top_k}`
      tanpa support filter argument.
    - Daripada modify shared helper (risk break ingest endpoints), kita bikin
      filter-aware variant di sini.
    - Reuse `_get_embeddings` from sub_agents (single source of truth untuk embeddings).
    """
    import warnings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    from app.agents.sub_agents import _get_embeddings

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
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def _build_is_active_filter(extra_must: Optional[list] = None) -> dict:
    """
    Build Qdrant filter that EXCLUDES chunks with metadata.is_active=false.

    Behavior:
    - Chunks with is_active=true → INCLUDED ✅
    - Chunks without is_active key (legacy) → INCLUDED ✅
    - Chunks with is_active=false → EXCLUDED ❌

    This means existing chunks ingested before Phase 4.1 (no is_active key)
    are not affected — they remain searchable. Only chunks explicitly marked
    inactive are filtered out.

    Args:
        extra_must: Additional filter conditions to AND with is_active filter.
                    e.g., [{"key": "metadata.feature", "match": {"value": "mata_peri"}}]

    Returns:
        Qdrant filter dict ready for `search_kwargs["filter"]`.
    """
    must_conditions = list(extra_must or [])

    # Build "must_not" — exclude only when is_active is explicitly false
    must_not_conditions = [
        {"key": "metadata.is_active", "match": {"value": False}}
    ]

    filter_dict: dict = {"must_not": must_not_conditions}
    if must_conditions:
        filter_dict["must"] = must_conditions

    return filter_dict


# =============================================================================
# Tool 1: search_dental_knowledge
# =============================================================================


def make_search_dental_knowledge_tool(
    embedding_provider_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
):
    """
    Factory: build search_dental_knowledge tool with embedding override closure.

    Args:
        embedding_provider_override: RnD-mode override for embedding provider
        embedding_model_override: RnD-mode override for embedding model

    Returns:
        @tool decorated async function.
    """

    @tool
    async def search_dental_knowledge(query: str, top_k: int = 3) -> dict[str, Any]:
        """Search the dental health knowledge base for evidence-based information.

        Use this tool when the user asks about:
        - Why something happens dentally ("kenapa gigi anak berlubang?")
        - How to prevent or treat dental issues
        - General dental hygiene advice
        - Specific conditions (cavities, gingivitis, fluorosis, etc.)
        - Diet and dental health
        - When to see a dentist

        Args:
            query: A focused search query in Indonesian or English.
                   Refine the user's question into key concepts (e.g.
                   "anak saya umur 5 tahun gigi belakang sakit" →
                   "rasa sakit gigi geraham anak balita").
            top_k: Number of relevant documents to retrieve (default 3, max 10).

        Returns a dict with keys:
        - docs: list of document content strings (knowledge base passages)
        - source_count: int

        Use the docs as evidence in your response. Cite naturally,
        don't dump raw text. If no docs found (source_count=0), tell
        the user honestly and offer general advice.

        NOTE: This tool automatically filters out documents with
        metadata.is_active=false (e.g., draft content, content for features
        not yet released). Only active content is returned.
        """
        from app.config.observability import trace_node, _safe_dict_for_trace

        # Clamp top_k to safe range
        top_k = max(1, min(int(top_k or 3), 10))

        async with trace_node(
            name="tool:search_dental_knowledge",
            state=None,
            input_data={
                "query": query[:300] if isinstance(query, str) else "",
                "top_k": top_k,
                "collection": settings.QDRANT_COLLECTION,
            },
        ) as span:
            docs: list[str] = []
            try:
                # Phase 4.1: Auto-filter is_active != false
                # We exclude only chunks where metadata.is_active is explicitly false.
                # Chunks without is_active key (legacy / pre-Phase 4.1) are INCLUDED.
                search_kwargs = {
                    "k": top_k,
                    "filter": _build_is_active_filter(),
                }
                retriever = _get_qdrant_retriever_with_filter(
                    collection=settings.QDRANT_COLLECTION,
                    search_kwargs=search_kwargs,
                    provider_override=embedding_provider_override,
                    model_override=embedding_model_override,
                )
                results = await retriever.ainvoke(query)
                docs = [doc.page_content for doc in results]

                if span:
                    span.update(output={
                        "doc_count": len(docs),
                        "is_active_filter_applied": True,
                        "docs_preview": [d[:200] for d in docs[:3]],
                        "docs": _safe_dict_for_trace(docs),
                        "total_chars": sum(len(d) for d in docs),
                    })
            except Exception as e:
                logger.warning(f"[tool:search_dental_knowledge] error: {e}")
                if span:
                    span.update(
                        output={
                            "doc_count": 0,
                            "error": str(e)[:200],
                            "fallback": "no_rag",
                        },
                        level="ERROR",
                        status_message=str(e)[:200],
                    )

            return {"docs": docs, "source_count": len(docs)}

    return search_dental_knowledge


# =============================================================================
# Tool 2: search_app_faq
# =============================================================================


def make_search_app_faq_tool(
    embedding_provider_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
):
    """
    Factory: build search_app_faq tool with embedding override closure.
    """

    @tool
    async def search_app_faq(
        query: str,
        top_k: int = 3,
        feature_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search the Peri Bugi app FAQ knowledge base.

        ✅ USE THIS TOOL when user asks about:
        - How to use Peri Bugi app features
        - Why a feature isn't working as expected
        - Account, settings, notifications, or technical questions about the app
        - "Gimana cara X di aplikasi?", "Kenapa X tidak muncul?", "X-nya gimana?"

        ❌ DO NOT use this tool for:
        - Dental health questions (use search_dental_knowledge instead)
        - User-specific data (use get_brushing_stats, get_scan_history, etc.)

        Args:
            query: A focused search query about the app.
            top_k: Number of relevant FAQ entries to retrieve (default 3, max 10).
            feature_filter: Optional filter to narrow search to specific feature.
                Valid values:
                - None (default) — search across all features
                - "rapot_peri" — Rapot Peri (sikat gigi log, streak, badge, kuesioner risiko karies)
                - "mata_peri" — Mata Peri (scan gigi 5-view + analisis foto chat)
                - "cerita_peri" — Cerita Peri (modul edukasi orang tua)
                - "tanya_peri" — Tanya Peri (chatbot AI ini sendiri)
                - "janji_peri" — Janji Peri (booking dokter)
                - "dunia_game" — Dunia Game (game edukasi anak)
                - "profile_account" — profil, settings, login, password
                - "general" — pertanyaan umum aplikasi (privacy, langganan, support)

                INSTRUKSI MEMILIH FILTER:
                - User mention nama fitur explicit → pakai filter sesuai
                  ("Mata Peri gimana?" → "mata_peri")
                - User tanya about UI/login/account → "profile_account"
                - User tanya general (privacy, kontak, dll) → "general"
                - Kalau tidak yakin / bisa multi-fitur → JANGAN set filter
                  (None default = search semua)

        Returns a dict with keys:
        - docs: list of FAQ content strings
        - source_count: int
        - feature_filter_applied: str | None — echo of what filter was used

        Note: This tool falls back to hardcoded FAQ if the Qdrant FAQ
        collection is unavailable. Always check source_count > 0 before
        relying on the docs.
        """
        from app.agents.sub_agents import _get_qdrant_retriever, _get_hardcoded_faq
        from app.config.observability import trace_node, _safe_dict_for_trace

        top_k = max(1, min(int(top_k or 3), 10))

        # Validate feature_filter
        VALID_FEATURES = {
            "rapot_peri", "mata_peri", "cerita_peri", "tanya_peri",
            "janji_peri", "dunia_game", "profile_account", "general",
        }
        if feature_filter is not None and feature_filter not in VALID_FEATURES:
            logger.warning(
                f"[tool:search_app_faq] invalid feature_filter='{feature_filter}', "
                f"falling back to no filter"
            )
            feature_filter = None

        async with trace_node(
            name="tool:search_app_faq",
            state=None,
            input_data={
                "query": query[:300] if isinstance(query, str) else "",
                "top_k": top_k,
                "feature_filter": feature_filter,
            },
        ) as span:
            docs: list[str] = []
            used_fallback = False
            try:
                faq_collection = getattr(
                    settings, "QDRANT_FAQ_COLLECTION", "peri_bugi_faq"
                )

                # Phase 4.1: Always apply is_active filter (exclude is_active=false)
                # Optionally add feature filter on top.
                extra_must = None
                if feature_filter:
                    extra_must = [
                        {
                            "key": "metadata.feature",
                            "match": {"value": feature_filter},
                        }
                    ]

                search_kwargs = {
                    "k": top_k,
                    "filter": _build_is_active_filter(extra_must=extra_must),
                }
                retriever = _get_qdrant_retriever_with_filter(
                    collection=faq_collection,
                    search_kwargs=search_kwargs,
                    provider_override=embedding_provider_override,
                    model_override=embedding_model_override,
                )

                results = await retriever.ainvoke(query)
                docs = [doc.page_content for doc in results]
            except Exception as e:
                logger.warning(f"[tool:search_app_faq] qdrant error: {e}")
                docs = _get_hardcoded_faq(query)
                used_fallback = True

            if span:
                span.update(output={
                    "doc_count": len(docs),
                    "used_fallback": used_fallback,
                    "feature_filter_applied": feature_filter,
                    "is_active_filter_applied": True,
                    "docs_preview": [d[:200] for d in docs[:3]],
                    "docs": _safe_dict_for_trace(docs),
                    "total_chars": sum(len(d) for d in docs),
                })

            return {
                "docs": docs,
                "source_count": len(docs),
                "feature_filter_applied": feature_filter,
            }

    return search_app_faq


# =============================================================================
# ToolSpec registration — Bagian C: registry pattern
# =============================================================================
from app.agents.tools.registry import ToolSpec, register_tool, BridgeContext


# ── search_dental_knowledge ─────────────────────────────────────────────────

def _bridge_dental_kb(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: extract docs to agent_results['kb_dental'] + update retrieved_docs."""
    docs = result.get("docs", []) or []
    agent_results["kb_dental"] = {
        "docs": docs,
        "source_count": result.get("source_count", len(docs)),
    }
    if docs:
        ctx.retrieved_docs = docs


def _inject_dental_kb(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject KB dental docs to system prompt as 'Referensi dari knowledge base'."""
    docs = (data or {}).get("docs", []) or []
    if not docs:
        return ""
    # Take top 3, mirror existing generate.py behavior (lines 105-108 baseline)
    docs_text = "\n\n".join(docs[:3])
    return f"\n\nReferensi dari knowledge base:\n{docs_text}"


register_tool(ToolSpec(
    tool_name="search_dental_knowledge",
    agent_key="kb_dental",
    required_agent="kb_dental",
    bridge_handler=_bridge_dental_kb,
    prompt_injector=_inject_dental_kb,
    thinking_label="Mencari info kesehatan gigi...",
))


# ── search_app_faq ──────────────────────────────────────────────────────────

def _bridge_app_faq(result: dict, agent_results: dict, ctx: BridgeContext) -> None:
    """Bridge: extract FAQ docs to agent_results['app_faq']."""
    agent_results["app_faq"] = {
        "docs": result.get("docs", []),
        "source_count": result.get("source_count", 0),
    }


def _inject_app_faq(data: dict, child_name: str, prompts: dict, response_mode: str) -> str:
    """Inject FAQ docs to system prompt as 'Info aplikasi yang relevan'."""
    docs = (data or {}).get("docs", []) or []
    if not docs:
        return ""
    faq_text = "\n\n".join(docs[:3])
    return f"\n\nInfo aplikasi yang relevan:\n{faq_text}"


register_tool(ToolSpec(
    tool_name="search_app_faq",
    agent_key="app_faq",
    required_agent="app_faq",
    bridge_handler=_bridge_app_faq,
    prompt_injector=_inject_app_faq,
    thinking_label="Mencari info aplikasi...",
))
