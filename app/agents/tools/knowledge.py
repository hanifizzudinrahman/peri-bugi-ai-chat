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
        """
        from app.agents.sub_agents import _get_qdrant_retriever
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
                retriever = _get_qdrant_retriever(
                    collection=settings.QDRANT_COLLECTION,
                    top_k=top_k,
                    provider_override=embedding_provider_override,
                    model_override=embedding_model_override,
                )
                results = await retriever.ainvoke(query)
                docs = [doc.page_content for doc in results]

                if span:
                    span.update(output={
                        "doc_count": len(docs),
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
    async def search_app_faq(query: str, top_k: int = 3) -> dict[str, Any]:
        """Search the Peri Bugi app FAQ knowledge base.

        Use this tool when the user asks about:
        - How to use the Peri Bugi app
        - App features (Mata Peri, Rapot Peri, Cerita Peri, Janji Peri, etc.)
        - Account, settings, notifications, or technical questions about the app
        - Why a feature isn't working as expected

        Args:
            query: A focused search query about the app.
            top_k: Number of relevant FAQ entries to retrieve (default 3, max 10).

        Returns a dict with keys:
        - docs: list of FAQ content strings
        - source_count: int

        Note: This tool falls back to hardcoded FAQ if the Qdrant FAQ
        collection is unavailable. Always check source_count > 0 before
        relying on the docs.

        Do NOT use this tool for:
        - Dental health questions (use search_dental_knowledge instead)
        - User-specific data (use get_brushing_stats, get_scan_history, etc.)
        """
        from app.agents.sub_agents import _get_qdrant_retriever, _get_hardcoded_faq
        from app.config.observability import trace_node, _safe_dict_for_trace

        top_k = max(1, min(int(top_k or 3), 10))

        async with trace_node(
            name="tool:search_app_faq",
            state=None,
            input_data={
                "query": query[:300] if isinstance(query, str) else "",
                "top_k": top_k,
            },
        ) as span:
            docs: list[str] = []
            used_fallback = False
            try:
                faq_collection = getattr(
                    settings, "QDRANT_FAQ_COLLECTION", "peri_bugi_faq"
                )
                retriever = _get_qdrant_retriever(
                    collection=faq_collection,
                    top_k=top_k,
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
                    "docs_preview": [d[:200] for d in docs[:3]],
                    "docs": _safe_dict_for_trace(docs),
                    "total_chars": sum(len(d) for d in docs),
                })

            return {"docs": docs, "source_count": len(docs)}

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
