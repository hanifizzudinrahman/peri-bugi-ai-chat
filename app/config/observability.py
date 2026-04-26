"""
Observability — Langfuse callback handler singleton.

Pattern:
- Pattern 1 (graceful degradation): Kalau Langfuse offline/unreachable,
  app tetap jalan normal. Langfuse SDK sudah handle async batching +
  silent failure di background thread.
- Pattern 2 (explicit toggle): LANGFUSE_ENABLED env flag. Default False
  supaya safe by default. Set =true di .env untuk aktifkan.

Usage:
    from app.config.observability import get_langfuse_handler

    handler = get_langfuse_handler()
    if handler:
        # observability enabled
        ...

Default behavior:
- LANGFUSE_ENABLED=false        → return None (zero overhead)
- Keys missing                  → return None + warning log
- Init failed                   → return None + warning log
- All OK                        → return CallbackHandler instance
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from app.config.settings import settings

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Module-level state untuk avoid spam log warning di setiap call
_init_attempted = False
_init_succeeded = False


@lru_cache(maxsize=1)
def get_langfuse_handler() -> Optional["BaseCallbackHandler"]:
    """
    Return Langfuse CallbackHandler instance kalau enabled + reachable.
    Return None kalau disabled / config invalid / init gagal.

    Cached via lru_cache supaya init cuma dijalankan sekali per process.
    Thread-safe karena Python GIL + lru_cache implementation.

    Reset cache (untuk testing) dengan: get_langfuse_handler.cache_clear()
    """
    global _init_attempted, _init_succeeded

    # ── Step 1: Check feature flag ────────────────────────────────────────────
    if not settings.LANGFUSE_ENABLED:
        if not _init_attempted:
            logger.info("Langfuse observability DISABLED (LANGFUSE_ENABLED=false)")
            _init_attempted = True
        return None

    # ── Step 2: Validate keys ─────────────────────────────────────────────────
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        if not _init_attempted:
            logger.warning(
                "Langfuse ENABLED tapi keys missing "
                "(LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
                "Observability disabled."
            )
            _init_attempted = True
        return None

    # ── Step 3: Try import + instantiate ──────────────────────────────────────
    try:
        # Import lazy supaya kalau langfuse package not installed,
        # cuma fail saat user beneran enable, bukan saat startup
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        if not _init_attempted or not _init_succeeded:
            logger.info(
                f"Langfuse observability ENABLED — host={settings.LANGFUSE_HOST}"
            )
            _init_attempted = True
            _init_succeeded = True

        return handler

    except ImportError:
        if not _init_attempted:
            logger.warning(
                "Langfuse ENABLED tapi package belum terinstall. "
                "Run: pip install langfuse>=3.0.0"
            )
            _init_attempted = True
        return None

    except Exception as e:
        if not _init_attempted:
            logger.warning(
                f"Langfuse init gagal: {type(e).__name__}: {e}. "
                "Observability disabled, app tetap jalan normal."
            )
            _init_attempted = True
        return None


def is_observability_enabled() -> bool:
    """
    True kalau Langfuse handler successfully initialized.
    Useful untuk conditional log info di startup, atau health endpoint.
    """
    return get_langfuse_handler() is not None


def build_trace_metadata(
    state: Optional[dict] = None,
    agent_name: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build metadata dict untuk per-call Langfuse trace.

    Args:
        state: AgentState dict — extract session_id, user_id, trace_id, dll.
        agent_name: e.g. "router", "supervisor", "generate", "view_hint",
                    "memory_summary", "kb_dental_retriever".
        extra: tambahan metadata custom.

    Returns:
        dict metadata yang bisa di-pass ke RunnableConfig:
        ```
        config = {"callbacks": [...], "metadata": {...}, "tags": [...]}
        await llm.ainvoke(messages, config=config)
        ```

    Aman dipanggil walau state None / partial — cuma return dict apa yang ada.
    """
    metadata: dict = {}

    if state:
        # AgentState fields yang berguna untuk filter di Langfuse dashboard
        for key in ("session_id", "trace_id", "chat_message_id", "response_mode"):
            value = state.get(key)
            if value:
                metadata[key] = value

        user_context = state.get("user_context") or {}
        user_id = user_context.get("user_id") or user_context.get("id")
        if user_id:
            metadata["user_id"] = str(user_id)

    if agent_name:
        metadata["agent"] = agent_name

    if extra:
        metadata.update(extra)

    return metadata


def build_trace_config(
    state: Optional[dict] = None,
    agent_name: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    extra_tags: Optional[list[str]] = None,
) -> dict:
    """
    Build RunnableConfig dict siap pass ke `.ainvoke(config=...)` /
    `.astream(config=...)`.

    Returns dict dengan:
    - callbacks: list — handler yang aktif (atau empty kalau disabled)
    - metadata:  dict — context dari state + agent_name
    - tags:      list[str] — quick-filter tags

    Pattern usage di call site:
        from app.config.observability import build_trace_config

        config = build_trace_config(state=state, agent_name="router")
        result = await llm.ainvoke(messages, config=config)

    Kalau handler None (disabled / failed), config tetap valid —
    cuma callbacks empty, gak break apa pun.
    """
    handler = get_langfuse_handler()

    config: dict = {
        "callbacks": [handler] if handler else [],
        "metadata": build_trace_metadata(state, agent_name, extra_metadata),
    }

    tags: list[str] = []
    if agent_name:
        tags.append(f"agent:{agent_name}")
    if state and state.get("response_mode"):
        tags.append(f"mode:{state['response_mode']}")
    if extra_tags:
        tags.extend(extra_tags)

    if tags:
        config["tags"] = tags

    return config
