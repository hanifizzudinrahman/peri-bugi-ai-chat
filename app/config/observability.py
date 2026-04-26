"""
Observability — Langfuse callback handler singleton.

Pattern:
- Pattern 1 (graceful degradation): Kalau Langfuse offline/unreachable,
  app tetap jalan normal. Langfuse SDK sudah handle async batching +
  silent failure di background thread.
- Pattern 2 (explicit toggle): LANGFUSE_ENABLED env flag. Default False
  supaya safe by default. Set =true di .env untuk aktifkan.

SDK v3 init pattern (OTEL-based):
  1. Init Langfuse global client SEKALI dengan credentials
     (Langfuse(public_key=..., secret_key=..., host=...))
  2. CallbackHandler() di-construct TANPA args — dia ambil credential
     dari global client yang udah ke-init.
  3. v3 berbeda dari v2 yang `CallbackHandler(public_key=..., secret_key=...)`.

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

    # ── Step 3: Init global Langfuse client + create CallbackHandler ──────────
    # SDK v3 pattern: init Langfuse client global dulu (singleton via get_client),
    # baru CallbackHandler() bisa di-construct tanpa args. Reference:
    # https://langfuse.com/integrations/frameworks/langchain
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Initialize global Langfuse client (singleton)
        # Aman dipanggil multiple times — internal SDK guard idempotent.
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        # Construct CallbackHandler — ambil credential dari global client.
        # v3 NO LONGER accepts public_key/secret_key/host as constructor args.
        handler = CallbackHandler()

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


def get_langfuse_client():
    """
    Return Langfuse global client (Langfuse instance) kalau enabled,
    atau None kalau disabled / not initialized.

    Dipakai untuk create parent span via langfuse.start_as_current_observation(),
    yang akan jadi root trace untuk seluruh request — supaya semua LLM call
    children nge-nest di bawah 1 trace yang sama.

    NOTE: get_langfuse_handler() harus dipanggil dulu (di app startup atau
    di pertama call) supaya global Langfuse() client ke-init.
    """
    # Trigger handler init kalau belum (idempotent)
    handler = get_langfuse_handler()
    if handler is None:
        return None

    try:
        from langfuse import get_client
        return get_client()
    except Exception as e:
        logger.warning(f"Failed to get Langfuse client: {e}")
        return None


def extract_user_message(state: Optional[dict]) -> Optional[str]:
    """
    Extract pesan user terakhir dari AgentState untuk trace input.
    Return None kalau gak ada pesan user.
    """
    if not state:
        return None
    messages = state.get("messages") or []
    user_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
    ]
    if not user_msgs:
        return None
    content = user_msgs[-1].get("content", "")
    return str(content) if content else None


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

    Note: Untuk v3, gunakan key `langfuse_session_id`, `langfuse_user_id`,
    `langfuse_tags` supaya Langfuse SDK extract jadi top-level trace properties
    (bukan cuma metadata). Reference:
    https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3
    """
    metadata: dict = {}

    if state:
        # AgentState fields → langfuse-specific keys (v3 convention)
        session_id = state.get("session_id")
        if session_id:
            metadata["langfuse_session_id"] = session_id

        user_context = state.get("user_context") or {}
        user_id = user_context.get("user_id") or user_context.get("id")
        if user_id:
            metadata["langfuse_user_id"] = str(user_id)

        # Custom metadata (bukan top-level v3 properties)
        for key in ("trace_id", "chat_message_id", "response_mode"):
            value = state.get(key)
            if value:
                metadata[key] = value

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
    - metadata:  dict — context dari state + agent_name (with langfuse_* keys)
    - tags:      list[str] — quick-filter tags (also via langfuse_tags metadata)

    Pattern usage di call site:
        from app.config.observability import build_trace_config

        config = build_trace_config(state=state, agent_name="router")
        result = await llm.ainvoke(messages, config=config)

    Kalau handler None (disabled / failed), config tetap valid —
    cuma callbacks empty, gak break apa pun.
    """
    handler = get_langfuse_handler()

    metadata = build_trace_metadata(state, agent_name, extra_metadata)

    tags: list[str] = []
    if agent_name:
        tags.append(f"agent:{agent_name}")
    if state and state.get("response_mode"):
        tags.append(f"mode:{state['response_mode']}")
    if extra_tags:
        tags.extend(extra_tags)

    # v3: also expose tags via metadata key supaya muncul di Langfuse trace UI
    if tags:
        metadata["langfuse_tags"] = tags

    config: dict = {
        "callbacks": [handler] if handler else [],
        "metadata": metadata,
    }

    if tags:
        config["tags"] = tags

    return config


# =============================================================================
# Phase 3: HTTP Call Instrumentation
# =============================================================================
#
# Helper untuk wrap HTTP call eksternal (httpx) sebagai child observation
# di parent span Langfuse. Menutup gap arsitektur dimana mata_peri/rapot_peri/
# cerita_peri/memory_job call internal API via HTTP yang tidak ke-trace
# otomatis oleh callback handler (yang cuma cover LangChain).
#
# Pattern: async context manager yang gracefully degrade kalau Langfuse off.

from contextlib import asynccontextmanager


def _redact_url(url: str, max_len: int = 200) -> str:
    """
    Redact URL untuk privacy — strip query params yang mungkin contain PII,
    truncate kalau terlalu panjang.

    Tetap preserve path biar visible di Langfuse trace.
    """
    if not url:
        return ""
    # Strip query string (might contain user_id, token, etc)
    base = url.split("?", 1)[0]
    if len(base) > max_len:
        return base[:max_len] + "..."
    return base


@asynccontextmanager
async def trace_http_call(
    name: str,
    method: str,
    url: str,
    body_keys: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
):
    """
    Context manager untuk trace HTTP call eksternal sebagai child observation
    di parent span Langfuse (kalau enabled).

    Privacy-aware: hanya log method, redacted URL, dan body keys (bukan values).
    Caller bertanggung jawab call `span.update(output=...)` dengan field aman.

    Usage:
        async with trace_http_call(
            name="mata-peri-analyze-image",
            method="POST",
            url=url,
            body_keys=list(json_body.keys()),
        ) as span:
            response = await client.post(url, json=json_body)
            result = response.json()
            if span:
                span.update(output={
                    "status": result.get("status"),
                    "scan_session_id": result.get("scan_session_id"),
                })

    Behavior:
    - Langfuse enabled  → yield Langfuse span object (observation type=span)
    - Langfuse disabled → yield None (caller harus check sebelum span.update)
    - Instrumentation error → yield None (graceful, jangan break HTTP call)

    Span otomatis nest di bawah parent (e.g. "tanya-peri-message") via OTEL
    context propagation.
    """
    langfuse = get_langfuse_client()

    if langfuse is None:
        # Langfuse disabled — yield None so caller's span.update is no-op
        yield None
        return

    try:
        span_input: dict = {
            "method": method.upper(),
            "url": _redact_url(url),
        }
        if body_keys:
            span_input["body_keys"] = sorted(list(body_keys))

        with langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            input=span_input,
            metadata=metadata or {},
        ) as span:
            yield span
    except Exception as e:
        # Defensive: kalau Langfuse SDK error, jangan break HTTP call.
        # Just log dan yield None.
        logger.warning(f"trace_http_call instrumentation failed for {name}: {e}")
        yield None


# =============================================================================
# Phase 4: Node-level + Generation Instrumentation
# =============================================================================
#
# Helpers untuk wrap LangGraph node logic + LLM generation sebagai child
# observation di Langfuse parent span. Goal: full diagnostic visibility —
# Hanif bisa lihat per-node input/output, prompt content yang masuk LLM,
# dan response content yang keluar.
#
# Pattern: 2 async context managers
#   - trace_node      → as_type="span" untuk node logic (router, supervisor, agents)
#   - trace_generation → as_type="generation" untuk LLM calls dengan rich UI
#
# Storage limits: 5000 chars per field (truncate auto), Langfuse render OK.


def _truncate(value, max_len: int = 5000) -> str:
    """
    Truncate string ke max_len chars dengan suffix indicator.
    Tidak break di middle word kalau bisa.
    """
    if value is None:
        return ""
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [truncated {len(s) - max_len} chars]"


def _safe_messages_for_trace(lc_messages, max_per_msg: int = 1500) -> list[dict]:
    """
    Convert list LangChain messages ke format aman untuk Langfuse trace.
    Handle Image content blocks (skip base64 data, just metadata).
    """
    out = []
    for m in lc_messages or []:
        try:
            role = type(m).__name__.replace("Message", "").lower()
            content = getattr(m, "content", "")

            # Handle multimodal content (list of content parts)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "unknown")
                        if ptype == "text":
                            parts.append({"type": "text", "text": _truncate(part.get("text", ""), max_per_msg)})
                        elif ptype in ("image_url", "image"):
                            # Skip base64 — too big untuk trace storage
                            parts.append({"type": ptype, "url_redacted": True})
                        else:
                            parts.append({"type": ptype})
                    else:
                        parts.append({"type": "raw", "value": _truncate(str(part), 200)})
                out.append({"role": role, "content_parts": parts})
            else:
                out.append({"role": role, "content": _truncate(content, max_per_msg)})
        except Exception:
            # Defensive — skip message yang gak bisa di-parse
            continue
    return out


@asynccontextmanager
async def trace_node(
    name: str,
    state: Optional[dict] = None,
    input_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Context manager untuk wrap LangGraph node logic sebagai child span
    di parent trace (kalau Langfuse enabled).

    Usage:
        async with trace_node(
            name="supervisor",
            state=state,
            input_data={"user_message": msg, "has_image": True}
        ) as span:
            # ... node logic ...
            if span:
                span.update(output={"agents_selected": [...], "decision": "..."})

    Behavior:
    - Langfuse enabled  → yield Langfuse span object
    - Langfuse disabled → yield None (caller's span.update is no-op)
    - Instrumentation error → yield None (graceful, jangan break node logic)
    """
    langfuse = get_langfuse_client()

    if langfuse is None:
        yield None
        return

    try:
        # Truncate input_data values defensively
        safe_input = {}
        if input_data:
            for k, v in input_data.items():
                if isinstance(v, str):
                    safe_input[k] = _truncate(v, 2000)
                elif isinstance(v, (list, dict, bool, int, float, type(None))):
                    safe_input[k] = v
                else:
                    safe_input[k] = _truncate(str(v), 500)

        with langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            input=safe_input or None,
            metadata=metadata or {},
        ) as span:
            yield span
    except Exception as e:
        logger.warning(f"trace_node instrumentation failed for {name}: {e}")
        yield None


@asynccontextmanager
async def trace_generation(
    name: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None,
    user_message: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Context manager untuk wrap LLM generation sebagai 'generation' type
    observation. Langfuse render khusus untuk generation: input/output
    formatted nicely, model name di highlight, token count visible.

    Usage:
        async with trace_generation(
            name="generate",
            model="gemini-3.1-flash-lite",
            system_prompt=full_system_prompt,
            messages=lc_messages,
            user_message=last_user_msg,
            metadata={"agents_used": [...], "response_mode": "simple"},
        ) as gen_span:
            # ... LLM streaming ...
            if gen_span:
                gen_span.update(
                    output=full_response,
                    usage_details={"output_tokens": ...},
                )

    Behavior:
    - Langfuse enabled  → yield Langfuse generation span
    - Langfuse disabled → yield None
    - Instrumentation error → yield None (graceful)

    Privacy: system_prompt + messages truncated ke 5000 chars.
    Multimodal images SKIP base64 data (too big).
    """
    langfuse = get_langfuse_client()

    if langfuse is None:
        yield None
        return

    try:
        # Build input dict — comprehensive for diagnostic
        input_data: dict = {}
        if system_prompt is not None:
            input_data["system_prompt"] = _truncate(system_prompt, 5000)
        if messages is not None:
            input_data["messages"] = _safe_messages_for_trace(messages)
        if user_message is not None:
            input_data["user_message"] = _truncate(user_message, 1000)

        kwargs: dict = {
            "as_type": "generation",
            "name": name,
            "input": input_data or None,
            "metadata": metadata or {},
        }
        if model:
            kwargs["model"] = model

        with langfuse.start_as_current_observation(**kwargs) as gen_span:
            yield gen_span
    except Exception as e:
        logger.warning(f"trace_generation instrumentation failed for {name}: {e}")
        yield None
