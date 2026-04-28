"""
Shared HTTP helpers for Phase 2 tools.

Extracted from app/agents/sub_agents/phase2_agents.py to be reusable across
tools layer. Logic and Langfuse trace integration preserved 1:1.

Phase 2 design principle: tools call HTTP via these helpers, not via httpx
directly. This guarantees:
- Consistent trace span naming
- Consistent timeout/error handling
- Consistent INTERNAL_SECRET auth header
- Consistent response shape ({"has_data": bool, "error"?: str, ...})
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


# Internal auth header — shared dengan peri-bugi-api
_INTERNAL_HEADERS = {"X-Internal-Secret": settings.INTERNAL_SECRET}

# Default timeout 10s — analyze_chat_image overrides to 120s
_DEFAULT_TIMEOUT = 10


# =============================================================================
# Public helpers
# =============================================================================


async def call_internal_get(path: str, *, timeout: Optional[int] = None) -> dict:
    """
    GET ke peri-bugi-api internal endpoint.

    Args:
        path: Path relatif (e.g. "/api/v1/internal/agent/brushing-stats/{uid}")
        timeout: Override default 10s timeout

    Returns:
        dict — response body kalau success, atau error envelope:
            {"error": "...", "has_data": False}
    """
    if not settings.PERI_API_URL:
        return {"error": "PERI_API_URL tidak diset", "has_data": False}

    url = f"{settings.PERI_API_URL.rstrip('/')}{path}"
    actual_timeout = timeout or _DEFAULT_TIMEOUT

    # Langfuse trace span (graceful degradation kalau Langfuse off)
    from app.config.observability import trace_http_call, _safe_dict_for_trace
    span_name = f"http-internal-get-{path.split('/')[-1].split('?')[0] or 'root'}"

    async with trace_http_call(
        name=span_name,
        method="GET",
        url=url,
        metadata={"timeout_sec": actual_timeout},
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=actual_timeout) as client:
                resp = await client.get(url, headers=_INTERNAL_HEADERS)
                resp.raise_for_status()
                result = resp.json()
            if span:
                span.update(output={
                    "status_code": resp.status_code,
                    "has_data": result.get("has_data"),
                    "response_body": _safe_dict_for_trace(result),
                })
            return result
        except httpx.TimeoutException:
            logger.warning(f"[tools._http] GET timeout: {path}")
            if span:
                span.update(
                    output={"error": "timeout"},
                    level="ERROR",
                    status_message=f"timeout after {actual_timeout}s",
                )
            return {"error": "timeout", "has_data": False}
        except Exception as e:
            logger.warning(f"[tools._http] GET error {path}: {e}")
            if span:
                span.update(
                    output={"error": str(e)[:200]},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
            return {"error": str(e), "has_data": False}


async def call_internal_post(
    path: str,
    json_body: dict,
    *,
    extra_headers: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """
    POST ke peri-bugi-api internal endpoint.

    Args:
        path: Path relatif
        json_body: JSON body untuk POST
        extra_headers: Optional headers (e.g. X-Request-ID untuk trace propagation)
        timeout: Override default 10s timeout (analyze_chat_image pakai 120s)

    Returns:
        dict — response body kalau success, atau error envelope:
            {"status": "failed", "error": "...", "error_code": "..."}
    """
    if not settings.PERI_API_URL:
        return {"status": "failed", "error": "PERI_API_URL tidak diset"}

    url = f"{settings.PERI_API_URL.rstrip('/')}{path}"
    headers = dict(_INTERNAL_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    actual_timeout = timeout or _DEFAULT_TIMEOUT

    from app.config.observability import trace_http_call, _safe_dict_for_trace
    span_name = f"http-internal-post-{path.split('/')[-1].split('?')[0] or 'root'}"

    async with trace_http_call(
        name=span_name,
        method="POST",
        url=url,
        body=json_body,
        body_keys=list(json_body.keys()) if json_body else None,
        metadata={"timeout_sec": actual_timeout},
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=actual_timeout) as client:
                resp = await client.post(url, json=json_body, headers=headers)
                resp.raise_for_status()
                result = resp.json()
            if span:
                span.update(output={
                    "status_code": resp.status_code,
                    "status": result.get("status"),
                    "scan_session_id": result.get("scan_session_id"),
                    "error_code": result.get("error_code"),
                    "response_body": _safe_dict_for_trace(result),
                })
            return result
        except httpx.TimeoutException:
            logger.warning(f"[tools._http] POST timeout: {path}")
            if span:
                span.update(
                    output={"error": "timeout"},
                    level="ERROR",
                    status_message=f"timeout after {actual_timeout}s",
                )
            return {"status": "failed", "error": "timeout", "error_code": "api_timeout"}
        except httpx.HTTPStatusError as e:
            logger.warning(f"[tools._http] POST HTTP {e.response.status_code}: {path}")
            if span:
                span.update(
                    output={"status_code": e.response.status_code, "error_code": "api_http_error"},
                    level="ERROR",
                    status_message=f"HTTP {e.response.status_code}",
                )
            return {
                "status": "failed",
                "error": f"HTTP {e.response.status_code}",
                "error_code": "api_http_error",
            }
        except Exception as e:
            logger.exception(f"[tools._http] POST error {path}: {e}")
            if span:
                span.update(
                    output={"error": str(e)[:200], "error_code": "api_unknown_error"},
                    level="ERROR",
                    status_message=str(e)[:200],
                )
            return {
                "status": "failed",
                "error": str(e)[:200],
                "error_code": "api_unknown_error",
            }
