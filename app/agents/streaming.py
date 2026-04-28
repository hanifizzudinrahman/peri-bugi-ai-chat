"""
Stream adapter — convert LangGraph astream_events() output → existing SSE format.

CRITICAL: Preserves existing FE contract 1:1. FE tidak butuh perubahan.

SSE events yang di-emit (sama dengan custom orchestrator sebelumnya):
- thinking      : per-step indicator
- token         : per-token streaming
- clarify       : view_hint clarification
- quick_reply   : interactive options
- tool          : agent tool calls (rare di Phase 1, lebih banyak di Phase 2)
- suggestions   : suggestion chips after response
- done          : final marker dengan metadata
- error         : kegagalan

Pattern:
1. Wrap dengan Langfuse parent span (preserve observability)
2. Build initial Pydantic state
3. Iterate astream_events() — map per event type ke SSE
4. Heartbeat task background untuk long-running nodes
5. Submit LLM call logs ke api setelah selesai
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from app.agents.state import AgentState, ImageInput
from app.schemas.chat import (
    ChatRequest,
    make_thinking_event,
    make_token_event,
    make_clarify_event,
    make_quick_reply_event,
    make_done_event,
    make_error_event,
    make_suggestions_event,
    make_tool_event,
)

logger = logging.getLogger(__name__)


# Heartbeat config (preserved dari old graph.py)
HEARTBEAT_INTERVAL_SEC = 12.0
HEARTBEAT_LONG_RUNNING_THRESHOLD_SEC = 30.0


# =============================================================================
# Public entry — main streaming function
# =============================================================================


async def langgraph_to_sse_stream(initial_state: AgentState) -> AsyncIterator[str]:
    """
    Top-level streaming entry: invoke LangGraph + emit SSE events.

    Wrapped dengan Langfuse parent span untuk observability.
    Submit LLM call logs ke api setelah selesai.
    """
    from app.config.observability import get_langfuse_client

    langfuse = get_langfuse_client()

    if langfuse is None:
        async for event in _stream_internal(initial_state):
            yield event
        return

    # Langfuse parent span (logic preserved dari graph.py original)
    user_message = _extract_user_msg(initial_state)
    has_image = initial_state.image is not None
    response_mode = initial_state.session.response_mode

    captured_output = {"text": "", "needs_clarification": False, "error": None, "tokens": 0}

    with langfuse.start_as_current_observation(
        as_type="span",
        name="tanya-peri-message",
        input={
            "message": user_message,
            "has_image": has_image,
            "source": initial_state.session.source,
        },
    ) as span:
        try:
            tags = ["tanya-peri", f"mode:{response_mode}"]
            if has_image:
                tags.append("with-image")

            # Update trace dengan user info dari state
            ctx = initial_state.user_context.model_dump()
            user = ctx.get("user") or {}
            user_id = user.get("id") or "unknown"

            span.update_trace(
                user_id=user_id,
                session_id=initial_state.session.session_id,
                tags=tags,
                metadata={
                    "response_mode": response_mode,
                    "allowed_agents": initial_state.control.allowed_agents,
                    "source": initial_state.session.source,
                },
            )

            async for event in _stream_internal(initial_state, captured_output):
                yield event

            # After completion — capture output
            output_summary = {
                "text": captured_output.get("text", "")[:1000],
                "tokens": captured_output.get("tokens", 0),
                "needs_clarification": captured_output.get("needs_clarification", False),
            }
            if captured_output.get("error"):
                output_summary["error"] = captured_output["error"]
                span.update(level="ERROR")

            span.update(output=output_summary)

        except Exception as e:
            logger.error(f"[stream_adapter] Top-level error: {e}", exc_info=True)
            span.update(output={"error": str(e)}, level="ERROR")
            yield make_error_event(f"Maaf, terjadi kesalahan: {str(e)[:100]}")


# =============================================================================
# Internal streaming — actual graph invocation + event mapping
# =============================================================================


async def _stream_internal(
    initial_state: AgentState,
    captured: Optional[dict] = None,
) -> AsyncIterator[str]:
    """
    Invoke compiled LangGraph + map astream_events() output ke SSE.

    Heartbeat: separate background task yang emit thinking events kalau
    no other event muncul dalam HEARTBEAT_INTERVAL_SEC.
    """
    from app.agents.builder import get_compiled_graph

    if captured is None:
        captured = {"text": "", "needs_clarification": False, "error": None, "tokens": 0}

    try:
        graph = await get_compiled_graph()
    except Exception as e:
        logger.error(f"[stream_adapter] Graph compile failed: {e}", exc_info=True)
        yield make_error_event("Maaf, sistem belum siap. Coba lagi sebentar.")
        captured["error"] = str(e)
        return

    # Thread config (untuk checkpointer)
    thread_id = initial_state.session.session_id or f"thread-{uuid.uuid4()}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    # State tracking untuk extract clarification/quick_reply/suggestions
    # post-graph (event aggregator pattern)
    final_state_snapshot: Optional[dict] = None

    # Heartbeat tracking
    last_event_time = time.monotonic()
    started_time = last_event_time
    event_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    heartbeat_active = asyncio.Event()
    heartbeat_active.set()

    async def heartbeat_loop():
        """Emit thinking heartbeat every HEARTBEAT_INTERVAL_SEC saat node lama."""
        nonlocal last_event_time
        try:
            while heartbeat_active.is_set():
                await asyncio.sleep(2.0)  # Check every 2s
                now = time.monotonic()
                idle = now - last_event_time
                total_running = now - started_time
                if (
                    idle >= HEARTBEAT_INTERVAL_SEC
                    and total_running >= HEARTBEAT_LONG_RUNNING_THRESHOLD_SEC
                ):
                    label = "Masih memproses, sebentar ya..."
                    await event_queue.put(make_thinking_event(99, label, done=False))
                    last_event_time = now
        except asyncio.CancelledError:
            pass

    heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def graph_runner():
        """Run graph + emit SSE events ke queue."""
        nonlocal final_state_snapshot, last_event_time
        try:
            current_node: Optional[str] = None
            generate_token_emitted = False  # untuk suppress duplicate tokens

            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                last_event_time = time.monotonic()
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # ── Per-node start: emit thinking ──────────────────────────
                if event_type == "on_chain_start" and event_name in (
                    "supervisor", "agent_dispatcher", "generate"
                ):
                    current_node = event_name
                    # Thinking emitted by node itself (returned in state update);
                    # we don't double-emit here.

                # ── Per-node end: capture state update ─────────────────────
                if event_type == "on_chain_end" and event_name in (
                    "supervisor", "agent_dispatcher", "generate"
                ):
                    # event_data["output"] = node return dict (state update)
                    output = event_data.get("output") or {}

                    # Emit new thinking_steps yang muncul di update ini
                    new_thinking = output.get("thinking_steps") or []
                    if isinstance(new_thinking, list) and new_thinking:
                        # Hanya emit yang TERAKHIR (yang baru ditambah node ini)
                        latest = new_thinking[-1]
                        if isinstance(latest, dict):
                            await event_queue.put(
                                make_thinking_event(
                                    step=latest.get("step", 1),
                                    label=latest.get("label", ""),
                                    done=latest.get("done", True),
                                )
                            )
                        elif hasattr(latest, "model_dump"):
                            d = latest.model_dump()
                            await event_queue.put(
                                make_thinking_event(
                                    step=d.get("step", 1),
                                    label=d.get("label", ""),
                                    done=d.get("done", True),
                                )
                            )

                    # Emit tool events kalau ada tool_calls baru
                    new_tools = output.get("tool_calls") or []
                    if isinstance(new_tools, list):
                        for tc in new_tools[-3:]:  # Last 3 only (avoid spam)
                            if isinstance(tc, dict):
                                await event_queue.put(
                                    make_tool_event(
                                        tool=tc.get("tool", "unknown"),
                                        input=tc.get("input", {}),
                                        result=tc.get("result", {}),
                                    )
                                )

                # ── LLM token streaming ────────────────────────────────────
                if event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content
                        if token and isinstance(token, str):
                            captured["text"] += token
                            captured["tokens"] += 1
                            await event_queue.put(make_token_event(token))
                            generate_token_emitted = True

            # ── After graph done: get final state snapshot ──────────────────
            try:
                state_obj = await graph.aget_state(config)
                if state_obj and state_obj.values:
                    if hasattr(state_obj.values, "model_dump"):
                        final_state_snapshot = state_obj.values.model_dump()
                    elif isinstance(state_obj.values, dict):
                        final_state_snapshot = state_obj.values
            except Exception as e:
                logger.warning(f"[stream_adapter] aget_state failed: {e}")

        except Exception as e:
            logger.error(f"[stream_adapter] Graph execution error: {e}", exc_info=True)
            captured["error"] = str(e)
            await event_queue.put(make_error_event(f"Maaf, terjadi kesalahan: {str(e)[:100]}"))
        finally:
            heartbeat_active.clear()
            await event_queue.put(None)  # Sentinel

    runner_task = asyncio.create_task(graph_runner())

    # Drain event queue
    try:
        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event
    finally:
        heartbeat_active.clear()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ── Post-graph: emit clarify / quick_reply / suggestions / done ───────────
    if final_state_snapshot is None:
        if not captured.get("error"):
            yield make_error_event("Maaf, gagal memuat respons.")
            captured["error"] = "no_final_state"
        return

    needs_clarify = final_state_snapshot.get("needs_clarification", False)
    clarification_data = final_state_snapshot.get("clarification_data")

    if needs_clarify and clarification_data:
        captured["needs_clarification"] = True
        try:
            yield make_clarify_event(
                question=clarification_data.get("question", "Pilih opsi:"),
                options=clarification_data.get("options", []),
                allow_multiple=clarification_data.get("allow_multiple", False),
            )
            yield make_done_event(
                content="",
                metadata=final_state_snapshot.get("llm_metadata", {}),
            )
        except Exception as e:
            logger.error(f"[stream_adapter] Clarify event error: {e}")
            yield make_error_event("Format clarification tidak valid.")
        # Submit LLM logs sebelum return
        await _submit_llm_logs(final_state_snapshot)
        return

    quick_reply_data = final_state_snapshot.get("quick_reply_data")
    if quick_reply_data:
        try:
            yield make_quick_reply_event(
                title=quick_reply_data.get("title", "Pilih:"),
                description=quick_reply_data.get("description"),
                options=quick_reply_data.get("options", []),
                ui_variant=quick_reply_data.get("ui_variant", "list"),
                meta=quick_reply_data.get("meta"),
            )
        except Exception as e:
            logger.error(f"[stream_adapter] Quick reply error: {e}")
            yield make_error_event("Format quick reply tidak valid.")

    # Suggestion chips
    suggestion_chips = final_state_snapshot.get("suggestion_chips") or []
    if suggestion_chips:
        yield make_suggestions_event(suggestion_chips)

    # Done event
    final_response = final_state_snapshot.get("final_response", "") or captured.get("text", "")
    metadata = final_state_snapshot.get("llm_metadata", {}) or {}
    yield make_done_event(content=final_response, metadata=metadata)

    # Submit LLM call logs (background — don't block stream)
    await _submit_llm_logs(final_state_snapshot)


# =============================================================================
# Helpers
# =============================================================================


def _extract_user_msg(state: AgentState) -> str:
    """Extract last user message dari state.messages untuk Langfuse trace."""
    for msg in reversed(state.messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            if isinstance(content, str):
                return content[:500]
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")[:500]
    return ""


async def _submit_llm_logs(final_state: dict) -> None:
    """Submit LLM call logs ke peri-bugi-api (best-effort, non-blocking)."""
    try:
        logs = final_state.get("llm_call_logs") or []
        if not logs:
            return

        from app.utils.llm_log_submitter import submit_llm_call_logs

        # Convert to legacy format yang api expect
        payloads = []
        session_id = final_state.get("session", {}).get("session_id") if isinstance(
            final_state.get("session"), dict
        ) else None

        for log in logs:
            if isinstance(log, dict):
                payload = dict(log)
                if not payload.get("session_id"):
                    payload["session_id"] = session_id
                payloads.append(payload)
            elif hasattr(log, "model_dump"):
                payload = log.model_dump()
                if not payload.get("session_id"):
                    payload["session_id"] = session_id
                payloads.append(payload)

        if payloads:
            asyncio.create_task(submit_llm_call_logs(payloads))
    except Exception as e:
        logger.warning(f"[stream_adapter] Submit LLM logs error: {e}")
