"""
Stream adapter — Hybrid LangGraph + native async generator pattern.

PHASE 2 STREAMING APPROACH (post-Step 2a — Single-Pass Tools):

  Step 1: invoke compiled graph (pre_router → agent → tools → tool_bridge → END)
          - Capture state updates, emit thinking/tool events
          - Graph terminates di END setelah tool_bridge selesai
  Step 2: Get final state snapshot dari checkpointer
  Step 3: Call generate_node(state) langsung sebagai async generator
          - Per-token streaming via `yield make_token_event(token)`
          - Identical pattern dengan original (proven work)
          - LangChain LLM .astream() emit chunks real-time

Kenapa hybrid:
- LangGraph astream(stream_mode="messages") TIDAK reliable emit per-token chunks
  untuk Gemini via langchain_google_genai. Buffering issue di langgraph 0.2.60.
- Original pattern (yield from async generator) PROVEN work — kita preserve untuk
  generate node yang punya streaming concern.
- Graph nodes (pre_router/agent/tools/tool_bridge) tidak perlu streaming, jadi
  cocok di graph dengan stream_mode="updates" untuk emit thinking + tool events.

Phase 6 (cleanup) nanti akan refactor generate.py jadi clean — saat itu kita
bisa eliminate _build_legacy_dict_state shim dan mungkin migrate streaming
pattern juga (tergantung apa Gemini astream sudah reliable saat itu).
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
    Top-level streaming entry: invoke graph + emit SSE events.

    Wrapped dengan Langfuse parent span untuk observability.
    """
    from app.config.observability import get_langfuse_client

    langfuse = get_langfuse_client()

    if langfuse is None:
        async for event in _stream_internal(initial_state):
            yield event
        return

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
# Internal streaming — graph invocation + generate as async generator
# =============================================================================


async def _stream_internal(
    initial_state: AgentState,
    captured: Optional[dict] = None,
) -> AsyncIterator[str]:
    """
    Hybrid streaming:
    - Step 1: invoke graph (pre_router + agent + tools + tool_bridge) via astream(stream_mode="updates")
    - Step 2: get final state, call generate_node(state) as async generator
    - Step 3: emit clarify/quick_reply/suggestions/done after generate selesai
    """
    from app.agents.builder import get_compiled_graph
    from app.agents.nodes.generate import generate_node

    if captured is None:
        captured = {"text": "", "needs_clarification": False, "error": None, "tokens": 0}

    try:
        graph = await get_compiled_graph()
    except Exception as e:
        logger.error(f"[stream_adapter] Graph compile failed: {e}", exc_info=True)
        yield make_error_event("Maaf, sistem belum siap. Coba lagi sebentar.")
        captured["error"] = str(e)
        return

    # CRITICAL FIX: per-turn thread_id (bukan per-session).
    #
    # Original design (pre-Phase 1): TIDAK pakai LangGraph checkpointer.
    # Tiap chat message build state FRESH dari build_initial_state(request).
    # State lama TIDAK di-resume cross-turn.
    #
    # Phase 1 add LangGraph + PostgresSaver checkpointer. Kalau pakai
    # thread_id = session_id, tiap turn berikutnya akan RESUME state lama
    # (image_url, agent_results, image_analysis, agents_selected, dll).
    # Hasilnya: user kirim text-only di session yang pernah upload foto →
    # image_url checkpoint lama bocor → mata_peri_agent tetap aktif → minta
    # clarification ulang. BUG.
    #
    # Solusi: thread_id unique per turn (per chat message). Checkpoint masih
    # tersimpan untuk debug/audit + mid-turn recovery, tapi tidak ada
    # cross-turn state leak. Sesuai design original (fresh state per turn).
    #
    # Memory cross-turn (riwayat percakapan, fakta user) tetap di-handle
    # pakai existing memory_context dari peri-bugi-api (TIDAK via checkpointer).
    # Phase 5 nanti akan revisit kalau perlu cross-turn checkpoint resume.
    base_thread = initial_state.session.session_id or f"thread-{uuid.uuid4()}"
    turn_id = (
        initial_state.session.chat_message_id
        or initial_state.session.trace_id
        or str(uuid.uuid4())
    )
    thread_id = f"{base_thread}::{turn_id}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    final_state: Optional[AgentState] = None
    last_event_time = time.monotonic()
    started_time = last_event_time

    emitted_thinking_steps: set = set()
    emitted_tool_calls: set = set()

    # ========================================================================
    # STEP 1: Run graph (pre_router + agent + tools + tool_bridge) — emit thinking/tool events
    # ========================================================================
    try:
        async for event_tuple in graph.astream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            last_event_time = time.monotonic()

            if not isinstance(event_tuple, dict):
                continue

            for node_name, update in event_tuple.items():
                if not isinstance(update, dict):
                    continue

                # Emit new thinking_steps
                thinking_steps = update.get("thinking_steps") or []
                if isinstance(thinking_steps, list):
                    for step in thinking_steps:
                        step_dict = (
                            step if isinstance(step, dict)
                            else step.model_dump() if hasattr(step, "model_dump")
                            else None
                        )
                        if step_dict is None:
                            continue
                        step_key = (step_dict.get("step"), step_dict.get("label"))
                        if step_key in emitted_thinking_steps:
                            continue
                        emitted_thinking_steps.add(step_key)
                        yield make_thinking_event(
                            step=step_dict.get("step", 1),
                            label=step_dict.get("label", ""),
                            done=step_dict.get("done", True),
                        )

                # Emit tool_calls
                tool_calls = update.get("tool_calls") or []
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        tc_dict = (
                            tc if isinstance(tc, dict)
                            else tc.model_dump() if hasattr(tc, "model_dump")
                            else None
                        )
                        if tc_dict is None:
                            continue
                        tc_key = (
                            tc_dict.get("tool"),
                            tc_dict.get("agent"),
                            json.dumps(tc_dict.get("input", {}), sort_keys=True, default=str),
                        )
                        if tc_key in emitted_tool_calls:
                            continue
                        emitted_tool_calls.add(tc_key)
                        yield make_tool_event(
                            tool=tc_dict.get("tool", "unknown"),
                            input=tc_dict.get("input", {}),
                            result=tc_dict.get("result", {}),
                        )

        # ========================================================================
        # STEP 2: Get final state snapshot dari graph (post-tool_bridge)
        # ========================================================================
        try:
            state_obj = await graph.aget_state(config)
            if state_obj and state_obj.values:
                if isinstance(state_obj.values, AgentState):
                    final_state = state_obj.values
                elif isinstance(state_obj.values, dict):
                    # Reconstruct AgentState from dict
                    final_state = AgentState(**state_obj.values)
        except Exception as e:
            logger.warning(f"[stream_adapter] aget_state failed: {e}", exc_info=True)
            final_state = initial_state  # fallback

        if final_state is None:
            final_state = initial_state

    except Exception as e:
        logger.error(f"[stream_adapter] Graph execution error: {e}", exc_info=True)
        captured["error"] = str(e)
        yield make_error_event(f"Maaf, terjadi kesalahan: {str(e)[:100]}")
        return

    # ========================================================================
    # STEP 3: Generate node sebagai async generator (per-token streaming)
    # ========================================================================
    # CRITICAL: generate_node (legacy) expect dict-state karena pakai
    # state["key"] = value (setitem) untuk mutate final_response, llm_metadata,
    # prompt_debug. Pydantic AgentState immutable by design — pakai dict shim.
    #
    # Pattern: convert Pydantic AgentState → legacy dict (via
    # _build_legacy_dict_state — kept in agent_dispatcher.py for now even
    # though dispatcher itself isn't in graph anymore). After generate_node
    # selesai, extract final_response, llm_metadata, prompt_debug dari dict
    # untuk submit_llm_logs + langfuse capture.
    #
    # Phase 6 (refactor generate.py) akan eliminate dict shim — generate jadi
    # pure Pydantic-aware async generator.
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state

    legacy_dict_state = _build_legacy_dict_state(final_state)

    try:
        async for sse_event in generate_node(legacy_dict_state):
            # generate_node yields SSE strings already (make_thinking_event, make_token_event,
            # make_clarify_event, make_quick_reply_event, make_suggestions_event)
            # NOTE: generate_node TIDAK yield make_done_event — itu tanggung jawab caller (kita).
            #
            # Track captured tokens untuk Langfuse output summary.
            #
            # SSE format (per app/schemas/chat.py:_sse):
            #   data: {"event": "token", "data": "Halo"}\n\n
            # Fields: "event" (event type) + "data" (payload, untuk token = string)
            if "data:" in sse_event and '"event": "token"' in sse_event:
                # Try extract token text untuk captured.text
                try:
                    json_part = sse_event.split("data:", 1)[1].strip()
                    if json_part.endswith("\n\n"):
                        json_part = json_part.rstrip("\n")
                    parsed = json.loads(json_part)
                    if parsed.get("event") == "token":
                        captured["tokens"] += 1
                        # Field is "data" per _sse() helper, NOT "content"/"token".
                        # For token events, data is the token string itself.
                        token_text = parsed.get("data", "")
                        if isinstance(token_text, str):
                            captured["text"] += token_text
                except Exception:
                    pass
            yield sse_event
    except Exception as e:
        logger.error(f"[stream_adapter] generate_node error: {e}", exc_info=True)
        captured["error"] = str(e)
        yield make_error_event(f"Maaf, terjadi kesalahan: {str(e)[:100]}")
        return

    # ========================================================================
    # STEP 4: Emit make_done_event — CRITICAL untuk FE!
    #
    # Mirror logic dari original graph.py line 285-305:
    # - Early return kalau needs_clarification + no final_response (FE handle clarify event sendiri)
    # - Early return kalau quick_reply_data + no final_response
    # - Otherwise: yield make_done_event dengan content + metadata + llm_call_logs
    #
    # Tanpa done event: FE stuck di "Sedang menjawab..." forever, tombol thumbs up/down
    # tidak muncul, tombol stop tidak hilang.
    # ========================================================================
    needs_clarification = legacy_dict_state.get("needs_clarification", False)
    quick_reply_data = legacy_dict_state.get("quick_reply_data")
    final_response = legacy_dict_state.get("final_response", "") or ""

    if needs_clarification and not final_response:
        # Clarify event sudah di-yield generate_node — FE handle sendiri, tidak perlu done
        captured["needs_clarification"] = True
        return

    if quick_reply_data and not final_response:
        # Quick reply sudah di-yield generate_node — FE handle, tidak perlu done
        return

    # Build metadata persis seperti original graph.py
    agents_used = list(legacy_dict_state.get("agent_results", {}).keys())
    llm_call_logs = legacy_dict_state.get("llm_call_logs", []) or []
    session_id = legacy_dict_state.get("session_id", "") or ""

    # Stamp session_id ke logs yang belum punya (parity dengan original)
    for log in llm_call_logs:
        if isinstance(log, dict) and not log.get("session_id"):
            log["session_id"] = session_id

    yield make_done_event(
        content=final_response,
        metadata={
            **legacy_dict_state.get("llm_metadata", {}),
            "agents_used": agents_used,
            "llm_call_logs": llm_call_logs,
        },
    )

    # ========================================================================
    # STEP 5: Submit LLM logs (best-effort, fire-and-forget)
    # Extract LLM call logs dari legacy_dict_state (yang di-mutate generate_node)
    # ========================================================================
    await _submit_llm_logs_from_legacy_dict(legacy_dict_state, final_state)


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


async def _submit_llm_logs_from_legacy_dict(
    legacy_dict_state: dict,
    fallback_state: AgentState,
) -> None:
    """Submit LLM call logs ke peri-bugi-api (best-effort, non-blocking).

    Generate_node mutate legacy_dict_state["llm_call_logs"] (append baru).
    Kita extract dari sana, fallback ke fallback_state.llm_call_logs kalau
    legacy_dict_state tidak punya logs (defensive).
    """
    try:
        # Prefer legacy dict (yang di-mutate generate_node)
        logs = legacy_dict_state.get("llm_call_logs") or []
        if not logs and fallback_state is not None:
            # Fallback: ambil dari Pydantic state (sebelum generate)
            logs = [l.model_dump() if hasattr(l, "model_dump") else dict(l)
                    for l in fallback_state.llm_call_logs]

        if not logs:
            return

        session_id = (
            legacy_dict_state.get("session_id")
            or (fallback_state.session.session_id if fallback_state else None)
        )

        payloads = []
        for log in logs:
            if isinstance(log, dict):
                payload = dict(log)
            elif hasattr(log, "model_dump"):
                payload = log.model_dump()
            else:
                continue
            if not payload.get("session_id"):
                payload["session_id"] = session_id
            payloads.append(payload)

        if payloads:
            from app.services.llm_logger import fire_and_forget_logs
            fire_and_forget_logs(payloads, session_id=session_id)
    except Exception as e:
        logger.warning(f"[stream_adapter] Submit LLM logs error: {e}")


async def _submit_llm_logs(final_state: AgentState) -> None:
    """[LEGACY] Submit LLM call logs dari Pydantic state.

    Kept for compat — kalau ada code lain yang panggil. Phase 1 main path
    pakai _submit_llm_logs_from_legacy_dict.
    """
    try:
        logs = final_state.llm_call_logs or []
        if not logs:
            return

        session_id = final_state.session.session_id

        payloads = []
        for log in logs:
            if hasattr(log, "model_dump"):
                payload = log.model_dump()
            elif isinstance(log, dict):
                payload = dict(log)
            else:
                continue
            if not payload.get("session_id"):
                payload["session_id"] = session_id
            payloads.append(payload)

        if payloads:
            from app.services.llm_logger import fire_and_forget_logs
            fire_and_forget_logs(payloads, session_id=session_id)
    except Exception as e:
        logger.warning(f"[stream_adapter] Submit LLM logs error: {e}")
