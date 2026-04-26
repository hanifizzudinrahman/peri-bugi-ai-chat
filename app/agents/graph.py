"""
graph.py — LangGraph StateGraph untuk Tanya Peri v2.

Phase 1: kb_dental, user_profile, app_faq
Phase 2: rapot_peri, cerita_peri, mata_peri  ← aktif sekarang
Phase 4: janji_peri
"""
import asyncio
from typing import AsyncIterator

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.nodes.generate import generate_node
from app.agents.sub_agents import (
    kb_dental_agent,
    user_profile_agent,
    app_faq_agent,
)
from app.agents.sub_agents.phase2_agents import (
    rapot_peri_agent,
    cerita_peri_agent,
    mata_peri_agent,
)
from app.schemas.chat import (
    ChatRequest,
    RnDChatRequest,
    make_done_event,
    make_error_event,
    make_thinking_event,
)

# Registry agent → function
_AGENT_REGISTRY = {
    # Phase 1
    "kb_dental": kb_dental_agent,
    "user_profile": user_profile_agent,
    "app_faq": app_faq_agent,
    # Phase 2
    "rapot_peri": rapot_peri_agent,
    "cerita_peri": cerita_peri_agent,
    "mata_peri": mata_peri_agent,
    # Phase 4 — uncomment saat siap:
    # "janji_peri": janji_peri_agent,
}


async def run_agent(initial_state: AgentState) -> AsyncIterator[str]:
    """
    Entry point utama — jalankan seluruh graph dan yield SSE events.

    Wraps execution dengan Langfuse parent span (kalau enabled) supaya semua
    child LLM call ter-nest di bawah 1 trace utuh. Kalau Langfuse disabled,
    langsung jalankan internal logic tanpa overhead.

    Behavior:
    - LANGFUSE_ENABLED=false → jalankan _run_agent_internal langsung
    - LANGFUSE_ENABLED=true  → wrap dengan span "tanya-peri-message" yang
      otomatis jadi root trace; LLM callback handler nge-detect parent
      context via OTEL → semua child observation nest di bawah trace ini.
    """
    from app.config.observability import (
        get_langfuse_client,
        extract_user_message,
    )

    langfuse = get_langfuse_client()

    # Path 1: Langfuse disabled → run normally
    if langfuse is None:
        async for event in _run_agent_internal(initial_state):
            yield event
        return

    # Path 2: Langfuse enabled → wrap dengan parent span
    user_message = extract_user_message(initial_state)
    session_id = initial_state.get("session_id")
    user_context = initial_state.get("user_context") or {}
    user_id = user_context.get("user_id") or user_context.get("id")
    response_mode = initial_state.get("response_mode", "simple")
    has_image = bool(initial_state.get("image_url"))
    source = initial_state.get("source")

    # Trace input — ringkasan request
    trace_input: dict = {}
    if user_message:
        trace_input["message"] = user_message
    if has_image:
        trace_input["has_image"] = True
    if source:
        trace_input["source"] = source

    # Tags untuk filter di dashboard
    tags = ["tanya-peri", f"mode:{response_mode}"]
    if has_image:
        tags.append("with-image")
    if source:
        tags.append(f"source:{source}")

    # Try import propagate_attributes, fallback graceful kalau API berubah
    try:
        from langfuse import propagate_attributes
    except ImportError:
        propagate_attributes = None  # type: ignore

    # Buffer untuk capture final response — diset saat done event muncul
    captured_output: dict = {"text": "", "needs_clarification": False, "error": None}

    async def _run_with_span():
        """Inner generator yang yield events sambil capture output untuk trace."""
        with langfuse.start_as_current_observation(
            as_type="span",
            name="tanya-peri-message",
            input=trace_input or None,
        ) as span:
            try:
                attrs_kwargs = {}
                if user_id:
                    attrs_kwargs["user_id"] = str(user_id)
                if session_id:
                    attrs_kwargs["session_id"] = str(session_id)
                if tags:
                    attrs_kwargs["tags"] = tags

                if propagate_attributes and attrs_kwargs:
                    # Propagate ke seluruh child observation (LLM calls, retrievers)
                    with propagate_attributes(**attrs_kwargs):
                        async for event in _run_agent_internal(initial_state):
                            _capture_response(event, captured_output)
                            yield event
                else:
                    # Fallback: gak ada propagate_attributes (langfuse v lama?)
                    async for event in _run_agent_internal(initial_state):
                        _capture_response(event, captured_output)
                        yield event

                # Update trace output di akhir
                trace_output = _build_trace_output(captured_output, initial_state)
                span.update(output=trace_output)

            except Exception as e:
                # Capture error di trace
                span.update(
                    output={"error": str(e), "type": type(e).__name__},
                    level="ERROR",
                    status_message=str(e),
                )
                raise

    async for event in _run_with_span():
        yield event


def _capture_response(event_str: str, captured: dict) -> None:
    """Parse SSE event untuk capture final response (untuk trace output)."""
    import json
    try:
        if not event_str.startswith("data: "):
            return
        parsed = json.loads(event_str[6:])
        ev_type = parsed.get("event")

        if ev_type == "token":
            token = parsed.get("data", {}).get("token", "")
            if token:
                captured["text"] += token
        elif ev_type == "done":
            data = parsed.get("data", {})
            content = data.get("content")
            if content:
                captured["text"] = content  # final, override token accumulation
        elif ev_type == "error":
            captured["error"] = parsed.get("data", {}).get("message")
        elif ev_type == "clarification":
            captured["needs_clarification"] = True
    except Exception:
        # Defensive — jangan break execution kalau parsing gagal
        pass


def _build_trace_output(captured: dict, state: AgentState) -> dict:
    """Build dict output untuk trace berdasarkan hasil capture + state."""
    output: dict = {}

    if captured.get("error"):
        output["error"] = captured["error"]
    if captured.get("text"):
        # Limit text supaya gak overflow Langfuse storage (mostly cosmetic)
        text = captured["text"]
        output["response"] = text[:5000] if len(text) > 5000 else text
    if captured.get("needs_clarification"):
        output["needs_clarification"] = True

    # Metadata tambahan dari state
    agents_used = list(state.get("agent_results", {}).keys())
    if agents_used:
        output["agents_used"] = agents_used

    if state.get("image_analysis"):
        output["image_analyzed"] = True

    return output if output else {"response": "(no output captured)"}


async def _run_agent_internal(initial_state: AgentState) -> AsyncIterator[str]:
    """
    Internal logic — original run_agent body, unchanged.
    Dipisah dari run_agent supaya wrapper Langfuse bisa di-apply transparently
    tanpa modify graph orchestration logic.
    """
    state = initial_state

    state.setdefault("thinking_steps", [])
    state.setdefault("tool_calls", [])
    state.setdefault("retrieved_docs", [])
    state.setdefault("agent_results", {})
    state.setdefault("image_analysis", None)
    state.setdefault("needs_clarification", False)
    state.setdefault("clarification_data", None)
    state.setdefault("quick_reply_data", None)
    state.setdefault("suggestion_chips", None)
    state.setdefault("final_response", "")
    state.setdefault("llm_metadata", {})
    state.setdefault("llm_call_logs", [])
    state.setdefault("agents_selected", [])
    state.setdefault("execution_plan", {})
    state.setdefault("allowed_agents", [])
    state.setdefault("agent_configs", {})
    state.setdefault("response_mode", "simple")
    state.setdefault("memory_context", {})
    state.setdefault("top_k_docs", 3)
    state.setdefault("force_intent", None)
    state.setdefault("llm_provider_override", None)
    state.setdefault("llm_model_override", None)
    state.setdefault("llm_temperature_override", None)
    state.setdefault("llm_max_tokens_override", None)
    state.setdefault("embedding_provider_override", None)
    state.setdefault("embedding_model_override", None)
    state.setdefault("include_prompt_debug", False)
    state.setdefault("prompt_debug", None)
    state.setdefault("quick_reply_option_id", None)

    try:
        # Step 1: Supervisor
        async for event in supervisor_node(state):
            yield event

        agents_selected = state.get("agents_selected", [])
        execution_plan = state.get("execution_plan", {})

        # Step 2: Run selected agents
        if agents_selected:
            step_num = len(state.get("thinking_steps", [])) + 1
            label = _thinking_label(agents_selected)

            yield make_thinking_event(step=step_num, label=label, done=False)

            mode = execution_plan.get("mode", "sequential")
            # Bug #5 fix — agents jalan async (await fn(state)) yang
            # bisa lama (e.g. mata_peri call ai-cv via api internal,
            # potential 60-120s). Tanpa heartbeat, FE diam total selama
            # itu — user kira app stuck.
            #
            # Pakai heartbeat-aware runner: emit thinking event berkala
            # supaya FE tahu app masih hidup, dengan label yang berubah-ubah
            # (e.g. "Menganalisis foto..." → "Hampir selesai..." → ...)
            # untuk feedback visual yang lebih hidup.
            if mode == "parallel" and len(agents_selected) > 1:
                async for event in _run_agents_parallel_with_heartbeat(
                    state, agents_selected, base_step=step_num, base_label=label
                ):
                    yield event
            else:
                async for event in _run_agents_sequential_with_heartbeat(
                    state, agents_selected, base_step=step_num, base_label=label
                ):
                    yield event

            yield make_thinking_event(step=step_num, label=label, done=True)
            state["thinking_steps"].append({"step": step_num, "label": label, "done": True})

        # Step 3: Generate
        async for event in generate_node(state):
            yield event

        # Step 4: Emit done
        if state.get("needs_clarification") and not state.get("final_response"):
            return
        if state.get("quick_reply_data") and not state.get("final_response"):
            return

        agents_used = list(state.get("agent_results", {}).keys())
        llm_call_logs = state.get("llm_call_logs", [])
        session_id = state.get("session_id", "")
        for log in llm_call_logs:
            if not log.get("session_id"):
                log["session_id"] = session_id

        yield make_done_event(
            content=state["final_response"],
            metadata={
                **state.get("llm_metadata", {}),
                "agents_used": agents_used,
                "llm_call_logs": llm_call_logs,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield make_error_event(f"Maaf, Tanya Peri sedang mengalami gangguan: {str(e)}")


def _thinking_label(agents_selected: list[str]) -> str:
    if len(agents_selected) > 1:
        return f"Mengumpulkan info dari {len(agents_selected)} sumber..."
    labels = {
        "kb_dental": "Mencari referensi kesehatan gigi...",
        "user_profile": "Membaca data profil...",
        "app_faq": "Mencari info aplikasi...",
        "rapot_peri": "Mengecek rapot sikat gigi...",
        "cerita_peri": "Mengecek progress Cerita Peri...",
        "mata_peri": "Menganalisis data gigi...",
        "janji_peri": "Mencari informasi dokter...",
    }
    return labels.get(agents_selected[0], "Mengumpulkan informasi...")


async def _run_agents_parallel(state: AgentState, agent_keys: list[str]) -> None:
    """Original parallel runner — dipakai oleh heartbeat wrapper."""
    tasks = []
    valid_keys = []
    for key in agent_keys:
        fn = _AGENT_REGISTRY.get(key)
        if fn:
            tasks.append(fn(state))
            valid_keys.append(key)
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for key, result in zip(valid_keys, results):
        state["agent_results"][key] = {"error": str(result)} if isinstance(result, Exception) else result


async def _run_agents_sequential(state: AgentState, agent_keys: list[str]) -> None:
    """Original sequential runner — dipakai oleh heartbeat wrapper."""
    for key in agent_keys:
        fn = _AGENT_REGISTRY.get(key)
        if fn:
            try:
                result = await fn(state)
                state["agent_results"][key] = result
            except Exception as e:
                state["agent_results"][key] = {"error": str(e)}


# =============================================================================
# Heartbeat-aware runners (Bug #5 fix)
# =============================================================================
#
# Wrapper yang yield periodic `thinking` events selama agent execution,
# supaya FE punya visual feedback hidup saat agent lama (e.g. mata_peri
# image analysis yang harus call ai-cv pipeline 30-90s).
#
# Pattern:
# 1. Spawn agent execution sebagai asyncio.Task (background).
# 2. Di main loop, asyncio.wait dengan timeout HEARTBEAT_INTERVAL.
# 3. Kalau task belum done, yield thinking event (label rotating).
# 4. Repeat sampai task done.
# 5. Hard timeout HEARTBEAT_MAX_DURATION untuk safety.

# Heartbeat setiap 12 detik — cukup sering supaya FE feel alive,
# cukup jarang supaya tidak spam SSE stream.
HEARTBEAT_INTERVAL_SEC = 12.0

# Hard ceiling — kalau agent belum selesai dalam waktu ini,
# emit warning thinking event tapi tetap tunggu sampai timeout HTTP-nya
# (timeout HTTP di-set di phase2_agents._call_internal_api_post = 120s).
HEARTBEAT_LONG_RUNNING_THRESHOLD_SEC = 30.0


def _heartbeat_label(base_label: str, elapsed_sec: float, agent_keys: list[str]) -> str:
    """
    Rotating label untuk heartbeat — kasih sense of progress ke user.
    Lebih spesifik untuk image analysis karena itu yang paling sering
    lama (ai-cv pipeline pakai YOLO + segmentation models).
    """
    is_image_flow = "mata_peri" in agent_keys
    if is_image_flow:
        if elapsed_sec < 8:
            return "Menganalisis foto gigi..."
        if elapsed_sec < 20:
            return "Membaca area gigi & karies..."
        if elapsed_sec < 40:
            return "Hampir selesai, sebentar ya..."
        # Long-running fallback — minta user sabar
        return "Analisis butuh waktu lebih lama, terima kasih sudah sabar 🙏"
    # Non-image flow — fallback ke base label
    return base_label


async def _run_agents_sequential_with_heartbeat(
    state: AgentState,
    agent_keys: list[str],
    base_step: int,
    base_label: str,
) -> AsyncIterator[str]:
    """
    Sequential runner dengan periodic heartbeat events.

    Identik dengan _run_agents_sequential di sisi state mutation,
    tapi yield thinking events setiap HEARTBEAT_INTERVAL_SEC supaya
    FE tidak diam terlalu lama saat agent slow.
    """
    task = asyncio.create_task(_run_agents_sequential(state, agent_keys))
    async for event in _heartbeat_loop(task, base_step, base_label, agent_keys):
        yield event


async def _run_agents_parallel_with_heartbeat(
    state: AgentState,
    agent_keys: list[str],
    base_step: int,
    base_label: str,
) -> AsyncIterator[str]:
    """Parallel version — sama pattern dengan sequential."""
    task = asyncio.create_task(_run_agents_parallel(state, agent_keys))
    async for event in _heartbeat_loop(task, base_step, base_label, agent_keys):
        yield event


async def _heartbeat_loop(
    task: asyncio.Task,
    base_step: int,
    base_label: str,
    agent_keys: list[str],
) -> AsyncIterator[str]:
    """
    Tunggu task selesai sambil yield heartbeat thinking events.

    Defensive:
    - Kalau task raise exception, propagate setelah loop selesai supaya
      caller graph.py bisa catch dan emit make_error_event seperti biasa.
    - Pakai asyncio.wait dengan timeout — kalau task selesai sebelum
      heartbeat tick, langsung exit loop (tidak ada heartbeat extra).
    """
    import time
    start_time = time.monotonic()
    long_running_warned = False

    while not task.done():
        try:
            # Tunggu task selesai atau timeout
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL_SEC)
        except asyncio.TimeoutError:
            # Task masih jalan — emit heartbeat thinking event
            elapsed = time.monotonic() - start_time
            label = _heartbeat_label(base_label, elapsed, agent_keys)
            yield make_thinking_event(step=base_step, label=label, done=False)

            # Log warning kalau long-running (untuk observability)
            if not long_running_warned and elapsed >= HEARTBEAT_LONG_RUNNING_THRESHOLD_SEC:
                long_running_warned = True
                import logging
                logging.getLogger(__name__).warning(
                    f"[heartbeat] Agent execution > {HEARTBEAT_LONG_RUNNING_THRESHOLD_SEC}s: "
                    f"agents={agent_keys} elapsed={elapsed:.1f}s"
                )
        except asyncio.CancelledError:
            # Caller cancel — propagate
            task.cancel()
            raise

    # Task done — kalau exception, propagate
    if task.exception() is not None:
        # Re-raise sehingga _run_agents_sequential's try/except bisa handle
        # via state.agent_results (sudah di-handle di runner).
        # Di sini kita tidak raise karena runner sudah catch internal.
        # Cukup log untuk visibility.
        import logging
        logging.getLogger(__name__).warning(
            f"[heartbeat] Agent task ended with exception: {task.exception()}"
        )


# =============================================================================
# State builders
# =============================================================================

def build_initial_state(request: ChatRequest) -> AgentState:
    return AgentState(
        session_id=request.session_id,
        user_context=request.user_context,
        messages=request.messages,
        prompts=request.prompts,
        image_url=request.image_url,
        image_url_public=request.image_url_public,
        clarification_selected=request.clarification_selected,
        quick_reply_option_id=request.quick_reply_option_id,
        chat_message_id=request.chat_message_id,
        trace_id=request.trace_id,
        source=request.source,
        allowed_agents=request.allowed_agents,
        agent_configs=request.agent_configs,
        response_mode=request.response_mode,
        memory_context=request.memory_context,
        llm_provider_override=None,
        llm_model_override=None,
        llm_temperature_override=None,
        llm_max_tokens_override=None,
        embedding_provider_override=None,
        embedding_model_override=None,
        top_k_docs=3,
        force_intent=None,
        include_prompt_debug=False,
        agents_selected=[],
        execution_plan={},
        agent_results={},
        retrieved_docs=[],
        image_analysis=None,
        scan_session_id=None,
        needs_clarification=False,
        clarification_data=None,
        quick_reply_data=None,
        suggestion_chips=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )


def build_rnd_state(request: RnDChatRequest) -> AgentState:
    messages = list(request.conversation_history)
    messages.append({"role": "user", "content": request.message})

    prompts = dict(request.custom_prompts)
    if request.system_prompt:
        prompts["_override_system"] = request.system_prompt

    return AgentState(
        session_id=f"rnd-{request.experiment_id or 'test'}",
        user_context=request.user_context,
        messages=messages,
        prompts=prompts,
        image_url=None,
        image_url_public=None,
        clarification_selected=None,
        quick_reply_option_id=None,
        chat_message_id=None,
        trace_id=None,
        source=None,
        allowed_agents=request.allowed_agents or list(_AGENT_REGISTRY.keys()),
        agent_configs={},
        response_mode=request.response_mode or "simple",
        memory_context={},
        llm_provider_override=request.provider,
        llm_model_override=request.model,
        llm_temperature_override=request.temperature,
        llm_max_tokens_override=request.max_tokens,
        embedding_provider_override=request.embedding_provider,
        embedding_model_override=request.embedding_model,
        top_k_docs=request.top_k_docs,
        force_intent=request.force_intent,
        include_prompt_debug=request.include_prompt_in_response,
        agents_selected=[],
        execution_plan={},
        agent_results={},
        retrieved_docs=[],
        image_analysis=None,
        scan_session_id=None,
        needs_clarification=False,
        clarification_data=None,
        quick_reply_data=None,
        suggestion_chips=None,
        thinking_steps=[],
        tool_calls=[],
        final_response="",
        llm_metadata={},
        llm_call_logs=[],
        prompt_debug=None,
    )
