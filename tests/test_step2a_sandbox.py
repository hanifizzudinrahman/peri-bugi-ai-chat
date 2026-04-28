#!/usr/bin/env python
"""
Sandbox functional test for Phase 2 Step 2a nodes.

Tests each node in isolation with mock AgentState. Does NOT call real LLM/API.

Test scenarios:
1. pre_router with image → forced_tool_calls populated
2. pre_router with text-only → forced_tool_calls empty
3. agent_node with forced_tool_calls → AIMessage emits forced calls (no LLM)
4. agent_node with empty allowed_agents → empty AIMessage (no crash)
5. tools_node with no AIMessage → pass through
6. tool_bridge with no ToolMessages → pass through
7. tool_bridge with kb_dental ToolMessage → bridges to agent_results+retrieved_docs
8. tool_bridge with analyze_chat_image clarification → sets needs_clarification
9. system prompt builder with full context

Run: PYTHONPATH=. python tests/test_step2a_sandbox.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Disable Langfuse + use ollama (no API key/SSL needed) for testing
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("INTERNAL_SECRET", "test-secret")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "test")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.state import (
    AgentState, SessionMeta, UserContextData, AgentControl,
    ImageInput, RnDOverrides, MemorySnapshot,
)


def _build_state(
    *,
    image_url: str | None = None,
    clarification_selected: list[str] | None = None,
    allowed_agents: list[str] | None = None,
    user_message: str = "halo",
    response_mode: str = "simple",
) -> AgentState:
    """Build a representative mock state."""
    image = None
    if image_url:
        image = ImageInput(
            image_url=image_url,
            clarification_selected=clarification_selected,
        )

    return AgentState(
        session=SessionMeta(
            session_id="test-session",
            response_mode=response_mode,
            chat_message_id="msg-123",
            trace_id="trace-456",
        ),
        user_context=UserContextData(
            user={"id": "u-test", "nickname": "Bunda Test", "gender": "F"},
            child={"id": "c-test", "nickname": "Adek", "age_years": 5},
            brushing={"current_streak": 7},
        ),
        control=AgentControl(
            allowed_agents=(
                allowed_agents
                if allowed_agents is not None
                else [
                    "kb_dental", "app_faq", "user_profile",
                    "rapot_peri", "cerita_peri", "mata_peri",
                ]
            ),
        ),
        image=image,
        prompts={},
        rnd=RnDOverrides(),
        memory=MemorySnapshot(),
        messages=[HumanMessage(content=user_message)],
    )


async def test_1_pre_router_with_image():
    print("\n[Test 1] pre_router with image → forced_tool_calls=[analyze_chat_image]")
    from app.agents.nodes.pre_router import pre_router_node
    state = _build_state(image_url="https://example.com/test.jpg")
    update = await pre_router_node(state)
    forced = update.get("forced_tool_calls", [])
    assert len(forced) == 1, f"Expected 1 forced call, got {len(forced)}"
    assert forced[0]["name"] == "analyze_chat_image", f"Wrong tool: {forced[0]}"
    assert len(update.get("thinking_steps", [])) == 1
    print("    ✅ PASSED")


async def test_2_pre_router_text_only():
    print("\n[Test 2] pre_router text-only → forced_tool_calls=[]")
    from app.agents.nodes.pre_router import pre_router_node
    state = _build_state(user_message="halo peri")
    update = await pre_router_node(state)
    assert update.get("forced_tool_calls") == [], f"Expected empty, got {update.get('forced_tool_calls')}"
    print("    ✅ PASSED")


async def test_3_pre_router_image_no_mata_peri():
    print("\n[Test 3] pre_router image but mata_peri not allowed → no forced call (graceful)")
    from app.agents.nodes.pre_router import pre_router_node
    state = _build_state(
        image_url="https://example.com/test.jpg",
        allowed_agents=["kb_dental"],  # no mata_peri
    )
    update = await pre_router_node(state)
    assert update.get("forced_tool_calls") == []
    print("    ✅ PASSED (graceful, no crash)")


async def test_4_agent_node_with_forced():
    print("\n[Test 4] agent_node with forced_tool_calls → emits AIMessage with that call (no LLM)")
    from app.agents.nodes.agent import agent_node
    state = _build_state(image_url="https://example.com/test.jpg")
    # Simulate: pre_router already set forced_tool_calls
    state = state.model_copy(update={
        "forced_tool_calls": [{"name": "analyze_chat_image", "args": {}}],
    })
    update = await agent_node(state)
    msgs = update.get("messages", [])
    assert len(msgs) == 1, f"Expected 1 message, got {len(msgs)}"
    ai_msg = msgs[0]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "analyze_chat_image"
    # No llm_call_logs since LLM was skipped
    assert "llm_call_logs" not in update or len(update.get("llm_call_logs", [])) == 0
    print("    ✅ PASSED (forced path, no LLM invocation)")


async def test_5_agent_node_no_tools():
    print("\n[Test 5] agent_node with empty allowed_agents → empty AIMessage (no LLM, no crash)")
    from app.agents.nodes.agent import agent_node
    state = _build_state(allowed_agents=[])
    update = await agent_node(state)
    msgs = update.get("messages", [])
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].tool_calls == []
    print("    ✅ PASSED")


async def test_6_tools_node_no_ai_message():
    print("\n[Test 6] tools_node with no recent AIMessage → pass through (empty update)")
    from app.agents.nodes.tools_node import tools_node
    state = _build_state()  # only has HumanMessage
    update = await tools_node(state)
    assert update == {} or update.get("messages") is None
    print("    ✅ PASSED")


async def test_7_tool_bridge_no_tool_messages():
    print("\n[Test 7] tool_bridge with no ToolMessages → pass through (empty update)")
    from app.agents.nodes.tool_bridge import tool_bridge_node
    state = _build_state()
    update = await tool_bridge_node(state)
    assert update == {}
    print("    ✅ PASSED")


async def test_8_tool_bridge_kb_dental():
    print("\n[Test 8] tool_bridge with kb_dental ToolMessage → bridges to agent_results")
    from app.agents.nodes.tool_bridge import tool_bridge_node
    state = _build_state()
    state = state.model_copy(update={
        "messages": [
            *state.messages,
            AIMessage(content="", tool_calls=[
                {"name": "search_dental_knowledge", "args": {"query": "kenapa"}, "id": "c1", "type": "tool_call"}
            ]),
            ToolMessage(
                content=json.dumps({"docs": ["doc1", "doc2"], "source_count": 2}),
                name="search_dental_knowledge",
                tool_call_id="c1",
            ),
        ],
    })
    update = await tool_bridge_node(state)
    ar = update.get("agent_results", {})
    assert "kb_dental" in ar, f"agent_results keys: {list(ar.keys())}"
    assert ar["kb_dental"]["docs"] == ["doc1", "doc2"]
    assert update.get("retrieved_docs") == ["doc1", "doc2"]
    assert "kb_dental" in update.get("agents_selected", [])
    print("    ✅ PASSED")


async def test_9_tool_bridge_clarification():
    print("\n[Test 9] tool_bridge with analyze_chat_image clarification → sets needs_clarification")
    from app.agents.nodes.tool_bridge import tool_bridge_node
    state = _build_state(image_url="https://example.com/test.jpg")
    clarification_payload = {
        "type": "single_select",
        "question": "Foto ini gigi bagian mana?",
        "options": [{"id": "front", "label": "Depan"}],
    }
    state = state.model_copy(update={
        "messages": [
            *state.messages,
            AIMessage(content="", tool_calls=[
                {"name": "analyze_chat_image", "args": {}, "id": "c1", "type": "tool_call"}
            ]),
            ToolMessage(
                content=json.dumps({
                    "has_data": False,
                    "needs_clarification": True,
                    "clarification_data": clarification_payload,
                    "mode": "clarification_pending",
                }),
                name="analyze_chat_image",
                tool_call_id="c1",
            ),
        ],
    })
    update = await tool_bridge_node(state)
    assert update.get("needs_clarification") is True
    assert update.get("clarification_data") == clarification_payload
    assert "mata_peri" in update.get("agents_selected", [])
    print("    ✅ PASSED")


async def test_10_tool_bridge_image_analysis_success():
    print("\n[Test 10] tool_bridge with analyze_chat_image SUCCESS → sets image_analysis + scan_session_id")
    from app.agents.nodes.tool_bridge import tool_bridge_node
    state = _build_state(image_url="https://example.com/test.jpg")
    image_analysis_payload = {"session_summary": {"caries": "moderate"}, "results": []}
    state = state.model_copy(update={
        "messages": [
            *state.messages,
            AIMessage(content="", tool_calls=[
                {"name": "analyze_chat_image", "args": {}, "id": "c1", "type": "tool_call"}
            ]),
            ToolMessage(
                content=json.dumps({
                    "has_data": True,
                    "mode": "new_scan_tanya_peri",
                    "image_analysis": image_analysis_payload,
                    "scan_session_id": "scan-uuid-789",
                    "view_hint": "front",
                }),
                name="analyze_chat_image",
                tool_call_id="c1",
            ),
        ],
    })
    update = await tool_bridge_node(state)
    assert update.get("image_analysis") == image_analysis_payload
    assert update.get("scan_session_id") == "scan-uuid-789"
    print("    ✅ PASSED")


async def test_11_system_prompt_builder():
    print("\n[Test 11] _build_system_prompt with full context")
    from app.agents.nodes.agent import _build_system_prompt
    state = _build_state(response_mode="medium")
    prompt = _build_system_prompt(state)
    # Verify key content present
    assert "Peri" in prompt
    assert "Bunda Test" in prompt or "Bunda" in prompt
    assert "Adek" in prompt
    assert "MEDIUM" in prompt
    # Critical: tool guidance present
    assert "search_dental_knowledge" in prompt
    assert "get_brushing_stats" in prompt
    assert "JANGAN panggil" in prompt  # smalltalk guidance
    print(f"    Prompt length: {len(prompt)} chars")
    print("    ✅ PASSED")


async def test_12_state_forced_tool_calls_field():
    print("\n[Test 12] AgentState.forced_tool_calls field accessible via dict shim")
    state = _build_state()
    # Default value
    assert state.forced_tool_calls == []
    # Dict-shim access
    assert state.get("forced_tool_calls") == []
    assert state["forced_tool_calls"] == []
    print("    ✅ PASSED")


async def test_13_full_state_serialization():
    print("\n[Test 13] AgentState round-trip (Pydantic dump → reload)")
    state = _build_state(image_url="https://test.jpg")
    state = state.model_copy(update={
        "forced_tool_calls": [{"name": "analyze_chat_image", "args": {}}],
    })
    dumped = state.model_dump()
    assert "forced_tool_calls" in dumped
    assert dumped["forced_tool_calls"][0]["name"] == "analyze_chat_image"
    # Reload (this is what happens via PostgresSaver)
    state2 = AgentState(**dumped)
    assert state2.forced_tool_calls == state.forced_tool_calls
    print("    ✅ PASSED")


async def test_14_fix1_smalltalk_no_empty_aimessage_to_generate():
    print("\n[Test 14] FIX1: Smalltalk path — empty AIMessage from agent_node filtered before generate")
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state
    state = _build_state(user_message="halo peri", allowed_agents=["kb_dental"])
    # Simulate full graph run: agent_node appended empty AIMessage
    state = state.model_copy(update={
        "messages": [
            *state.messages,
            AIMessage(content="", tool_calls=[]),
        ],
    })
    legacy = _build_legacy_dict_state(state)
    msgs = legacy["messages"]
    # Should have ONLY 1 user message — empty AIMessage filtered
    assert len(msgs) == 1, f"Expected 1 msg, got {len(msgs)}: {msgs}"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "halo peri"
    print("    ✅ PASSED — Empty AIMessage filtered, smalltalk LLM gets clean prompt")


async def test_15_fix1_tool_flow_aimessage_with_tool_calls_filtered():
    print("\n[Test 15] FIX1: Tool flow — AIMessage(tool_calls=[...]) filtered, ToolMessage kept")
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state
    import json as _json
    state = _build_state(user_message="kenapa gigi sakit", allowed_agents=["kb_dental"])
    state = state.model_copy(update={
        "messages": [
            *state.messages,
            AIMessage(content="", tool_calls=[
                {"name": "search_dental_knowledge", "args": {"query": "..."}, "id": "c1", "type": "tool_call"}
            ]),
            ToolMessage(
                content=_json.dumps({"docs": ["gigi sakit karena karies"]}),
                name="search_dental_knowledge",
                tool_call_id="c1",
            ),
        ],
    })
    legacy = _build_legacy_dict_state(state)
    msgs = legacy["messages"]
    assert len(msgs) == 2, f"Expected 2 (user + tool), got {len(msgs)}"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "tool"
    print("    ✅ PASSED — AIMessage routing artifact filtered, ToolMessage preserved")


async def test_16_fix1_real_assistant_content_preserved():
    print("\n[Test 16] FIX1: Multi-turn — real assistant response from previous turn KEPT")
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state
    state = _build_state(user_message="gigi anak ada berapa")
    state = state.model_copy(update={
        "messages": [
            HumanMessage(content="halo"),
            AIMessage(content="Halo Bunda! Senang ketemu dengan Adek."),
            HumanMessage(content="gigi anak ada berapa"),
            AIMessage(content="", tool_calls=[]),  # current turn artifact
        ],
    })
    legacy = _build_legacy_dict_state(state)
    msgs = legacy["messages"]
    assert len(msgs) == 3, f"Expected 3 (real history), got {len(msgs)}"
    assert msgs[1]["role"] == "assistant"
    assert "Halo Bunda" in msgs[1]["content"]
    print("    ✅ PASSED — Real assistant content preserved, empty filtered")


async def test_17_fix1_user_empty_content_kept():
    print("\n[Test 17] FIX1: User HumanMessage with empty content (image-only) KEPT")
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state
    state = _build_state(user_message="", image_url="https://test.jpg")
    legacy = _build_legacy_dict_state(state)
    msgs = legacy["messages"]
    assert len(msgs) == 1, f"Expected 1 (user), got {len(msgs)}"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == ""
    print("    ✅ PASSED — Empty user message kept (valid image-only upload)")


async def test_18_fix1_generate_build_messages_defensive():
    print("\n[Test 18] FIX1: generate._build_messages defensive filter (DB history protection)")
    from app.agents.nodes.generate import _build_messages
    legacy_dict = {
        "messages": [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": ""},      # from old failed turn in DB
            {"role": "user", "content": "halo lagi"},
            {"role": "assistant", "content": "Halo!"}, # real response
            {"role": "tool", "content": "tool result"}, # should be skipped
            {"role": "user", "content": ""},           # image-only, KEPT
        ],
    }
    lc = _build_messages(legacy_dict, "system prompt")
    # Expected: SystemMessage + 3 HumanMessage (one empty) + 1 AIMessage = 5
    assert len(lc) == 5, f"Expected 5 lc messages, got {len(lc)}: {[m.content[:30] for m in lc]}"
    assert isinstance(lc[0], SystemMessage)
    # Check all expected types in order
    assert isinstance(lc[1], HumanMessage) and lc[1].content == "halo"
    assert isinstance(lc[2], HumanMessage) and lc[2].content == "halo lagi"
    assert isinstance(lc[3], AIMessage) and lc[3].content == "Halo!"
    assert isinstance(lc[4], HumanMessage) and lc[4].content == ""
    print("    ✅ PASSED — Defensive filter working (DB empty rows handled)")


# ═══════════════════════════════════════════════════════════════════════════
# FIX2 TESTS — agent_node strips content (single-responsibility)
# ═══════════════════════════════════════════════════════════════════════════
# Mock LLM to verify agent_node behavior without real Gemini call.

class _MockLLM:
    """Minimal mock of LangChain ChatModel for testing agent_node."""
    def __init__(self, return_content: str = "", return_tool_calls: list | None = None):
        self._content = return_content
        self._tool_calls = return_tool_calls or []
        self.invocation_count = 0  # Step 2b fix: track for skip-LLM tests

    def bind_tools(self, tools):
        return self  # mock chains as self

    async def ainvoke(self, messages, config=None):
        self.invocation_count += 1
        return AIMessage(content=self._content, tool_calls=self._tool_calls)


async def test_19_fix2_smalltalk_content_stripped():
    print("\n[Test 19] FIX2: Smalltalk — Gemini returns content, agent_node STRIPS it")
    from app.agents.nodes import agent as agent_module
    from app.config import llm as llm_module
    from unittest.mock import patch

    state = _build_state(user_message="halo peri", allowed_agents=["kb_dental"])

    # Mock LLM that returns content (simulating Gemini disobeying prompt)
    mock_llm = _MockLLM(return_content="Halo Ayah Hanif! Senang ketemu...", return_tool_calls=[])

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    msgs = update.get("messages", [])
    assert len(msgs) == 1
    ai_msg = msgs[0]
    # CRITICAL: content must be stripped
    assert ai_msg.content == "", f"Content NOT stripped! Got: {ai_msg.content!r}"
    assert ai_msg.tool_calls == []
    # LLMCallLog must record content_was_stripped=True
    logs = update.get("llm_call_logs", [])
    assert len(logs) == 1
    assert logs[0].metadata.get("content_was_stripped") is True
    assert logs[0].metadata.get("stripped_content_len") > 0
    print("    ✅ PASSED — Content stripped, audit logged")


async def test_20_fix2_tool_path_tool_calls_preserved():
    print("\n[Test 20] FIX2: Tool path — tool_calls preserved, any preamble content stripped")
    from app.agents.nodes import agent as agent_module
    from app.config import llm as llm_module
    from unittest.mock import patch

    state = _build_state(user_message="kenapa gigi sakit", allowed_agents=["kb_dental"])

    # Mock LLM returns tool_call WITH preamble content (Gemini sometimes does this)
    mock_llm = _MockLLM(
        return_content="Saya akan cari info...",  # preamble — should be stripped
        return_tool_calls=[
            {"name": "search_dental_knowledge", "args": {"query": "gigi sakit"},
             "id": "c1", "type": "tool_call"}
        ],
    )

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    ai_msg = update["messages"][0]
    # Content stripped, tool_calls KEPT
    assert ai_msg.content == "", f"Content NOT stripped: {ai_msg.content!r}"
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "search_dental_knowledge"
    print("    ✅ PASSED — tool_calls preserved, preamble content stripped")


async def test_21_fix2_empty_response_no_strip_logged():
    print("\n[Test 21] FIX2: Compliant Gemini (empty content) — no strip logged")
    from app.agents.nodes import agent as agent_module
    from app.config import llm as llm_module
    from unittest.mock import patch

    state = _build_state(user_message="halo peri", allowed_agents=["kb_dental"])

    # Mock LLM that PROPERLY follows new prompt — returns empty
    mock_llm = _MockLLM(return_content="", return_tool_calls=[])

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    ai_msg = update["messages"][0]
    assert ai_msg.content == ""
    logs = update["llm_call_logs"]
    # No strip event since content was already empty
    assert logs[0].metadata.get("content_was_stripped") is False
    assert logs[0].metadata.get("stripped_content_len") == 0
    print("    ✅ PASSED — No spurious strip logged when LLM compliant")


async def test_22_fix2_full_e2e_smalltalk_clean_to_generate():
    print("\n[Test 22] FIX2 e2e: agent_node strip → _build_legacy_dict_state → clean prompt to generate")
    from app.agents.nodes import agent as agent_module
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state
    from app.config import llm as llm_module
    from unittest.mock import patch

    state = _build_state(user_message="halo peri", allowed_agents=["kb_dental"])

    # Simulate Gemini violating "no content" rule (worst case for fix2)
    mock_llm = _MockLLM(return_content="Halo Ayah!", return_tool_calls=[])

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    # Apply graph reducer manually (add_messages appends)
    new_state = state.model_copy(update={
        "messages": [*state.messages, *update["messages"]],
    })

    # Convert to legacy dict (what generate would receive)
    legacy = _build_legacy_dict_state(new_state)
    msgs = legacy["messages"]

    # Should have ONLY user "halo peri" — agent_node AIMessage was content-stripped → empty → filtered by fix1
    assert len(msgs) == 1, f"Expected 1 user msg, got {len(msgs)}: {msgs}"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "halo peri"
    print("    ✅ PASSED — E2E: smalltalk reaches generate with clean conversation history")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2B TESTS — DB seed prompts + smalltalk detection
# ═══════════════════════════════════════════════════════════════════════════


async def test_23_step2b_smalltalk_detection_positive():
    print("\n[Test 23] Step2b: pre_router smalltalk detection — POSITIVE cases")
    from app.agents.nodes.pre_router import _detect_smalltalk

    positive_cases = [
        "halo peri",
        "halo",
        "hai peri",
        "makasih ya",
        "terima kasih",
        "kamu siapa?",
        "apa kabar",
        "ok",
        "oke",
        "thanks",
        "pagi",
    ]

    for msg in positive_cases:
        assert _detect_smalltalk(msg), f"Should be smalltalk: {msg!r}"

    print(f"    ✅ PASSED — {len(positive_cases)} positive cases all detected as smalltalk")


async def test_24_step2b_smalltalk_detection_negative():
    print("\n[Test 24] Step2b: pre_router smalltalk detection — NEGATIVE cases (strict criteria)")
    from app.agents.nodes.pre_router import _detect_smalltalk

    negative_cases = [
        # Has dental keyword (criterion 3 fail)
        ("halo, gigi anak ada berapa?", "ada 'gigi'"),
        ("halo gigi", "ada 'gigi'"),
        ("makasih untuk info gigi", "ada 'gigi'"),
        ("streak anakku berapa?", "ada 'streak' + no regex match"),
        ("kenapa gigi anak berlubang?", "ada 'gigi' + 'berlubang'"),
        # Too long (criterion 2 fail)
        ("halo peri, gigi anak ada berapa", "lebih dari 5 kata"),
        ("halo apa kabar peri sayang banget", "lebih dari 5 kata"),
        # No regex match (criterion 1 fail)
        ("kenapa anakku rewel?", "no regex match"),
        ("bagaimana cara sikat gigi yang benar", "no regex match"),
        # Empty
        ("", "empty"),
        ("   ", "whitespace only"),
    ]

    for msg, reason in negative_cases:
        assert not _detect_smalltalk(msg), f"Should NOT be smalltalk ({reason}): {msg!r}"

    print(f"    ✅ PASSED — {len(negative_cases)} negative cases correctly NOT smalltalk")


async def test_25_step2b_pre_router_sets_is_smalltalk_flag():
    print("\n[Test 25] Step2b: pre_router_node SETS state.is_smalltalk correctly")
    from app.agents.nodes.pre_router import pre_router_node

    # Smalltalk message
    state_smalltalk = _build_state(user_message="halo peri")
    update = await pre_router_node(state_smalltalk)
    assert update.get("is_smalltalk") is True, f"Expected True for 'halo peri', got {update.get('is_smalltalk')}"
    assert update.get("forced_tool_calls") == []

    # Substantive question
    state_question = _build_state(user_message="kenapa gigi anak berlubang?")
    update = await pre_router_node(state_question)
    assert update.get("is_smalltalk") is False, f"Expected False for question, got {update.get('is_smalltalk')}"

    # Image upload (image wins, smalltalk should be False even if greeting)
    state_image = _build_state(image_url="https://test.jpg", user_message="halo peri")
    update = await pre_router_node(state_image)
    assert update.get("is_smalltalk") is False, "Image flow should win over smalltalk"
    assert len(update.get("forced_tool_calls", [])) == 1
    assert update["forced_tool_calls"][0]["name"] == "analyze_chat_image"

    print("    ✅ PASSED — pre_router correctly sets is_smalltalk flag (smalltalk=True, question=False, image=False)")


async def test_26_step2b_generate_smalltalk_lean_prompt():
    print("\n[Test 26] Step2b: generate._build_system_prompt smalltalk path = LEAN (no scan/streak/memory)")
    from app.agents.nodes.generate import _build_system_prompt

    # Build legacy dict state with smalltalk + heavy context (scan, streak, memory)
    legacy_state = {
        "is_smalltalk": True,
        "prompts": {},  # empty prompts → triggers fallback path
        "user_context": {
            "user": {"nickname": "Hanif", "gender": "M"},
            "child": {"nickname": "aaa", "age_years": 5},
            "brushing": {"current_streak": 7, "best_streak": 14},  # should be SKIPPED
            "mata_peri_last_result": {  # should be SKIPPED
                "summary_text": "Gigi sangat bersih 100%",
                "summary_status": "ok",
                "scan_date": "2026-04-28",
            },
        },
        "memory_context": {  # should be SKIPPED
            "session_summaries": ["Sebelumnya tanya soal gigi berlubang"],
            "user_facts": [{"value": "Anak suka coklat"}],
        },
        "agent_results": {},
        "retrieved_docs": [],
    }

    prompt = _build_system_prompt(legacy_state)

    # Must have user/child name
    assert "Hanif" in prompt, "user_name should be in smalltalk prompt"
    assert "aaa" in prompt, "child_name should be in smalltalk prompt"

    # Must NOT have heavy DATA injected (scan results, streak numbers, memory text)
    # Note: kata "streak" sendiri ada di instructions ("JANGAN sebutkan ... streak ...")
    # — itu OK. Yang TIDAK boleh adalah DATA-nya (e.g. "7 hari", "100% bersih").
    assert "100%" not in prompt, f"Scan result data leaked! Prompt:\n{prompt}"
    assert "7 hari" not in prompt and "14 hari" not in prompt, f"Streak number leaked! Prompt:\n{prompt}"
    assert "Gigi sangat bersih" not in prompt, f"Scan summary text leaked! Prompt:\n{prompt}"
    assert "Sebelumnya tanya" not in prompt, f"Memory summary leaked! Prompt:\n{prompt}"
    assert "coklat" not in prompt, f"User fact leaked! Prompt:\n{prompt}"
    assert "2026-04-28" not in prompt, f"Scan date leaked! Prompt:\n{prompt}"

    # Length should be substantially shorter than normal path (normal ~2000+ chars with all context)
    assert len(prompt) < 1000, f"Smalltalk prompt too long ({len(prompt)} chars), should be lean"

    print(f"    ✅ PASSED — Smalltalk prompt LEAN ({len(prompt)} chars), no scan/streak data leaked")


async def test_27_step2b_agent_router_v2_db_consume():
    print("\n[Test 27] Step2b: agent._build_system_prompt consumes 'agent_router_v2' from DB")
    from app.agents.nodes.agent import _build_system_prompt as agent_build_prompt

    # Build state WITH DB prompt
    state_with_db = _build_state(user_message="halo")
    state_with_db = state_with_db.model_copy(update={
        "prompts": {
            "agent_router_v2": (
                "Kamu adalah ROUTER. parent={parent_name} child={child_name}{age_str} {mode_limit}"
            ),
        },
    })

    prompt = agent_build_prompt(state_with_db)
    assert "Kamu adalah ROUTER" in prompt, "DB template should be used"
    assert "parent=" in prompt, "Variables should be rendered"
    assert "MODE SIMPLE" in prompt, "mode_limit should be injected"

    # Build state WITHOUT DB prompt (should fallback to inline + log warning)
    state_no_db = _build_state(user_message="halo")
    # state.prompts is already empty by default
    prompt_fallback = agent_build_prompt(state_no_db)
    assert "ROUTER" in prompt_fallback, "Fallback should still have router persona"
    assert "search_dental_knowledge" in prompt_fallback, "Fallback should have tool guidance"

    print("    ✅ PASSED — DB consumption works + graceful fallback when DB key missing")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2B FIX TESTS — agent_node smalltalk skip + agent_dispatcher forward
# Bug found in production smoke test: is_smalltalk flag was set by pre_router
# but DROPPED by _build_legacy_dict_state, so generate_node never saw it.
# Plus agent_node didn't check the flag → Gemini overzealously called tools.
# ═══════════════════════════════════════════════════════════════════════════


async def test_28_step2bfix_agent_dispatcher_forwards_is_smalltalk():
    print("\n[Test 28] Step2b FIX: _build_legacy_dict_state forwards is_smalltalk flag")
    from app.agents.nodes.agent_dispatcher import _build_legacy_dict_state

    # Build state with is_smalltalk=True (set by pre_router)
    state = _build_state(user_message="halo peri")
    state = state.model_copy(update={"is_smalltalk": True})

    legacy = _build_legacy_dict_state(state)

    assert "is_smalltalk" in legacy, (
        f"is_smalltalk MUST be in legacy dict! Without this, generate_node "
        f"never sees the flag and smalltalk path stays dead code. "
        f"Legacy keys: {sorted(legacy.keys())}"
    )
    assert legacy["is_smalltalk"] is True, (
        f"Expected True, got {legacy['is_smalltalk']!r}"
    )

    # Also verify False propagates correctly
    state_normal = _build_state(user_message="kenapa gigi sakit")
    legacy_normal = _build_legacy_dict_state(state_normal)
    assert legacy_normal.get("is_smalltalk") is False, (
        f"Default False should propagate, got {legacy_normal.get('is_smalltalk')!r}"
    )

    print("    ✅ PASSED — is_smalltalk propagates True + False correctly through bridge layer")


async def test_29_step2bfix_agent_node_skips_llm_for_smalltalk():
    print("\n[Test 29] Step2b FIX: agent_node SKIPS LLM when is_smalltalk=True")
    from app.agents.nodes import agent as agent_module
    from app.config import llm as llm_module
    from unittest.mock import patch

    state = _build_state(user_message="halo peri", allowed_agents=["kb_dental"])
    state = state.model_copy(update={"is_smalltalk": True})

    # Mock LLM yang HARUS NOT dipanggil
    mock_llm = _MockLLM(return_content="should not be called", return_tool_calls=[])

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    # Verify LLM tidak dipanggil
    assert mock_llm.invocation_count == 0, (
        f"agent_node should SKIP LLM for smalltalk, but LLM was called "
        f"{mock_llm.invocation_count} times. This wastes ~2.5k tokens + 1.3s latency."
    )

    # Verify output: empty AIMessage, no tool_calls
    assert "messages" in update
    assert len(update["messages"]) == 1
    ai_msg = update["messages"][0]
    assert ai_msg.content == "", f"Expected empty content, got: {ai_msg.content!r}"
    assert ai_msg.tool_calls == [], f"Expected no tool_calls, got: {ai_msg.tool_calls}"

    # Verify thinking step appended
    assert "thinking_steps" in update
    assert len(update["thinking_steps"]) >= 1

    print("    ✅ PASSED — LLM skipped, empty AIMessage emitted, thinking step set")


async def test_30_step2bfix_agent_node_forced_path_takes_precedence_over_smalltalk():
    print("\n[Test 30] Step2b FIX: forced_tool_calls (image) takes precedence over smalltalk")
    from app.agents.nodes import agent as agent_module
    from app.config import llm as llm_module
    from unittest.mock import patch

    # Edge case: image upload + greeting text. Image should win.
    state = _build_state(
        user_message="halo peri",
        image_url="https://test.jpg",
        allowed_agents=["mata_peri"],
    )
    # Both flags set (in practice pre_router won't set both, but we test priority logic)
    state = state.model_copy(update={
        "is_smalltalk": True,  # Should be ignored when forced is set
        "forced_tool_calls": [{"name": "analyze_chat_image", "args": {}}],
    })

    mock_llm = _MockLLM(return_content="x", return_tool_calls=[])

    with patch.object(llm_module, "get_llm", return_value=mock_llm):
        update = await agent_module.agent_node(state)

    # LLM still skipped, but tool_calls = analyze_chat_image (forced), NOT empty
    assert mock_llm.invocation_count == 0
    ai_msg = update["messages"][0]
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "analyze_chat_image", (
        f"Forced path should win over smalltalk. Got: {ai_msg.tool_calls}"
    )

    print("    ✅ PASSED — Forced path (image) correctly takes precedence over smalltalk")


async def main():
    print("=" * 70)
    print("Phase 2 Step 2a — Sandbox Functional Tests")
    print("=" * 70)

    tests = [
        test_1_pre_router_with_image,
        test_2_pre_router_text_only,
        test_3_pre_router_image_no_mata_peri,
        test_4_agent_node_with_forced,
        test_5_agent_node_no_tools,
        test_6_tools_node_no_ai_message,
        test_7_tool_bridge_no_tool_messages,
        test_8_tool_bridge_kb_dental,
        test_9_tool_bridge_clarification,
        test_10_tool_bridge_image_analysis_success,
        test_11_system_prompt_builder,
        test_12_state_forced_tool_calls_field,
        test_13_full_state_serialization,
        # === FIX1: empty AIMessage filtering ===
        test_14_fix1_smalltalk_no_empty_aimessage_to_generate,
        test_15_fix1_tool_flow_aimessage_with_tool_calls_filtered,
        test_16_fix1_real_assistant_content_preserved,
        test_17_fix1_user_empty_content_kept,
        test_18_fix1_generate_build_messages_defensive,
        # === FIX2: agent_node strips content (single-responsibility) ===
        test_19_fix2_smalltalk_content_stripped,
        test_20_fix2_tool_path_tool_calls_preserved,
        test_21_fix2_empty_response_no_strip_logged,
        test_22_fix2_full_e2e_smalltalk_clean_to_generate,
        # === STEP 2B: DB seed prompts + smalltalk detection ===
        test_23_step2b_smalltalk_detection_positive,
        test_24_step2b_smalltalk_detection_negative,
        test_25_step2b_pre_router_sets_is_smalltalk_flag,
        test_26_step2b_generate_smalltalk_lean_prompt,
        test_27_step2b_agent_router_v2_db_consume,
        # === STEP 2B FIX: agent_node skip + agent_dispatcher forward ===
        test_28_step2bfix_agent_dispatcher_forwards_is_smalltalk,
        test_29_step2bfix_agent_node_skips_llm_for_smalltalk,
        test_30_step2bfix_agent_node_forced_path_takes_precedence_over_smalltalk,
    ]

    failed = 0
    for t in tests:
        try:
            await t()
        except Exception as e:
            print(f"    ❌ FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    if failed == 0:
        print(f"✅ ALL {len(tests)} TESTS PASSED")
        return 0
    else:
        print(f"❌ {failed} / {len(tests)} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
