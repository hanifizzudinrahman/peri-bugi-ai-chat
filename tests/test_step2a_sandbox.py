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
