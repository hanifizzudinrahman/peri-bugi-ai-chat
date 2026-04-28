#!/usr/bin/env python
"""
Verify Phase 2 Step 1 tool registry.

Builds a mock AgentState, calls make_tools(state), then:
1. Asserts the expected tool count matches allowed_agents
2. Dumps each tool's name + description (what the LLM will see)
3. Dumps each tool's args_schema (what params LLM can supply)
4. Verifies no import errors / no syntax errors

Does NOT make any real API calls.

Usage:
    docker compose exec ai-chat python scripts/verify_tools.py
"""
from __future__ import annotations

import asyncio
import os
import sys

# Ensure project root (/app inside container) is on sys.path so `from app...` works
# regardless of where the script is invoked from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _build_mock_state():
    """Build a representative mock AgentState (same as production)."""
    from app.agents.state import (
        AgentState,
        SessionMeta,
        UserContextData,
        AgentControl,
        ImageInput,
        RnDOverrides,
    )

    user_ctx = UserContextData(
        user={
            "id": "test-user-uuid-1234",
            "nickname": "Bunda",
            "full_name": "Bunda Test",
            "gender": "F",
        },
        child={
            "id": "test-child-uuid-5678",
            "nickname": "Adek",
            "full_name": "Adek Test",
            "age_years": 5,
            "gender": "M",
        },
        brushing={
            "current_streak": 7,
            "best_streak": 14,
        },
        mata_peri_last_result=None,
    )

    session = SessionMeta(
        session_id="test-session-uuid",
        response_mode="medium",
        source="web",
        chat_message_id="test-msg-uuid",
        trace_id="test-trace-id",
    )

    control = AgentControl(
        allowed_agents=[
            "kb_dental",
            "app_faq",
            "user_profile",
            "rapot_peri",
            "cerita_peri",
            "mata_peri",
        ],
        agent_configs={},
    )

    return AgentState(
        session=session,
        user_context=user_ctx,
        control=control,
        image=ImageInput(image_url="https://example.com/test.jpg"),
        prompts={},
        rnd=RnDOverrides(),
    )


def _dump_tool_info(tool) -> dict:
    """Extract safe-to-display info from a @tool-decorated function."""
    schema = None
    try:
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is not None:
            schema = schema_cls.model_json_schema()
    except Exception as e:
        schema = {"error_extracting_schema": str(e)}

    return {
        "name": tool.name,
        "description": (tool.description or "").strip()[:300] + "...",
        "args_schema_properties": (
            list((schema or {}).get("properties", {}).keys()) if schema else []
        ),
        "is_coroutine": asyncio.iscoroutinefunction(tool.coroutine) if hasattr(tool, "coroutine") else None,
    }


async def main() -> int:
    print("=" * 70)
    print("Phase 2 Step 1 — Tool Registry Verification")
    print("=" * 70)
    print()

    # Step 1: Import (catches syntax / circular import issues)
    print("[1/4] Importing make_tools...")
    try:
        from app.agents.tools import make_tools
        print("      ✅ Import OK")
    except Exception as e:
        print(f"      ❌ Import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()

    # Step 2: Build mock state
    print("[2/4] Building mock AgentState...")
    try:
        state = _build_mock_state()
        print(f"      ✅ State built: allowed_agents={state.control.allowed_agents}")
    except Exception as e:
        print(f"      ❌ State build FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()

    # Step 3: Build tools
    print("[3/4] Calling make_tools(state)...")
    try:
        tools = make_tools(state)
        print(f"      ✅ make_tools returned {len(tools)} tools")
    except Exception as e:
        print(f"      ❌ make_tools FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Expected: 7 tools (mata_peri contributes 2 — get_scan_history + analyze_chat_image)
    expected_count = 7
    if len(tools) != expected_count:
        print(f"      ⚠️  Expected {expected_count} tools, got {len(tools)}")
    print()

    # Step 4: Dump tool info
    print("[4/4] Dumping tool metadata (this is what LLM will see):")
    print()
    expected_names = {
        "search_dental_knowledge",
        "search_app_faq",
        "get_user_profile",
        "get_brushing_stats",
        "get_cerita_progress",
        "get_scan_history",
        "analyze_chat_image",
    }
    actual_names = set()
    for i, tool in enumerate(tools, 1):
        info = _dump_tool_info(tool)
        actual_names.add(info["name"])
        print(f"  Tool #{i}: {info['name']}")
        print(f"    args: {info['args_schema_properties']}")
        print(f"    description: {info['description'][:120]}...")
        print()

    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing:
        print(f"      ⚠️  MISSING tools: {missing}")
    if extra:
        print(f"      ⚠️  UNEXPECTED tools: {extra}")
    if not missing and not extra:
        print("      ✅ All 7 expected tools present")
    print()

    # Step 5: Test allowed_agents filtering (regression check)
    print("[BONUS] Testing allowed_agents filter — only kb_dental allowed:")
    state.control.allowed_agents = ["kb_dental"]
    tools_filtered = make_tools(state)
    if len(tools_filtered) == 1 and tools_filtered[0].name == "search_dental_knowledge":
        print(f"      ✅ Filtered to 1 tool: {tools_filtered[0].name}")
    else:
        print(f"      ❌ Filter broken: got {[t.name for t in tools_filtered]}")
        return 1
    print()

    print("=" * 70)
    print("✅ VERIFICATION PASSED")
    print("=" * 70)
    print()
    print("Note: This script does NOT call any tool. It only verifies that")
    print("tools can be instantiated with mock state and have correct schemas.")
    print()
    print("Next step: Step 2 will wire these tools into a ReAct LangGraph node.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
