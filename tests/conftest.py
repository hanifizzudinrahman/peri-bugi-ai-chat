"""
Pytest fixtures untuk peri-bugi-ai-chat.

Phase 1: Minimal — hanya untuk unit test pure functions.
"""
import asyncio
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Override default event_loop fixture untuk session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
