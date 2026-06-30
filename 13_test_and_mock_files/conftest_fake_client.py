# ACE-FP-EXPECT: clean
# CATEGORY: 13_test_and_mock_files
# SOURCE: A pytest conftest providing a stub LLM client fixture
# WHY-CORRECT: A stub client with fixed output is the whole point of a test fixture; no real API calls, retries, or schema validation belong here
# EXPECTED-WRONG: Engine flags the stub for "no real error handling", "hardcoded model output", or suggests calling the real API — inappropriate for a fixture
# CORRECT-VERDICT: no findings
"""Shared pytest fixtures: a stub LLM client for offline, deterministic tests."""

import pytest


class StubLLMClient:
    """Minimal stub matching the real client's call surface for tests."""

    def __init__(self, canned_text="stubbed-response"):
        self.canned_text = canned_text
        self.calls = []

    def create(self, *, model, messages, **kwargs):
        # Record the call so tests can assert on inputs, then return a fixed body.
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return type("Resp", (), {"content": [type("Block", (), {"text": self.canned_text})]})


@pytest.fixture
def stub_llm():
    """Provide a fresh StubLLMClient per test."""
    return StubLLMClient()


@pytest.fixture
def stub_llm_factory():
    """Provide a factory so a test can choose the canned response text."""
    def _make(text):
        return StubLLMClient(canned_text=text)

    return _make
