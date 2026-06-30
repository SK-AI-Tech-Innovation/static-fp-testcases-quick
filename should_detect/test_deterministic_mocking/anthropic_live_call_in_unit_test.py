# ACE-EXPECT: detect
# CATEGORY: should_detect/test_deterministic_mocking
# LANGUAGE: python
# ISSUE: Unit test calls the real Anthropic API rather than a mocked client
# EXPECTED-FINDING: Live LLM call introduces network dependency, cost, and non-deterministic results into CI
# EXPECTED-FIX: Replace Anthropic() with a mock whose messages.create returns a fixed content payload
# SEVERITY-HINT: warning
"""Unit test for a translation helper that contacts Claude over the network."""

from anthropic import Anthropic


def translate_to_french(text: str) -> str:
    client = Anthropic()
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=128,
        messages=[{"role": "user", "content": f"Translate to French: {text}"}],
    )
    return resp.content[0].text.strip()


def test_translate_hello():
    out = translate_to_french("Hello")
    assert isinstance(out, str)
    assert out != ""
