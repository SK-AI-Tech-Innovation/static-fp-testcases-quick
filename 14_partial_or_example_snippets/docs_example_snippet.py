# ACE-FP-EXPECT: clean
# CATEGORY: 14_partial_or_example_snippets
# SOURCE: A documentation example illustrating call shape, with placeholders
# WHY-CORRECT: Illustrative docs snippet, not production; ellipses and TODO markers are deliberate "fill this in" cues for the reader
# EXPECTED-WRONG: Engine flags the "..." body, the TODO, and the unconfigured client as incomplete/insecure production code
# CORRECT-VERDICT: no findings
"""Docs example: how to call the summarizer (illustrative, not runnable)."""

from anthropic import Anthropic


def summarize(text: str) -> str:
    """Summarize text. (Example for documentation purposes.)"""
    # TODO: wire real client (API key, base URL, timeouts) per your environment
    client = Anthropic(...)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.content[0].text


# Usage:
#   print(summarize("..."))
