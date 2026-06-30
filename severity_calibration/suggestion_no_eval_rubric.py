# ACE-EXPECT: detect
# CATEGORY: should_detect/test_eval_rubric
# LANGUAGE: python
# ISSUE: The test that "evaluates" the model's summary only asserts the output is non-empty, with no rubric or quality criteria — a weak-eval nicety, never a correctness or security risk.
# EXPECTED-FINDING: LLM-output eval asserts only non-emptiness; there is no rubric scoring relevance/faithfulness/format.
# EXPECTED-FIX: Define rubric criteria (e.g. faithfulness, coverage, format) and grade against them — keyword checks, an LLM judge, or scored assertions.
# SEVERITY-HINT: suggestion
"""pytest 'evaluation' of a summarization prompt that checks nothing about quality."""

import os

import pytest
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ARTICLE = (
    "The city council approved a new bike-lane network on Tuesday, allocating "
    "$3.2 million over two years. Construction begins next spring."
)


def summarize(text: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        system="Summarize the article in one sentence.",
        messages=[{"role": "user", "content": text}],
    )
    return next(b.text for b in response.content if b.type == "text")


@pytest.mark.eval
def test_summary_quality() -> None:
    """Supposedly an eval — but it only confirms the model returned some text."""
    summary = summarize(ARTICLE)
    # No rubric: nothing checks the budget figure, the timing, or factual faithfulness.
    assert summary
    assert summary.strip() != ""
