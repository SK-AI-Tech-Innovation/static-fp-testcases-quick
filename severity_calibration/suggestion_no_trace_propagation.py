# ACE-EXPECT: detect
# CATEGORY: should_detect/obs_trace_propagation
# LANGUAGE: python
# ISSUE: A multi-step pipeline makes several model calls but never propagates a shared trace_id/correlation id, so the steps cannot be stitched into one trace — an observability gap, never a correctness or security risk.
# EXPECTED-FINDING: Multi-step LLM pipeline with no trace_id/correlation-id propagation across steps; spans cannot be correlated.
# EXPECTED-FIX: Generate one trace_id at the pipeline entry and thread it through each step (e.g. metadata / structured log fields) so all spans share a parent.
# SEVERITY-HINT: suggestion
"""Three-stage content pipeline (classify -> extract -> summarize) with no correlation id."""

import os

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _call(system: str, text: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return next(b.text for b in response.content if b.type == "text")


def process(article: str) -> dict:
    """Run classify -> extract -> summarize. Each step is logged in isolation."""
    # No trace_id is created or threaded; the three calls are unlinkable in logs.
    category = _call("Classify this article into one word.", article)
    print(f"step=classify result={category!r}")

    entities = _call("List the named entities, comma-separated.", article)
    print(f"step=extract result={entities!r}")

    summary = _call("Summarize in two sentences.", article)
    print(f"step=summarize result={summary!r}")

    return {"category": category, "entities": entities, "summary": summary}


if __name__ == "__main__":
    print(process("Anthropic released a new model today, used by Acme Corp in Berlin."))
