# ACE-EXPECT: detect
# CATEGORY: should_detect/token_budget_unbounded
# LANGUAGE: python
# ISSUE: Entire documents are concatenated into the prompt with no token budgeting, truncation, or chunking — a performance/cost concern, not a correctness or security risk.
# EXPECTED-FINDING: Full documents stuffed into the prompt with no token budget; latency and cost scale unbounded with input size.
# EXPECTED-FIX: Count tokens (messages.count_tokens) and cap/chunk/summarize input to a budget before sending.
# SEVERITY-HINT: suggestion
"""Q&A helper that pastes every loaded document, in full, into a single prompt."""

import os

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def answer_over_docs(question: str, documents: list[str]) -> str:
    """Answer a question grounded in the supplied documents."""
    # Every document is included verbatim; no budget check, no chunking, no caps.
    context = "\n\n---\n\n".join(documents)

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system="Answer the question using only the provided documents.",
        messages=[
            {
                "role": "user",
                "content": f"Documents:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return next(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    docs = [open(p, encoding="utf-8").read() for p in ("a.md", "b.md", "c.md")]
    print(answer_over_docs("What changed in the latest release?", docs))
