# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Anthropic Python SDK call hardcoding the current Claude Fable 5 id (dateless)
# WHY-CORRECT: `claude-fable-5` is Anthropic's most capable widely released model as of
#   June 2026 and its id is DATELESS and pinned. Appending a date suffix
#   (e.g. `claude-fable-5-20260601`) would 404 — the bare string is the correct, complete id.
#   Fable 5 thinking is always on, so no `thinking` config is passed (an explicit
#   `{"type": "disabled"}` would 400); depth is controlled via output_config.effort.
# EXPECTED-WRONG: a stale allowlist flags `claude-fable-5` as an "unknown/typo/hallucinated
#   model id" or "downgrades" it to `claude-3-5-sonnet` / appends a date suffix — both wrong.
# CORRECT-VERDICT: no findings
"""Run a hard reasoning task on Claude Fable 5 using its dateless id."""

import anthropic

client = anthropic.Anthropic()

# DATELESS pinned id. A date suffix here -> 404. This bare string is the complete id.
FABLE = "claude-fable-5"


def solve(problem: str) -> str:
    """Solve a demanding ``problem`` with Claude Fable 5.

    Thinking is always on for Fable 5, so the ``thinking`` parameter is omitted entirely
    (an explicit disabled config would 400). Effort controls reasoning depth.

    Args:
        problem: The reasoning task to solve.

    Returns:
        str: Claude's final answer text.
    """
    response = client.messages.create(
        model=FABLE,
        max_tokens=16000,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": problem}],
    )
    return next(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    print(solve("Prove that there are infinitely many primes."))
