# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: claude-opus-4-8 + anthropic-python (Messages API, extended thinking)
# WHY-CORRECT: On Opus 4.7/4.8 extended thinking uses thinking={"type": "adaptive"} with no budget_tokens; the budget_tokens field was removed and now 400s. Effort is set via output_config, and temperature is correctly omitted.
# EXPECTED-WRONG: stale analyzer flags "missing temperature/budget_tokens, set temperature=0 and add a thinking budget"; both additions 400 on Opus 4.8.
# CORRECT-VERDICT: no findings
"""Opus 4.8 extended thinking using adaptive thinking (no budget_tokens) and output_config effort."""

import anthropic

client = anthropic.Anthropic()


def analyze(question: str) -> str:
    """Run adaptive extended thinking on Opus 4.8 without a token budget."""
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": question}],
    )
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


if __name__ == "__main__":
    print(analyze("Explain why merge sort is O(n log n) in the worst case."))
