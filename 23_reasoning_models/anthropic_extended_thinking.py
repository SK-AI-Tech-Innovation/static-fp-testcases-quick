# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: claude-sonnet-4-5 + anthropic-python (extended thinking)
# WHY-CORRECT: thinking={"type": "enabled", "budget_tokens": N} with budget_tokens < max_tokens is the correct extended-thinking idiom for Sonnet 4.5;
#              the response is iterated by block.type, reading both 'thinking' and 'text' blocks.
# EXPECTED-WRONG: engine flags the 'thinking' param or the thinking-block reading loop as unknown/unsupported, but this is the documented extended-thinking shape for this model.
# CORRECT-VERDICT: no findings
"""Extended thinking with Claude Sonnet 4.5, reading thinking and text blocks."""

import anthropic

client = anthropic.Anthropic()


def reason(problem: str) -> tuple[str, str]:
    """Return (thinking_summary, answer) from an extended-thinking request."""
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 4000},
        messages=[{"role": "user", "content": problem}],
    )

    thinking_text = ""
    answer_text = ""
    for block in message.content:
        if block.type == "thinking":
            thinking_text += block.thinking
        elif block.type == "text":
            answer_text += block.text
    return thinking_text, answer_text


if __name__ == "__main__":
    thoughts, answer = reason("How many distinct ways can 8 rooks be placed on a chessboard with none attacking another?")
    print(answer)
