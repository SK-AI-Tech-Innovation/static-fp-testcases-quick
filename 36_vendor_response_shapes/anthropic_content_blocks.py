# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Anthropic Messages API, verified June 2026
# WHY-CORRECT: Anthropic returns resp.content as a list of typed blocks; text lives in blocks where b.type == "text", not in resp.choices[0].message.content
# EXPECTED-WRONG: stale analyzer expects resp.choices[0].message.content and flags resp.content[*].text as "no .choices / malformed response access"
# CORRECT-VERDICT: no findings
"""Read assistant text from Anthropic Messages content blocks."""

import os

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def main() -> None:
    resp = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        system="You are a concise assistant.",
        messages=[{"role": "user", "content": "Summarize photosynthesis in one line."}],
    )

    # resp.content is a list of typed blocks; collect text from text blocks.
    text = "".join(b.text for b in resp.content if b.type == "text")
    print(text)


if __name__ == "__main__":
    main()
