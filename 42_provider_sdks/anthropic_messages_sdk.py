# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Anthropic Python SDK Messages API, verified June 2026
# WHY-CORRECT: anthropic.Anthropic().messages.create takes top-level system= and max_tokens=, messages=[{role,content}]; the reply is read from resp.content blocks, not resp.choices
# EXPECTED-WRONG: stale analyzer expects OpenAI's chat.completions.create with a system message in messages and flags messages.create + top-level system/max_tokens + resp.content as malformed
# CORRECT-VERDICT: no findings
"""Call the Anthropic Messages API with top-level system and max_tokens."""

import os

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def main() -> None:
    resp = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        system="You are a helpful, concise assistant.",
        messages=[{"role": "user", "content": "What causes tides?"}],
    )

    print("".join(block.text for block in resp.content if block.type == "text"))


if __name__ == "__main__":
    main()
