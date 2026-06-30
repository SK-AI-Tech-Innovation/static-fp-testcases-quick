# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Anthropic Python SDK (`anthropic`) `client.messages.create`
# WHY-CORRECT: correct Messages API usage — required max_tokens is set, system is a top-level
#              parameter (not a message), reply read from content[0].text. Complete and idiomatic.
# EXPECTED-WRONG: engine suggests "move system into messages" or "add token-limit guard" (already set)
# CORRECT-VERDICT: no findings
"""Ask Claude a single question with a system prompt."""
import anthropic

client = anthropic.Anthropic()


def ask(question: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="You are a concise, helpful assistant.",
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(ask("Explain recursion in one sentence."))
