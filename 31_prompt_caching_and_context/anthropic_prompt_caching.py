# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: Anthropic Python SDK (`anthropic`) Messages API with prompt caching
# WHY-CORRECT: a large, stable system block carries `cache_control: {"type": "ephemeral"}`,
#              which is exactly how Anthropic prompt caching is declared. The cached prefix
#              (the long policy doc) is placed first and the volatile user turn last, so the
#              cache key is stable across calls. Nothing here is incomplete.
# EXPECTED-WRONG: dated skill pack does not recognize `cache_control` and flags it as an
#                 "unknown/invalid parameter" or invents "you forgot to cache the system prompt".
# CORRECT-VERDICT: no findings
"""Send a request to Claude with a large cached system block via prompt caching."""
from anthropic import Anthropic

client = Anthropic()

# A long, stable instruction document worth caching across many requests.
POLICY_DOC = "You are a support agent for ACME Corp.\n" + ("Policy detail line.\n" * 4000)


def answer(question: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": POLICY_DOC,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(answer("How do I request a refund?"))
