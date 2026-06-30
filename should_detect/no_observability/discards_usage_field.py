# ACE-EXPECT: detect
# CATEGORY: should_detect/no_observability
# LANGUAGE: python
# ISSUE: A streaming chat call receives a final message event that carries a usage field, but the code explicitly pulls out only the text and throws the rest (including usage) away
# EXPECTED-FINDING: the usage field on the streamed final message is available but explicitly discarded; nothing records prompt/completion token counts
# EXPECTED-FIX: capture final_message.usage from the stream and record/emit the token counts (e.g. to logs or a metrics counter) instead of discarding it
# SEVERITY-HINT: warning
"""Streams a chat reply and deliberately keeps only the text, dropping usage."""

from anthropic import Anthropic

client = Anthropic()


def chat(user_message: str) -> str:
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=800,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for _ in stream.text_stream:
            pass  # we don't even forward tokens, just drain
        final = stream.get_final_message()

    # final.usage has input_tokens / output_tokens — we throw it on the floor.
    return final.content[0].text


if __name__ == "__main__":
    print(chat("Explain vector databases in two sentences."))
