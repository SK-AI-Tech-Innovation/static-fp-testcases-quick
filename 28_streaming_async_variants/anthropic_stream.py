# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: anthropic-sdk-python v0.x (Messages streaming helper)
# WHY-CORRECT: `with client.messages.stream(...) as stream:` plus `for text in stream.text_stream` is the documented Anthropic streaming helper; get_final_message() returns the assembled Message.
# EXPECTED-WRONG: engine may claim messages.stream does not exist, that max_tokens is missing (it is present), or that text_stream should be awaited.
# CORRECT-VERDICT: no findings
"""Streaming a Claude response via the Anthropic messages.stream helper."""

import anthropic

client = anthropic.Anthropic()


def stream_claude(prompt: str) -> str:
    chunks = []
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            print(text, end="", flush=True)

        final_message = stream.get_final_message()

    print()
    print("stop_reason:", final_message.stop_reason)
    return "".join(chunks)


if __name__ == "__main__":
    stream_claude("Explain backpropagation in two sentences.")
