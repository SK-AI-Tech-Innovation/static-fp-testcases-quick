# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: deepseek-reasoner (legacy DeepSeek-R1) + openai-python, multi-turn history
# WHY-CORRECT: legacy DeepSeek-R1 docs require STRIPPING reasoning_content before appending the assistant turn to
#              history; re-sending past CoT degrades quality and R1 explicitly does not accept it as input. Dropping
#              it here is the documented, correct behavior (the OPPOSITE rule from DeepSeek V3.2 thinking).
# EXPECTED-WRONG: engine flags discarding message.reasoning_content as "dropping model output is data loss", not
#                 recognizing that R1 mandates removal of prior reasoning from conversation history.
# CORRECT-VERDICT: no findings
"""Strip reasoning_content before appending the R1 assistant turn to history (correct for R1)."""

from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="${DEEPSEEK_API_KEY}",
)


def chat_round(history: list[dict], user_text: str) -> tuple[list[dict], str]:
    """Append one R1 turn to history WITHOUT its reasoning_content.

    Args:
        history: Prior messages (already free of reasoning_content).
        user_text: The new user message.

    Returns:
        Updated history and the final answer text.
    """
    history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=history,
    )
    msg = response.choices[0].message

    # Legacy R1: do NOT feed reasoning_content back into history. Keep only the
    # final content; re-sending CoT degrades quality and is rejected as input.
    history.append({"role": "assistant", "content": msg.content})
    return history, msg.content


if __name__ == "__main__":
    convo: list[dict] = []
    convo, answer = chat_round(convo, "Summarize why the sky is blue.")
    convo, answer = chat_round(convo, "Now explain it to a five-year-old.")
    print(answer)
