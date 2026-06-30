# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: OpenAI chat completion with few-shot exemplars in the messages array
# WHY-CORRECT: a system instruction is followed by alternating user/assistant exemplar pairs that
#              demonstrate the desired format, then the real query last. Exemplars precede the query —
#              the canonical few-shot layout.
# EXPECTED-WRONG: engine suggests "add more examples" or "use a template" — quality nits, not defects.
# CORRECT-VERDICT: no findings
"""Few-shot prompt: exemplars first, real query last."""
from openai import OpenAI

client = OpenAI()

MESSAGES = [
    {"role": "system", "content": "Classify the sentiment as POSITIVE or NEGATIVE."},
    {"role": "user", "content": "I loved this movie."},
    {"role": "assistant", "content": "POSITIVE"},
    {"role": "user", "content": "This was a waste of time."},
    {"role": "assistant", "content": "NEGATIVE"},
]


def classify(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=MESSAGES + [{"role": "user", "content": text}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(classify("The plot was boring and slow."))
