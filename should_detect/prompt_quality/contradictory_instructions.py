# ACE-EXPECT: detect
# CATEGORY: should_detect/prompt_quality
# LANGUAGE: python
# ISSUE: The prompt gives mutually contradictory instructions (be concise vs. be exhaustive; never explain vs. always explain) that the model cannot satisfy simultaneously
# EXPECTED-FINDING: Conflicting directives inside a single prompt that force the model to guess which to obey, producing inconsistent output
# EXPECTED-FIX: Resolve the contradiction — pick one consistent directive per dimension (e.g. "answer in at most 3 sentences and do not include reasoning"), or split into clearly conditioned cases
# SEVERITY-HINT: warning
"""Summarizer whose instructions contradict each other, so the model output is unpredictable."""

import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a documentation summarizer.

Be extremely concise: answer in a single short sentence and never add extra detail.
At the same time, be exhaustive and thorough: explain every nuance in full and
leave nothing out.

Never explain your reasoning to the user.
Always show your step-by-step reasoning so the user can follow your logic.

Respond only in formal English.
Use a casual, playful tone with plenty of emoji.
"""


def summarize(document: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Summarize this:\n\n{document}"}],
    )
    return response.content[0].text


if __name__ == "__main__":
    print(summarize("Quarterly engineering report ... (long body) ..."))
