# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x Responses API targeting a modern reasoning model (o4-mini) with an old f-string prompt style
# WHY-CORRECT: hand-built f-string prompt concatenation is still perfectly valid input; a modern reasoning model accepts plain text input and an optional reasoning effort setting. Old prompt construction + new reasoning model is fully compatible.
# EXPECTED-WRONG: engine may insist reasoning models require a special "developer" message schema or that f-string prompts are unsupported / must use a template framework.
# CORRECT-VERDICT: no findings
"""Feed an old-style f-string prompt to a modern reasoning model."""

from openai import OpenAI

client = OpenAI()

PROMPT_TEMPLATE = """You are a careful math tutor.
Solve the following problem and show your reasoning briefly.

Problem: {problem}
"""


def solve(problem: str) -> str:
    prompt = PROMPT_TEMPLATE.format(problem=problem)
    response = client.responses.create(
        model="o4-mini",
        input=prompt,
        reasoning={"effort": "medium"},
    )
    return response.output_text


if __name__ == "__main__":
    print(solve("If 3x + 7 = 22, what is x?"))
