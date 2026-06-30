# ACE-FP-EXPECT: clean
# CATEGORY: edge_cases/syntax_error
# LANGUAGE: python
# SOURCE: A Python file whose VALID portion is correct LLM code, truncated mid-statement at the end (incomplete file)
# WHY-CORRECT: Robustness case: partial/streamed files happen in real pipelines. The complete function is correct (schema-enforced output), so the only "defect" is the truncation, which the engine must ignore — not crash, not hallucinate, not flag the correct code
# EXPECTED-WRONG: ast.parse raises on the incomplete tail and the run errors, or the engine fabricates the missing code, or it flags the correct complete function
# CORRECT-VERDICT: no crash; no hallucinated findings; no finding on the correct complete function
"""Agent whose source was cut off mid-statement — valid part is correct; must not crash."""

from anthropic import Anthropic
from pydantic import BaseModel

client = Anthropic()


class Label(BaseModel):
    category: str


def classify(text: str) -> Label:
    # complete, correct function: schema-enforced structured output, nothing to flag
    resp = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=64,
        tools=[{"name": "label", "input_schema": Label.model_json_schema()}],
        tool_choice={"type": "tool", "name": "label"},
        messages=[{"role": "user", "content": f"Classify:\n{text}"}],
    )
    return Label.model_validate(resp.content[0].input)


def classify_batch(items: list[str]) -> list[Label]:
    results = []
    for item in items:
        results.append(classify(it
    # <-- file truncated here on purpose, mid-expression (unclosed call)
