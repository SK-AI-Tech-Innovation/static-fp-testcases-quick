# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a string variable containing python-looking LLM code that is never executed
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""A template string that merely contains, but never runs, LLM code."""

EXAMPLE_SNIPPET = """
import json
from openai import OpenAI

client = OpenAI()

def run(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    payload = json.loads(response.choices[0].message.content)
    return payload["answer"]
"""
