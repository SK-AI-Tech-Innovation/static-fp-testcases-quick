# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: OpenAI chat completions with a single forced tool/function call as the structured-output mechanism
# WHY-CORRECT: forcing exactly one tool via `tool_choice={"type": "function", "function": {"name": ...}}`
#              guarantees the model returns `tool_calls[0].function.arguments` matching the tool's JSON schema.
#              This is a recognized, valid structured-output technique — the arguments are schema-enforced JSON.
# EXPECTED-WRONG: engine only recognizes `response_format=`/`.parse(...)` as "structured output" and flags the
#                 forced-function-call pattern as "free-text parsing / not structured".
# CORRECT-VERDICT: no findings
"""Extract a typed person record using a single forced OpenAI function call."""
from __future__ import annotations

import json

from openai import OpenAI

client = OpenAI()

PERSON_TOOL = {
    "type": "function",
    "function": {
        "name": "record_person",
        "description": "Record the structured person details extracted from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
            },
            "required": ["name"],
        },
    },
}


def extract_person(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": text}],
        tools=[PERSON_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_person"}},
    )
    # Forced tool_choice guarantees one tool call whose arguments match the schema.
    call = response.choices[0].message.tool_calls[0]
    return json.loads(call.function.arguments)
