# ACE-FP-EXPECT: scoped-out
# CATEGORY: 02_out_of_scope_general_quality
# SOURCE: a correct Anthropic structured-output extraction call written with terse, poor variable names
# WHY-CORRECT: the LLM usage is idiomatic and correct — tool-based structured output with a schema,
#              explicit max_tokens, and the result parsed from a tool_use block. The ONLY defects are
#              general code-quality (single-letter names, no type hints), which static scope excludes.
# EXPECTED-WRONG: linting-style findings on naming ("rename x/r/d", "non-descriptive identifiers") or
#                 "add type hints" — pure code-style nits the LLM-pattern engine must not emit.
# CORRECT-VERDICT: no findings (naming/style is out-of-scope; the structured-output usage is already correct)
"""Extract invoice fields into a fixed JSON schema via Anthropic tool-use (terse names, but correct)."""
from __future__ import annotations

import json

import anthropic

c = anthropic.Anthropic()

t = [
    {
        "name": "emit_invoice",
        "description": "Return the structured invoice fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "total": {"type": "number"},
                "due_date": {"type": "string"},
            },
            "required": ["vendor", "total", "due_date"],
        },
    }
]


def x(d):
    # `d` is the raw invoice text; `r` the response; `b` the tool_use block. Naming is bad, logic is fine.
    r = c.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=t,
        tool_choice={"type": "tool", "name": "emit_invoice"},
        messages=[{"role": "user", "content": f"Extract the invoice fields:\n{d}"}],
    )
    for b in r.content:
        if b.type == "tool_use" and b.name == "emit_invoice":
            return b.input
    return None


if __name__ == "__main__":
    print(json.dumps(x("ACME Corp — total $1,240.00 — due 2026-07-01"), indent=2))
