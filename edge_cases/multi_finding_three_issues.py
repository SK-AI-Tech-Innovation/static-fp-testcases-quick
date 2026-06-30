# ACE-EXPECT: detect
# CATEGORY: should_detect/structured_output_missing
# EXPECT-MIN-FINDINGS: 2
# LANGUAGE: python
# ISSUE: three distinct reliability/quality anti-patterns in one agent file
# EXPECTED-FINDING: (a) LLM call with no retry/fallback; (b) tool with vague generic dict schema and no descriptions; (c) json.loads on raw model text with no schema
# EXPECTED-FIX: add bounded retries+fallback; give the tool a typed schema with field descriptions; enforce structured output / validate parsed JSON
# SEVERITY-HINT: warning
"""A small agent with THREE separate, genuinely-present issues.

Each issue is independently citeable:
  (a) `single_shot_complete` calls the model with no retry and no fallback.
  (b) `LOOKUP_TOOL` declares a generic `dict`-typed schema with no per-field
      descriptions, so the model has no guidance on how to call it.
  (c) `parse_plan` runs json.loads on raw model text with no schema enforcement
      and no validation of the resulting structure.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

ANTHROPIC_MODEL = "claude-opus-4-20250514"

_client = anthropic.Anthropic()


# --- Issue (a): no retry, no fallback ---------------------------------------
def single_shot_complete(prompt: str) -> str:
    """One call, no retry, no fallback. A transient 429/5xx fails the request.

    There is no try/except, no backoff loop, and no degraded-mode response.
    """
    response = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


# --- Issue (b): vague, generic tool schema with no descriptions -------------
# The schema is just an open object of strings; the model gets no guidance on
# what the parameters mean or which values are valid.
LOOKUP_TOOL = {
    "name": "lookup",
    "description": "lookup",
    "input_schema": {
        "type": "object",
        "properties": {
            "args": {"type": "object"},
            "data": {"type": "object"},
        },
    },
}


def call_with_tool(prompt: str) -> Any:
    response = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        tools=[LOOKUP_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return None


# --- Issue (c): json.loads on raw model text, no schema, no validation ------
def parse_plan(goal: str) -> dict[str, Any]:
    """Ask the model for a JSON plan and json.loads the raw text.

    No structured-output enforcement, no schema, and the parsed result is
    returned without checking it has the expected shape.
    """
    raw = single_shot_complete(
        f"Return a JSON object describing a plan for: {goal}"
    )
    plan = json.loads(raw)  # raw model text -> dict, unchecked
    return plan


def run(goal: str) -> dict[str, Any]:
    plan = parse_plan(goal)
    plan["lookup"] = call_with_tool(goal)
    return plan
