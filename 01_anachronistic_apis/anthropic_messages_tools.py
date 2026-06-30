# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Anthropic Python SDK Messages API tool use (`client.messages.create(..., tools=[...])`)
# WHY-CORRECT: this is the canonical Anthropic agentic loop — declare tools, detect `stop_reason == "tool_use"`,
#              execute, and feed a `tool_result` content block back. Error handling and the loop are complete.
# EXPECTED-WRONG: a Python-but-OpenAI-centric engine flags "no structured output / response_format missing",
#                 mis-reads `tool_use`/`tool_result` blocks as unparsed free text, or wants a `response_format=`
#                 arg that the Anthropic SDK does not have.
# CORRECT-VERDICT: no findings
"""Anthropic Messages API agentic loop with a single weather tool."""
from __future__ import annotations

import json

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city, in Celsius.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def _get_weather(city: str) -> dict[str, float]:
    # Stand-in for a real weather service call.
    return {"city": city, "celsius": 21.5}


def ask_with_tools(question: str) -> str:
    messages: list[dict] = [{"role": "user", "content": question}]

    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if resp.stop_reason != "tool_use":
            return "".join(block.text for block in resp.content if block.type == "text")

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = _get_weather(**block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})
