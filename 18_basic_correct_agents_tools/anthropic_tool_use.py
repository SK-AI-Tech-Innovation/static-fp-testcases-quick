# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK (Messages API, manual tool-use loop)
# WHY-CORRECT: The tool carries a clear input_schema (typed properties with
#   descriptions, enum-constrained unit, required list). The roundtrip is the
#   documented one: append the assistant content (preserving tool_use blocks),
#   then send one tool_result per tool_use_id. Model is claude-opus-4-8 with
#   adaptive thinking. Nothing is missing or under-specified.
# EXPECTED-WRONG: engine may suggest "add a tool schema", "describe the
#   parameters", or "feed tool results back to the model" — all already done.
# CORRECT-VERDICT: no findings
"""Correct Anthropic tool_use roundtrip with a clear input_schema."""

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_current_weather",
        "description": (
            "Get the current weather for a city. Call this when the user asks "
            "about present-day weather conditions for a named location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state/country, e.g. 'Paris, France'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit to report.",
                },
            },
            "required": ["location", "unit"],
        },
    }
]


def get_current_weather(location: str, unit: str) -> str:
    """Look up the current weather (stubbed deterministic response)."""
    return f"21 degrees {unit} and sunny in {location}"


def ask_weather(user_message: str) -> str:
    """Run one tool-use turn and return the model's final text answer."""
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        return next((b.text for b in response.content if b.type == "text"), "")

    # Preserve the assistant content (including tool_use blocks).
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = get_current_weather(block.input["location"], block.input["unit"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
    messages.append({"role": "user", "content": tool_results})

    final = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=messages,
    )
    return next((b.text for b in final.content if b.type == "text"), "")


if __name__ == "__main__":
    print(ask_weather("What's the weather in Paris right now?"))
