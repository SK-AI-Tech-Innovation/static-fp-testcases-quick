# ACE-FP-EXPECT: clean
# CATEGORY: 15_correct_terse_code
# SOURCE: A single, well-described tool/function schema for tool use
# WHY-CORRECT: The schema already has clear field descriptions, types, constraints, and required fields. It is complete; nothing needs adding.
# EXPECTED-WRONG: Engine suggests adding more descriptions, examples, extra validation, or splitting the tool — redundant on an already well-specified schema
# CORRECT-VERDICT: no findings
"""A single, fully described tool schema for Claude tool use."""

from pydantic import BaseModel, Field


class GetWeatherInput(BaseModel):
    """Validated arguments for the get_weather tool."""

    city: str = Field(..., description="City name, e.g. 'Seoul' or 'San Francisco'.")
    units: str = Field(
        "celsius",
        description="Temperature unit to return.",
        pattern="^(celsius|fahrenheit)$",
    )
    days: int = Field(
        1,
        ge=1,
        le=7,
        description="Number of forecast days to return, from 1 to 7.",
    )


GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather forecast for a city. Use when the user asks "
    "about current or upcoming weather conditions.",
    "input_schema": GetWeatherInput.model_json_schema(),
}
