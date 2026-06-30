# ACE-EXPECT: detect
# CATEGORY: should_detect/test_boundary_inputs
# LANGUAGE: python
# ISSUE: Agent tool test covers only the successful lookup; no missing-key, empty-query, or malformed-arg cases
# EXPECTED-FINDING: Tool error handling and edge inputs are never exercised, only the happy path
# EXPECTED-FIX: Add cases for unknown symbol, empty string, whitespace, and oversized/special-character queries
# SEVERITY-HINT: suggestion
"""Tests for a weather-lookup tool exposed to a LangChain agent."""

import pytest
from langchain_core.tools import tool

_FAKE_DB = {"seoul": "22C, clear", "tokyo": "18C, rain"}


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return _FAKE_DB[city.lower()]


def test_known_city_returns_weather():
    result = get_weather.invoke({"city": "Seoul"})
    assert result == "22C, clear"
