# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A normalization step that must produce identical output across runs
# WHY-CORRECT: temperature=0 is a deliberate choice for reproducibility/determinism in a data pipeline, not an oversight
# EXPECTED-WRONG: Engine flags temperature=0 as "too rigid" or suggests raising temperature for diversity, contradicting the documented intent
# CORRECT-VERDICT: no findings
"""Deterministically canonicalize a free-text address using an LLM."""

from anthropic import Anthropic

client = Anthropic()


def canonicalize_address(raw_address: str) -> str:
    """Return a canonical, reproducible form of a postal address.

    Determinism matters: the same input must always map to the same output
    so downstream deduplication keys stay stable. temperature=0 is chosen
    deliberately for exactly this reason.

    Args:
        raw_address: An unstructured address string.

    Returns:
        str: The canonicalized address.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        temperature=0,  # intentional: reproducible, deduplication-stable output
        system="Rewrite the address in canonical USPS form. Output only the address.",
        messages=[{"role": "user", "content": raw_address}],
    )
    return response.content[0].text.strip()
