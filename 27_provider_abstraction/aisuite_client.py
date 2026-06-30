# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: aisuite unified client
# WHY-CORRECT: aisuite exposes an OpenAI-shaped client.chat.completions.create where the model is "provider:model" (e.g. "anthropic:claude-sonnet-4-5"). Iterating several provider:model strings through one client is the intended unified-API usage.
# EXPECTED-WRONG: engine may not recognize aisuite and flag the "anthropic:claude..." colon-prefixed model or the unified create() as a malformed/unknown LLM call.
"""Unified multi-provider chat via the aisuite client."""

import aisuite as ai


def make_client() -> "ai.Client":
    """Create a single aisuite client usable across providers."""
    return ai.Client()


def chat(model: str, prompt: str) -> str:
    """Run a chat completion for a 'provider:model' identifier."""
    client = make_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    for model in ("openai:gpt-4o", "anthropic:claude-sonnet-4-5"):
        print(model, "->", chat(model, "Explain a bloom filter briefly."))
