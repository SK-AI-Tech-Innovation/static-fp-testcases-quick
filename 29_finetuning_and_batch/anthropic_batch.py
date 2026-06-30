# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: Anthropic Message Batches API (anthropic SDK)
# WHY-CORRECT: client.messages.batches.create with a list of Request(custom_id, params=MessageCreateParamsNonStreaming(...)) is the documented Anthropic batch contract; each params block is a valid Messages API request (model + max_tokens + messages).
# EXPECTED-WRONG: engine may flag the nested batches.create / Request / MessageCreateParamsNonStreaming shape (not a plain messages.create) as an unknown or malformed chat call.
"""Submit an asynchronous Anthropic Message Batch."""

import os

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


def submit_batch(prompts: list[str]) -> str:
    """Create an Anthropic message batch from prompts; return the batch id."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    requests = [
        Request(
            custom_id=f"request-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-opus-4-8",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, prompt in enumerate(prompts)
    ]

    batch = client.messages.batches.create(requests=requests)
    return batch.id


if __name__ == "__main__":
    print("Created batch:", submit_batch(["Summarize TCP.", "Summarize UDP."]))
