# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: openai (async) + tenacity
# WHY-CORRECT: Concurrency is bounded by an asyncio.Semaphore to avoid overwhelming the provider, and each call retries with exponential backoff plus random jitter to prevent thundering-herd retries. Both rate-limit protection and retry are present.
# EXPECTED-WRONG: missing retry / missing rate limiting on concurrent LLM calls
# CORRECT-VERDICT: no findings
"""Run bounded-concurrency async LLM calls with jittered exponential retry."""

import asyncio

from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

client = AsyncOpenAI(timeout=30)

# Cap in-flight requests so we never exceed the provider's concurrency budget.
_semaphore = asyncio.Semaphore(5)


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _embed(text: str) -> list[float]:
    """Single embedding call with jittered backoff under the concurrency cap."""
    async with _semaphore:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding


async def embed_all(texts: list[str]) -> list[list[float]]:
    """Embed many texts concurrently while respecting the semaphore bound."""
    return await asyncio.gather(*(_embed(t) for t in texts))
