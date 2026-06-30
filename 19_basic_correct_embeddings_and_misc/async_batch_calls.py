# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: AsyncOpenAI + asyncio.gather with a Semaphore concurrency limit
# WHY-CORRECT: independent prompts run concurrently via gather, bounded by a semaphore so the API
#              isn't hammered. Results stay in input order (gather preserves order). Idiomatic async.
# EXPECTED-WRONG: engine claims "unbounded concurrency" or "no rate limiting" despite the explicit
#                 semaphore gating every call.
# CORRECT-VERDICT: no findings
"""Run independent LLM calls concurrently with a bounded semaphore."""
import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI()
semaphore = asyncio.Semaphore(5)


async def ask(prompt: str) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


async def ask_many(prompts: list[str]) -> list[str]:
    return await asyncio.gather(*(ask(prompt) for prompt in prompts))


if __name__ == "__main__":
    questions = ["What is 2+2?", "Name a primary color.", "Capital of Japan?"]
    print(asyncio.run(ask_many(questions)))
