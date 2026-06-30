# ACE-EXPECT: detect
# CATEGORY: should_detect/nonstandard_implementation
# LANGUAGE: python
# ISSUE: A hand-written retry loop reimplements retry/backoff for LLM calls, but it has an off-by-one (only retries attempts-1 times), uses fixed sleep with no exponential backoff and no jitter, causing thundering-herd retries and one fewer attempt than intended
# EXPECTED-FINDING: a custom retry loop with off-by-one attempt counting and constant (non-exponential, no-jitter) sleep
# EXPECTED-FIX: use a battle-tested retry library (tenacity with exponential backoff + jitter) or the SDK's built-in max_retries instead of hand-rolling the loop
# SEVERITY-HINT: warning
"""Custom, buggy retry wrapper around an LLM call."""

import time

from anthropic import Anthropic

client = Anthropic()


def call_with_retry(prompt: str, attempts: int = 3) -> str:
    last_err = None
    # BUG: range(1, attempts) runs only attempts-1 times (off-by-one).
    for i in range(1, attempts):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Constant 2s sleep: no exponential backoff, no jitter.
            time.sleep(2)
    raise RuntimeError(f"all retries failed: {last_err}")


if __name__ == "__main__":
    print(call_with_retry("Give me a haiku about retries."))
