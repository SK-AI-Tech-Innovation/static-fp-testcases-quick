# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Cerebras, verified base_url https://api.cerebras.ai/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Cerebras
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Cerebras via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the speed of light in a vacuum?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
