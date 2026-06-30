# ACE-FP-EXPECT: clean
# CATEGORY: edge_cases/encoding
# LANGUAGE: python
# SOURCE: Correct LLM code whose prompts contain emoji, RTL text, and assorted unicode
# WHY-CORRECT: Uses bounded retries and schema-enforced output; the emoji/unicode are data, not defects
# EXPECTED-WRONG: The engine garbles multi-byte characters in current_code (hallucination) or flags the unicode as a problem
# CORRECT-VERDICT: no findings; cited code (if any) stays byte-accurate
"""Correct moderation helper whose prompts include emoji ✅🚀 and mixed scripts (العربية, 日本語)."""

from anthropic import Anthropic
from pydantic import BaseModel

client = Anthropic()


class Verdict(BaseModel):
    allowed: bool
    reason: str


SYSTEM = "You are a content classifier. Decide if the message is allowed. 🚦 Reply with the schema."


def moderate(message: str) -> str:
    # emoji and mixed-script content flow through as ordinary text
    resp = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=128,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Message (✉️): {message} — 안전한가요? آمن؟"}],
    )
    return resp.content[0].text
