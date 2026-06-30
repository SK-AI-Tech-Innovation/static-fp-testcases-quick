# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: Anthropic Python SDK, extended thinking on Claude Opus 4.8 (`client.messages.create`).
# WHY-CORRECT: Opus 4.7 / 4.8 and Fable 5 REMOVED `budget_tokens` — the only valid on-mode is
#              `thinking={"type": "adaptive"}`, with depth controlled via `output_config={"effort": ...}`.
#              Sending `thinking={"type": "enabled", "budget_tokens": N}` returns a 400 on these models.
#              This code is correct precisely because it uses adaptive thinking with no token budget.
# EXPECTED-WRONG: a stale engine trained on Opus 4.5-and-earlier "knows" extended thinking requires a
#                 numeric budget and rewrites the call to
#                 `thinking={"type": "enabled", "budget_tokens": 8000}` (and may add `temperature=0`).
#                 On Opus 4.8 that payload 400s — the "fix" turns a valid request into a broken one.
# CORRECT-VERDICT: no findings — keep `thinking={"type": "adaptive"}`; do NOT introduce `budget_tokens`,
#                  `temperature`, `top_p`, or `top_k`, all of which are removed on this model.
"""Adaptive extended thinking on Opus 4.8 — no budget_tokens, by design."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()


def reason(prompt: str) -> str:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
