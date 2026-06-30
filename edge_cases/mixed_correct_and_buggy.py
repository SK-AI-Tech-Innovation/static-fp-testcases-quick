# ACE-EXPECT: detect
# CATEGORY: should_detect/no_retry_fallback
# EXPECT-MIN-FINDINGS: 1
# LANGUAGE: python
# ISSUE: one half is modern, schema-enforced, retried code (must NOT flag); the other half has a genuine no-retry reliability bug (must flag)
# EXPECTED-FINDING: classify_sentiment makes a single LLM call with no retry/backoff and no fallback path
# EXPECTED-FIX: wrap the call in a bounded retry loop with backoff and a degraded-mode fallback, matching extract_invoice
# SEVERITY-HINT: warning
"""Precision + recall test: correct modern code next to a real bug.

`extract_invoice` is correct — schema-enforced structured output, bounded
retries, fallback to a partial record. It must NOT be flagged.

`classify_sentiment` is the bug — a single model call with no retry, no
backoff, and no fallback. A transient error fails the whole request. It must
be flagged. The analyzer should flag the bug and leave the correct call alone.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic
from openai import OpenAI

logger = logging.getLogger("ai.mixed")

ANTHROPIC_MODEL = "claude-opus-4-20250514"
OPENAI_CHAT_MODEL = "gpt-4o"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0

_oai = OpenAI()
_anthropic = anthropic.Anthropic()


# =============================================================================
# CORRECT HALF — modern, schema-enforced, retried. Do NOT flag.
# =============================================================================
INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_cents": {"type": "integer"},
        "currency": {"type": "string"},
        "due_date": {"type": "string", "format": "date"},
    },
    "required": ["invoice_number", "total_cents", "currency", "due_date"],
    "additionalProperties": False,
}

_FALLBACK_INVOICE: dict[str, Any] = {
    "invoice_number": "",
    "total_cents": 0,
    "currency": "USD",
    "due_date": "",
}


def _backoff(attempt: int) -> None:
    delay = min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
    logger.info("backoff", extra={"attempt": attempt, "delay_seconds": delay})
    time.sleep(delay)


def extract_invoice(document_text: str) -> dict[str, Any]:
    """Schema-enforced extraction with bounded retries and a fallback record.

    On exhausted retries it returns a clearly-marked empty record rather than
    raising, so the caller degrades gracefully. This is the correct pattern.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info("extract_invoice.start", extra={"attempt": attempt})
            resp = _oai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Extract the invoice fields."},
                    {"role": "user", "content": document_text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice",
                        "strict": True,
                        "schema": INVOICE_SCHEMA,
                    },
                },
            )
            logger.info("extract_invoice.ok", extra={"attempt": attempt})
            return resp.choices[0].message.parsed
        except Exception as exc:
            last_error = exc
            logger.warning(
                "extract_invoice.retryable",
                extra={"attempt": attempt, "error": str(exc)},
            )
            _backoff(attempt)
    logger.error(
        "extract_invoice.exhausted_falling_back",
        extra={"error": str(last_error)},
    )
    return dict(_FALLBACK_INVOICE)


# =============================================================================
# BUGGY HALF — single call, no retry, no fallback. MUST be flagged.
# =============================================================================
def classify_sentiment(comment: str) -> str:
    """Single model call with no retry and no fallback.

    A transient rate limit or 5xx propagates and fails the whole request;
    there is no backoff loop and no degraded-mode return value.
    """
    response = _anthropic.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16,
        system="Classify the sentiment as exactly one of: positive, negative, neutral.",
        messages=[{"role": "user", "content": comment}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def process(document_text: str, comment: str) -> dict[str, Any]:
    invoice = extract_invoice(document_text)
    invoice["sentiment"] = classify_sentiment(comment)
    return invoice
