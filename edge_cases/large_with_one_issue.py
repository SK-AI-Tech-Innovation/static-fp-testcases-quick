# ACE-EXPECT: detect
# CATEGORY: should_detect/unvalidated_output_to_sink
# EXPECT-MIN-FINDINGS: 1
# LANGUAGE: python
# ISSUE: model-generated text is passed directly to eval() with no validation
# EXPECTED-FINDING: LLM output flowing into a dangerous sink (eval) — arbitrary code execution
# EXPECTED-FIX: never eval model output; parse with ast.literal_eval, a strict schema, or a sandboxed evaluator
# SEVERITY-HINT: critical
"""A large, mostly-correct analytics assistant with ONE planted issue.

Most of this module is correct (bounded retries, structured logging,
schema-enforced extraction). The needle is in `compute_metric`, where the
model is asked to produce a Python expression and the result is fed to
eval() with no validation — a critical unvalidated-output-to-sink bug.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import anthropic
from openai import OpenAI

logger = logging.getLogger("ai.analytics")

ANTHROPIC_MODEL = "claude-opus-4-20250514"
OPENAI_CHAT_MODEL = "gpt-4o"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


@dataclass
class ChatTurn:
    role: str
    content: str


def _sleep_backoff(attempt: int) -> None:
    delay = min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
    logger.info("backoff", extra={"attempt": attempt, "delay_seconds": delay})
    time.sleep(delay)


class AnthropicChatClient:
    """Anthropic Messages wrapper with bounded retries + structured logging."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, messages: Sequence[ChatTurn], max_tokens: int = 1024) -> str:
        payload = [{"role": t.role, "content": t.content} for t in messages]
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info("complete.start", extra={"attempt": attempt})
                response = self._client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=payload,
                )
                logger.info(
                    "complete.ok",
                    extra={"output_tokens": response.usage.output_tokens},
                )
                return "".join(b.text for b in response.content if b.type == "text")
            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                last_error = exc
                logger.warning("complete.retryable", extra={"error": str(exc)})
                _sleep_backoff(attempt)
        logger.error("complete.exhausted", extra={"error": str(last_error)})
        assert last_error is not None
        raise last_error


SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "trend": {"type": "string", "enum": ["up", "down", "flat"]},
        "confidence": {"type": "number"},
    },
    "required": ["headline", "trend", "confidence"],
    "additionalProperties": False,
}


class SummaryWriter:
    """Schema-enforced summary generation (correct, for contrast)."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI()

    def summarize(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Summarize the metric rows."},
                        {"role": "user", "content": str(rows)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "metric_summary",
                            "strict": True,
                            "schema": SUMMARY_SCHEMA,
                        },
                    },
                )
                return resp.choices[0].message.parsed
            except Exception as exc:
                last_error = exc
                logger.warning("summarize.retryable", extra={"error": str(exc)})
                _sleep_backoff(attempt)
        assert last_error is not None
        raise last_error


@dataclass
class MetricRequest:
    description: str
    variables: dict[str, float]


def _render_variable_block(variables: dict[str, float]) -> str:
    return "\n".join(f"{k} = {v}" for k, v in variables.items())


# --- THE PLANTED ISSUE -------------------------------------------------------
# compute_metric asks the model to emit a Python expression and then feeds that
# raw model text straight into eval(). This is the needle: untrusted LLM output
# reaching a code-execution sink. Everything around it is correct.
def compute_metric(chat: AnthropicChatClient, request: MetricRequest) -> float:
    """Ask the model for a formula, then evaluate it.

    The model returns a Python expression string; we pass it directly to
    eval() with the request variables as locals.
    """
    system = (
        "You are a quantitative analyst. Given a metric description and the "
        "available variables, respond with a single Python expression that "
        "computes the metric. Respond with ONLY the expression, no prose."
    )
    user = (
        f"Metric: {request.description}\n"
        f"Variables:\n{_render_variable_block(request.variables)}"
    )
    expression = chat.complete(
        system=system,
        messages=[ChatTurn(role="user", content=user)],
        max_tokens=128,
    ).strip()

    logger.info("compute_metric.eval", extra={"expression": expression})
    # CRITICAL: raw model output evaluated as code with no validation.
    result = eval(expression, {"__builtins__": {}}, dict(request.variables))
    return float(result)


def build_pipeline() -> tuple[AnthropicChatClient, SummaryWriter]:
    chat = AnthropicChatClient()
    writer = SummaryWriter()
    logger.info("analytics.pipeline.ready")
    return chat, writer


def run_report(request: MetricRequest, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """End-to-end: compute the metric and summarize the rows."""
    chat, writer = build_pipeline()
    value = compute_metric(chat, request)
    summary = writer.summarize(rows)
    summary["metric_value"] = value
    return summary
