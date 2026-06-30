# ACE-EXPECT: detect
# CATEGORY: should_detect/god_agent_single_responsibility
# EXPECT-MIN-FINDINGS: 2
# LANGUAGE: python
# ISSUE: three comparable design/reliability issues in one class — god object, no observability, no retry/fallback
# EXPECTED-FINDING: (a) god-object class doing HTTP, DB, email, billing, prompts, and tool execution; (b) multi-step agent loop with no structured logging/telemetry; (c) the LLM call has no retry/fallback on transient errors
# EXPECTED-FIX: split the god object into focused collaborators; add structured per-step logging/tracing; wrap the LLM call in bounded retries with a fallback
# SEVERITY-HINT: suggestion
"""A monolithic 'agent' class with three comparable (non-critical) issues.

  (a) `MegaAgent` is a god object: HTTP, DB writes, email, billing, prompt building, tools.
  (b) `agent_loop` runs a multi-step tool loop with zero logging/telemetry.
  (c) the model call in the loop has no retry/fallback for transient 429/5xx errors.
"""

from __future__ import annotations

import smtplib
import sqlite3
import urllib.request
from typing import Any

import anthropic

ANTHROPIC_MODEL = "claude-opus-4-20250514"


class MegaAgent:
    """God object: unrelated responsibilities crammed into one class."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()
        self._db = sqlite3.connect(":memory:")
        self._smtp = None
        self._invoices: list[dict[str, Any]] = []

    # ---- unrelated job: raw HTTP ----
    def fetch_url(self, url: str) -> bytes:
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            return resp.read()

    # ---- unrelated job: database writes ----
    def save_record(self, table: str, row: dict[str, Any]) -> None:
        cols = ",".join(row.keys())
        placeholders = ",".join("?" for _ in row)
        self._db.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
        self._db.commit()

    # ---- unrelated job: email ----
    def send_email(self, to: str, subject: str, body: str) -> None:
        if self._smtp is None:
            self._smtp = smtplib.SMTP("localhost")
        self._smtp.sendmail("agent@example.com", [to], f"Subject: {subject}\n\n{body}")

    # ---- unrelated job: billing ----
    def charge_invoice(self, customer_id: str, amount_cents: int) -> dict[str, Any]:
        invoice = {"customer": customer_id, "amount": amount_cents, "status": "paid"}
        self._invoices.append(invoice)
        return invoice

    def _build_prompt(self, task: str) -> str:
        return f"You are an automation agent. Complete this task:\n{task}"

    # --- Issues (b) no observability + (c) no retry/fallback, in one loop ---
    def agent_loop(self, task: str, max_steps: int = 6) -> str:
        """Run a tool loop. No logging/metrics/tracing of any step, and the model
        call has no retry or fallback for transient failures."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": self._build_prompt(task)}]
        for _ in range(max_steps):
            # no retry, no try/except, no fallback: a transient 429/5xx aborts the loop
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=1024, messages=messages
            )
            if response.stop_reason == "end_turn":
                return "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "continue"})  # tool step, still unlogged
        return "step budget exhausted"


def run(task: str) -> str:
    return MegaAgent().agent_loop(task)
