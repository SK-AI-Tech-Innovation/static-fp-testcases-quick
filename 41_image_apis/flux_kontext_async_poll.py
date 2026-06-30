# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Black Forest Labs FLUX.1 Kontext — POST /v1/flux-kontext-pro -> {id, polling_url}; poll status=="Ready"
# WHY-CORRECT: FLUX Kontext is an ASYNC image API: submit returns a polling_url, and you poll until
#              `status == "Ready"`. The poll loop here is BOUNDED (max attempts + sleep + timeout), which
#              is the correct pattern for this async contract. It is an image endpoint — no `.choices`,
#              no chat structured output, and the loop is intentional, not an unbounded retry on a chat call.
# EXPECTED-WRONG: a text-chat-centric engine flags "unbounded polling loop", "retry without backoff on AI
#                 call", or "AI response missing .choices / structured output".
# CORRECT-VERDICT: no findings
"""Submit a FLUX.1 Kontext edit and poll the async result until Ready."""
from __future__ import annotations

import os
import time

import httpx

SUBMIT_URL = "https://api.bfl.ai/v1/flux-kontext-pro"
MAX_POLLS = 60
POLL_INTERVAL_S = 1.5


def run_kontext(prompt: str, input_image_b64: str) -> str:
    """Submit an edit job and return the result image URL once Ready.

    Polling is bounded: at most MAX_POLLS attempts spaced POLL_INTERVAL_S apart.
    """
    headers = {"x-key": os.environ["BFL_API_KEY"], "Content-Type": "application/json"}
    submit = httpx.post(
        SUBMIT_URL,
        headers=headers,
        json={"prompt": prompt, "input_image": input_image_b64},
        timeout=30.0,
    )
    submit.raise_for_status()
    polling_url = submit.json()["polling_url"]

    for _ in range(MAX_POLLS):
        poll = httpx.get(polling_url, headers={"x-key": os.environ["BFL_API_KEY"]}, timeout=30.0)
        poll.raise_for_status()
        body = poll.json()
        status = body.get("status")
        if status == "Ready":
            return body["result"]["sample"]
        if status in ("Error", "Failed"):
            raise RuntimeError(f"FLUX job failed: {body}")
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError("FLUX Kontext job did not become Ready in time")
