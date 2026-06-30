# ACE-EXPECT: detect
# CATEGORY: should_detect/tool_idempotency
# LANGUAGE: python
# ISSUE: The payment-charge tool sends no idempotency key and is invoked inside a retry loop, so a retried call double-charges the customer.
# EXPECTED-FINDING: Non-idempotent side-effecting tool (charge_card) retried without an idempotency key — risk of duplicate charges.
# EXPECTED-FIX: Generate a stable idempotency key per logical charge and pass it to the payment API; the provider dedupes retries.
# SEVERITY-HINT: warning
"""Checkout agent: the model decides to charge, and the handler retries the charge on transient failure."""

import os
import time

import stripe
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
stripe.api_key = os.environ["STRIPE_API_KEY"]

TOOLS = [
    {
        "name": "charge_card",
        "description": "Charge the customer's saved card for the order total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Stripe customer id"},
                "amount_cents": {"type": "integer", "description": "Amount to charge in cents"},
            },
            "required": ["customer_id", "amount_cents"],
        },
    }
]


def charge_card(customer_id: str, amount_cents: int) -> str:
    """Charge with retry — but no idempotency key, so each retry is a fresh charge."""
    last_err = None
    for attempt in range(3):
        try:
            intent = stripe.PaymentIntent.create(
                customer=customer_id,
                amount=amount_cents,
                currency="usd",
                confirm=True,
                # MISSING: idempotency_key=... — a retried network error charges again.
            )
            return intent.id
        except stripe.error.APIConnectionError as e:  # transient
            last_err = e
            time.sleep(2**attempt)
    raise last_err


def run(order_request: str) -> object:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        tools=TOOLS,
        messages=[{"role": "user", "content": order_request}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "charge_card":
            return charge_card(**block.input)
    return None


if __name__ == "__main__":
    print(run("Place the order and charge cus_123 for $49.99."))
