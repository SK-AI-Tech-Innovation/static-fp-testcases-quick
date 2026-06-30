# ACE-EXPECT: detect
# CATEGORY: should_detect/prompt_context_hierarchy
# LANGUAGE: python
# ISSUE: A classification prompt lists 50 documents first and only states the labeling rules afterward
# EXPECTED-FINDING: The classification rules (the task definition) appear after a long batch of documents, so the model reads most of the input without knowing what to do with it
# EXPECTED-FIX: State the labels and rules first, then provide the documents to classify
# SEVERITY-HINT: warning
"""Batch-classify support tickets into categories with OpenAI."""

from openai import OpenAI

client = OpenAI()

CATEGORIES = ["billing", "bug", "feature_request", "account", "other"]


def classify_tickets(tickets: list[str]) -> str:
    docs_block = "\n".join(f"{i}. {t}" for i, t in enumerate(tickets))
    prompt = (
        f"Tickets to process:\n{docs_block}\n\n"
        # The rules — what the model is supposed to DO — come after all 50 docs.
        "Classify each numbered ticket above into exactly one of these categories: "
        f"{', '.join(CATEGORIES)}. "
        "Use 'billing' for payment/invoice issues, 'bug' for broken behavior, "
        "'feature_request' for new-capability asks, 'account' for login/access, "
        "and 'other' otherwise. Return JSON mapping ticket number to category."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
