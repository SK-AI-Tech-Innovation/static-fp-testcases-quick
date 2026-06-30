# ACE-FP-EXPECT: 1
# CATEGORY: 12_severity_miscalibration
# SOURCE: a correct embedding helper called repeatedly on a small, repeating set of labels (cacheable)
# WHY-CORRECT: the code is correct as written — it embeds each label and returns the nearest one. Because
#              the label set is small and stable, the per-label embeddings could be memoized/cached, but
#              correctness does not depend on it. A caching opportunity is a performance optimization, which
#              per the rubric is NEVER critical; at most it merits a single suggestion.
# EXPECTED-WRONG: escalating "should cache embeddings" to `critical`, or raising it as a correctness defect.
# CORRECT-VERDICT: at most 1 suggestion ("memoize label embeddings"), NEVER critical
"""Nearest-label classifier that re-embeds a fixed label set each call (correct; just cacheable)."""
from __future__ import annotations

import math

LABELS = ["billing", "technical", "account", "feedback"]


def embed(text: str) -> list[float]:
    vec = [0.0] * 32
    for i, ch in enumerate(text.lower()):
        vec[(ord(ch) + i) % 32] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def nearest_label(query: str) -> str:
    q = embed(query)
    best_label = LABELS[0]
    best_score = -1.0
    for label in LABELS:
        # The fixed LABELS could be embedded once and cached across calls, but it is correct as-is.
        score = sum(a * b for a, b in zip(q, embed(label)))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


if __name__ == "__main__":
    print(nearest_label("I was charged twice on my invoice"))
