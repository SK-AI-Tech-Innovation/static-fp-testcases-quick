# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1087 (gen_metric_abbrev.py) — ACE: "parsed['en'] KeyError 위험"; author confirmed FP
# WHY-CORRECT: the dict is indexed with parsed["en"] ONLY inside an `if parsed.get("en"):` guard. The
#              truthy short-circuit means the key is present (and non-empty) before the subscript runs,
#              so KeyError is unreachable. This is the standard get()-guard-then-index idiom.
# EXPECTED-WRONG: engine sees `parsed["en"]` and warns "KeyError if 'en' missing", missing that the
#                 enclosing `if parsed.get("en")` already proved the key exists.
# CORRECT-VERDICT: no findings
"""Parse an LLM expansion result with a get()-guard before indexing.

ACE flagged a KeyError risk on `parsed["en"]`, but the surrounding `if parsed.get("en")`
guarantees the key is present and truthy before it is read.
"""
import json


def verify_expansion(raw: str) -> dict | None:
    parsed = json.loads(raw)
    # Truthy guard: get("en") must be non-empty for the body to run.
    if parsed.get("en"):
        # Safe: the guard above proved "en" exists; KeyError cannot occur here.
        return {"en": parsed["en"], "ko": parsed.get("ko", [])}
    return None
