# ACE-EXPECT: detect
# CATEGORY: should_detect/hardcoded_tool_routing_heuristic
# LANGUAGE: python
# SOURCE: SK ATIT project feedback — Talent-AX (Notion: https://app.notion.com/p/sk-atit/Talent-AX-9e81772c74d583c3ae85819c9873985b)
# ISSUE: Tool selection for a free-form chat agent is forced by hardcoded Korean substring/length heuristics instead of letting the model decide
# EXPECTED-FINDING: Tool routing is driven by brittle hand-coded patterns (substring lists like "도 알려"/"도 추천" and a <=20-char length check) to auto-reselect the previous tool; in open-ended chat these heuristics cannot anticipate real inputs and will misroute, bypassing the LLM's own tool-choice
# EXPECTED-FIX: Drive tool selection through the LLM (system prompt + tool/function definitions, or a structured tool-choice step) rather than keyword/length heuristics; encode the "continue previous tool" intent in the prompt instead of substring matching
# SEVERITY-HINT: warning
"""Tool routing forced by hardcoded keyword + length heuristics.

For a free-text chat service you cannot enumerate every way a user might phrase a
follow-up. Hardcoding substring patterns and a length threshold to *force* the previous
tool is fragile — it both fires on unintended inputs and misses genuine continuations.
The robust path is to let the model choose the tool from the system prompt + tool specs.
"""
from typing import Iterator


def plan_tools(
    current_message_clean: str,
    previous_tool_with_reask: str | None,
    allowed_tool_names: set[str],
    state: dict,
) -> Iterator[dict]:
    # If the previous turn re-asked with a tool, try to auto-continue with the same tool.
    if previous_tool_with_reask and previous_tool_with_reask in allowed_tool_names:
        # Short message (<=20 chars) or a "~도 알려줘"-style pattern -> force same tool.
        continuation_patterns = ["도 알려", "도 추천", "도 조회", "도 보여", "도 해줘", "도 해주"]
        has_continuation_pattern = any(
            pattern in current_message_clean for pattern in continuation_patterns
        )

        # Verb keyword check.
        action_keywords = ["적용", "추천", "조회", "선택", "확인", "보기", "알려", "해줘", "해주"]
        has_action_keyword = any(
            keyword in current_message_clean for keyword in action_keywords
        )

        # Brittle: any of these substrings, or just a short message, forces the tool.
        if has_continuation_pattern or (
            len(current_message_clean) <= 20
            and not has_action_keyword
            and current_message_clean
        ):
            final_plan = [
                {
                    "step_id": "step_1",
                    "name": previous_tool_with_reask,
                    "params": {},
                    "hint": current_message_clean,
                    "tool_message": "",
                }
            ]
            state.update({"plan": final_plan})
            yield {"plan": state.get("plan")}
            return

    # ... otherwise fall through to real planning ...
    yield {"plan": state.get("plan", [])}
