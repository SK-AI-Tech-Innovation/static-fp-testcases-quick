# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: argparse CLI + interactive input() confirmation prompts (stdlib only)
# WHY-CORRECT: "prompt" here means the terminal prompt shown to a human via input(); it is
#              standard interactive CLI UX. No LLM, no chat messages, no system prompt.
# EXPECTED-WRONG: keywords "prompt", "system_prompt", "PROMPT" -> false "LLM prompting"
#                 detection -> spurious findings about prompt templates / injection.
# CORRECT-VERDICT: no findings
"""Interactive CLI that prompts the user to confirm a destructive action. Stdlib only."""
from __future__ import annotations

import argparse
import sys

CONFIRM_PROMPT = "Type the project name to confirm deletion: "
YES_NO_PROMPT = "Proceed? [y/N] "


def confirm_yes_no(prompt: str = YES_NO_PROMPT) -> bool:
    """Prompt the user for a yes/no answer at the terminal."""
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def prompt_for_name(expected: str) -> bool:
    """Require the user to retype a name exactly before continuing."""
    typed = input(CONFIRM_PROMPT).strip()
    return typed == expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete a project by name.")
    parser.add_argument("project", help="project name to delete")
    parser.add_argument(
        "--yes", action="store_true", help="skip the interactive prompt"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.yes:
        if not prompt_for_name(args.project) or not confirm_yes_no():
            print("Aborted.", file=sys.stderr)
            return 1
    print(f"Deleted project {args.project!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
