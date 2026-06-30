"""Determinism guard for the static engine's DETERMINISTIC rules.

The two AST rules — `detect_structured_output` and `detect_unvalidated_sink`
(backend/services/static_analysis/stages/deterministic_rules.py) — are the part of
the engine that MUST be reproducible: same source in => same verdict out, every run,
regardless of PYTHONHASHSEED, dict/set iteration order, etc. The LLM layer is
inherently non-deterministic and is NOT checked here.

This runs each detector N times over every Python file in the corpus and flags any
file whose verdict (verdict + line + evidence) differs between runs. It needs NO LLM,
NO DB, NO stack — just the importable backend package — so it's cheap enough for CI.

Usage (from repo root):
    uv run python static-fp-testcases/check_determinism.py [--repeat N] [--dir NN_x]

Exit code is non-zero if any non-determinism is found (so it can gate CI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.services.static_analysis.stages.deterministic_rules import (
    detect_structured_output,
    detect_unvalidated_sink,
)

ROOT = Path(__file__).resolve().parent
DETECTORS = {
    "structured_output": detect_structured_output,
    "unvalidated_sink": detect_unvalidated_sink,
}


def _verdict_signature(rv: object) -> tuple:
    """Stable, comparable snapshot of a RuleVerdict."""
    return (
        getattr(getattr(rv, "verdict", None), "value", str(getattr(rv, "verdict", None))),
        getattr(rv, "line", None),
        getattr(rv, "evidence", None),
    )


def discover(sub: str | None) -> list[Path]:
    base = ROOT / sub if sub else ROOT
    return [
        p
        for p in sorted(base.rglob("*.py"))
        if p.name != Path(__file__).name and "results" not in p.parts
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, help="runs per detector per file (default 5)")
    ap.add_argument("--dir", help="limit to one subdir")
    args = ap.parse_args()

    files = discover(args.dir)
    if not files:
        print("No Python files found.")
        return 0

    print(f"Determinism check: {len(files)} files x {len(DETECTORS)} detectors x {args.repeat} runs\n")
    unstable: list[dict] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:  # unreadable file shouldn't kill the whole run
            print(f"  [skip] {path.name}: {type(e).__name__}: {e}")
            continue
        for name, fn in DETECTORS.items():
            sigs = set()
            for _ in range(args.repeat):
                try:
                    sigs.add(_verdict_signature(fn(source)))
                except Exception as e:
                    sigs.add(("EXCEPTION", type(e).__name__, str(e)))
            if len(sigs) > 1:
                rel = str(path.relative_to(ROOT))
                unstable.append({"file": rel, "detector": name, "verdicts": sorted(map(str, sigs))})
                print(f"  ✗ NON-DETERMINISTIC  {rel}  [{name}] -> {len(sigs)} distinct verdicts")

    print(f"\n{'='*60}")
    if unstable:
        print(f"  FAIL: {len(unstable)} (file, detector) pairs are non-deterministic")
        for u in unstable:
            print(f"    - {u['file']} [{u['detector']}]: {u['verdicts']}")
        print(f"{'='*60}")
        return 1
    print(f"  PASS: all {len(files)} files give stable verdicts across {args.repeat} runs")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
