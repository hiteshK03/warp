#!/usr/bin/env python3
"""Get a skeptical Claude review of a change before it is committed.

Usage:
    python3 tools/claude_review.py --step "what this step does" \
        [--diff-base metal-1.17] [--files path ...] [--question "..."] \
        [--context-file notes.md] [--out review.md]

Design notes (learned the hard way):
  - ALWAYS inline the code/diff into the prompt and pass --tools "" so Claude
    answers instead of spending its whole turn budget reading files. A review
    with Read enabled burned 13 turns / $0.91 and returned an empty result.
  - --max-turns 1 with no tools forces a single prose answer.
  - Ask for concrete failing inputs, not general impressions; then VERIFY each
    claim by execution before accepting it. Claude's df64 review was right on
    every tested claim, but "right on the tested ones" is not "all correct".
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

PREAMBLE = """You are a skeptical senior reviewer of a fork of NVIDIA Warp that adds an Apple Metal
backend (MSL: no double, strict pointer address spaces, no struct inheritance).
Be adversarial, not agreeable. Prioritize correctness risks over style. For every
bug you claim, give a concrete input that breaks it and the expected vs actual
result, so it can be verified by execution. If the approach is wrong, say so.
Do not pad with praise. If something is genuinely fine, say so briefly and move on.
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, help="one-line description of the change under review")
    ap.add_argument("--diff-base", default=None, help="review committed diff vs this ref")
    ap.add_argument("--staged", action="store_true", help="review the working-tree diff (uncommitted)")
    ap.add_argument("--files", nargs="*", default=[], help="extra files to inline in full")
    ap.add_argument("--context-file", default=None, help="markdown notes to inline (design/measurements)")
    ap.add_argument("--question", action="append", default=[], help="specific question (repeatable)")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    parts = [PREAMBLE, "\n## The step under review\n", args.step, "\n"]

    if args.diff_base:
        parts.append(f"\n## Diff vs {args.diff_base}\n```diff\n{git('diff', args.diff_base)}\n```\n")
    if args.staged:
        parts.append(f"\n## Uncommitted diff\n```diff\n{git('diff')}\n```\n")
    for f in args.files:
        p = pathlib.Path(f)
        if not p.is_absolute():
            p = REPO / p
        parts.append(f"\n## File: {f}\n```\n{p.read_text()}\n```\n")
    if args.context_file:
        p = pathlib.Path(args.context_file)
        if not p.is_absolute():
            p = REPO / p
        parts.append(f"\n## Context / measurements\n{p.read_text()}\n")

    if args.question:
        parts.append("\n## Specific questions\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(args.question)))
    parts.append(
        "\nAnswer now in prose. No tools are available; everything you need is above. "
        "Be concise and specific. Where you claim a bug, give the concrete failing input."
    )
    prompt = "".join(parts)

    proc = subprocess.run(
        ["claude", "-p", "--model", args.model, "--effort", args.effort,
         "--tools", "", "--max-turns", "1", "--output-format", "json"],
        input=prompt, capture_output=True, text=True, cwd=REPO,
    )
    try:
        d = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("claude returned non-JSON:", proc.stdout[:500], proc.stderr[:500], file=sys.stderr)
        return 1

    result = d.get("result", "")
    print(f"[review] subtype={d.get('subtype')} turns={d.get('num_turns')} "
          f"cost=${d.get('total_cost_usd', 0):.3f} chars={len(result)}", file=sys.stderr)
    if not result:
        print("EMPTY RESULT — do not treat as approval.", file=sys.stderr)
        return 1
    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.write_text(result)
        print(f"[review] written to {out}", file=sys.stderr)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
