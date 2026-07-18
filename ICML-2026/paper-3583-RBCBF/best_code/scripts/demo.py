#!/usr/bin/env python3
"""RBCBF single-prompt demo.

Runs one prompt through (a) the baseline generator and (b) the RBCBF-controlled
generator, prints live trigger / rollback / Δh events, and reports a comparison.

Usage:
    python scripts/demo.py
    python scripts/demo.py --prompt "..." --verbose 2
    python scripts/demo.py --prompt_index 3 --max_new_tokens 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from inside scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbcbf import RBCBFRunner  # noqa: E402


def _load_demo_prompt(prompt_index: int) -> str:
    demo_path = ROOT / "data" / "demo_prompts.json"
    items = json.loads(demo_path.read_text(encoding="utf-8"))
    if not items:
        raise SystemExit("data/demo_prompts.json is empty.")
    prompt_index = max(0, min(prompt_index, len(items) - 1))
    return items[prompt_index]["prompt"]


def main() -> int:
    parser = argparse.ArgumentParser(description="RBCBF single-prompt demo.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt; if omitted, uses --prompt_index from demo_prompts.json.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=0,
        help="Index into data/demo_prompts.json (default: 0).",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rbcbf_default.json"),
        help="Path to RBCBF config JSON.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base LM (default: Qwen/Qwen2.5-7B-Instruct as in config).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="0=silent  1=event-level (trigger/Δh)  2=per-token streaming.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip the no-control baseline pass; run only RBCBF.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt else _load_demo_prompt(args.prompt_index)

    runner = RBCBFRunner.from_config(
        args.config,
        device=args.device,
        base_model=args.base_model,
    )

    if not args.skip_baseline:
        print("\n=== BASELINE (no control) ===")
        baseline = runner.generate(
            prompt,
            control=False,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print()
        print(baseline.summary())
        print("\n--- baseline final response (truncated) ---")
        print(baseline.text[:500] + ("..." if len(baseline.text) > 500 else ""))
    else:
        baseline = None

    print("\n=== RBCBF (controlled) ===")
    controlled = runner.generate(
        prompt,
        control=True,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print()
    print(controlled.summary())
    print("\n--- controlled final response (truncated) ---")
    print(controlled.text[:500] + ("..." if len(controlled.text) > 500 else ""))

    if baseline is not None:
        print("\n=== COMPARISON ===")
        print(f"  Baseline    h_final: {baseline.h_final:+.3f}  triggered: {baseline.triggered}")
        print(f"  Controlled  h_final: {controlled.h_final:+.3f}  triggered: {controlled.triggered}")
        if controlled.triggered:
            improvement = controlled.h_final - baseline.h_final
            print(f"  Safety margin improvement: {improvement:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
