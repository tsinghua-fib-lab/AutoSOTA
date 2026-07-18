#!/usr/bin/env python3
"""RBCBF multi-prompt runner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbcbf import RBCBFRunner  # noqa: E402


def _load_prompts(path: Path, max_prompts: int) -> List[Dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if max_prompts > 0:
        items = items[:max_prompts]
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            out.append({"id": f"prompt_{i:04d}", "prompt": it})
        else:
            out.append({
                "id": it.get("id", f"prompt_{i:04d}"),
                "prompt": it.get("prompt", ""),
                "meta": {k: v for k, v in it.items() if k not in ("id", "prompt")},
            })
    return out


def _result_to_record(prompt_item: Dict[str, Any], result, control: bool) -> Dict[str, Any]:
    return {
        "id": prompt_item["id"],
        "prompt": prompt_item["prompt"],
        "control": control,
        "generated_text": result.text,
        "n_tokens": len(result.token_ids),
        "triggered": result.triggered,
        "t_u": result.t_u,
        "t_star": result.t_star,
        "h_trajectory": [(s, round(h, 4)) for s, h in result.h_trajectory],
        "h_min": round(result.h_min, 4),
        "h_final": round(result.h_final, 4),
        "delta_h": round(result.delta_h, 4),
        "n_interventions": len(result.interventions),
        "interventions": result.interventions,
        "elapsed_sec": round(result.elapsed_sec, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RBCBF batch runner.")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "rbcbf_default.json"))
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--max_prompts", type=int, default=-1, help="-1 means all.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Only run the controlled pass per prompt.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not prompts_path.is_file():
        print(f"ERROR: prompts file not found: {prompts_path}", file=sys.stderr)
        return 2

    runner = RBCBFRunner.from_config(
        args.config, device=args.device, base_model=args.base_model
    )
    items = _load_prompts(prompts_path, args.max_prompts)
    if not items:
        print(f"ERROR: no prompts loaded from {prompts_path}", file=sys.stderr)
        return 2

    def _safe_generate(prompt: str, *, control: bool):
        """Run a single generate(); return GenerationResult, or None on failure."""
        try:
            return runner.generate(
                prompt,
                control=control,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [WARN] generation failed (control={control}): {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None

    # Atomic write: stage into .tmp, fsync, then rename. Crash-safe.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_written = 0
    n_failed = 0
    t_start = time.time()
    with tmp_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(items):
            prompt = item["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"  [SKIP] [{idx + 1}/{len(items)}] {item['id']}: empty prompt")
                n_failed += 1
                continue
            print(f"\n>>> [{idx + 1}/{len(items)}] {item['id']}")

            if not args.skip_baseline:
                base_res = _safe_generate(prompt, control=False)
                if base_res is not None:
                    fout.write(
                        json.dumps(_result_to_record(item, base_res, control=False),
                                   ensure_ascii=False) + "\n"
                    )
                    n_written += 1
                else:
                    n_failed += 1

            ctrl_res = _safe_generate(prompt, control=True)
            if ctrl_res is not None:
                fout.write(
                    json.dumps(_result_to_record(item, ctrl_res, control=True),
                               ensure_ascii=False) + "\n"
                )
                n_written += 1
            else:
                n_failed += 1
            fout.flush()

    tmp_path.replace(out_path)  # atomic move

    elapsed = time.time() - t_start
    print(f"\nDone. Wrote {n_written} records to {out_path} in {elapsed:.1f}s "
          f"({n_failed} failed).")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
