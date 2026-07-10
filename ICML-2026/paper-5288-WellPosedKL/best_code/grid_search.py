#!/usr/bin/env python3
"""
Grid search harness for Paper 5288 SOTA optimization (CODE-03).

Reads a JSON config, iterates over parameter grids, runs eval_wkl.py,
parses results, and reports top configurations.
"""
import argparse
import json
import subprocess
import sys
from itertools import product


def parse_result_line(line):
    """Parse RESULT: KEY=VALUE lines from eval output."""
    if line.startswith("RESULT:"):
        parts = line.split("=", 1)
        if len(parts) == 2:
            key = parts[0].replace("RESULT:", "").strip()
            try:
                value = float(parts[1].strip())
            except ValueError:
                value = parts[1].strip()
            return key, value
    return None, None


def run_eval(params, eval_script="python3 eval_wkl.py", timeout=60):
    """Run eval_wkl.py with given parameters and parse results."""
    args = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])

    cmd = eval_script.split() + args + ["--json-output"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"_error": f"exit={result.returncode}", "_stderr": result.stderr[:500]}

        # Try JSON parse first
        try:
            data = json.loads(result.stdout.strip())
            data["_stable_wkl"] = data.get("WKL_Spectral_Radius", 999) < 1.0
            data["_stable_kl"] = data.get("KL_Spectral_Radius", 999) < 1.0
            return data
        except json.JSONDecodeError:
            # Fall back to line parsing
            data = {}
            for line in result.stdout.split("\n"):
                key, value = parse_result_line(line)
                if key is not None:
                    data[key] = value
            data["_stable_wkl"] = data.get("WKL_Spectral_Radius", 999) < 1.0
            data["_stable_kl"] = data.get("KL_Spectral_Radius", 999) < 1.0
            return data
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}
    except Exception as e:
        return {"_error": str(e)}


def grid_search(config):
    """Run a grid search over parameter combinations."""
    param_names = config["params"]
    param_values = config["values"]
    eval_script = config.get("eval_script", "python3 eval_wkl.py")
    primary_metric = config.get("primary_metric", "WKL_Frobenius_Norm")

    # Build all combinations
    keys = list(param_names)
    value_lists = [param_values[k] for k in keys]
    combinations = list(product(*value_lists))

    print(f"Grid search: {len(combinations)} combinations over {keys}")
    print(f"Primary metric: {primary_metric}")

    results = []
    stable_count = 0
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        data = run_eval(params, eval_script)

        if "_error" in data:
            print(f"  [{i+1}/{len(combinations)}] ERROR: {data['_error']}", file=sys.stderr)
            continue

        wkl_norm = data.get(primary_metric)
        wkl_stable = data.get("_stable_wkl", False)
        kl_norm = data.get("KL_Baseline_Frobenius_Norm")

        if wkl_norm is not None:
            gap = wkl_norm - kl_norm if kl_norm is not None else None
            entry = {
                "params": params,
                "WKL_Frobenius_Norm": wkl_norm,
                "KL_Baseline_Frobenius_Norm": kl_norm,
                "WKL_KL_Gap": gap,
                "WKL_Spectral_Radius": data.get("WKL_Spectral_Radius"),
                "WKL_Stability_Margin": data.get("WKL_Stability_Margin"),
                "stable_wkl": wkl_stable,
            }
            results.append(entry)
            if wkl_stable:
                stable_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(combinations)}] processed, {stable_count} stable so far")

    print(f"\nDone. {len(results)} evaluated, {stable_count} stable.")

    # Sort by primary metric (higher is better)
    results.sort(key=lambda r: r.get(primary_metric, 0), reverse=True)

    # Report top-10 stable
    stable_results = [r for r in results if r["stable_wkl"]]
    unstable_results = [r for r in results if not r["stable_wkl"]]

    print(f"\n=== Top 10 Stable Configurations ===")
    for i, r in enumerate(stable_results[:10]):
        print(f"  #{i+1}: WKL={r['WKL_Frobenius_Norm']:.6f} KL={r['KL_Baseline_Frobenius_Norm']:.6f} "
              f"Gap={r['WKL_KL_Gap']:.6f} Margin={r['WKL_Stability_Margin']:.6f}")
        print(f"       params={r['params']}")

    if unstable_results:
        print(f"\n=== Top 5 Unstable (rejected) ===")
        for i, r in enumerate(unstable_results[:5]):
            print(f"  WKL={r['WKL_Frobenius_Norm']:.6f} SR={r['WKL_Spectral_Radius']:.6f}")
            print(f"  params={r['params']}")

    return {
        "total_combinations": len(combinations),
        "total_evaluated": len(results),
        "stable_count": stable_count,
        "best_stable": stable_results[0] if stable_results else None,
        "top10_stable": stable_results[:10],
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search harness for Paper 5288")
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config file with params and values")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    summary = grid_search(config)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Return best stable result info for the caller
    if summary["best_stable"]:
        best = summary["best_stable"]
        print(f"\nBEST_STABLE: WKL={best['WKL_Frobenius_Norm']:.6f} params={best['params']}")
    else:
        print("\nBEST_STABLE: NONE (no stable configurations found)")


if __name__ == "__main__":
    main()
