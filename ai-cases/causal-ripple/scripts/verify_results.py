"""Validate bundled results and selected source integrity without model dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/paper_results.json").read_text())
    outcomes = json.loads((ROOT / "results/outcomes.json").read_text())
    improvements = []
    for benchmark, n in report["tasks"].items():
        reference = outcomes[benchmark]["ripple"]
        for method, values in outcomes[benchmark].items():
            assert len(values) == n, (benchmark, method, "incomplete")
            assert values.keys() == reference.keys(), (benchmark, method, "task IDs")
            assert all(type(value) is bool for value in values.values())
            if method in report["successful_tasks"]:
                assert sum(values.values()) == report["successful_tasks"][method][benchmark]
        counts = report["successful_tasks"]
        best = max(v[benchmark] for k, v in counts.items()
                   if k != "ripple" and benchmark in v)
        improvement = 100 * (counts["ripple"][benchmark] - best) / best
        improvements.append(improvement)
        row = next(r for r in report["main_comparison"] if r["benchmark"] == benchmark)
        assert row["baseline_correct"] == best
        assert row["ripple"] == counts["ripple"][benchmark]
        assert row["additional_correct"] == row["ripple"] - best
        assert abs(row["relative_gain_percent"] - improvement) < 1e-10
        cascade = report["cascade"][benchmark]
        key = "cascade_input" if benchmark == "ds1000" else "ripple"
        assert sum(outcomes[benchmark][key].values()) == cascade["input_passed"]
        assert sum(outcomes[benchmark]["visible_cascade"].values()) == cascade["cascade_passed"]
    mean = sum(improvements) / len(improvements)
    assert abs(mean - report["macro_average_relative_gain_percent"]) < 1e-10
    assert f"{mean:.1f}" == "5.0"
    for point in json.loads((ROOT / "results/analysis_results.json").read_text())["points"]:
        assert point["correct"] == report["successful_tasks"][point["method"]][point["benchmark"]]
    hashes = json.loads((ROOT / "evidence/selected_code_sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print("PASS: complete task vectors, result counts, baseline comparisons, Cascade inputs,")
    print("      figure counts, and selected runtime source hashes.")
    print(f"Macro-average relative improvement: {mean:.6f}% (reported: {mean:.1f}%).")


if __name__ == "__main__":
    main()
