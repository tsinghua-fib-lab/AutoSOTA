"""Evaluate complete code task sets with the pinned official benchmark tools."""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .jsonl import file_digest, read_jsonl


def score_task(benchmark, problem, sample, evaluator_dir=None, gt_time=None, oracle=None):
    start = time.perf_counter()
    if sample["generation_status"] != "generated":
        return dict(task_id=sample["task_id"], passed=False, result="context_limit")
    program = sample["program"]
    if benchmark == "ds1000":
        sys.path.insert(0, str(evaluator_dir))
        import execution

        harness = (
            problem["code_context"] + "\n" + f"code = {program!r}\n" + "test_execution(code)\n"
        )
        if "test_string(" in problem["code_context"]:
            harness += "test_string(code)\n"
        result = execution.check_correctness(
            harness, timeout=120, completion_id=problem["metadata"]["problem_id"]
        )
        row = dict(
            task_id=sample["task_id"], passed=bool(result["passed"]), result=result["result"]
        )
    elif benchmark == "bigcodebench":
        from bigcodebench.eval import untrusted_check
        from bigcodebench.sanitize import sanitize

        solution = sanitize(program, problem["entry_point"])
        calibrated = problem["code_prompt"] + "\n    pass\n" + solution
        status, details = untrusted_check(
            calibrated,
            problem["test"],
            problem["entry_point"],
            30 * 1024,
            30 * 1024,
            10,
            1,
            gt_time if gt_time is not None else 20,
        )
        row = dict(
            task_id=sample["task_id"],
            passed=status == "pass",
            status=status,
            details=details,
            solution_sha256=hashlib.sha256(solution.encode()).hexdigest(),
        )
    else:
        from evalplus.evaluate import check_correctness
        from evalplus.sanitize import sanitize

        solution = sanitize(program, entrypoint=problem["entry_point"])
        result = check_correctness(
            "mbpp", 0, problem, solution, oracle, fast_check=False, identifier=sample["task_id"]
        )
        row = dict(
            task_id=sample["task_id"],
            passed=result["base"][0] == result["plus"][0] == "pass",
            base_passed=result["base"][0] == "pass",
            base_status=result["base"][0],
            plus_status=result["plus"][0],
        )
    row["elapsed_seconds"] = time.perf_counter() - start
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=["ds1000", "mbppplus", "bigcodebench"], required=True
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, help="Pinned DS-1000 checkout")
    parser.add_argument("--groundtruth", type=Path, help="BigCodeBench ground-truth timing JSONL")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.benchmark == "ds1000" and args.evaluator_dir is None:
        parser.error("DS-1000 requires --evaluator-dir")
    if args.benchmark == "bigcodebench" and args.groundtruth is None:
        parser.error("BigCodeBench requires --groundtruth")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if manifest["benchmark"] != args.benchmark:
        raise ValueError("Prepared data belong to a different benchmark")
    problems_path = args.data_dir / "problems.jsonl"
    public_path = args.data_dir / "public.jsonl"
    if (
        file_digest(problems_path) != manifest["evaluation_sha256"]
        or file_digest(public_path) != manifest["public_sha256"]
    ):
        raise ValueError("Prepared data have changed")
    problems = {row["task_id"]: row for row in read_jsonl(problems_path)}
    public = {row["task_id"]: row for row in read_jsonl(public_path)}
    rows = read_jsonl(args.samples)
    samples = {row["task_id"]: row for row in rows}
    if len(samples) != len(rows) or samples.keys() != problems.keys():
        raise ValueError("Scoring requires exactly one completion for every official task")
    if any(row["prompt_sha256"] != public[row["task_id"]]["prompt_sha256"] for row in rows):
        raise ValueError("Generated samples do not match the prepared prompts")
    timings = {}
    oracle = {}
    if args.benchmark == "bigcodebench":
        timings = {row["task_id"]: row["time"] for row in read_jsonl(args.groundtruth)}
        if timings.keys() != problems.keys():
            raise ValueError("Calibration timing cache does not cover the official task set")
    if args.benchmark == "mbppplus":
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import get_groundtruth

        problems = get_mbpp_plus(version="v0.2.0")
        if problems.keys() != samples.keys():
            raise ValueError("EvalPlus task set differs from the prepared data")
        oracle = get_groundtruth(
            problems, get_mbpp_plus_hash(version="v0.2.0"), MBPP_OUTPUT_NOT_NONE_TASKS
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        benchmark=args.benchmark,
        samples_sha256=file_digest(args.samples),
        dataset_sha256=manifest["evaluation_sha256"],
        scorer_sha256=file_digest(Path(__file__)),
        calibration_sha256=file_digest(args.groundtruth) if args.groundtruth else None,
    )
    sidecar = args.output.with_suffix(".manifest.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != protocol:
            raise ValueError("Resume scoring inputs differ from the original manifest")
    else:
        if args.output.exists():
            raise ValueError("Existing score file lacks a manifest")
        sidecar.write_text(json.dumps(protocol, indent=2) + "\n")
    completed = read_jsonl(args.output) if args.output.exists() else []
    seen = {row["task_id"] for row in completed}
    if len(seen) != len(completed) or not seen <= samples.keys():
        raise ValueError("Score file has duplicated or unknown task IDs")
    pending = [task for task in samples if task not in seen]
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        args.output.open("a", encoding="utf-8", buffering=1) as stream,
    ):
        futures = [
            pool.submit(
                score_task,
                args.benchmark,
                problems[task],
                samples[task],
                args.evaluator_dir,
                timings.get(task),
                oracle.get(task),
            )
            for task in pending
        ]
        for future in as_completed(futures):
            row = future.result()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed.append(row)
            if len(completed) % 10 == 0 or len(completed) == len(samples):
                print(f"scored={len(completed)}/{len(samples)}", flush=True)
    summary = dict(
        benchmark=args.benchmark,
        n=len(completed),
        expected_n=len(samples),
        complete=len(completed) == len(samples),
        passed=sum(bool(row["passed"]) for row in completed),
    )
    summary["pass_at_1"] = summary["passed"] / summary["n"]
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
