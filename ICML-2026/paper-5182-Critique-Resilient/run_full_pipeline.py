import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple


def _append_args(cmd: List[str], flag: str, values: Iterable[str] | None) -> None:
    if not values:
        return
    cmd.append(flag)
    cmd.extend(values)


def _run_step(name: str, cmd: List[str], timeout_seconds: int) -> int:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n==> {name}")
    print(rendered)
    try:
        subprocess.run(cmd, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"{name} timed out after {timeout_seconds} seconds.", file=sys.stderr)
        return 124
    except subprocess.CalledProcessError as exc:
        print(f"{name} failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    return 0


def _terminate_processes(processes: Iterable[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _run_steps_parallel(steps: List[Tuple[str, List[str]]], timeout_seconds: int) -> int:
    processes: List[Tuple[str, subprocess.Popen, float]] = []
    timed_out_steps: List[str] = []
    for name, cmd in steps:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n==> {name}")
        print(rendered)
        try:
            proc = subprocess.Popen(cmd)
        except OSError as exc:
            print(f"{name} failed to start: {exc}.", file=sys.stderr)
            _terminate_processes([proc for _, proc, _ in processes])
            return 1
        processes.append((name, proc, time.monotonic()))

    while processes:
        for name, proc, start_time in list(processes):
            return_code = proc.poll()
            if return_code is not None:
                processes.remove((name, proc, start_time))
                if return_code != 0:
                    print(
                        f"{name} failed with exit code {return_code}.",
                        file=sys.stderr,
                    )
                    _terminate_processes([proc for _, proc, _ in processes])
                    return return_code
                continue

            if time.monotonic() - start_time > timeout_seconds:
                print(
                    f"{name} timed out after {timeout_seconds} seconds.",
                    file=sys.stderr,
                )
                _terminate_processes([proc])
                processes.remove((name, proc, start_time))
                timed_out_steps.append(name)
                continue

        time.sleep(0.1)

    if timed_out_steps:
        print(
            "\nPipeline completed with timeouts in parallel steps:",
            file=sys.stderr,
        )
        for name in timed_out_steps:
            print(f"  - {name}", file=sys.stderr)
        return 124
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark pipeline with per-step timeouts."
    )
    parser.add_argument("--runs-file", type=Path, default=Path("configs/runs.json"))
    parser.add_argument("--topic-info-file", type=Path, default=Path("configs/topic_info.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-batch", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--timeout-hours", type=float, default=3.0)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-question-attempts", type=int, default=None)
    parser.add_argument("--debate-rounds", type=int, default=5)
    parser.add_argument("--allow-self-answering", action="store_true")
    parser.add_argument("--rerun-answer-failures", action="store_true")
    parser.add_argument("--self-improve-critiques", action="store_true")
    parser.add_argument("--no-allow-concede", action="store_true")
    parser.add_argument("--overwrite-judgments", action="store_true")
    parser.add_argument("--allow-no-debate", action="store_true")
    parser.add_argument("--force-correct-critiques", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--critique-modes", nargs="*", default=["contradictor", "evaluator"])
    parser.add_argument("--benchmark-models", nargs="*")
    parser.add_argument("--answer-models", nargs="*")
    parser.add_argument("--critique-models", nargs="*")
    parser.add_argument("--judge-models", nargs="*")
    parser.add_argument(
        "--run-parallel",
        action="store_true",
        help="Run all pipeline steps concurrently instead of sequentially.",
    )
    args = parser.parse_args()

    timeout_seconds = int(args.timeout_hours * 3600)
    base_python = sys.executable

    steps: List[Tuple[str, List[str]]] = []

    benchmark_cmd = [
        base_python,
        "generate_benchmark.py",
        "--runs-file",
        str(args.runs_file),
        "--topic-info-file",
        str(args.topic_info_file),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.benchmark_dir),
        "--log-level",
        args.log_level,
    ]
    if args.limit is not None:
        benchmark_cmd.extend(["--limit", str(args.limit)])
    if args.disable_batch:
        benchmark_cmd.append("--disable-batch")
    if args.max_rounds is not None:
        benchmark_cmd.extend(["--max-rounds", str(args.max_rounds)])
    if args.max_question_attempts is not None:
        benchmark_cmd.extend(["--max-question-attempts", str(args.max_question_attempts)])
    _append_args(benchmark_cmd, "--models", args.benchmark_models)
    steps.append(("Generate benchmark questions", benchmark_cmd))

    answers_cmd = [
        base_python,
        "generate_answers.py",
        "--config",
        str(args.config),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--output-dir",
        str(args.answers_dir),
        "--log-level",
        args.log_level,
    ]
    if args.limit is not None:
        answers_cmd.extend(["--limit", str(args.limit)])
    if args.disable_batch:
        answers_cmd.append("--disable-batch")
    if args.max_rounds is not None:
        answers_cmd.extend(["--max-rounds", str(args.max_rounds)])
    if args.allow_self_answering:
        answers_cmd.append("--allow-self-answering")
    if args.rerun_answer_failures:
        answers_cmd.append("--rerun-failures")
    _append_args(answers_cmd, "--models", args.answer_models)
    steps.append(("Generate answers", answers_cmd))

    for mode in args.critique_modes:
        critiques_cmd = [
            base_python,
            "generate_critiques.py",
            "--mode",
            mode,
            "--config",
            str(args.config),
            "--benchmark-dir",
            str(args.benchmark_dir),
            "--answers-dir",
            str(args.answers_dir),
            "--output-dir",
            str(args.critiques_dir),
            "--log-level",
            args.log_level,
        ]
        if args.limit is not None:
            critiques_cmd.extend(["--limit", str(args.limit)])
        if args.disable_batch:
            critiques_cmd.append("--disable-batch")
        if args.max_rounds is not None:
            critiques_cmd.extend(["--max-rounds", str(args.max_rounds)])
        if args.self_improve_critiques:
            critiques_cmd.append("--self-improve")
        _append_args(critiques_cmd, "--models", args.critique_models)
        steps.append((f"Generate critiques ({mode})", critiques_cmd))

    debate_illposed_cmd = [
        base_python,
        "debate.py",
        "--mode",
        "ill-posed",
        "--config",
        str(args.config),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--answers-dir",
        str(args.answers_dir),
        "--critiques-dir",
        str(args.critiques_dir),
        "--output-dir",
        str(args.debates_dir),
        "--rounds",
        str(args.debate_rounds),
        "--log-level",
        args.log_level,
    ]
    if args.limit is not None:
        debate_illposed_cmd.extend(["--limit", str(args.limit)])
    if args.no_allow_concede:
        debate_illposed_cmd.append("--no-allow-concede")
    steps.append(("Debate ill-posed claims", debate_illposed_cmd))

    debate_critique_cmd = [
        base_python,
        "debate.py",
        "--mode",
        "critique",
        "--config",
        str(args.config),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--answers-dir",
        str(args.answers_dir),
        "--critiques-dir",
        str(args.critiques_dir),
        "--output-dir",
        str(args.debates_dir),
        "--rounds",
        str(args.debate_rounds),
        "--log-level",
        args.log_level,
    ]
    if args.limit is not None:
        debate_critique_cmd.extend(["--limit", str(args.limit)])
    if args.no_allow_concede:
        debate_critique_cmd.append("--no-allow-concede")
    steps.append(("Debate critiques", debate_critique_cmd))

    judge_cmd = [
        base_python,
        "automated_judge.py",
        "--mode",
        "all",
        "--config",
        str(args.config),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--answers-dir",
        str(args.answers_dir),
        "--critiques-dir",
        str(args.critiques_dir),
        "--debates-dir",
        str(args.debates_dir),
        "--output-dir",
        str(args.automated_dir),
        "--log-level",
        args.log_level,
    ]
    if args.limit is not None:
        judge_cmd.extend(["--limit", str(args.limit)])
    if args.disable_batch:
        judge_cmd.append("--disable-batch")
    if args.batch_size is not None:
        judge_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.overwrite_judgments:
        judge_cmd.append("--overwrite")
    if args.allow_no_debate:
        judge_cmd.append("--allow-no-debate")
    if args.force_correct_critiques:
        judge_cmd.append("--force-correct-critiques")
    _append_args(judge_cmd, "--models", args.judge_models)
    steps.append(("Automated judging", judge_cmd))

    if args.run_parallel:
        code = _run_steps_parallel(steps, timeout_seconds)
        if code != 0:
            return code
    else:
        timed_out_steps: List[str] = []
        for name, cmd in steps:
            code = _run_step(name, cmd, timeout_seconds)
            if code == 124:
                timed_out_steps.append(name)
                continue
            if code != 0:
                return code

        if timed_out_steps:
            print("\nPipeline completed with timeouts:", file=sys.stderr)
            for name in timed_out_steps:
                print(f"  - {name}", file=sys.stderr)
            return 124

    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
