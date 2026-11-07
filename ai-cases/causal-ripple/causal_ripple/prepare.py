"""Prepare public generation inputs and separate official evaluation data."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from .adapters import adapt_ds1000
from .config import MODEL_ID, MODEL_REVISION
from .jsonl import write_jsonl

COUNTS = {"ds1000": 1000, "mbppplus": 378, "bigcodebench": 1140}
DS1000_COMMIT = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"
BCB_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
MBPPPLUS_VERSION = "v0.2.0"


def load_source(benchmark, source):
    if source is not None:
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if benchmark == "bigcodebench":
        from datasets import load_dataset

        return list(load_dataset("bigcode/bigcodebench", split="v0.1.4", revision=BCB_REVISION))
    url = (
        f"https://raw.githubusercontent.com/xlang-ai/DS-1000/{DS1000_COMMIT}/data/ds1000.jsonl.gz"
        if benchmark == "ds1000"
        else f"https://github.com/evalplus/mbppplus_release/releases/download/{MBPPPLUS_VERSION}/MbppPlus.jsonl.gz"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        content = gzip.decompress(response.read()).decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def public_row(benchmark, problem):
    if benchmark == "ds1000":
        row = adapt_ds1000(
            dict(
                task_id=f"DS1000/{problem['metadata']['problem_id']}",
                prompt=problem["prompt"],
                metadata=problem["metadata"],
            )
        )
        row["cleaning_mode"] = "corrected_adapted"
        return row
    prompt = (
        problem["prompt"].strip() + "\n" if benchmark == "mbppplus" else problem["complete_prompt"]
    )
    contract = dict(
        benchmark=benchmark,
        task_id=problem["task_id"],
        entry_point=problem["entry_point"],
        output_names=[],
        provided_names=[],
        examples=[],
        assertions=[],
        doctest_text="",
    )
    if benchmark == "mbppplus":
        contract["assertions"] = [
            line.strip() for line in prompt.splitlines() if line.strip().startswith("assert ")
        ]
    else:
        contract["doctest_text"] = prompt
    return dict(
        task_id=problem["task_id"],
        prompt=prompt,
        completion_prefix="",
        official_prefix=prompt,
        contract=contract,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        cleaning_mode="legacy_mbpp" if benchmark == "mbppplus" else "legacy_body",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=COUNTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Local official JSONL or JSONL.GZ file")
    parser.add_argument("--model", default=MODEL_ID, help="Checkpoint used to count prompt tokens")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    problems = load_source(args.benchmark, args.source)
    if len(problems) != COUNTS[args.benchmark]:
        raise ValueError("Official dataset has an unexpected task count")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    rows = [public_row(args.benchmark, problem) for problem in problems]
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Official dataset has duplicated task IDs")
    for row in rows:
        row["prompt_tokens"] = len(tokenizer(row["prompt"])["input_ids"])
    if args.benchmark == "ds1000":
        problems = [dict(problem, task_id=row["task_id"]) for problem, row in zip(problems, rows)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.output_dir / "public.jsonl"
    evaluation_path = args.output_dir / "problems.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if any(path.exists() for path in [public_path, evaluation_path, manifest_path]):
        raise FileExistsError("Use a new output directory to preserve prepared dataset snapshots")
    write_jsonl(public_path, rows)
    write_jsonl(evaluation_path, problems)
    manifest = dict(
        benchmark=args.benchmark,
        count=len(rows),
        model_revision=MODEL_REVISION,
        public_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
        evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        dataset_revision={
            "ds1000": DS1000_COMMIT,
            "mbppplus": MBPPPLUS_VERSION,
            "bigcodebench": BCB_REVISION,
        }[args.benchmark],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
