"""Prepare and optionally submit SWE-Bench predictions for JiSi outputs."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import jsonlines


def extract_diff(response: str | None) -> str:
    """Extract a patch from tagged, fenced, or plain model responses."""
    if response is None:
        return ""

    diff_matches: list[str] = []
    other_matches: list[str] = []

    tag_pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in tag_pattern.findall(response):
        if code.lower() in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)

    fence_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in fence_pattern.findall(response):
        if (code or "").lower() in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)

    if diff_matches:
        return diff_matches[0].strip()
    if other_matches:
        return other_matches[0].strip()
    return response.split("</s>")[0].strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_index_map(index_map_path: Path | None, benchmark_file: Path | None) -> dict[int, str]:
    """Load a JiSi-local index to SWE-Bench instance_id mapping."""
    if index_map_path is not None:
        raw_map = load_json(index_map_path)
        return {int(key): str(value) for key, value in raw_map.items()}

    if benchmark_file is not None:
        raw_data = load_json(benchmark_file)
        if isinstance(raw_data, dict):
            records = raw_data.get("records", [])
        elif isinstance(raw_data, list):
            records = raw_data
        else:
            records = []
        return {
            int(record["index"]): str(record["instance_id"])
            for record in records
            if "index" in record and "instance_id" in record
        }

    return {}


def load_result_rows(result_path: Path) -> list[dict[str, Any]]:
    with jsonlines.Reader(result_path.open("r", encoding="utf-8-sig")) as reader:
        return list(reader)


def build_predictions(
    rows: list[dict[str, Any]],
    index_map: dict[int, str],
    model_name: str,
    dataset_keyword: str,
    response_key: str,
) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    missing_indices: list[int] = []

    for row in rows:
        dataset = str(row.get("dataset", "")).lower()
        if dataset_keyword.lower() not in dataset:
            continue

        index = int(row["index"])
        instance_id = row.get("instance_id") or index_map.get(index)
        if not instance_id:
            missing_indices.append(index)
            continue

        predictions[str(instance_id)] = {
            "model_patch": extract_diff(row.get(response_key)),
            "model_name_or_path": model_name,
        }

    if missing_indices:
        preview = ", ".join(str(idx) for idx in missing_indices[:10])
        raise ValueError(
            "Missing SWE-Bench instance_id mapping for "
            f"{len(missing_indices)} rows. First missing indices: {preview}"
        )

    if not predictions:
        raise ValueError(f"No rows matched dataset keyword: {dataset_keyword}")

    return predictions


def submit_predictions(
    predictions_path: Path,
    run_id: str,
    api_key: str,
    benchmark: str,
    split: str,
) -> None:
    env = os.environ.copy()
    env["SWEBENCH_API_KEY"] = api_key
    command = [
        "sb-cli",
        "submit",
        benchmark,
        split,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
    ]
    subprocess.run(command, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert JiSi result.jsonl SWE-Bench rows to the sb-cli prediction "
            "format and optionally submit them for SWE-Bench verification."
        )
    )
    parser.add_argument("--res_path", required=True, type=Path, help="Path to JiSi result.jsonl")
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Path for the SWE-Bench prediction JSON. Defaults to <result_dir>/swe_result.json.",
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        default=None,
        help="Optional JSON mapping from JiSi-local SWE indices to SWE-Bench instance ids",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Optional benchmark_bank SWE-Bench JSON file with records containing index and instance_id",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model or run name written as model_name_or_path. Defaults to --run-id or the result directory name.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="SWE-Bench submission run id. Defaults to --model-name or the result directory name.",
    )
    parser.add_argument("--dataset-keyword", default="swe", help="Substring used to select SWE-Bench rows")
    parser.add_argument("--response-key", default="response", help="Field containing the generated patch response")
    parser.add_argument("--submit", action="store_true", help="Submit the prediction JSON with sb-cli")
    parser.add_argument(
        "--api-key",
        default=None,
        help="SWE-Bench API key. If omitted, SWEBENCH_API_KEY is read from the environment.",
    )
    parser.add_argument("--benchmark", default="swe-bench_verified", help="sb-cli benchmark name")
    parser.add_argument("--split", default="test", help="sb-cli split name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_dir_name = args.res_path.resolve().parent.name
    run_id = args.run_id or args.model_name or result_dir_name
    model_name = args.model_name or run_id
    output_path = args.output or (args.res_path.resolve().parent / "swe_result.json")

    index_map = load_index_map(args.index_map, args.benchmark_file)
    rows = load_result_rows(args.res_path)
    predictions = build_predictions(
        rows=rows,
        index_map=index_map,
        model_name=model_name,
        dataset_keyword=args.dataset_keyword,
        response_key=args.response_key,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(predictions)} SWE-Bench predictions to {output_path}")

    if args.submit:
        api_key = args.api_key or os.environ.get("SWEBENCH_API_KEY")
        if not api_key:
            raise ValueError("Set --api-key or SWEBENCH_API_KEY before using --submit")
        submit_predictions(
            predictions_path=output_path,
            run_id=run_id,
            api_key=api_key,
            benchmark=args.benchmark,
            split=args.split,
        )


if __name__ == "__main__":
    main()
