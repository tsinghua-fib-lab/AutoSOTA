"""Post-evaluate JiSi aggregation outputs with the bundled evaluators."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines
from tqdm import tqdm

from evaluation.factory import EvaluatorFactory


DEFAULT_DATASETS = [
    "aime",
    "gpqa",
    "livemathbench",
    "mmlupro",
    "livecodebench",
    "hle",
    "simpleqa",
    "arenahard",
]


def wrap_evaluate(evaluate_fn):
    def wrap(*args, **kwargs):
        run_id = args[0]
        remain_args = args[1:]
        return run_id, evaluate_fn(*remain_args, **kwargs)

    return wrap


def _parse_datasets(value: str) -> list[str]:
    if value.strip().lower() == "paper":
        return DEFAULT_DATASETS
    return [dataset.strip() for dataset in value.split(",") if dataset.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate JiSi aggregation result.jsonl files")
    parser.add_argument("--res_path", type=str, required=True, help="Path to JiSi result.jsonl")
    parser.add_argument(
        "--datasets",
        type=str,
        default="paper",
        help="Comma-separated dataset list, or 'paper' for the default JiSi paper set",
    )
    parser.add_argument("--max-process", type=int, default=16, help="Maximum evaluator workers")
    args = parser.parse_args()

    res_path = args.res_path
    res_dir = os.path.dirname(os.path.abspath(res_path))
    eval_datasets = _parse_datasets(args.datasets)
    max_process = max(1, args.max_process)

    with jsonlines.Reader(open(res_path, "r", encoding="utf-8")) as reader:
        res_list = list(reader)

    for dataset in eval_datasets:
        output_res_path = os.path.join(res_dir, f"{dataset}_result.jsonl")
        output_summary_path = os.path.join(res_dir, f"{dataset}_summary.json")
        final_res = []

        if os.path.exists(output_res_path):
            with jsonlines.Reader(open(output_res_path, "r", encoding="utf-8")) as reader:
                done_list = list(reader)
            done_run_id = [q["run_id"] for q in done_list]
        else:
            done_list = []
            done_run_id = []

        if done_run_id:
            have_done_corr = sum(q["is_correct"] for q in done_list)
            final_res.extend([False] * int(len(done_list) - have_done_corr) + [True] * int(have_done_corr))

        print(f"Begin to evaluate {dataset}...")
        res_list_dataset = [r for r in res_list if r["dataset"] == dataset]
        if not res_list_dataset:
            print(f"Skip {dataset}: no matching rows in {res_path}")
            continue

        evaluator = EvaluatorFactory().get_evaluator(dataset)
        eval_data = evaluator.load_data(res_list_dataset[0]["split"])
        batch_index = list(range(0, len(res_list_dataset), max_process)) + [len(res_list_dataset)]
        cnt = len(done_run_id)

        for i in tqdm(range(len(batch_index) - 1)):
            batch_index_range = list(range(batch_index[i], batch_index[i + 1]))
            pending_indices = [idx for idx in batch_index_range if idx not in done_run_id]
            if not pending_indices:
                continue

            tasks = [
                [idx, eval_data[res_list_dataset[idx]["index"] - 1], res_list_dataset[idx]["response"]]
                for idx in pending_indices
            ]
            with ThreadPoolExecutor(max_workers=min(len(tasks), max_process)) as executor:
                futures = [executor.submit(wrap_evaluate(evaluator.evaluate), *task) for task in tasks]
                response_pred_dict_list = [future.result() for future in as_completed(futures)]

            response_pred_dict_list = sorted(response_pred_dict_list, key=lambda x: x[0])
            cnt += len(pending_indices)

            for run_id, result_dict_i in response_pred_dict_list:
                log_dict = {
                    "query": res_list_dataset[run_id]["query"],
                    "dataset": res_list_dataset[run_id]["dataset"],
                    "index": res_list_dataset[run_id]["index"],
                    "split": res_list_dataset[run_id]["split"],
                    "response": res_list_dataset[run_id]["response"],
                    "run_id": run_id,
                    "gt": res_list_dataset[run_id]["gt"],
                }
                log_dict.update(result_dict_i)
                final_res.append(result_dict_i["is_correct"])
                summary = {
                    "corr": sum(final_res),
                    "wrong": len(final_res) - sum(final_res),
                    "acc": sum(final_res) / len(final_res),
                    "schedule": f"{cnt}/{len(res_list_dataset)}",
                }
                with jsonlines.Writer(open(output_res_path, "a", encoding="utf-8")) as writer:
                    writer.write(log_dict)
                with open(output_summary_path, "w", encoding="utf-8") as fo:
                    fo.write(json.dumps(summary, indent=2, ensure_ascii=False))

        if final_res:
            print(f"{dataset} acc: {sum(final_res) / len(final_res):.4f}")


if __name__ == "__main__":
    main()
