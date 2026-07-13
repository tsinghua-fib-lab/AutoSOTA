import argparse
import copy
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from rlie.utils.logger_config import LoggerConfig

# Support direct execution of this file during development.
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlie.datasets.data_loader import load_config
from rlie.core.task_runner import (
    _set_global_info_enabled,
    run_single_task,
    save_json,
)
from rlie.utils import concurrent_config, eval_utils


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
LOGGER = logging.getLogger("rlie")

LoggerConfig.setup_logger()


def run(config_path: str = "configs/default.yaml"):
    raw_config = load_config(config_path)
    evaluation_modes = eval_utils.parse_evaluation_modes(raw_config)
    LOGGER.info("Evaluation modes: %s", ", ".join(evaluation_modes))

    logging_cfg = raw_config.get("logging", {}) or {}
    base_output_dir = logging_cfg.get("output_dir", "outputs")
    timestamp_root = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(base_output_dir, timestamp_root)
    os.makedirs(base_output_dir, exist_ok=True)
    LOGGER.info("Top-level output directory: %s", base_output_dir)

    repeat = int(raw_config.get("repeat", 3))
    if repeat <= 0:
        raise ValueError("config.repeat must be a positive integer")

    data_roots = raw_config.get("data_roots")
    if data_roots:
        if not isinstance(data_roots, list) or not data_roots:
            raise ValueError("config.data_roots must be a non-empty list when provided")
        task_list = data_roots
    else:
        single_root = raw_config.get("data_root")
        if not single_root:
            raise ValueError("No data_root specified")
        task_list = [single_root]

    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base_seed = int(raw_config.get("data_seed", 42))

    for task_root in task_list:
        LOGGER.info("Starting task %s with %d repeats (parallel)", task_root, repeat)

        total_concurrent_limit = None
        try:
            llms_config = raw_config.get("llms", {})
            client_cfg = llms_config.get("client", {})
            api_keys = concurrent_config.extract_api_keys(client_cfg)
            num_keys = len(api_keys)

            eval_config = llms_config.get("evaluation", {})
            retry_overhead = float(eval_config.get("retry_overhead", 0.15))
            total_concurrent_limit = concurrent_config.calculate_total_concurrent_limit(
                eval_config,
                num_keys,
                retry_overhead,
            )
            LOGGER.info(
                "总并发预算: %d (基于 %d API keys, max_concurrent_per_key=%s, retry_overhead=%.0f%%)",
                total_concurrent_limit,
                num_keys,
                eval_config.get("max_concurrent_per_key"),
                retry_overhead * 100,
            )
        except (ValueError, KeyError) as exc:
            LOGGER.warning("无法计算并发预算，将使用配置中的原始值: %s", exc)

        futures = {}
        max_workers = max(1, min(repeat, (os.cpu_count() or repeat)))
        suppress_logs = repeat > 1
        if suppress_logs:
            _set_global_info_enabled(False)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for repeat_index in range(repeat):
                task_config = copy.deepcopy(raw_config)
                task_config["data_root"] = task_root
                task_config["data_roots"] = None
                task_config["repeat"] = repeat
                task_config["data_seed"] = base_seed + repeat_index

                if total_concurrent_limit is not None:
                    per_repeat_budget = max(1, total_concurrent_limit // repeat)
                    task_config["_concurrent_budget"] = per_repeat_budget
                    LOGGER.debug(
                        "Repeat %d/%d 分配并发预算: %d",
                        repeat_index + 1,
                        repeat,
                        per_repeat_budget,
                    )

                futures[
                    executor.submit(
                        run_single_task,
                        task_config,
                        evaluation_modes,
                        task_root,
                        base_output_dir,
                        repeat_index,
                        repeat,
                        config_path,
                    )
                ] = repeat_index + 1

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    summaries = future.result()
                except Exception as exc:  # pragma: no cover
                    LOGGER.error("Task %s repeat %d failed: %s", task_root, idx, exc)
                    continue
                for mode, metrics in summaries.items():
                    bucket = aggregated[task_root][mode]
                    for metric_name, value in metrics.items():
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            continue
                        bucket[metric_name].append(value)
        if suppress_logs:
            _set_global_info_enabled(True)
        LOGGER.info("[%s] All %d repeats completed", task_root, repeat)

    for task_root, mode_dict in aggregated.items():
        if not mode_dict:
            continue
        task_id = task_root.replace("/", "_") if task_root else "default"
        summary_output = {}
        for mode, metrics in mode_dict.items():
            summary_output[mode] = {}
            for metric_name, values in metrics.items():
                if not values:
                    continue
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary_output[mode][metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "formatted": f"{mean_val:.4f}±{std_val:.4f}",
                    "values": values,
                }
        if not summary_output:
            continue

        summary_path = os.path.join(base_output_dir, f"{task_id}_summary.json")
        save_json(summary_path, summary_output)
        LOGGER.info("[%s] Summary saved to %s", task_root, summary_path)
        for mode, metrics in summary_output.items():
            acc = metrics.get("accuracy")
            f1 = metrics.get("f1")
            if acc or f1:
                LOGGER.info(
                    "[%s][%s] accuracy=%s | f1=%s",
                    task_root,
                    mode,
                    acc["formatted"] if acc else "n/a",
                    f1["formatted"] if f1 else "n/a",
                )


def main():
    parser = argparse.ArgumentParser(description="Run RLIE rule learning experiments.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the experiment configuration file.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
