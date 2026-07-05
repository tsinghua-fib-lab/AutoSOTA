from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch_frame
from relbench.base import TaskType

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from proxies.grad_norm import GradNormEvaluator
from proxies.nas_wot import NWTEvaluator
from proxies.ntk_condition_num import NTKCondNumEvaluator
from proxies.ntk_trace import NTKTraceEvaluator
from proxies.ntk_trace_approx import NTKTraceApproxEvaluator
from proxies.prune_fisher import FisherEvaluator
from proxies.prune_grasp import GraspEvaluator
from proxies.prune_snip import SnipEvaluator
from proxies.prune_synflow import SynFlowEvaluator
from search_space.block_mixed import PTNASBlockMixed
from utils.resource import get_text_embedder_cfg
from utils.table_data import TableData

DEFAULT_SPACE_FILE = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "blockmixed.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "proxy_score" / "classic"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "batch_cache"

PROXY_TO_EVALUATOR = {
    "GradNorm": GradNormEvaluator,
    "NASWOT": NWTEvaluator,
    "NTKCond": NTKCondNumEvaluator,
    "NTKTrace": NTKTraceEvaluator,
    "NTKTrAppx": NTKTraceApproxEvaluator,
    "Fisher": FisherEvaluator,
    "GraSP": GraspEvaluator,
    "SNIP": SnipEvaluator,
    "SynFlow": SynFlowEvaluator,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score space_diverse with classic baseline proxies.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--space_file", type=Path, default=DEFAULT_SPACE_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--proxies", type=str, default=",".join(PROXY_TO_EVALUATOR.keys()))
    parser.add_argument("--max_models", type=int, default=None)
    return parser.parse_args()


def load_space_rows(path: Path, max_models: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if max_models is not None:
        rows = rows[:max_models]
    return rows


def load_dataset_context(dataset_name: str):
    data_dir = PROJECT_ROOT / "datasets" / "fit-medium-table" / dataset_name
    table_data = TableData.load_from_dir(str(data_dir))
    if not table_data.is_materialize:
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    return table_data, stype_encoder_dict


def build_or_load_cached_batch(
    dataset_name: str,
    table_data: TableData,
    stype_encoder_dict,
    channels: int,
    batch_size: int,
    cache_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_name}_c{channels}_b{batch_size}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["batch_x"], payload["batch_y"]

    loader = torch_frame.data.DataLoader(
        table_data.train_tf,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    batch = next(iter(loader)).to("cpu")
    ref_model = PTNASBlockMixed(
        channels=channels,
        out_channels=1,
        num_layers=2,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_specs=[("mlp", channels, "layer_norm", "relu", 0.1, "residual", 0)] * 2,
    ).to("cpu")
    ref_model.eval()
    with torch.no_grad():
        encoded_x, _ = ref_model.encoder(batch)
        batch_x = encoded_x.reshape(encoded_x.size(0), -1).detach().cpu()
        batch_y = batch.y.float().detach().cpu()
    torch.save({"batch_x": batch_x, "batch_y": batch_y}, cache_path)
    return batch_x, batch_y


def task_space_name(task_type: TaskType) -> str:
    if task_type == TaskType.REGRESSION:
        return "block_mixed:regression"
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "block_mixed:binary"
    raise ValueError(f"Unsupported task type: {task_type}")


def main() -> None:
    args = parse_args()
    proxies = [name.strip() for name in args.proxies.split(",") if name.strip()]
    unknown = [name for name in proxies if name not in PROXY_TO_EVALUATOR]
    if unknown:
        raise ValueError(f"Unknown proxies: {unknown}")

    rows = load_space_rows(args.space_file, args.max_models)
    table_data, stype_encoder_dict = load_dataset_context(args.dataset)
    batch_x, batch_y = build_or_load_cached_batch(
        dataset_name=args.dataset,
        table_data=table_data,
        stype_encoder_dict=stype_encoder_dict,
        channels=args.channels,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    task_name = task_space_name(table_data.task_type)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict] = []

    for i, record in enumerate(rows, start=1):
        block_specs_json = json.dumps(record["block_specs"], ensure_ascii=True)

        for proxy_name in proxies:
            torch.manual_seed(42 + int(record["rank"]))
            np.random.seed(42 + int(record["rank"]))
            model = PTNASBlockMixed.from_space_record(
                record,
                channels=args.channels,
                out_channels=1,
                col_stats=table_data.col_stats,
                col_names_dict=table_data.col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
            ).to(device)
            model.train()
            model.zero_grad(set_to_none=True)
            evaluator = PROXY_TO_EVALUATOR[proxy_name]()
            t0 = time.time()
            error = ""
            try:
                score = evaluator.evaluate(model, str(device), batch_x, batch_y, task_name)
                score = float(score)
                if not np.isfinite(score):
                    error = "non_finite_score"
                    score = float(np.nan_to_num(score, nan=0.0, posinf=1e8, neginf=-1e8))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                score = float("nan")
            elapsed = time.time() - t0
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            score_rows.append(
                {
                    "dataset": args.dataset,
                    "proxy": proxy_name,
                    "rank": int(record["rank"]),
                    "group": record["group"],
                    "variant_kind": record["variant_kind"],
                    "ref_capacity": int(record["ref_capacity"]),
                    "depth": int(record["depth"]),
                    "block_specs": block_specs_json,
                    "num_params": int(num_params),
                    "proxy_score": score,
                    "proxy_time_seconds": float(elapsed),
                    "error": error,
                }
            )
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        if i % 25 == 0 or i == len(rows):
            print(f"[{args.dataset}] scored {i}/{len(rows)} models for {','.join(proxies)}", flush=True)

    df = pd.DataFrame(score_rows).sort_values(["rank", "proxy"]).reset_index(drop=True)
    score_csv = args.output_dir / f"score_{args.dataset}.csv"
    df.to_csv(score_csv, index=False)
    print(f"[saved] {score_csv}")


if __name__ == "__main__":
    main()
