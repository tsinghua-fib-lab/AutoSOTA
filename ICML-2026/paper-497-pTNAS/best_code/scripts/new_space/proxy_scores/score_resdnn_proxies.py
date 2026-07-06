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
from proxies.ptproxy_blockmixed import ptproxy_blockmixed_score
from search_space.resnet import PTNASResNet
from utils.resource import get_text_embedder_cfg
from utils.table_data import TableData


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "proxy_score" / "ptproxy"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "batch_cache"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "training" / "resnet_pool_results.csv"

CLASSIC_PROXY_TO_EVALUATOR = {
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
    parser = argparse.ArgumentParser(description="Score PTNASResNet pool with pTProxy or classic zero-cost proxies.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results_csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--variants", type=str, default="v1")
    parser.add_argument(
        "--merge_variants",
        action="store_true",
        default=False,
        help="Write one score_<dataset>.csv with a proxy column instead of one file per variant.",
    )
    parser.add_argument("--max_models", type=int, default=None)
    parser.add_argument("--force_rebuild_cache", action="store_true", default=False)
    return parser.parse_args()


def load_dataset_context(dataset_name: str):
    data_dir = PROJECT_ROOT / "datasets" / "fit-medium-table" / dataset_name
    table_data = TableData.load_from_dir(str(data_dir))
    if not table_data.is_materialize:
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    return table_data, stype_encoder_dict


def load_architectures(results_csv: Path, dataset: str, max_models: int | None = None) -> list[str]:
    df = pd.read_csv(results_csv)
    df = df[(df["dataset"] == dataset) & (df["space_name"] == "resnet")].copy()
    archs = sorted(df["architecture"].astype(str).unique())
    if max_models is not None:
        archs = archs[:max_models]
    return archs


def build_or_load_cached_batch(
    dataset_name: str,
    table_data: TableData,
    stype_encoder_dict,
    batch_size: int,
    cache_dir: Path,
    force_rebuild: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_name}_resnet_b{batch_size}.pt"
    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["batch_x"], payload["batch_y"], int(payload["channels"])

    loader = torch_frame.data.DataLoader(
        table_data.train_tf,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    batch = next(iter(loader)).to("cpu")

    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    ref_model = PTNASResNet(
        channels=num_cols,
        out_channels=1,
        num_layers=2,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_widths=[32, 32],
        normalization="layer_norm",
        dropout_prob=0.2,
    ).to("cpu")
    ref_model.eval()
    with torch.no_grad():
        encoded_x, _ = ref_model.encoder(batch)
        batch_x = encoded_x.reshape(encoded_x.size(0), -1).detach().cpu()
        batch_y = batch.y.float().detach().cpu()

    torch.save({"batch_x": batch_x, "batch_y": batch_y, "channels": num_cols}, cache_path)
    return batch_x, batch_y, num_cols


def task_space_name(task_type: TaskType) -> str:
    if task_type == TaskType.REGRESSION:
        return "resnet:regression"
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "resnet:binary"
    raise ValueError(f"Unsupported task type: {task_type}")


def build_model(arch_str: str, table_data: TableData, stype_encoder_dict, channels: int, device: torch.device):
    block_widths = [int(x) for x in arch_str.split("-")]
    model = PTNASResNet(
        channels=channels,
        out_channels=1,
        num_layers=len(block_widths),
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_widths=block_widths,
        normalization="layer_norm",
        dropout_prob=0.2,
    ).to(device)
    return model


def score_v1(model: PTNASResNet, batch_x: torch.Tensor, batch_y: torch.Tensor, device: str) -> tuple[float, float, str]:
    try:
        score, elapsed = ptproxy_blockmixed_score(
            arch=model,
            batch_data=batch_x,
            batch_labels=batch_y,
            device=device,
            respect_input=True,
        )
        return float(score), float(elapsed), ""
    except Exception as exc:
        return float("nan"), 0.0, f"{type(exc).__name__}: {exc}"


def score_classic(proxy_name: str, model: PTNASResNet, batch_x: torch.Tensor, batch_y: torch.Tensor,
                  device: str, task_name: str) -> tuple[float, float, str]:
    evaluator = CLASSIC_PROXY_TO_EVALUATOR[proxy_name]()
    model.train()
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    try:
        score = evaluator.evaluate(model, device, batch_x, batch_y, task_name)
        score = float(score)
        if not np.isfinite(score):
            return float(np.nan_to_num(score, nan=0.0, posinf=1e8, neginf=-1e8)), time.time() - t0, "non_finite_score"
        return score, time.time() - t0, ""
    except Exception as exc:
        return float("nan"), time.time() - t0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    table_data, stype_encoder_dict = load_dataset_context(args.dataset)
    batch_x, batch_y, channels = build_or_load_cached_batch(
        dataset_name=args.dataset,
        table_data=table_data,
        stype_encoder_dict=stype_encoder_dict,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )
    archs = load_architectures(args.results_csv, args.dataset, args.max_models)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    task_name = task_space_name(table_data.task_type)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows_by_variant: dict[str, list[dict]] = {variant: [] for variant in variants}

    for i, arch_str in enumerate(archs, start=1):
        depth = len(arch_str.split("-"))
        num_params = None

        for variant in variants:
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Some classic proxies mutate module state in-place (e.g. Fisher
            # replaces forward methods), so each variant must score a fresh
            # model instance to keep results isolated and reproducible.
            model = build_model(arch_str, table_data, stype_encoder_dict, channels, device)
            if num_params is None:
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            if variant == "v1":
                score, elapsed, error = score_v1(model, batch_x, batch_y, str(device))
            else:
                score, elapsed, error = score_classic(variant, model, batch_x, batch_y, str(device), task_name)

            score_rows_by_variant[variant].append(
                {
                    "dataset": args.dataset,
                    "variant": variant,
                    "architecture": arch_str,
                    "depth": depth,
                    "num_params": int(num_params),
                    "proxy_score": float(score),
                    "proxy_time_seconds": float(elapsed),
                    "error": error,
                }
            )

            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        if i % 50 == 0 or i == len(archs):
            print(f"[{args.dataset}] scored {i}/{len(archs)} architectures for {','.join(variants)}", flush=True)

    if args.merge_variants:
        df = pd.concat(
            [pd.DataFrame(score_rows_by_variant[variant]) for variant in variants],
            ignore_index=True,
        )
        df = df.rename(columns={"variant": "proxy"})
        df = df.sort_values(["architecture", "proxy"]).reset_index(drop=True)
        score_csv = args.output_dir / f"score_{args.dataset}.csv"
        df.to_csv(score_csv, index=False)
        print(f"[saved] {score_csv}")
        return

    for variant in variants:
        df = pd.DataFrame(score_rows_by_variant[variant]).sort_values("architecture").reset_index(drop=True)
        score_csv = args.output_dir / f"score_{args.dataset}_{variant}.csv"
        summary_json = args.output_dir / f"summary_{args.dataset}_{variant}.json"
        df.to_csv(score_csv, index=False)
        summary = {
            "dataset": args.dataset,
            "variant": variant,
            "n_models": int(len(df)),
            "n_failures": int((df["error"] != "").sum()),
            "device": str(device),
            "batch_size": args.batch_size,
            "channels": channels,
            "mean_proxy_time_seconds": float(df["proxy_time_seconds"].mean()),
            "median_proxy_time_seconds": float(df["proxy_time_seconds"].median()),
            "mean_proxy_score": float(df["proxy_score"].replace([np.inf, -np.inf], np.nan).mean()),
            "std_proxy_score": float(df["proxy_score"].replace([np.inf, -np.inf], np.nan).std(ddof=0)),
        }
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[saved] {score_csv}")
        print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
