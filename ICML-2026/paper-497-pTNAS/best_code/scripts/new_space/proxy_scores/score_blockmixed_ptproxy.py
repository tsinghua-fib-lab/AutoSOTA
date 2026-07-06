from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch_frame

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from proxies.ptproxy_blockmixed import ptproxy_blockmixed_score
from search_space.block_mixed import PTNASBlockMixed
from utils.resource import get_text_embedder_cfg
from utils.table_data import TableData


DEFAULT_SPACE_FILE = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "blockmixed.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "proxy_score" / "ptproxy"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "batch_cache"

FINAL_VARIANTS = {"v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all models in blockmixed.txt on one dataset with the final BlockMixed pTProxy."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--space_file", type=Path, default=DEFAULT_SPACE_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--variants", type=str, default="v1")
    parser.add_argument("--max_models", type=int, default=None)
    parser.add_argument("--force_rebuild_cache", action="store_true", default=False)
    return parser.parse_args()


def load_space_rows(path: Path, max_models: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if max_models is not None:
        rows = rows[:max_models]
    return rows


def load_dataset_context(
    dataset_name: str,
):
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
    force_rebuild: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_name}_c{channels}_b{batch_size}.pt"
    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["batch_x"], payload["batch_y"]

    loader = torch_frame.data.DataLoader(
        table_data.train_tf,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    batch = next(iter(loader)).to("cpu")

    torch.manual_seed(0)
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

def corr_pair(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 2:
        return float("nan")
    return float(sub[x].corr(sub[y], method=method))


def score_one_variant(
    variant: str,
    model: PTNASBlockMixed,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    if variant == "v1":
        return ptproxy_blockmixed_score(
            arch=model,
            batch_data=batch_x,
            batch_labels=batch_y,
            device=device,
            respect_input=True,
        )
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in FINAL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    rows = load_space_rows(args.space_file, args.max_models)
    table_data, stype_encoder_dict = load_dataset_context(args.dataset)
    batch_x, batch_y = build_or_load_cached_batch(
        dataset_name=args.dataset,
        table_data=table_data,
        stype_encoder_dict=stype_encoder_dict,
        channels=args.channels,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows_by_variant: dict[str, list[dict]] = {variant: [] for variant in variants}

    for i, record in enumerate(rows, start=1):
        torch.manual_seed(42 + int(record["rank"]))
        np.random.seed(42 + int(record["rank"]))

        model = PTNASBlockMixed.from_space_record(
            record,
            channels=args.channels,
            out_channels=1,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        ).to(args.device)

        block_specs_json = json.dumps([list(spec) for spec in model.block_specs], ensure_ascii=True)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for variant in variants:
            score, elapsed = score_one_variant(
                variant=variant,
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                device=args.device,
            )
            score_rows_by_variant[variant].append(
                {
                    "dataset": args.dataset,
                    "variant": variant,
                    "rank": int(record["rank"]),
                    "group": record["group"],
                    "variant_kind": record["variant_kind"],
                    "ref_capacity": int(record["ref_capacity"]),
                    "depth": int(record["depth"]),
                    "block_specs": block_specs_json,
                    "num_params": int(num_params),
                    "proxy_score": float(score),
                    "proxy_time_seconds": float(elapsed),
                }
            )

        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

        if i % 25 == 0 or i == len(rows):
            print(f"[{args.dataset}] scored {i}/{len(rows)} models for {','.join(variants)}", flush=True)

    for variant in variants:
        df = pd.DataFrame(score_rows_by_variant[variant]).sort_values("rank").reset_index(drop=True)
        score_csv = args.output_dir / f"score_{args.dataset}_{variant}.csv"
        summary_json = args.output_dir / f"summary_{args.dataset}_{variant}.json"

        df.to_csv(score_csv, index=False)
        summary = {
            "dataset": args.dataset,
            "variant": variant,
            "n_models": int(len(df)),
            "device": args.device,
            "batch_size": args.batch_size,
            "channels": args.channels,
            "space_file": str(args.space_file),
            "cache_dir": str(args.cache_dir),
            "mean_proxy_time_seconds": float(df["proxy_time_seconds"].mean()),
            "median_proxy_time_seconds": float(df["proxy_time_seconds"].median()),
            "mean_proxy_score": float(df["proxy_score"].mean()),
            "std_proxy_score": float(df["proxy_score"].std(ddof=0)),
        }
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[saved] {score_csv}")
        print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
