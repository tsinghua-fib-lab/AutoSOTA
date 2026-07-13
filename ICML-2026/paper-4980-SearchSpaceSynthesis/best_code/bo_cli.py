# =============================================================================
# CLI FOR BO EXPERIMENTS (COMPANION CODE FOR ICML 2026 PAPER)
# =============================================================================
#
# Companion code used in the ICML 2026 submission "Search Space Synthesis for
# Parametric Functions" (poster ref: https://icml.cc/virtual/2026/poster/62530).
#
# This module implements the command-line interface used to launch,
# aggregate and inspect experiments. Primary entry points are `run`,
# `aggregate`, `index` and `plot`. The `run` subcommand drives the BO
# experiments via the Ask/Tell-style optimizer interface (initialize,
# suggest, observe) implemented in `bo_runner.py` and compatible with the
# refactored cosy-examples Ask/Tell implementation.
#
# =============================================================================

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bo_runner import run_experiment
from bo_experiment_config import (
    EVAL_BUDGET,
    INITIAL_SAMPLE_SIZE,
    RANKING_POOL_SIZE,
    DEFAULT_DISTANCE_KERNEL,
    DISTANCE_KERNEL_OPTIONS,
)
from bo_runner import (
    MAX_TREE_DEPTH,
    MUTATION_RATE,
    POPULATION_SIZE,
    RECOMBINATION_RATE,
)

from bo_plotting import load_aggregated_results, plot_optimization_curve, plot_runtime_curve, plot_distance_curves, \
    plot_ranking_metrics

from bayesian_optimization.examples.damg_nas.damg_targets import (
    target_len_2,
    target_len_3,
    target_len_4,
    target_len_5,
    target_len_6,
    target_len_3_refined_1, target_len_3_refined_2, target_len_3_refined_3,
    target_len_5_refined_1, target_len_5_refined_2, target_len_5_refined_3,
    target_len_4_refined_1, target_to_name
)

AVAILABLE_TARGETS = {
    target_to_name(target_len_2): target_len_2,
    target_to_name(target_len_3): target_len_3,
    target_to_name(target_len_4): target_len_4,
    target_to_name(target_len_5): target_len_5,
    target_to_name(target_len_6): target_len_6,
    target_to_name(target_len_3_refined_1): target_len_3_refined_1,
    target_to_name(target_len_3_refined_2): target_len_3_refined_2,
    target_to_name(target_len_3_refined_3): target_len_3_refined_3,
    target_to_name(target_len_5_refined_1): target_len_5_refined_1,
    target_to_name(target_len_5_refined_2): target_len_5_refined_2,
    target_to_name(target_len_5_refined_3): target_len_5_refined_3,
    target_to_name(target_len_4_refined_1): target_len_4_refined_1,
}

AVAILABLE_KERNELS = [
    "wl",
    "damg",
    "combined",
    "noisy_hierarchical_wl_kernel",
    "noisy_hierarchical_damg_kernel",
    "noisy_combined_hierarchical_kernel",
]


def run_command(
        target_names: List[str],
        kernels: List[str],
        seeds: List[int],
        *,
        results_root: str = "results",
        eval_budget: int = EVAL_BUDGET,
        initial_sample_size: int = INITIAL_SAMPLE_SIZE,
        ranking_pool_size: int = RANKING_POOL_SIZE,
        kernel_optimizer: str = "fmin_l_bfgs_b",
        n_restarts_kernel_optimizer: int = 20,
        optimizer_population_size: int = POPULATION_SIZE,
        optimizer_mutation_rate: float = MUTATION_RATE,
        optimizer_recombination_rate: float = RECOMBINATION_RATE,
        max_depth: int = MAX_TREE_DEPTH,
        distance_kernel_name: str | None = DEFAULT_DISTANCE_KERNEL,
        show_progress: bool = True,
        config_path: str | None = None,
        # Debugging: enable saving EI sanity-check plots/arrays per BO run
        save_debug_plots: bool = False,
        debug_plots_dir: str | None = None,
        sanity_n_sanity: int = 10000,
):
    """Run experiments for combinations of targets × kernels × seeds.

	Calls `run_experiment` from the runner and writes a simple manifest.json into
	each experiment folder returned in the export_paths map.
	"""
    # Suppress sklearn's GP ConvergenceWarning for the duration of CLI runs. The
    # GP kernel optimizer hits the iteration limit frequently in early BO
    # iterations (small training sets); the messages flood stdout and hide real
    # problems. Library callers see warnings unchanged.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    config = _load_json_config(config_path)
    for tname in target_names:
        if tname not in AVAILABLE_TARGETS:
            raise ValueError(f"Unknown target: {tname}. Available: {list(AVAILABLE_TARGETS.keys())}")
        target_obj = AVAILABLE_TARGETS[tname]
        for kernel in kernels:
            print(f"[CLI] Running target={tname} kernel={kernel} seeds={seeds}")
            trace_df, ranking_df, export_paths = run_experiment(
                target_obj,
                kernel,
                seeds,
                eval_budget=eval_budget,
                initial_sample_size=initial_sample_size,
                ranking_pool_size=ranking_pool_size,
                results_root=results_root,
                kernel_optimizer=kernel_optimizer,
                n_restarts_kernel_optimizer=n_restarts_kernel_optimizer,
                optimizer_population_size=optimizer_population_size,
                optimizer_mutation_rate=optimizer_mutation_rate,
                optimizer_recombination_rate=optimizer_recombination_rate,
                max_depth=max_depth,
                distance_kernel_name=distance_kernel_name,
                show_progress=show_progress,
                sanity_n_sanity=sanity_n_sanity,
                save_debug_plots=save_debug_plots,
                debug_plots_dir=debug_plots_dir,
            )

            # For each exported experiment entry, write a manifest alongside CSVs
            for exp_key, files in export_paths.items():
                # find parent folder from any path
                any_path = next(iter(files.values()))
                parent = Path(any_path).parent
                manifest = {
                    "experiment_key": exp_key,
                    "target": tname,
                    "kernel": kernel,
                    "seeds": seeds,
                    "generated_at": datetime.now().isoformat(),
                    "experiment_params": {
                        "results_root": str(Path(results_root).resolve()),
                        "config_path": str(Path(config_path).resolve()) if config_path else None,
                        "eval_budget": eval_budget,
                        "initial_sample_size": initial_sample_size,
                        "ranking_pool_size": ranking_pool_size,
                        "kernel_optimizer": kernel_optimizer,
                        "n_restarts_kernel_optimizer": n_restarts_kernel_optimizer,
                        "optimizer_population_size": optimizer_population_size,
                        "optimizer_mutation_rate": optimizer_mutation_rate,
                        "optimizer_recombination_rate": optimizer_recombination_rate,
                        "max_depth": max_depth,
                        "distance_kernel_name": distance_kernel_name,
                    },
                    "files": files,
                }
                try:
                    (parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
                except Exception as e:
                    print(f"[CLI] Warning: failed to write manifest for {exp_key}: {e}")
            print(f"[CLI] Completed target={tname} kernel={kernel}")


def aggregate_command(results_root: str = "results"):
    """Aggregate trace and ranking CSVs across all experiment folders under `results_root`.

	The function searches for CSV files with names containing "trace" or "ranking",
	concatenates them and writes aggregated CSVs into a new folder
  results_root/aggregated__<timestamp>/, then refreshes the central manifest index
  and writes a compact manifest summary CSV.
	"""
    base = Path(results_root)
    if not base.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    trace_frames = []
    ranking_frames = []
    aggregated_trace = None
    aggregated_ranking = None

    # scan one level deep for folders (experiment folders)
    for folder in sorted(base.glob("*__*")):
        if not folder.is_dir():
            continue
        # collect trace CSVs
        for trace_file in folder.glob("*_trace.csv"):
            try:
                df = pd.read_csv(trace_file)
                trace_frames.append(df)
            except Exception:
                print(f"[CLI] Warning: failed to read {trace_file}")
        for rank_file in folder.glob("*_ranking.csv"):
            try:
                df = pd.read_csv(rank_file)
                ranking_frames.append(df)
            except Exception:
                print(f"[CLI] Warning: failed to read {rank_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = base / f"aggregated__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if trace_frames:
        aggregated_trace = pd.concat(trace_frames, ignore_index=True)
        trace_path = out_dir / "aggregated_trace.csv"
        aggregated_trace.to_csv(trace_path, index=False)
        print(f"[CLI] Wrote aggregated trace to {trace_path}")
    else:
        print("[CLI] No trace CSVs found to aggregate.")

    if ranking_frames:
        aggregated_ranking = pd.concat(ranking_frames, ignore_index=True)
        ranking_path = out_dir / "aggregated_ranking.csv"
        aggregated_ranking.to_csv(ranking_path, index=False)
        print(f"[CLI] Wrote aggregated ranking to {ranking_path}")
    else:
        print("[CLI] No ranking CSVs found to aggregate.")

    # simple summary: group by method and compute mean bo_overhead_time and objective_value
    if trace_frames:
        try:
            summary = aggregated_trace.groupby(["method"]).agg(
                evaluations=("evaluation", "count"),
                mean_objective=("objective_value", "mean"),
                mean_overhead=("bo_overhead_time", "mean"),
            ).reset_index()
            summary_path = out_dir / "summary_by_method.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[CLI] Wrote summary to {summary_path}")
        except Exception as e:
            print(f"[CLI] Warning: failed to compute/write summary: {e}")

    manifest_index_path = None
    manifest_index = None
    try:
        manifest_index_path, manifest_index = build_manifest_index(results_root=results_root)
    except Exception as e:
        print(f"[CLI] Warning: failed to refresh manifest index: {e}")

    manifest_summary_path = None
    if manifest_index and manifest_index.get("entries"):
        try:
            manifest_summary = pd.DataFrame(
                [
                    {
                        "experiment_key": entry.get("experiment_key"),
                        "target": entry.get("target"),
                        "kernel": entry.get("kernel"),
                        "seeds": json.dumps(entry.get("seeds", [])),
                        "manifest_path": entry.get("manifest_path"),
                        "experiment_folder": entry.get("experiment_folder"),
                        "file_count": len(entry.get("files", {}) or {}),
                        "files": json.dumps(entry.get("files", {})),
                        "experiment_params": json.dumps(entry.get("experiment_params", {})),
                    }
                    for entry in manifest_index["entries"]
                ]
            )
            manifest_summary_path = out_dir / "manifest_summary.csv"
            manifest_summary.to_csv(manifest_summary_path, index=False)
            print(f"[CLI] Wrote manifest summary to {manifest_summary_path}")
        except Exception as e:
            print(f"[CLI] Warning: failed to write manifest summary: {e}")

    return {
        "output_dir": str(out_dir),
        "trace_path": str(trace_path) if trace_frames else None,
        "ranking_path": str(ranking_path) if ranking_frames else None,
        "summary_path": str(summary_path) if trace_frames else None,
        "manifest_index_path": str(manifest_index_path) if manifest_index_path else None,
        "manifest_summary_path": str(manifest_summary_path) if manifest_summary_path else None,
    }


def build_manifest_index(results_root: str = "results", output_filename: str = "manifest_index.json"):
    """Build a central index over all per-experiment manifest.json files.

    The index is written directly under `results_root` and contains one entry per
    manifest file as well as a grouping by experiment_key for convenience.
    """
    base = Path(results_root)
    if not base.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    manifest_files = sorted(base.rglob("manifest.json"))
    entries = []
    by_experiment_key: dict[str, list[dict]] = {}

    for manifest_file in manifest_files:
        try:
            manifest = json.loads(manifest_file.read_text())
        except Exception as exc:
            print(f"[CLI] Warning: failed to read manifest {manifest_file}: {exc}")
            continue

        record = {
            **manifest,
            "manifest_path": str(manifest_file.resolve()),
            "experiment_folder": str(manifest_file.parent.resolve()),
        }
        entries.append(record)

        experiment_key = record.get("experiment_key")
        if experiment_key is not None:
            by_experiment_key.setdefault(str(experiment_key), []).append(record)

    index = {
        "generated_at": datetime.now().isoformat(),
        "results_root": str(base.resolve()),
        "manifest_count": len(entries),
        "entries": entries,
        "by_experiment_key": by_experiment_key,
    }

    out_path = base / output_filename
    out_path.write_text(json.dumps(index, indent=2))
    print(f"[CLI] Wrote manifest index to {out_path}")
    return out_path, index


def plot_command(
        results_root: str,
        *,
        output_dir: str | None = None,
        methods: List[str] | None = None,
        targets: List[str] | None = None,
        kernels: List[str] | None = None,
):
    """Generate the experimental plots from the most recent aggregated CSVs.

    The actual plotting functions in `bo_plotting` are still stubs that raise
    `NotImplementedError`; this command loads the data, invokes each plot
    function defensively, and reports per-plot which ones are not yet
    implemented. It exists so the persistence side of the pipeline can be
    smoke-tested end-to-end without blocking on the final plotting style.
    """
    data = load_aggregated_results(results_root)
    trace_df = data.get("trace", pd.DataFrame())
    ranking_df = data.get("ranking", pd.DataFrame())

    if trace_df.empty and ranking_df.empty:
        print(
            f"[CLI] No aggregated results found under {results_root}. "
            "Run `bo_cli aggregate` first."
        )
        return

    target_output = Path(output_dir) if output_dir else Path(results_root) / "bo_plots"
    target_output.mkdir(parents=True, exist_ok=True)

    # Optional filters — applied identically to both DataFrames.
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series([True] * len(df), index=df.index)
        if methods:
            mask &= df["method"].isin(methods)
        if targets:
            mask &= df["target_name"].isin(targets)
        if kernels:
            mask &= df["kernel_name"].isin(kernels)
        return df.loc[mask]

    trace_df = _filter(trace_df)
    ranking_df = _filter(ranking_df)

    print(f"[CLI] Plotting from results_root={results_root} → {target_output}")
    print(f"[CLI] trace rows={len(trace_df)} ranking rows={len(ranking_df)}")

    plot_calls = (
        ("plot_optimization_curve", plot_optimization_curve, trace_df),
        ("plot_runtime_curve", plot_runtime_curve, trace_df),
        ("plot_ranking_metrics", plot_ranking_metrics, ranking_df),
        ("plot_distance_curves", plot_distance_curves, trace_df),
    )
    for name, fn, df in plot_calls:
        try:
            fn(df, target_output)
            print(f"[CLI] {name}: OK")
        except NotImplementedError as exc:
            print(f"[CLI] {name}: not implemented yet — skipping ({exc})")
        except Exception as exc:
            print(f"[CLI] {name}: failed — {exc}")


def _load_json_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config


def _parse_seed_list(seed_str: str) -> List[int]:
    parts = [s.strip() for s in seed_str.split(",") if s.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="BO experiment runner and aggregator CLI")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run experiments for target×kernel×seeds")
    p_run.add_argument(
        "--targets",
        default=None,
        help="Comma separated target names (or set via 'targets' key in --config JSON)",
    )
    p_run.add_argument(
        "--kernels",
        default=None,
        help="Comma separated kernel names (or set via 'kernels' key in --config JSON)",
    )
    p_run.add_argument(
        "--seeds",
        default=None,
        help="Comma separated list of integer seeds (or set via 'seeds' key in --config JSON)",
    )
    p_run.add_argument("--config", default=None, help="Optional JSON file with shared run parameters")
    p_run.add_argument("--results-root", default=None, help="Root folder for timestamped experiment outputs")
    p_run.add_argument("--eval-budget", type=int, default=None, help="Number of BO iterations after presamples")
    p_run.add_argument("--initial-sample-size", type=int, default=None,
                       help="Number of initial samples shared across methods")
    p_run.add_argument("--ranking-pool-size", type=int, default=None,
                       help="Number of programs used to compute ranking metrics")
    p_run.add_argument("--kernel-optimizer", default=None, help="Kernel optimizer passed to BO")
    p_run.add_argument("--n-restarts-kernel-optimizer", type=int, default=None, help="Kernel optimizer restarts")
    p_run.add_argument("--optimizer-population-size", type=int, default=None,
                       help="Population size for acquisition optimizer")
    p_run.add_argument("--optimizer-mutation-rate", type=float, default=None,
                       help="Mutation rate for acquisition optimizer")
    p_run.add_argument("--optimizer-recombination-rate", type=float, default=None,
                       help="Recombination rate for acquisition optimizer")
    p_run.add_argument("--max-depth", type=int, default=None, help="Maximum derivation tree depth")
    p_run.add_argument(
        "--distance-kernel",
        choices=list(DISTANCE_KERNEL_OPTIONS),
        default=None,
        help="Structural distance kernel for Plot 5/6 (default: wl).",
    )
    p_run.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        default=None,
        help="Disable tqdm progress bars (e.g. when redirecting output to a log file).",
    )
    p_run.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="Enable EI sanity-check debug output (save debug plots/arrays).",
    )
    p_run.add_argument(
        "--debug-plots-dir",
        dest="debug_plots_dir",
        default=None,
        help="Optional directory to write debug plots/arrays (default: per-experiment results folder).",
    )
    p_run.add_argument(
        "--sanity-n-sanity",
        dest="sanity_n_sanity",
        type=int,
        default=None,
        help="Number of random candidates to sample for EI sanity-check (default: 1000).",
    )

    sub.add_parser("list-targets", help="List available target names")
    sub.add_parser("list-kernels", help="List supported kernel names")

    p_agg = sub.add_parser("aggregate", help="Aggregate results under results/ directory")
    p_agg.add_argument("--results-root", default="results", help="Root folder containing experiment folders")

    p_index = sub.add_parser("index", help="Build a central manifest index under the results root")
    p_index.add_argument("--results-root", default="results", help="Root folder containing experiment folders")
    p_index.add_argument("--output-filename", default="manifest_index.json", help="Filename for the central index")

    p_plot = sub.add_parser(
        "plot",
        help="Generate plots from the most recent aggregated CSVs (functions still stubbed)",
    )
    p_plot.add_argument("--results-root", default="results", help="Root folder containing aggregated__* subfolders")
    p_plot.add_argument("--output-dir", default=None, help="Where to write PDFs (default: <results-root>/bo_plots)")
    p_plot.add_argument("--methods", default=None, help="Comma-separated method filter (random, bo)")
    p_plot.add_argument("--targets", default=None, help="Comma-separated target_name filter")
    p_plot.add_argument("--kernels", default=None, help="Comma-separated kernel_name filter")

    args = parser.parse_args()
    if args.command == "list-targets":
        for k in sorted(AVAILABLE_TARGETS.keys()):
            print(k)
        return
    if args.command == "list-kernels":
        for k in AVAILABLE_KERNELS:
            print(k)
        return

    if args.command == "run":
        config = _load_json_config(args.config)

        def _parse_csv(s: str | None) -> List[str] | None:
            if not s:
                return None
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts or None

        targets = _parse_csv(args.targets) or config.get("targets")
        kernels = _parse_csv(args.kernels) or config.get("kernels")
        if args.seeds:
            seeds = _parse_seed_list(args.seeds)
        else:
            seeds = config.get("seeds")
            if seeds is not None:
                seeds = [int(s) for s in seeds]

        if not targets or not kernels or not seeds:
            raise SystemExit(
                "run requires targets, kernels and seeds — provide them either via "
                "the CLI (--targets/--kernels/--seeds) or in the JSON config under "
                "the keys 'targets'/'kernels'/'seeds'."
            )
        run_command(
            targets,
            kernels,
            seeds,
            results_root=args.results_root if args.results_root is not None else config.get("results_root", "results"),
            eval_budget=args.eval_budget if args.eval_budget is not None else config.get("eval_budget", EVAL_BUDGET),
            initial_sample_size=args.initial_sample_size if args.initial_sample_size is not None else config.get(
                "initial_sample_size", INITIAL_SAMPLE_SIZE),
            ranking_pool_size=args.ranking_pool_size if args.ranking_pool_size is not None else config.get(
                "ranking_pool_size", RANKING_POOL_SIZE),
            kernel_optimizer=args.kernel_optimizer if args.kernel_optimizer is not None else config.get(
                "kernel_optimizer", "fmin_l_bfgs_b"),
            n_restarts_kernel_optimizer=args.n_restarts_kernel_optimizer if args.n_restarts_kernel_optimizer is not None else config.get(
                "n_restarts_kernel_optimizer", 20),
            optimizer_population_size=args.optimizer_population_size if args.optimizer_population_size is not None else config.get(
                "optimizer_population_size", POPULATION_SIZE),
            optimizer_mutation_rate=args.optimizer_mutation_rate if args.optimizer_mutation_rate is not None else config.get(
                "optimizer_mutation_rate", MUTATION_RATE),
            optimizer_recombination_rate=args.optimizer_recombination_rate if args.optimizer_recombination_rate is not None else config.get(
                "optimizer_recombination_rate", RECOMBINATION_RATE),
            max_depth=args.max_depth if args.max_depth is not None else config.get("max_depth", MAX_TREE_DEPTH),
            distance_kernel_name=args.distance_kernel if args.distance_kernel is not None else config.get(
                "distance_kernel_name", DEFAULT_DISTANCE_KERNEL),
            show_progress=args.show_progress if args.show_progress is not None else bool(
                config.get("show_progress", True)),
            config_path=args.config,
            # debug flags
            save_debug_plots=(args.debug if args.debug is not None else bool(config.get("save_debug_plots", False))),
            debug_plots_dir=(
                args.debug_plots_dir if args.debug_plots_dir is not None else config.get("debug_plots_dir", None)),
            sanity_n_sanity=(
                args.sanity_n_sanity if args.sanity_n_sanity is not None else config.get("sanity_n_sanity", 1000)),
        )
        return

    if args.command == "aggregate":
        aggregate_command(results_root=args.results_root)
        return

    if args.command == "index":
        build_manifest_index(results_root=args.results_root, output_filename=args.output_filename)
        return

    if args.command == "plot":
        def _split(s):
            return [t.strip() for t in s.split(",") if t.strip()] if s else None

        plot_command(
            results_root=args.results_root,
            output_dir=args.output_dir,
            methods=_split(args.methods),
            targets=_split(args.targets),
            kernels=_split(args.kernels),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
