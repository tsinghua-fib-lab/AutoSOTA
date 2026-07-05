#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


#### TODO : import à modifier en fonction de là on décale.

RBAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RBAL_ROOT.parent
BASELINE_DIR = RBAL_ROOT / "baseline"
PYCIPHOD_SRC = Path.home() / "code" / "pyCIPHOD" / "src"

for p in [PROJECT_ROOT, RBAL_ROOT, BASELINE_DIR, PYCIPHOD_SRC]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pyciphod.utils.time_series.data_format import ts_to_lagged_df
from pyciphod.utils.stat_tests.equality_tests import LinearRegressionCoefficientEqualityTest

from generator import SETTINGS, SETTING_SEED_INDEX, CHANGE_MODELS, generate_one_run
from pyciphod.causal_discovery.difference.ts_diff_constraint_based import TsLDiffPC, TsDCI
from metrics_ts import evaluate_all_ts

from baseline.MBGH import learn_ddag
# from baseline.microcause import micro_cause
# from baseline.rcd import top_k_rc, BINS

DEFAULT_LAGS = {
    "setting1_lag2": [2],
    "setting2_lag1": [1, 2],
    "setting3_contemporaneous_with_self_lag": [0, 1, 2],
    "setting4_iid": [0, 1, 2],
}

ALGOS = ["tsldiffpc", "tsldiffpc_pc", "tsdci", "tsdci_pc", "tsMBGH", "microcause", "rcd", ]  # tsiscan
GRAPH_ALGOS = {"tsldiffpc", "tsldiffpc_pc", "tsdci", "tsdci_pc", "tsMBGH", }  # "tsiscan"


####

def ensure(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def seed_block(setting: str, p: int, n: int, rep: int, base_seed: int) -> dict[str, int]:
    """Generate reproducible random seeds for graph generation and data simulation."""
    base = base_seed + SETTING_SEED_INDEX[setting] * 10_000_000 + p * 100_000 + n + rep
    return {
        "structure_seed": base,
        "coef_seed": base + 10_000,
        "dataset_seed": base + 100_000,
        "change_seed": base + 200_000,
    }


def prf(pred: set[Any], true: set[Any]) -> dict[str, Any]:
    """Compute precision, recall, F1-score and confusion counts for set predictions."""
    tp, fp, fn = len(pred & true), len(pred - true), len(true - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def lagged(X1_ts: pd.DataFrame, X2_ts: pd.DataFrame, user_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert two time-series datasets into lagged representations."""
    X1 = ts_to_lagged_df(X1_ts, window_size=user_lag + 1, dropna=True)
    X2 = ts_to_lagged_df(X2_ts, window_size=user_lag + 1, dropna=True)
    if list(X1.columns) != list(X2.columns):
        raise RuntimeError("Lagged X1 and X2 do not have the same columns/order.")
    if not np.isfinite(X1.to_numpy(dtype=float)).all() or not np.isfinite(X2.to_numpy(dtype=float)).all():
        raise RuntimeError("Lagged input contains NaN/inf.")
    return X1, X2


def run_graph(algo_name: str, X1_ts: pd.DataFrame, X2_ts: pd.DataFrame, user_lag: int, args: argparse.Namespace) -> \
dict[str, Any]:
    """
    :param algo_name: name of the temporal graph algorithm to run. Supported values are tsMBGH, tsldiffpc, tsldiffpc_pc, tsdci and tsdci_pc.
    :param X1_ts: time-series dataframe from the normal regime.
    :param X2_ts: time-series dataframe from the anomalous regime.
    :param user_lag: maximum lag used to construct the lagged temporal representation.
    :param args: command-line arguments containing algorithm hyperparameters.
    :return: dictionary containing the predicted directed edges, undirected edges and number of tests.
    This function first converts both regimes into lagged dataframes. It then runs the selected temporal difference-graph algorithm.
    For tsMBGH, it estimates a directed difference graph with the MBGH procedure.
    For tsldiffpc and tsdci, it runs the corresponding constraint-based difference-graph algorithm. When the algorithm name ends with _pc, the remaining unresolved edges are further oriented using tPC on the normal regime.
    """
    X1, X2 = lagged(X1_ts, X2_ts, user_lag)

    if algo_name == "tsMBGH":
        Z1, Z2 = X1.to_numpy(dtype=float), X2.to_numpy(dtype=float)
        lam = args.mbgh_lam
        if lam is None:
            lam = float(args.mbgh_lam_mult * np.sqrt(np.log(max(Z1.shape[1], 2)) / max(Z1.shape[0], 1)))
        ddag = learn_ddag(
            Z1, Z2,
            estimator="admm",
            lam=lam,
            diagonal_tol=0.05,
            edge_tol=0.05,
            diagonal_tol_fraction=args.mbgh_diag_frac,
            edge_tol_fraction=args.mbgh_edge_frac,
            admm_kwargs=None,
        )
        cols = list(X1.columns)
        oriented = set()
        for child_idx, child in enumerate(cols):
            if int(child.time) != 0:
                continue
            for parent_idx, parent in enumerate(cols):
                if child_idx != parent_idx and ddag[child_idx, parent_idx] != 0 and int(parent.time) <= int(child.time):
                    oriented.add(((str(parent.name), int(parent.time)), (str(child.name), int(child.time))))
        return {"oriented": oriented, "undirected": set(), "nb_tests": 0}

    if algo_name == "tsiscan":
        return run_tsiscan(X1_ts, X2_ts, user_lag=user_lag, )

    if algo_name.startswith("tsldiffpc"):
        algo = TsLDiffPC(sparsity=args.sparsity, seed=args.seed, eq_test=LinearRegressionCoefficientEqualityTest)
    elif algo_name.startswith("tsdci"):
        algo = TsDCI(sparsity=args.sparsity, seed=args.seed)
    else:
        raise ValueError(f"Unknown graph algorithm: {algo_name}")

    algo.run(df1=X1, df2=X2, max_sepset_size=args.max_sepset_size)
    base_tests = int(getattr(algo, "nb_ci_tests", getattr(algo, "nb_tests", 0)) or 0)

    if algo_name.endswith("_pc"):
        algo.add_tspc_orientation(
            df1=X1,
            pc_sparsity=args.pc_sparsity if args.pc_sparsity is not None else args.sparsity,
            pc_max_sepset_size=args.pc_max_sepset_size,
        )

    pred = {
        "oriented": {((str(u.name), int(u.time)), (str(v.name), int(v.time))) for u, v in
                     algo.g_hat.get_directed_edges()},
        "undirected": {((str(u.name), int(u.time)), (str(v.name), int(v.time))) for u, v in
                       algo.g_hat.get_undirected_edges()},
        "nb_tests": int(getattr(algo, "nb_ci_tests", getattr(algo, "nb_tests", 0)) or 0),
    }
    pred["nb_tests"] = max(pred["nb_tests"], base_tests)
    return pred


def run_tsiscan(X1_ts: pd.DataFrame, X2_ts: pd.DataFrame, user_lag: int, ) -> dict[str, Any]:
    import iscan

    X1, X2 = lagged(X1_ts, X2_ts, user_lag)
    nodes = list(X1.columns)

    pred_shifted, est_order, var_ratio_dict = iscan.est_node_shifts(
        X1.to_numpy(dtype=float),
        X2.to_numpy(dtype=float),
        eta_G=0.05,
        eta_H=0.05,
        normalize_var=False,
        shifted_node_thres=2.0,
        elbow=True,
        elbow_thres=30.0,
        elbow_online=True,
        use_both_rank=True,
        verbose=False,
    )

    estimated_order = [nodes[int(i)] for i in est_order]
    order_pos = {node: pos for pos, node in enumerate(estimated_order)}

    shifted_nodes = [nodes[int(i)] for i in pred_shifted]
    shifted_current_nodes = [node for node in shifted_nodes if int(node.time) == 0]

    oriented_edges = {
        (
            (str(parent.name), int(parent.time)),
            (str(target.name), int(target.time)),
        )
        for target in shifted_current_nodes
        for parent in estimated_order[:order_pos[target]]
        if parent != target and int(parent.time) <= int(target.time)
    }

    return {"oriented": oriented_edges, "undirected": set(), "nb_tests": 0, }


def run_microcause(X1_ts: pd.DataFrame, X2_ts: pd.DataFrame, user_lag: int, args: argparse.Namespace, ) -> list[str]:
    """
    :param X1_ts: time-series dataframe from the normal regime.
    :param X2_ts: time-series dataframe from the anomalous regime.
    :param user_lag: maximum temporal lag used by MicroCause.
    :param args: command-line arguments containing the MicroCause hyperparameters.
    :return: list of predicted root-cause nodes.
     This function concatenates the normal and anomalous regimes into a single time series, sets the change point at the beginning of the anomalous regime, and runs the MicroCause root-cause analysis algorithm.
     The returned list contains the predicted root-cause nodes.
    """
    from baseline.microcause import micro_cause

    random.seed(args.seed)
    np.random.seed(args.seed)

    cols = list(X1_ts.columns)
    data = pd.concat([X1_ts, X2_ts[cols]], ignore_index=True, )
    change_point = len(X1_ts)

    roots = micro_cause(
        data=data,
        anomalous_nodes=cols,
        anomalies_start_time={node: change_point for node in cols},
        anomaly_length=len(X2_ts),
        gamma_max=int(user_lag),
        sig_threshold=args.microcause_sig_threshold,
    )
    return list(dict.fromkeys(str(x) for x in (roots or [])))


def run_rcd(X1_ts: pd.DataFrame, X2_ts: pd.DataFrame, args: argparse.Namespace, ) -> list[str]:
    """
    :param X1_ts: time-series dataframe from the normal regime.
    :param X2_ts: time-series dataframe from the anomalous regime.
    :param args: command-line arguments containing the RCD hyperparameters.
    :return: list of predicted root-cause nodes.
     This function runs the Root Cause Discovery (RCD) algorithm on the normal and anomalous regimes. The returned list contains the top-k predicted root-cause nodes.
    """
    from baseline.rcd import top_k_rc, BINS

    np.random.seed(args.seed)

    normal_df = X1_ts.copy()
    anomalous_df = X2_ts[normal_df.columns].copy()

    result = top_k_rc(
        normal_df=normal_df,
        anomalous_df=anomalous_df,
        k=args.rcd_k,
        bins=BINS,
        localized=args.rcd_local,
        verbose=False,
    )

    return list(dict.fromkeys(str(x) for x in result["root_cause"]))[: args.rcd_k]


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate temporal difference-graph algorithms and local RBAL baselines.")
    p.add_argument("--settings", nargs="+", default=["setting2_lag1"], choices=list(SETTINGS.keys()))
    p.add_argument("--algos", nargs="+", default=GRAPH_ALGOS, choices=ALGOS)
    p.add_argument("--p-list", nargs="+", type=int, default=[3])
    p.add_argument("--n-list", nargs="+", type=int, default=[1000])
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--user-lags", nargs="+", type=int, default=[1])

    p.add_argument("--edge-prob", type=float, default=0.3)
    p.add_argument("--base-seed", type=int, default=12000)
    p.add_argument("--burn-in", type=int, default=200)
    p.add_argument("--min-abs-change", type=float, default=0.5)
    p.add_argument("--change-model", choices=sorted(CHANGE_MODELS), default="all_parents_min2")
    p.add_argument("--min-incoming-parents", type=int, default=2)

    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-sepset-size", type=int, default=None)
    p.add_argument("--pc-sparsity", type=float, default=None)
    p.add_argument("--pc-max-sepset-size", type=int, default=None)

    p.add_argument("--mbgh-lam", type=float, default=None)
    p.add_argument("--mbgh-lam-mult", type=float, default=2.0)
    p.add_argument("--mbgh-diag-frac", type=float, default=0.05)
    p.add_argument("--mbgh-edge-frac", type=float, default=0.05)

    p.add_argument("--microcause-sig-threshold", type=float, default=0.05)
    p.add_argument("--rcd-k", type=int, default=1)
    p.add_argument("--rcd-local", action="store_true")
    p.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = args_parser()

    rows = []
    failures = []

    print("ROOT:", RBAL_ROOT)
    print("BASELINE_DIR:", BASELINE_DIR)
    print("Settings:", args.settings)
    print("Algorithms:", args.algos)
    print("Change model:", args.change_model)

    for setting in args.settings:
        lags = args.user_lags if args.user_lags is not None else DEFAULT_LAGS[setting]

        for p in args.p_list:
            for n in args.n_list:
                for rep in range(1, args.n_reps + 1):
                    run_id = f"{setting}_p{p}_n{n}_rep{rep:03d}"
                    print(f"\n=== {run_id} ===")

                    try:
                        seeds = seed_block(setting=setting, p=p, n=n, rep=rep, base_seed=args.base_seed, )
                        generated = generate_one_run(
                            setting_name=setting,
                            p=p,
                            n=n,
                            rep=rep,
                            edge_prob=args.edge_prob,
                            structure_seed=seeds["structure_seed"],
                            coef_seed=seeds["coef_seed"],
                            dataset_seed=seeds["dataset_seed"],
                            change_seed=seeds["change_seed"],
                            burn_in=args.burn_in,
                            min_abs_change=args.min_abs_change,
                            change_model=args.change_model,
                            min_incoming_parents=args.min_incoming_parents,
                        )

                        X1_ts = generated["X1"]
                        X2_ts = generated["X2"]
                        truth = generated["truth"]
                        true_edges = {((str(source.name), int(source.time)), (str(target.name), int(target.time)),) for
                                      source, target in truth["changed_edges_lagged"]}
                        shifted_nodes = set(map(str, truth["shifted_nodes"]))

                    except Exception as exc:
                        failures.append(
                            {"run_id": run_id, "algo": "GENERATOR", "error": type(exc).__name__, "message": str(exc), })
                        print(f"[GENERATOR ERROR] {type(exc).__name__}: {exc}")
                        if not args.continue_on_error:
                            raise
                        continue

                    for algo in args.algos:
                        algo_lags = [-1] if algo == "rcd" else lags

                        for user_lag in algo_lags:
                            print(f"\n--- {algo} / user_lag={user_lag} ---")

                            try:
                                if algo in GRAPH_ALGOS:
                                    pred = run_graph(algo_name=algo, X1_ts=X1_ts, X2_ts=X2_ts, user_lag=int(user_lag),
                                                     args=args, )
                                    scores = evaluate_all_ts(true_edges=true_edges, pred_oriented=set(pred["oriented"]),
                                                             pred_undirected=set(pred["undirected"]),
                                                             shifted_nodes=shifted_nodes, )

                                    for metric, values in scores.items():
                                        row = {
                                            "setting": setting,
                                            "change_model": args.change_model,
                                            "p": p,
                                            "n": n,
                                            "rep": rep,
                                            "algo": algo,
                                            "user_lag": int(user_lag),
                                            "metric": metric,
                                            "nb_tests": pred.get("nb_tests", 0),
                                            "n_pred_oriented": len(pred["oriented"]),
                                            "n_pred_undirected": len(pred["undirected"]),
                                            **values,
                                        }
                                        rows.append(row)

                                elif algo == "microcause":
                                    pred_nodes = run_microcause(X1_ts=X1_ts, X2_ts=X2_ts, user_lag=int(user_lag),
                                                                args=args, )
                                    values = prf(set(pred_nodes), shifted_nodes)

                                    row = {
                                        "setting": setting,
                                        "change_model": args.change_model,
                                        "p": p,
                                        "n": n,
                                        "rep": rep,
                                        "algo": algo,
                                        "user_lag": int(user_lag),
                                        "metric": "node_f1",
                                        "pred_nodes": ";".join(pred_nodes),
                                        "true_nodes": ";".join(sorted(shifted_nodes)),
                                        **values,
                                    }
                                    rows.append(row)

                                elif algo == "rcd":
                                    pred_nodes = run_rcd(X1_ts=X1_ts, X2_ts=X2_ts, args=args, )
                                    values = prf(set(pred_nodes), shifted_nodes)

                                    row = {
                                        "setting": setting,
                                        "change_model": args.change_model,
                                        "p": p,
                                        "n": n,
                                        "rep": rep,
                                        "algo": algo,
                                        "user_lag": int(user_lag),
                                        "metric": f"node_top{args.rcd_k}",
                                        "pred_nodes": ";".join(pred_nodes),
                                        "true_nodes": ";".join(sorted(shifted_nodes)),
                                        **values,
                                    }
                                    rows.append(row)
                                else:
                                    raise ValueError(f"Unknown algorithm: {algo}")

                            except Exception as exc:
                                failures.append(
                                    {"run_id": run_id, "algo": algo, "user_lag": user_lag, "error": type(exc).__name__,
                                     "message": str(exc), })

                                print(f"[ERROR] {algo}: {type(exc).__name__}: {exc}")
                                if not args.continue_on_error:
                                    raise

    if rows:
        summary = (
            pd.DataFrame(rows)
            .groupby(
                [
                    "setting",
                    "change_model",
                    "p",
                    "n",
                    "user_lag",
                    "algo",
                    "metric",
                ],
                dropna=False,
            )
            .agg(
                n_ok=("f1", "size"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
                f1_std=("f1", "std"),
                f1_var=("f1", "var"),
            )
            .reset_index()
            .sort_values(["setting", "p", "user_lag", "algo", "metric"])
        )

        print("\nFINAL SUMMARY")
        print(summary.to_string(index=False))
    else:
        print("\nNo successful result.")

    if failures:
        print(f"\n{len(failures)} failures:")
        for failure in failures:
            print(
                f"- {failure.get('run_id')} | {failure.get('algo')} "
                f"| lag={failure.get('user_lag', '-')}: "
                f"{failure.get('error')}: {failure.get('message')}"
            )


if __name__ == "__main__":
    main()