#!/usr/bin/env python3
"""
plot_logs.py
Aggregate and visualize results from synthetic_hitl_causal_dpo.py runs.
Saves PNG + PDF for each plot.

Usage:
    python plot_logs.py --outdir ./runs/exp1 --pick_run seed1
"""

import argparse, os, glob, json, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # non-interactive


def savefig_dual(fig, outbase):
    fig.savefig(outbase + ".png", dpi=150)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)
    print("Saved", outbase + ".png", "and", outbase + ".pdf")


def load_runs(outdir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*_logs.json"))):
        prefix = os.path.basename(path).replace("_logs.json", "")
        with open(path) as f:
            logs = json.load(f)
        settings_path = os.path.join(outdir, f"{prefix}_settings.json")
        settings = None
        if os.path.exists(settings_path):
            with open(settings_path) as f:
                settings = json.load(f)
        runs[prefix] = {"logs": logs, "settings": settings}
    return runs


def plot_learning_curves(runs, outdir):
    fig = plt.figure(figsize=(10, 4))
    for prefix, run in runs.items():
        logs = run["logs"]
        rounds = [row["round"] for row in logs]
        # acc = [row["exist_acc@0.5"] for row in logs]
        entropy = [row["avg_pred_entropy"] for row in logs]
        ess = [row["ess"] for row in logs]
        # plt.subplot(1, 3, 1); plt.plot(rounds, acc, label=prefix)
        plt.subplot(1, 2, 1); plt.plot(rounds, entropy, label=prefix)
        plt.subplot(1, 2, 2); plt.plot(rounds, ess, label=prefix)
    # plt.subplot(1, 3, 1); plt.xlabel("Round"); plt.ylabel("Acc"); plt.title("Accuracy")
    plt.subplot(1, 2, 1); plt.xlabel("Round"); plt.ylabel("Entropy"); plt.title("Uncertainty")
    plt.subplot(1, 2, 2); plt.xlabel("Round"); plt.ylabel("ESS"); plt.title("ESS")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_overlay"))


def plot_mean_curves(runs, outdir):
    minT = min(len(run["logs"]) for run in runs.values())
    #acc_curves = np.array([[row["exist_acc@0.5"] for row in run["logs"][:minT]]
    #                       for run in runs.values()])
    entropy_curves = np.array([[row["avg_pred_entropy"] for row in run["logs"][:minT]]
                               for run in runs.values()])
    rounds = np.arange(1, minT+1)
    # mean_acc, std_acc = acc_curves.mean(0), acc_curves.std(0)
    mean_entropy, std_entropy = entropy_curves.mean(0), entropy_curves.std(0)

    fig = plt.figure(figsize=(6,4))
    # plt.subplot(1,2,1)
    # plt.fill_between(rounds, mean_acc-std_acc, mean_acc+std_acc, alpha=0.3)
    # plt.plot(rounds, mean_acc, label="mean acc")
    # plt.xlabel("Round"); plt.ylabel("Accuracy @0.5"); plt.title("Mean ± std")
    # plt.subplot(1,2,2)
    plt.fill_between(rounds, mean_entropy-std_entropy, mean_entropy+std_entropy,
                     alpha=0.3)
    plt.plot(rounds, mean_entropy, label="mean entropy")
    plt.xlabel("Round"); plt.ylabel("Avg pred entropy"); plt.title("Mean ± std")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_mean"))


def plot_scoring_curves(runs, outdir):
    fig = plt.figure(figsize=(10, 4))
    # ETCP
    plt.subplot(1,2,1)
    for prefix, run in runs.items():
        logs = run["logs"]
        rounds = [row["round"] for row in logs]
        etcp = [row.get("exp_true_class_prob", np.nan) for row in logs]
        plt.plot(rounds, etcp, label=prefix)
    plt.xlabel("Round"); plt.ylabel("ETCP"); plt.title("Expected true-class prob")
    # Brier
    plt.subplot(1,2,2)
    for prefix, run in runs.items():
        logs = run["logs"]
        rounds = [row["round"] for row in logs]
        brier = [row.get("brier", np.nan) for row in logs]
        plt.plot(rounds, brier, label=prefix)
    plt.xlabel("Round"); plt.ylabel("Brier score"); plt.title("Brier (lower=better)")
    plt.legend()
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_scoring_overlay"))


def plot_mean_scoring_curves(runs, outdir):
    minT = min(len(run["logs"]) for run in runs.values())
    etcp_curves = np.array([[row.get("exp_true_class_prob", np.nan) for row in run["logs"][:minT]]
                             for run in runs.values()])
    brier_curves = np.array([[row.get("brier", np.nan) for row in run["logs"][:minT]]
                              for run in runs.values()])
    rounds = np.arange(1, minT+1)
    mean_etcp, std_etcp = np.nanmean(etcp_curves, 0), np.nanstd(etcp_curves, 0)
    mean_brier, std_brier = np.nanmean(brier_curves, 0), np.nanstd(brier_curves, 0)

    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.fill_between(rounds, mean_etcp-std_etcp, mean_etcp+std_etcp, alpha=0.3)
    plt.plot(rounds, mean_etcp, label="mean ETCP")
    plt.xlabel("Round"); plt.ylabel("ETCP"); plt.title("Mean ± std")

    plt.subplot(1,2,2)
    plt.fill_between(rounds, mean_brier-std_brier, mean_brier+std_brier, alpha=0.3, color="red")
    plt.plot(rounds, mean_brier, color="red", label="mean Brier")
    plt.xlabel("Round"); plt.ylabel("Brier score"); plt.title("Mean ± std")

    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_scoring_mean"))

def plot_structural_curves_from_samples(runs, outdir):
    fig = plt.figure(figsize=(15,8))
    keys = [
        ("samp_skel_precision","Skel Precision","Prec"),
        ("samp_skel_recall","Skel Recall","Rec"),
        ("samp_skel_f1","Skel F1","F1"),
        ("samp_orient_precision","Orient Precision","Prec"),
        ("samp_orient_recall","Orient Recall","Rec"),
        ("samp_orient_f1","Orient F1","F1"),
    ]
    for k,(key,title,y) in enumerate(keys):
        plt.subplot(2,3,k+1)
        for prefix, run in runs.items():
            rounds = [row["round"] for row in run["logs"]]
            vals = [row.get(key, np.nan) for row in run["logs"]]
            plt.plot(rounds, vals, label=prefix)
        plt.xlabel("Round"); plt.ylabel(y); plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_structural_samples_overlay"))

    # SHD separate (different scale)
    fig = plt.figure(figsize=(6,4))
    for prefix, run in runs.items():
        rounds = [row["round"] for row in run["logs"]]
        vals = [row.get("samp_shd", np.nan) for row in run["logs"]]
        plt.plot(rounds, vals, label=prefix)
    plt.xlabel("Round"); plt.ylabel("SHD"); plt.title("SHD (sample-averaged)")
    plt.legend()
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_shd_samples_overlay"))


def plot_mean_structural_curves_from_samples(runs, outdir):
    minT = min(len(run["logs"]) for run in runs.values())
    keys = [
        ("samp_skel_precision","Skel Precision"),
        ("samp_skel_recall","Skel Recall"),
        ("samp_skel_f1","Skel F1"),
        ("samp_orient_precision","Orient Precision"),
        ("samp_orient_recall","Orient Recall"),
        ("samp_orient_f1","Orient F1"),
    ]
    fig = plt.figure(figsize=(15,8))
    rounds = np.arange(1, minT+1)
    for k,(key,title) in enumerate(keys):
        curves = np.array([[row.get(key, np.nan) for row in run["logs"][:minT]]
                           for run in runs.values()])
        mean = np.nanmean(curves, 0)
        std  = np.nanstd(curves, 0)
        plt.subplot(2,3,k+1)
        plt.fill_between(rounds, mean-std, mean+std, alpha=0.3)
        plt.plot(rounds, mean, label=f"mean {key}")
        plt.xlabel("Round"); plt.ylabel(title); plt.title(f"{title} (mean ± std)")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_structural_samples_mean"))

    # SHD mean
    curves = np.array([[row.get("samp_shd", np.nan) for row in run["logs"][:minT]]
                       for run in runs.values()])
    mean = np.nanmean(curves, 0)
    std  = np.nanstd(curves, 0)
    fig = plt.figure(figsize=(6,4))
    plt.fill_between(rounds, mean-std, mean+std, alpha=0.3)
    plt.plot(rounds, mean, label="mean samp_shd")
    plt.xlabel("Round"); plt.ylabel("SHD"); plt.title("SHD (sample-averaged mean ± std)")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, "curves_shd_samples_mean"))





def plot_heatmaps(outdir, pick_run):
    A_true = np.load(os.path.join(outdir, f"{pick_run}_A_star.npy"))
    marg = np.load(os.path.join(outdir, f"{pick_run}_posterior_marginals.npy"))
    fig = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(A_true, cmap="Greys", interpolation="none")
    plt.title("True adjacency"); plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(marg, cmap="viridis", interpolation="none", vmin=0, vmax=1)
    plt.title("Posterior marginals"); plt.colorbar()
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, f"{pick_run}_heatmaps"))


def plot_orientation_trajectories(outdir, pick_run, max_nonedges=4):
    with open(os.path.join(outdir, f"{pick_run}_logs.json")) as f:
        logs = json.load(f)
    A_true = np.load(os.path.join(outdir, f"{pick_run}_A_star.npy"))
    D = A_true.shape[0]
    rounds = [row["round"] for row in logs]

    edges_true = [(i,j) for i in range(D) for j in range(D) if i!=j and A_true[i,j]==1]
    edges_to_track = []
    for (i,j) in edges_true:
        edges_to_track.append((i,j))
        edges_to_track.append((j,i))

    all_non = [(i,j) for i in range(D) for j in range(D)
               if i!=j and A_true[i,j]==0 and A_true[j,i]==0]
    if all_non:
        sample_non = random.sample(all_non, min(max_nonedges, len(all_non)))
        edges_to_track.extend(sample_non)

    fig = plt.figure(figsize=(10,6))
    for (i,j) in edges_to_track:
        probs = [row["marginals"][i][j] for row in logs]
        if A_true[i,j]==1:
            style, color = "-", "blue"
        elif A_true[j,i]==1:
            style, color = "--", "red"
        else:
            style, color = ":", "gray"
        plt.plot(rounds, probs, style, color=color,
                 label=f"{i}->{j} (true={A_true[i,j]})")
    plt.xlabel("Round"); plt.ylabel("Posterior P(i->j)")
    plt.title(f"Orientation trajectories ({pick_run})")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir, f"{pick_run}_orient"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory with outputs")
    ap.add_argument("--pick_run", type=str, default=None,
                    help="Run prefix (e.g. seed1) for heatmaps/orientation")
    args = ap.parse_args()

    runs = load_runs(args.outdir)
    if not runs:
        print("No runs found in", args.outdir); return

    plot_learning_curves(runs, args.outdir)
    plot_mean_curves(runs, args.outdir)
    plot_scoring_curves(runs, args.outdir)
    plot_mean_scoring_curves(runs, args.outdir)
    plot_structural_curves_from_samples(runs, args.outdir)
    plot_mean_structural_curves_from_samples(runs, args.outdir)

    if args.pick_run:
        plot_heatmaps(args.outdir, args.pick_run)
        plot_orientation_trajectories(args.outdir, args.pick_run)
    else:
        for pick_run in runs.keys():
            plot_heatmaps(args.outdir, pick_run)
            plot_orientation_trajectories(args.outdir, pick_run)

if __name__ == "__main__":
    main()

