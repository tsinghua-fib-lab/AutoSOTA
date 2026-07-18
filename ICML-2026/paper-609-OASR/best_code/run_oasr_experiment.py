"""Run OASR experiment: DiscoGP with and without overlap penalty for multiple seeds."""

import sys, os, json, time, torch
from pathlib import Path

sys.path.insert(0, "/repo")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from circuit_discovery.run import (
    load_configs, load_model, load_task_dataset_from_config,
    train_loader_from_config, evaluate_circuit,
)
from circuit_discovery.algorithms.discogp import DiscoGP, DiscoGPConfig
from circuit_discovery.metrics import discogp_fidelity_loss, discogp_completeness_loss
from circuit_discovery.utils import set_seed
from circuit_discovery.circuit import complement

configs = load_configs()
params = configs["notebooks"]["01_oasr_alternative_sheaves"]["hyperparams"]

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODE = sys.argv[2] if len(sys.argv) > 2 else "independent"
OUTPUT_DIR = Path("/repo/experiment_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running {N_RUNS} runs in {MODE} mode")
print(f"Params: model={params['model_name']}, task={params['task']}, "
      f"train_size={params['train_size']}, test_size={params['test_size']}")

# Load data once
print("Loading data...")
data = load_task_dataset_from_config(params)
train_loader = train_loader_from_config(data.train.dataset, params)

warmup_e = int(0.8 * params["n_epochs_e"])
cooldown_e = max(params["n_epochs_e"] - warmup_e, 0)

def make_discogp_config(overlap_penalty=False):
    return DiscoGPConfig(
        model_name=params["model_name"],
        prune_edges=True, prune_weights=False,
        n_epochs_e=params["n_epochs_e"],
        batch_size=params["batch_size"],
        lr_e=params["lr_e"],
        edge_logit_init_mean=params["edge_logit_init_mean"],
        edge_logit_init_std=params["edge_logit_init_std"],
        random_mode=params["random_mode"],
        gs_temp_edge=params["gs_temp_edge"],
        lambda_sparse_e=params["lambda_sparse_e"],
        min_times_lambda_sparse_e=params["min_times_lambda_sparse_e"],
        max_times_lambda_sparse_e=params["max_times_lambda_sparse_e"],
        n_epoch_warmup_lambda_sparse_e=warmup_e,
        n_epoch_cooldown_lambda_sparse_e=cooldown_e,
        lambda_complete_e=params["lambda_complete_e"],
        completeness_start_frac=params["completeness_start_frac"],
        lambda_overlap_e=params["lambda_overlap_e"],
        min_times_lambda_overlap_e=1.0,
        max_times_lambda_overlap_e=1.0,
        n_epoch_warmup_lambda_overlap_e=0,
        n_epoch_cooldown_lambda_overlap_e=0,
        overlap_penalty=overlap_penalty,
        tqdm_disabled=False,
    )

# Helper: evaluate complement accuracy safely
def evaluate_complement_accuracy(model, dataloader, circuit):
    """Evaluate accuracy of the complement circuit (edges NOT in the circuit)."""
    comp = complement(circuit)
    # Reset weight masks to None to avoid shape issues
    for key, node in comp.nodes.items():
        for w_key in list(node.weight_masks.keys()):
            node.weight_masks[w_key] = None
    result = evaluate_circuit(model, dataloader, comp)
    return result["acc"]

all_reference_circuits = []
results = []

# Load model once, DiscoGP will use it (fresh edge logits per run)
print("Loading model (one-time)...")
base_model = load_model(params["model_name"], device="cuda")

for run_idx in range(N_RUNS):
    seed = run_idx
    use_overlap = (MODE == "overlap") and (len(all_reference_circuits) > 0)

    print(f"\n{'='*60}")
    print(f"Run {run_idx+1}/{N_RUNS}, seed={seed}, overlap={use_overlap}")
    print(f"{'='*60}")

    t0 = time.time()

    # Reuse model but create fresh DiscoGP instance (fresh edge logits)
    set_seed(seed)
    runner = DiscoGP(model=base_model, config=make_discogp_config(overlap_penalty=use_overlap))

    if use_overlap:
        # Use union of all previous circuits as reference for overlap penalty
        from circuit_discovery.circuit import union
        ref = all_reference_circuits[0]
        for c in all_reference_circuits[1:]:
            ref = union(ref, c)
        runner.load_reference_circuit(ref)

    circuit = runner.discover_circuit(
        train_loader,
        fidelity_loss_fn=discogp_fidelity_loss,
        completeness_loss_fn=discogp_completeness_loss,
        finalize=False,  # Don't finalize yet; we need complement
    )

    elapsed = time.time() - t0

    # Evaluate original circuit
    eval_result = evaluate_circuit(base_model, data.test, circuit)

    # Evaluate complement accuracy
    comp_acc = evaluate_complement_accuracy(base_model, data.test, circuit)

    # Now finalize for saving
    finalized = base_model.finalize_circuit(circuit)
    all_reference_circuits.append(finalized)

    result = {
        "run": run_idx + 1,
        "seed": seed,
        "mode": MODE,
        "use_overlap": use_overlap,
        "acc": eval_result["acc"],
        "complement_acc": comp_acc,
        "edge_density": eval_result["edge_density"],
        "num_kept_edges": eval_result["num_kept_edges"],
        "num_total_edges": eval_result.get("num_edges", 0),
        "elapsed_seconds": elapsed,
    }
    results.append(result)

    print(f"  acc={result['acc']:.4f}, comp_acc={result['complement_acc']:.4f}, "
          f"edge_density={result['edge_density']:.4f}, edges={result['num_kept_edges']}, "
          f"time={elapsed:.1f}s")

    # Save checkpoint (finalized circuit)
    torch.save({
        "circuit": finalized,
        "seed": seed,
        "algorithm": "discogp",
        "mode": MODE,
    }, OUTPUT_DIR / f"run_{run_idx+1:02d}_seed_{seed}.pt")

    # Periodic summary
    if (run_idx + 1) % 5 == 0 or run_idx == N_RUNS - 1:
        accs = [r["acc"] for r in results]
        comp_accs = [r["complement_acc"] for r in results]
        eds = [r["edge_density"] for r in results]
        ecs = [r["num_kept_edges"] for r in results]
        print(f"\n  >>> After {run_idx+1} runs: "
              f"acc={sum(accs)/len(accs):.4f} [{min(accs):.4f}-{max(accs):.4f}], "
              f"comp_acc={sum(comp_accs)/len(comp_accs):.4f}, "
              f"edge_density={sum(eds)/len(eds):.4f}, "
              f"edges={sum(ecs)/len(ecs):.1f}")

# Final summary
accs = [r["acc"] for r in results]
comp_accs = [r["complement_acc"] for r in results]
edge_densities = [r["edge_density"] for r in results]
edge_counts = [r["num_kept_edges"] for r in results]

summary = {
    "mode": MODE,
    "n_runs": N_RUNS,
    "mean_acc": sum(accs) / len(accs),
    "min_acc": min(accs),
    "max_acc": max(accs),
    "mean_complement_acc": sum(comp_accs) / len(comp_accs),
    "min_complement_acc": min(comp_accs),
    "max_complement_acc": max(comp_accs),
    "mean_edge_density": sum(edge_densities) / len(edge_densities),
    "min_edge_density": min(edge_densities),
    "max_edge_density": max(edge_densities),
    "mean_edge_count": sum(edge_counts) / len(edge_counts),
    "min_edge_count": min(edge_counts),
    "max_edge_count": max(edge_counts),
    "per_run": results,
}

print(f"\n{'='*60}")
print(f"FINAL SUMMARY ({MODE}, {N_RUNS} runs)")
print(f"{'='*60}")
print(f"Accuracy:      mean={summary['mean_acc']:.4f}, min={summary['min_acc']:.4f}, max={summary['max_acc']:.4f}")
print(f"Complement Acc: mean={summary['mean_complement_acc']:.4f}, min={summary['min_complement_acc']:.4f}, max={summary['max_complement_acc']:.4f}")
print(f"Edge Density:   mean={summary['mean_edge_density']:.4f}, min={summary['min_edge_density']:.4f}, max={summary['max_edge_density']:.4f}")
print(f"Edge Count:     mean={summary['mean_edge_count']:.1f}, min={summary['min_edge_count']}, max={summary['max_edge_count']}")

# Compare with paper values
print(f"\nPaper OASR:    acc=0.9959, comp_acc=0.4587, edge_density=0.0286, edges=928.5")
print(f"Paper Random:  acc=0.9995, comp_acc=0.4564, edge_density=0.0304, edges=987.0")

with open(OUTPUT_DIR / f"summary_{MODE}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}")
