"""Evaluate IOI circuits for paper 609 reproduction metrics.

Usage:
    python3 eval_ioi_circuits.py [circuit_path_or_dir]

If no argument, evaluates OASR circuits from circuits_discovered/oasr_ioi_circuits/
"""
import sys, os, json
from pathlib import Path

sys.path.insert(0, "/repo")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from circuit_discovery.run import load_configs, load_model, load_task_dataset_from_config, evaluate_circuit
from circuit_discovery.circuit import complement
from circuit_discovery.visualization import load_circuit

configs = load_configs()
params = configs["notebooks"]["01_oasr_alternative_sheaves"]["hyperparams"]

print("Loading GPT-2 Small model...")
model = load_model(params["model_name"], device="cuda")
print("Loading IOI test data...")
data = load_task_dataset_from_config(params)

def eval_with_complement(circuit, name="circuit"):
    result = evaluate_circuit(model, data.test, circuit)
    comp = complement(circuit)
    for key, node in comp.nodes.items():
        for w_key in list(node.weight_masks.keys()):
            node.weight_masks[w_key] = None
    comp_result = evaluate_circuit(model, data.test, comp)
    return {
        "name": name,
        "accuracy_pct": round(result["acc"] * 100, 2),
        "complement_accuracy_pct": round(comp_result["acc"] * 100, 2),
        "edge_density_pct": round(result["edge_density"] * 100, 2),
        "edge_count": result["num_kept_edges"],
    }

# Find circuits to evaluate
target = sys.argv[1] if len(sys.argv) > 1 else "/repo/circuits_discovered/oasr_ioi_circuits"
target_path = Path(target)

circuit_files = []
if target_path.is_dir():
    circuit_files = sorted(target_path.glob("*.pt"))
elif target_path.is_file():
    circuit_files = [target_path]
else:
    # Default: use OASR circuits
    default_dir = Path("/repo/circuits_discovered/oasr_ioi_circuits")
    circuit_files = sorted(default_dir.glob("*.pt"))

if not circuit_files:
    print("No circuit files found!")
    sys.exit(1)

results = []
for pt_file in circuit_files:
    print("Evaluating: " + pt_file.name)
    try:
        data_pt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
        circuit = data_pt.get("circuit", data_pt)
        r = eval_with_complement(circuit, pt_file.stem)
        results.append(r)
        print(json.dumps(r, indent=2))
    except Exception as e:
        print("  ERROR: " + str(e))

# Summary
if results:
    accs = [r["accuracy_pct"] for r in results]
    comps = [r["complement_accuracy_pct"] for r in results]
    eds = [r["edge_density_pct"] for r in results]
    ecs = [r["edge_count"] for r in results]
    summary = {
        "n_circuits": len(results),
        "mean_accuracy_pct": round(sum(accs) / len(accs), 2),
        "mean_complement_accuracy_pct": round(sum(comps) / len(comps), 2),
        "mean_edge_density_pct": round(sum(eds) / len(eds), 2),
        "mean_edge_count": round(sum(ecs) / len(ecs), 1),
        "per_circuit": results,
    }
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    with open("/repo/experiment_results/reproduction_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
