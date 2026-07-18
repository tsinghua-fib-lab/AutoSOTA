"""Evaluate all circuits for IOI metrics: accuracy, complement accuracy, edge density, edge count."""
import sys, os, json
sys.path.insert(0, "/repo")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from circuit_discovery.run import load_configs, load_model, load_task_dataset_from_config, evaluate_circuit
from circuit_discovery.circuit import complement
from circuit_discovery.visualization import load_circuit
import torch

configs = load_configs()
params = configs["notebooks"]["01_oasr_alternative_sheaves"]["hyperparams"]

print("Loading model...")
model = load_model(params["model_name"], device="cuda")
print("Loading data...")
data = load_task_dataset_from_config(params)

def eval_circuit_safe(name, circuit):
    """Evaluate circuit and its complement."""
    result = evaluate_circuit(model, data.test, circuit)
    # Complement: only invert edges, keep weight masks as None
    comp = complement(circuit)
    for key, node in comp.nodes.items():
        for w_key in list(node.weight_masks.keys()):
            node.weight_masks[w_key] = None
    comp_result = evaluate_circuit(model, data.test, comp)
    return {
        "name": name,
        "acc": result["acc"],
        "complement_acc": comp_result["acc"],
        "edge_density": result["edge_density"],
        "num_kept_edges": result["num_kept_edges"],
        "num_total_edges": result.get("num_edges", 0),
    }

results = []

# Pre-computed DiscoGP circuits
disco_dir = Path("/repo/circuits_discovered/discogp_circuits")
for pt_file in sorted(disco_dir.glob("*.pt")):
    name = pt_file.stem
    try:
        circuit = load_circuit(pt_file)
        r = eval_circuit_safe(name, circuit)
        results.append(r)
        print(f"  disco/{name}: acc={r['acc']:.4f}, comp_acc={r['complement_acc']:.4f}, "
              f"edge_density={r['edge_density']:.4f}, edges={r['num_kept_edges']}")
    except Exception as e:
        print(f"  disco/{name}: ERROR {e}")

# OASR IOI circuits
oasr_dir = Path("/repo/circuits_discovered/oasr_ioi_circuits")
for pt_file in sorted(oasr_dir.glob("*.pt")):
    name = pt_file.stem
    try:
        circuit = load_circuit(pt_file)
        r = eval_circuit_safe(name, circuit)
        results.append(r)
        print(f"  oasr/{name}: acc={r['acc']:.4f}, comp_acc={r['complement_acc']:.4f}, "
              f"edge_density={r['edge_density']:.4f}, edges={r['num_kept_edges']}")
    except Exception as e:
        print(f"  oasr/{name}: ERROR {e}")

# Experiment results
exp_dir = Path("/repo/experiment_results")
for pt_file in sorted(exp_dir.glob("run_*.pt")):
    name = pt_file.stem
    try:
        data_pt = torch.load(pt_file, map_location="cpu", weights_only=False)
        circuit = data_pt["circuit"]
        r = eval_circuit_safe(name, circuit)
        r["seed"] = data_pt.get("seed", "?")
        r["mode"] = data_pt.get("mode", "?")
        results.append(r)
        print(f"  exp/{name}: acc={r['acc']:.4f}, comp_acc={r['complement_acc']:.4f}, "
              f"edge_density={r['edge_density']:.4f}, edges={r['num_kept_edges']}")
    except Exception as e:
        print(f"  exp/{name}: ERROR {e}")

# Compute summary for the OASR circuits specifically
oasr_results = [r for r in results if "oasr" in r["name"] or "low_iou" in r["name"]]
if oasr_results:
    accs = [r["acc"] for r in oasr_results]
    comps = [r["complement_acc"] for r in oasr_results]
    eds = [r["edge_density"] for r in oasr_results]
    ecs = [r["num_kept_edges"] for r in oasr_results]
    print(f"\nOASR circuits summary ({len(oasr_results)} circuits):")
    print(f"  Accuracy: mean={sum(accs)/len(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")
    print(f"  Complement: mean={sum(comps)/len(comps):.4f}, min={min(comps):.4f}, max={max(comps):.4f}")
    print(f"  Edge Density: mean={sum(eds)/len(eds):.4f}, min={min(eds):.4f}, max={max(eds):.4f}")
    print(f"  Edge Count: mean={sum(ecs)/len(ecs):.1f}, min={min(ecs)}, max={max(ecs)}")

# Save results
with open("/repo/experiment_results/all_evaluations.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} evaluations to /repo/experiment_results/all_evaluations.json")
