
"""Iteration 1: CODE-02 enabled overlap schedule + stronger sparsity.
Trains 2 circuits with improved DiscoGP hyperparameters.
"""
import sys, os, json, time, shutil
from pathlib import Path

sys.path.insert(0, '/repo')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from circuit_discovery.run import (
    load_configs, load_model, load_task_dataset_from_config,
    train_loader_from_config, evaluate_circuit,
)
from circuit_discovery.algorithms.discogp import DiscoGP, DiscoGPConfig
from circuit_discovery.metrics import discogp_fidelity_loss, discogp_completeness_loss
from circuit_discovery.utils import set_seed
from circuit_discovery.circuit import complement, union

configs = load_configs()
params = configs['notebooks']['01_oasr_alternative_sheaves']['hyperparams']

# --- Iteration 1 hyperparameters ---
# CODE-02: enable overlap schedule (was hardcoded to 1.0/1.0/0/0 in run_oasr_experiment.py)
# + stronger sparsity: lambda_sparse_e 1.0 -> 1.5, max_times 20.0 -> 25.0
N_CIRCUITS = 2
LAMBDA_SPARSE_E = 1.5          # +50% from baseline 1.0
MAX_TIMES_SPARSE = 25.0         # +25% from baseline 20.0
LAMBDA_OVERLAP_E = 1.04         # same as config
MAX_TIMES_OVERLAP = 20.0        # CODE-02: was hardcoded to 1.0
MIN_TIMES_OVERLAP = 0.01        # CODE-02: was hardcoded to 1.0
N_EPOCHS = params['n_epochs_e'] # 40

warmup_e = int(0.8 * N_EPOCHS)
cooldown_e = max(N_EPOCHS - warmup_e, 0)

OUTPUT_DIR = Path('/repo/circuits_discovered/oasr_ioi_circuits')
BACKUP_DIR = Path('/repo/circuits_discovered/oasr_ioi_circuits_backup')

# Back up original circuits
if OUTPUT_DIR.exists() and not BACKUP_DIR.exists():
    shutil.copytree(str(OUTPUT_DIR), str(BACKUP_DIR))
    print(f'Backed up original circuits to {BACKUP_DIR}')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def make_config(overlap_penalty=False):
    return DiscoGPConfig(
        model_name=params['model_name'],
        prune_edges=True, prune_weights=False,
        n_epochs_e=N_EPOCHS,
        batch_size=params['batch_size'],
        lr_e=params['lr_e'],
        edge_logit_init_mean=params['edge_logit_init_mean'],
        edge_logit_init_std=params['edge_logit_init_std'],
        random_mode=params['random_mode'],
        gs_temp_edge=params['gs_temp_edge'],
        lambda_sparse_e=LAMBDA_SPARSE_E,
        min_times_lambda_sparse_e=params['min_times_lambda_sparse_e'],
        max_times_lambda_sparse_e=MAX_TIMES_SPARSE,
        n_epoch_warmup_lambda_sparse_e=warmup_e,
        n_epoch_cooldown_lambda_sparse_e=cooldown_e,
        lambda_complete_e=params['lambda_complete_e'],
        completeness_start_frac=params['completeness_start_frac'],
        lambda_overlap_e=LAMBDA_OVERLAP_E,
        # CODE-02: enable overlap schedule (was hardcoded to 1.0/1.0/0/0)
        min_times_lambda_overlap_e=MIN_TIMES_OVERLAP,
        max_times_lambda_overlap_e=MAX_TIMES_OVERLAP,
        n_epoch_warmup_lambda_overlap_e=warmup_e,
        n_epoch_cooldown_lambda_overlap_e=cooldown_e,
        overlap_penalty=overlap_penalty,
        tqdm_disabled=False,
    )

def evaluate_complement_accuracy(model, dataloader, circuit):
    comp = complement(circuit)
    for key, node in comp.nodes.items():
        for w_key in list(node.weight_masks.keys()):
            node.weight_masks[w_key] = None
    result = evaluate_circuit(model, dataloader, comp)
    return result['acc']

print('Loading data...')
data = load_task_dataset_from_config(params)
train_loader = train_loader_from_config(data.train.dataset, params)

print('Loading model...')
base_model = load_model(params['model_name'], device='cuda')

all_reference_circuits = []
results = []

for idx in range(N_CIRCUITS):
    seed = idx
    use_overlap = (len(all_reference_circuits) > 0)

    print(f'\n{"="*60}')
    print(f'Circuit {idx+1}/{N_CIRCUITS}, seed={seed}, overlap={use_overlap}')
    print(f'{"="*60}')

    t0 = time.time()
    set_seed(seed)
    runner = DiscoGP(model=base_model, config=make_config(overlap_penalty=use_overlap))

    if use_overlap:
        ref = all_reference_circuits[0]
        for c in all_reference_circuits[1:]:
            ref = union(ref, c)
        runner.load_reference_circuit(ref)

    circuit = runner.discover_circuit(
        train_loader,
        fidelity_loss_fn=discogp_fidelity_loss,
        completeness_loss_fn=discogp_completeness_loss,
        finalize=False,
    )

    elapsed = time.time() - t0

    # Evaluate
    eval_result = evaluate_circuit(base_model, data.test, circuit)
    comp_acc = evaluate_complement_accuracy(base_model, data.test, circuit)

    # Finalize and save
    finalized = base_model.finalize_circuit(circuit)
    all_reference_circuits.append(finalized)

    result = {
        'idx': idx + 1,
        'seed': seed,
        'acc': eval_result['acc'],
        'complement_acc': comp_acc,
        'edge_density': eval_result['edge_density'],
        'num_kept_edges': eval_result['num_kept_edges'],
        'elapsed_seconds': elapsed,
    }
    results.append(result)

    print(f'  acc={result["acc"]:.4f}, comp_acc={result["complement_acc"]:.4f}, '
          f'edge_density={result["edge_density"]:.4f}, edges={result["num_kept_edges"]}, '
          f'time={elapsed:.1f}s')

    # Save circuit
    save_path = OUTPUT_DIR / f'low_iou_{idx}.pt'
    torch.save({'circuit': finalized, 'seed': seed, 'algorithm': 'discogp_iter1'}, save_path)
    print(f'  Saved to {save_path}')

# Summary
accs = [r['acc'] for r in results]
comp_accs = [r['complement_acc'] for r in results]
eds = [r['edge_density'] for r in results]
ecs = [r['num_kept_edges'] for r in results]

summary = {
    'n_circuits': N_CIRCUITS,
    'mean_accuracy_pct': round(sum(accs)/len(accs)*100, 2),
    'mean_complement_accuracy_pct': round(sum(comp_accs)/len(comp_accs)*100, 2),
    'mean_edge_density_pct': round(sum(eds)/len(eds)*100, 2),
    'mean_edge_count': round(sum(ecs)/len(ecs), 1),
    'per_circuit': results,
}

print(f'\n{"="*60}')
print(f'ITERATION 1 SUMMARY')
print(f'{"="*60}')
print(json.dumps(summary, indent=2))

# Save summary for parsing
with open('/repo/experiment_results/iter1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\nIteration 1 training complete.')
