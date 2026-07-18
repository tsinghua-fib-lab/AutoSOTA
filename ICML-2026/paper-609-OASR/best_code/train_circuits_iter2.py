#!/usr/bin/env python3
"""Iteration 2: ALGO-06 GS temp annealing + CODE-02 overlap schedule + baseline seeds 42,43."""
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

# Iteration 2 hyperparameters
N_CIRCUITS = 2
SEEDS = [42, 43]
LAMBDA_SPARSE_E = params['lambda_sparse_e']
MAX_TIMES_SPARSE = params['max_times_lambda_sparse_e']
N_EPOCHS = params['n_epochs_e']

# CODE-02: enable overlap schedule
LAMBDA_OVERLAP_E = params['lambda_overlap_e']
MAX_TIMES_OVERLAP = 20.0
MIN_TIMES_OVERLAP = 0.01

# ALGO-06: GS temperature annealing
GS_TEMP_INIT = 2.0
GS_TEMP_FINAL = 0.1

warmup_e = int(0.8 * N_EPOCHS)
cooldown_e = max(N_EPOCHS - warmup_e, 0)

OUTPUT_DIR = Path('/repo/circuits_discovered/oasr_ioi_circuits')
BACKUP_DIR = Path('/repo/circuits_discovered/oasr_ioi_circuits_backup')

if OUTPUT_DIR.exists() and not BACKUP_DIR.exists():
    shutil.copytree(str(OUTPUT_DIR), str(BACKUP_DIR))
    print('Backed up original circuits to %s' % BACKUP_DIR)

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
        gs_temp_edge_init=GS_TEMP_INIT,
        gs_temp_edge_final=GS_TEMP_FINAL,
        lambda_sparse_e=LAMBDA_SPARSE_E,
        min_times_lambda_sparse_e=params['min_times_lambda_sparse_e'],
        max_times_lambda_sparse_e=MAX_TIMES_SPARSE,
        n_epoch_warmup_lambda_sparse_e=warmup_e,
        n_epoch_cooldown_lambda_sparse_e=cooldown_e,
        lambda_complete_e=params['lambda_complete_e'],
        completeness_start_frac=params['completeness_start_frac'],
        lambda_overlap_e=LAMBDA_OVERLAP_E,
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
    seed = SEEDS[idx]
    use_overlap = (len(all_reference_circuits) > 0)

    print('')
    print('=' * 60)
    print('Circuit %d/%d, seed=%d, overlap=%s' % (idx + 1, N_CIRCUITS, seed, use_overlap))
    print('  GS temp: %.1f -> %.1f (exponential)' % (GS_TEMP_INIT, GS_TEMP_FINAL))
    print('=' * 60)

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

    eval_result = evaluate_circuit(base_model, data.test, circuit)
    comp_acc = evaluate_complement_accuracy(base_model, data.test, circuit)

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

    print('  acc=%.4f, comp_acc=%.4f, edge_density=%.4f, edges=%d, time=%.1fs' % (
        result['acc'], result['complement_acc'], result['edge_density'],
        result['num_kept_edges'], elapsed))

    save_path = OUTPUT_DIR / ('low_iou_%d.pt' % idx)
    torch.save({'circuit': finalized, 'seed': seed, 'algorithm': 'discogp_iter2'}, save_path)
    print('  Saved to %s' % save_path)

# Summary
accs = [r['acc'] for r in results]
comp_accs = [r['complement_acc'] for r in results]
eds = [r['edge_density'] for r in results]
ecs = [r['num_kept_edges'] for r in results]

summary = {
    'iteration': 2,
    'idea': 'ALGO-06 + CODE-02',
    'n_circuits': N_CIRCUITS,
    'seeds': SEEDS,
    'gs_temp_init': GS_TEMP_INIT,
    'gs_temp_final': GS_TEMP_FINAL,
    'mean_accuracy_pct': round(sum(accs) / len(accs) * 100, 2),
    'mean_complement_accuracy_pct': round(sum(comp_accs) / len(comp_accs) * 100, 2),
    'mean_edge_density_pct': round(sum(eds) / len(eds) * 100, 2),
    'mean_edge_count': round(sum(ecs) / len(ecs), 1),
    'per_circuit': results,
}

print('')
print('=' * 60)
print('ITERATION 2 SUMMARY')
print('=' * 60)
print(json.dumps(summary, indent=2))

with open('/repo/experiment_results/iter2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('')
print('Iteration 2 training complete.')
