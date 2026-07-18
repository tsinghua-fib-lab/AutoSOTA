#!/usr/bin/env python3
"""Iteration 3: Extended epochs (60) + lower GS temp (0.5) + better schedule split.

Key insight from iters 1-2: the baseline warmup spends 80% of training ramping UP,
reaching max sparsity only briefly before cooling down. By extending epochs and
redistributing the schedule, we give the model more time at high sparsity pressure.

Changes:
- n_epochs_e: 40 -> 60 (50% more training)
- gs_temp_edge: 1.0 -> 0.5 (sharper masks during training)
- warmup: 20 epochs (fast ramp to max), cooldown: 40 epochs (slow descent)
- seeds 42, 43 (baseline seeds)
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

N_CIRCUITS = 2
SEEDS = [42, 43]
N_EPOCHS = 60  # up from 40
GS_TEMP = 0.5  # down from 1.0 - sharper masks
LAMBDA_SPARSE_E = params['lambda_sparse_e']  # 1.0
MAX_TIMES = params['max_times_lambda_sparse_e']  # 20.0

# Better schedule: fast ramp to max (20 epochs), slow descent (40 epochs)
WARMUP = 20
COOLDOWN = 40

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
        gs_temp_edge=GS_TEMP,
        lambda_sparse_e=LAMBDA_SPARSE_E,
        min_times_lambda_sparse_e=params['min_times_lambda_sparse_e'],
        max_times_lambda_sparse_e=MAX_TIMES,
        n_epoch_warmup_lambda_sparse_e=WARMUP,
        n_epoch_cooldown_lambda_sparse_e=COOLDOWN,
        lambda_complete_e=params['lambda_complete_e'],
        completeness_start_frac=params['completeness_start_frac'],
        lambda_overlap_e=params['lambda_overlap_e'],
        min_times_lambda_overlap_e=1.0,
        max_times_lambda_overlap_e=1.0,
        n_epoch_warmup_lambda_overlap_e=0,
        n_epoch_cooldown_lambda_overlap_e=0,
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
    print('  n_epochs=%d, gs_temp=%.1f, warmup=%d, cooldown=%d' % (
        N_EPOCHS, GS_TEMP, WARMUP, COOLDOWN))
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
    torch.save({'circuit': finalized, 'seed': seed, 'algorithm': 'discogp_iter3'}, save_path)
    print('  Saved to %s' % save_path)

# Summary
accs = [r['acc'] for r in results]
comp_accs = [r['complement_acc'] for r in results]
eds = [r['edge_density'] for r in results]
ecs = [r['num_kept_edges'] for r in results]

summary = {
    'iteration': 3,
    'idea': 'More epochs + lower GS temp + better schedule split',
    'n_epochs': N_EPOCHS,
    'gs_temp': GS_TEMP,
    'warmup': WARMUP,
    'cooldown': COOLDOWN,
    'n_circuits': N_CIRCUITS,
    'seeds': SEEDS,
    'mean_accuracy_pct': round(sum(accs) / len(accs) * 100, 2),
    'mean_complement_accuracy_pct': round(sum(comp_accs) / len(comp_accs) * 100, 2),
    'mean_edge_density_pct': round(sum(eds) / len(eds) * 100, 2),
    'mean_edge_count': round(sum(ecs) / len(ecs), 1),
    'per_circuit': results,
}

print('')
print('=' * 60)
print('ITERATION 3 SUMMARY')
print('=' * 60)
print(json.dumps(summary, indent=2))

with open('/repo/experiment_results/iter3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('')
print('Iteration 3 training complete.')
