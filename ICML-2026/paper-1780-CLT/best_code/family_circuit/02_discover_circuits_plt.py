"""
Circuit Discovery for PLT
Identifies sparse sub-networks (circuits) within the PLT that detect specific protein families.
Pipeline: Train Linear Probe -> Compute Attribution (Train) -> Greedy Selection (Val) -> Evaluation (Test).
"""
import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
sys.path.append('../training_transcoder')
sys.path.append('..')
from family_utils import get_data, train_probe, evaluate_circuit, split_data
from circuit_utils.plt_circuit import CircuitDiscovererPLT
from circuit_utils.circuit_utils import compute_attribution, rank_nodes, circuit_search
import gc
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

PLT_CHECKPOINT = os.environ.get("PLT_CHECKPOINT")
ESM_WEIGHTS = os.environ.get("ESM_WEIGHTS")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "families")
MASTER_NAME = os.environ.get("MASTER_NPZ_NAME", "all_acts.npz")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_limit", type=int, default=128)
    parser.add_argument("--recovery_ratio", type=float, default=0.7) 
    parser.add_argument("--step_size", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_freeze_attention", action="store_true", help="Disable frozen attention.")
    parser.add_argument("--source", type=str, default="mlp_output", choices=["mlp_output", "layer_output"], help="Target for reconstruction: 'mlp_output' (default) or 'layer_output'.")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    discoverer = CircuitDiscovererPLT(device)
    save_subdir = "PLT_no_frozen" if args.no_freeze_attention else "PLT"
    freeze_attn = not args.no_freeze_attention
    
    # Load Data
    path = Path(OUTPUT_DIR) / (f"{args.target}.npz" if args.target else MASTER_NAME)
    Path(os.path.join(OUTPUT_DIR, save_subdir)).mkdir(parents=True, exist_ok=True)
    if not path.exists(): sys.exit(f"Data not found at {path}")
    data = np.load(path, allow_pickle=True)
    unique_ids, counts = np.unique(data["interpro_ids"], return_counts=True)
    counts_dict = dict(zip(unique_ids, counts))
    if args.target:
        targets = [args.target]
    else:
        sorted_indices = np.argsort(-counts)
        targets = unique_ids[sorted_indices]
    if args.limit: 
        targets = targets[:args.limit]
    valid_targets = [t for t in targets if t != "NEGATIVE" and counts_dict.get(t, 0) >= 6]

    print(f"Starting PLT discovery using source: {args.source}")
    for family in tqdm(valid_targets, desc="Scanning Families (PLT)", position=0):
        if not args.overwrite and (Path(OUTPUT_DIR) / save_subdir / f"{family}.json").exists():
            continue
        
        X, y, seqs, n_total_pos = get_data(family, data["embeddings"], data["sequences"], data["interpro_ids"])
        if X is None: continue
        
        # 1. Split data
        X_train, X_val, X_test, y_train, y_val, y_test, seq_train, seq_val, seq_test = split_data(X, y, seqs, args.test_size, args.val_limit)
                
        # 2. Train probe
        probe = train_probe(X_train, y_train, device)

        # 3. Check the max possible F1. Define a ceiling based on the performance of max_f1_val
        probe.eval()
        with torch.no_grad():
            X_val_tensor = torch.as_tensor(X_val, device=device, dtype=torch.float32)
            logits = probe(X_val_tensor).squeeze(-1) # (Batch, 1) -> (Batch,)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
        clean_f1_val = f1_score(y_val, preds)
        max_f1_val = evaluate_circuit(discoverer, probe, seq_val, y_val, None, BATCH_SIZE, mean_pooled=True, freeze_attention=freeze_attn, source=args.source)['f1']
        standard_target = clean_f1_val * args.recovery_ratio
        if max_f1_val < standard_target:
            target_f1 = max_f1_val
            tqdm.write(f"[{family}] Target adjusted: Max F1 ({max_f1_val:.3f}) < Standard ({standard_target:.3f}). New Target: {target_f1:.3f}")
        else:
            target_f1 = standard_target
            
        # 4. Compute attribution
        pos_val_seqs = seq_val[y_val==1]            
        global_attr = compute_attribution(discoverer, probe, pos_val_seqs, BATCH_SIZE, freeze_attention=freeze_attn, source=args.source)
        ranking = rank_nodes(global_attr)

        # 5. Circuit Selection
        eval_fn = lambda d, p, s, y_true, nodes, bs: evaluate_circuit(d, p, s, y_true, nodes, bs, mean_pooled=True, freeze_attention=freeze_attn, source=args.source)['f1']
        best_nodes, best_k, val_recovered_f1 = circuit_search(
            discoverer, 
            probe, 
            ranking, 
            seq_val, 
            y_val, 
            target_metric=target_f1,
            metric_fn=eval_fn,
            step_size=args.step_size,
            max_nodes=args.max_nodes,
            desc=f"[{family}]"
        )

        # 6. Final Evaluation (Test Set)
        with torch.no_grad():
            X_test_tensor = torch.as_tensor(X_test, device=device, dtype=torch.float32)
            logits = probe(X_test_tensor).squeeze(-1) # (Batch, 1) -> (Batch,)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
        clean_f1_test = f1_score(y_test, preds)
        max_metrics_test = evaluate_circuit(discoverer, probe, seq_test, y_test, None, BATCH_SIZE, mean_pooled=True, gt_embeddings=X_test, freeze_attention=freeze_attn, source=args.source)
        base_metrics_test = evaluate_circuit(discoverer, probe, seq_test, y_test, {}, BATCH_SIZE, mean_pooled=True, gt_embeddings=X_test, freeze_attention=freeze_attn, source=args.source)
        test_metrics = evaluate_circuit(discoverer, probe, seq_test, y_test, best_nodes, BATCH_SIZE, mean_pooled=True, gt_embeddings=X_test, freeze_attention=freeze_attn, source=args.source)
        test_f1 = test_metrics['f1']
        test_nmse = test_metrics['nmse']
        recovered_ratio = 0.0
        denom = clean_f1_test - base_metrics_test['f1']
        if denom > 1e-6:
            recovered_ratio = (test_f1 - base_metrics_test['f1']) / denom
        tqdm.write(f"[{family}] n={n_total_pos} | Clean: {clean_f1_test:.2f} | Max: {max_metrics_test['f1']:.2f} | Recov F1: {test_f1:.2f} | Recov NMSE: {test_nmse:.4f} | Nodes: {best_k}")
        discoverer.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save
        res = {
            "family": family,
            "n_sequences": int(n_total_pos),
            "k": best_k,
            "source": args.source,
            "freeze_attention": freeze_attn,
            "clean_f1": float(clean_f1_test),
            "max_f1": float(max_metrics_test['f1']),
            "max_nmse": float(max_metrics_test['nmse']),
            "base_f1": float(base_metrics_test['f1']),
            "base_nmse": float(base_metrics_test['nmse']),
            "recovered_f1": float(test_f1),
            "recovered_nmse": float(test_nmse),
            "recovered_ratio": recovered_ratio,
            "nodes": {str(l): list(s) for l, s in best_nodes.items()}
        }
        with open(Path(OUTPUT_DIR) / save_subdir / f"{family}.json", "w") as f:
            json.dump(res, f, indent=2)
            
if __name__ == "__main__":
    main()