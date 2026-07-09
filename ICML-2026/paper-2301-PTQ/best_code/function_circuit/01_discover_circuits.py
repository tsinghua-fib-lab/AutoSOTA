#!/usr/bin/env python3
"""
DMS Circuit Discovery Pipeline (CNN Probe + CLT & PLT)
"""
import sys
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
sys.path.append('../training')
sys.path.append('../training_transcoder')
sys.path.append('../family_circuit')
sys.path.append('..')
from function_utils import CNNProbe, precompute_embeddings, estimate_memmap_size, EmbeddingDataset, train_probe_cnn, evaluate_circuit, evaluate_probe_direct
from circuit_utils.circuit_utils import compute_attribution, rank_nodes, circuit_search
from circuit_utils.esm_activation import ESMInference
from circuit_utils.clt_circuit import CircuitDiscovererCLT
from circuit_utils.plt_circuit import CircuitDiscovererPLT
import gc
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

BATCH_SIZE = 128
BATCH_SIZE_CIRCUIT = 64
CNN_TOTAL_STEPS = 10000
CNN_WARMUP = 100
ESM_WEIGHTS = os.environ.get("ESM_WEIGHTS")
CLT_CHECKPOINT = os.environ.get("CLT_CHECKPOINT")
PLT_CHECKPOINT = os.environ.get("PLT_CHECKPOINT")

# Define Method Configs
METHOD_CONFIGS = {
    "CLT_direct": {
        "discoverer_cls": CircuitDiscovererCLT,
        "ckpt": CLT_CHECKPOINT,
        "flags": {
            "sequential": False,        # Direct Mode
            "freeze_attention": True,  # Standard assumption
            "source": "mlp_output"      # Targets MLP specifically
        }
    },
    "PLT": {
        "discoverer_cls": CircuitDiscovererPLT,
        "ckpt": PLT_CHECKPOINT,
        "flags": {
            "sequential": True,         # Sequential Mode
            "freeze_attention": True,   # Isolate MLP/PLT
            "source": "mlp_output"
        }
    },
    "CLT_sequential": {
        "discoverer_cls": CircuitDiscovererCLT,
        "ckpt": CLT_CHECKPOINT,
        "flags": {
            "sequential": True,        # Sequential Mode
            "freeze_attention": True,  # Standard assumption
            "source": "mlp_output"      # Targets MLP specifically
        }
    },
    "CLT_sequential_no_frozen": {
        "discoverer_cls": CircuitDiscovererCLT,
        "ckpt": CLT_CHECKPOINT,
        "flags": {
            "sequential": True,        # Direct Mode
            "freeze_attention": False,  # Standard assumption
            "source": "mlp_output"      # Targets MLP specifically
        }
    },
    "PLT_no_frozen": {
        "discoverer_cls": CircuitDiscovererPLT,
        "ckpt": PLT_CHECKPOINT,
        "flags": {
            "sequential": True,         # Sequential Mode
            "freeze_attention": False,   # Isolate MLP/PLT
            "source": "mlp_output"
        }
    }
}

class MemmapSubset:
    """Helper to slice a master memmap without copying data."""
    def __init__(self, memmap_obj, indices):
        self.memmap = memmap_obj
        self.indices = indices
    
    def __getitem__(self, idx):
        # Handle slicing (e.g., [0:8]) by slicing the indices array first
        global_indices = self.indices[idx] 
        return self.memmap[global_indices]
    
    def __len__(self):
        return len(self.indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_root", default="DMS", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recovery_ratio", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mem_limit_gb", type=float, default=100.0, help="Max GB for embeddings before chunking")
    parser.add_argument("--step_size", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--val_limit", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=320)
    parser.add_argument("--output_dir", type=str, default="functions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Initialize
    inference = ESMInference(device, esm_weights_path=ESM_WEIGHTS, num_layers=args.layers, d_model=args.hidden_size)    
    discoverers = {}
    for name, cfg in METHOD_CONFIGS.items():
        if cfg["ckpt"]:
            print(f"Loading {name} Discoverer...")
            discoverers[name] = cfg["discoverer_cls"](device, ckpt_path=cfg["ckpt"], esm_weights_path=ESM_WEIGHTS)

    # 2. Specify DMS Directories
    base_dir = Path(args.dms_root)
    cache_root = base_dir.parent / "embeddings_cache"
    subfolders = ["cv_folds_single_substitutions", "cv_folds_multiples_substitutions"]
    
    for subfolder in subfolders:
        input_dir = base_dir / subfolder
        if not input_dir.exists(): continue
        
        # Load files
        dtype = "single" if "single" in subfolder else "multiples"
        print(f"\n{'='*10} Processing {dtype} {'='*10}")        
        if args.layers == 6:
            probe_dir = Path(f"probe/{dtype}")
        elif args.layers == 12:
            probe_dir = Path(f"probe_35M/{dtype}")
        else:
            probe_dir = Path(f"probe_L{args.layers}/{dtype}")
        func_dirs = {name: Path(f"{args.output_dir}/{name}/{dtype}") for name in METHOD_CONFIGS}
        for p in [probe_dir] + list(func_dirs.values()):
            p.mkdir(parents=True, exist_ok=True)
        csv_files = [
            f for f in input_dir.glob("*.csv") 
        ]
        
        for csv_file in tqdm(csv_files, desc=f"Datasets ({dtype})"):
            dms_name = csv_file.stem
            cache_key = f"{dms_name}_{dtype}"
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                df = pd.read_csv(csv_file)
                seqs = df["mutated_sequence"].values
                labels = df["DMS_score"].values
            except Exception as e:
                print(f"Error loading {dms_name}: {e}")
                continue

            if not args.overwrite:
                fold_cols_check = [c for c in df.columns if "fold_" in c]
                all_done = True
                for fc in fold_cols_check:
                    m_name = fc.replace("fold_", "")
                    for i in range(5):
                        for name in METHOD_CONFIGS:
                            if not (func_dirs[name] / dms_name / f"{m_name}_fold{i}.json").exists():
                                all_done = False
                if all_done: 
                    continue

            # Check if all folds exist
            if not args.overwrite:
                fold_cols_check = [c for c in df.columns if "fold_" in c]
                all_done = True
                for fc in fold_cols_check:
                    m_name = fc.replace("fold_", "")
                    for i in range(5):
                        for name in METHOD_CONFIGS:
                            if not (func_dirs[name] / dms_name / f"{m_name}_fold{i}.json").exists():
                                all_done = False
                if all_done: continue

            # 3. Compute embeddings. If embeddings too large, use chunking
            L_est = len(seqs[0])
            D_est = inference.model.embed_dim
            est_gb = estimate_memmap_size(len(seqs), L_est, D_est)
            use_chunking = est_gb > args.mem_limit_gb
            master_embeddings = None            
            if not use_chunking:
                try:
                    master_embeddings = precompute_embeddings(
                        seqs, inference, cache_key, 
                        target_layer=-1, 
                        indices=None,   
                        suffix="",      
                        cache_dir=str(cache_root),
                        batch_size=BATCH_SIZE
                    )
                except Exception as e:
                    print(f"Failed to create master embedding: {e}")
                    continue
            else:
                tqdm.write(f"Dataset {dms_name} too large ({est_gb:.2f} GB). Using Chunking Strategy.")

            # 4. Cross Validation Loop
            fold_cols = [c for c in df.columns if "fold_" in c]
            for fold_col in fold_cols:
                folds = df[fold_col].values
                method_name = fold_col.replace("fold_", "")
                
                for i in range(5): 
                    expt_name = f"{method_name}_fold{i}"
                    probe_path = probe_dir / dms_name / f"{method_name}_fold{i}_cnn.pt"
                    probe_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Split Indices
                    fold_vals = folds - folds.min() 
                    test_mask = (fold_vals == i)
                    val_fold = (i + 1) % 5
                    val_mask = (fold_vals == val_fold)
                    train_mask = ~(test_mask | val_mask)
                    train_indices = np.where(train_mask)[0]
                    val_indices = np.where(val_mask)[0]
                    test_indices = np.where(test_mask)[0]

                    input_dim = D_est
                    cnn_probe = CNNProbe(input_dim).to(device)
                    
                    # 5. Train or Load Probe
                    if probe_path.exists() and not args.overwrite:
                        cnn_probe.load_state_dict(torch.load(probe_path, map_location=device))
                    else:
                        if not use_chunking:
                            train_ds = EmbeddingDataset(master_embeddings, labels[train_indices], indices=train_indices)
                            val_ds = EmbeddingDataset(master_embeddings, labels[val_indices], indices=val_indices)
                            cnn_probe, _, _, _ = train_probe_cnn(
                                train_ds, val_ds, input_dim, device, 
                                total_steps=CNN_TOTAL_STEPS, batch_size=BATCH_SIZE,
                                recycle_data=True
                            )
                        else:
                            # 5.1. Compute Val Data 
                            val_embs = precompute_embeddings(
                                seqs, inference, cache_key, indices=val_indices,
                                suffix=f"val_fold{val_fold}", cache_dir=str(cache_root), batch_size=BATCH_SIZE
                            )
                            val_ds = EmbeddingDataset(val_embs, labels[val_indices])

                            # 5.2. Set train chunks and compute each chunk individually for training (to prevent loading all training chunks at once)
                            train_fold_ids = [f for f in range(5) if f != i and f != val_fold]
                            probe_state = {
                                "probe": cnn_probe, "optimizer": None, "scheduler": None,
                                "start_step": 0, "best_val_loss": float('inf'), "best_val_corr": 0,
                                "patience_counter": 0, "best_state": None
                            }
                            chunk_done = False
                            for chunk_id, f_id in enumerate(train_fold_ids):
                                if chunk_done: break
                                chunk_mask = (fold_vals == f_id)
                                chunk_indices = np.where(chunk_mask)[0]

                                chunk_fname = f"{cache_key}_train_chunk_fold{f_id}_L{args.layers - 1}_mlp_seq.dat"
                                chunk_path = cache_root / chunk_fname
                                if chunk_path.exists():
                                    try:
                                        print(f"Removing orphan chunk: {chunk_fname}")
                                        os.remove(chunk_path)
                                    except OSError:
                                        print(f"Warning: Could not delete {chunk_fname}")
                                
                                # 5.3. Compute temp chunk
                                chunk_embs = precompute_embeddings(
                                    seqs, inference, cache_key, indices=chunk_indices,
                                    suffix=f"train_chunk_fold{f_id}", cache_dir=str(cache_root), batch_size=BATCH_SIZE
                                )
                                chunk_ds = EmbeddingDataset(chunk_embs, labels[chunk_indices])
                                
                                # 5.4. Train CNN. If early stopping condition met, break. Otherwise, iterate through other chunks
                                # Split total steps across chunks (approx 3 chunks for 5-fold CV)
                                steps_per_chunk = int(CNN_TOTAL_STEPS / 3)
                                cnn_probe, _, state, stopped = train_probe_cnn(
                                    chunk_ds, val_ds, input_dim, device,
                                    total_steps=steps_per_chunk, batch_size=BATCH_SIZE,
                                    recycle_data=False, **probe_state
                                )
                                probe_state = state
                                probe_state["probe"] = cnn_probe
                                
                                # 5.5. Delete chunk immediately to save space
                                del chunk_ds
                                del chunk_embs
                                chunk_path = cache_root / f"{cache_key}_train_chunk_fold{f_id}_L{args.layers - 1}_mlp_seq.dat"
                                if chunk_path.exists(): os.remove(chunk_path)
                                if stopped: chunk_done = True
                                
                            del val_embs # Cleanup val
                        
                        # Save Probe
                        torch.save(cnn_probe.state_dict(), probe_path)

                    # 6. Circuit discovery
                    print(f"Starting discovery")
                    # Limit size of test set in large DMS assays
                    limit_datasets = ["CAPSD_AAV2S_Sinai_2021", "GRB2_HUMAN_Faure_2021", "HIS7_YEAST_Pokusaeva_2019", "SPG1_STRSG_Olson_2014", "YAP1_HUMAN_Araya_2012"]
                    if dms_name in limit_datasets and args.layers > 6 and len(test_indices) > 1024:
                        rng_eval = np.random.RandomState(42)
                        test_indices = rng_eval.choice(test_indices, 1024, replace=False)
                        tqdm.write(f"[{dms_name}] Limited test indices to 1024 (Model L{args.layers} > 6).")
                    if not use_chunking:
                        test_ds_wrap = EmbeddingDataset(master_embeddings, labels[test_indices], indices=test_indices)
                        val_ds_wrap  = EmbeddingDataset(master_embeddings, labels[val_indices], indices=val_indices)
                        test_embs = MemmapSubset(master_embeddings, test_indices)
                    else:
                        test_embs = precompute_embeddings(
                            seqs, inference, cache_key, indices=test_indices,
                            suffix=f"test_fold{i}", cache_dir=str(cache_root), batch_size=BATCH_SIZE
                        )
                        test_ds_wrap = EmbeddingDataset(test_embs, labels[test_indices]) # Already sliced 
                        # Re-compute Val (if deleted)
                        val_embs_search = precompute_embeddings(
                             seqs, inference, cache_key, indices=val_indices, 
                             suffix=f"val_fold{val_fold}", cache_dir=str(cache_root), batch_size=BATCH_SIZE
                        )
                        val_ds_wrap = EmbeddingDataset(val_embs_search, labels[val_indices])

                    # 7. Compute clean and max possible spearman
                    clean_spearman_val = evaluate_probe_direct(cnn_probe, val_ds_wrap, labels[val_indices], device, batch_size=BATCH_SIZE_CIRCUIT)

                    # 8. Setup subsets for Attribution and Greedy Search
                    y_val_full = labels[val_mask]
                    seq_val_full = seqs[val_mask]
                    median_score = np.median(y_val_full)
                    # 8.1. Prepare Attribution Set (Top performers to get strong gradient signal)
                    top_indices = np.where(y_val_full >= median_score)[0]
                    rng_attr = np.random.RandomState(42 + i)
                    if len(top_indices) > args.val_limit:
                        subset_attr = rng_attr.choice(top_indices, args.val_limit, replace=False)
                    else:
                        subset_attr = top_indices
                    seq_attr = seq_val_full[subset_attr]
                    # 8.2. Prepare Search Set (Balanced for Spearman stability)
                    # Sample half from high, half from low
                    # Only downsample if we are in the heavy 12-layer/multiples scenario
                    if args.layers == 12 and "multiples" in subfolder and len(y_val_full) > args.val_limit:
                        hi_idx = np.where(y_val_full >= median_score)[0]
                        lo_idx = np.where(y_val_full < median_score)[0]
                        n_half = args.val_limit // 2
                        rng_search = np.random.RandomState(43 + i)
                        hi_sample = rng_search.choice(hi_idx, min(len(hi_idx), n_half), replace=False)
                        lo_sample = rng_search.choice(lo_idx, min(len(lo_idx), n_half), replace=False)
                        search_indices = np.concatenate([hi_sample, lo_sample])
                        seq_search = seq_val_full[search_indices]
                        y_search = y_val_full[search_indices]
                        tqdm.write(f"Downsampled search set to {len(y_search)} sequences for speed.")
                    else:
                        # Use full validation set for singles or 6-layer runs
                        seq_search = seq_val_full
                        y_search = y_val_full

                    # 9. Run discovery on each method
                    for name, cfg in METHOD_CONFIGS.items():
                        disc = discoverers[name]
                        if not disc: continue
                        
                        out_json = func_dirs[name] / dms_name / f"{method_name}_fold{i}.json"
                        out_json.parent.mkdir(parents=True, exist_ok=True)
                        if out_json.exists() and not args.overwrite: continue

                        flags = cfg["flags"]

                        # 9.1. Define ceiling 
                        max_spearman_val = evaluate_circuit(disc, cnn_probe, seq_search, y_search, None, batch_size=BATCH_SIZE_CIRCUIT, cnn=True, **flags)['spearman']
                        standard_target = clean_spearman_val * args.recovery_ratio
                        if max_spearman_val < standard_target:
                            target_spearman = max_spearman_val
                            tqdm.write(f"[{name} {dms_name} {expt_name} F{i}] Target adjusted: Max Spearman ({max_spearman_val:.3f}) < Standard ({standard_target:.3f}). New Target: {target_spearman:.3f}")
                        else:
                            target_spearman = standard_target

                        # 9.2. Compute attribution
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        global_attr = compute_attribution(disc, cnn_probe, seq_attr, batch_size=32, **flags)
                        ranking = rank_nodes(global_attr)
                        disc.clear_cache()
                        gc.collect()

                        # 9.3. Circuit selection
                        eval_fn = lambda d, p, s, y_true, nodes, bs: (evaluate_circuit(d, p, s, y_true, nodes, bs, cnn=True, **flags), d.clear_cache(), torch.cuda.empty_cache())[0]['spearman']
                        best_nodes, best_k, val_recovered_spearman = circuit_search(
                            disc,
                            cnn_probe,
                            ranking,
                            seq_search,
                            y_search,
                            target_metric=target_spearman,
                            metric_fn=eval_fn,
                            step_size=args.step_size,
                            max_nodes=args.max_nodes,
                            batch_size=BATCH_SIZE_CIRCUIT,
                            desc=f"[{expt_name}]",
                            **flags
                        )

                        # 9.4. Final Evaluation (Test Set)
                        clean_spearman_test = evaluate_probe_direct(cnn_probe, test_ds_wrap, labels[test_indices], device, batch_size=BATCH_SIZE_CIRCUIT)
                        max_metrics_test = evaluate_circuit(disc, cnn_probe, seqs[test_indices], labels[test_indices], None, BATCH_SIZE_CIRCUIT, cnn=True, gt_embeddings=test_embs, **flags)
                        disc.clear_cache()
                        gc.collect()
                        base_metrics_test = evaluate_circuit(disc, cnn_probe, seqs[test_indices], labels[test_indices], {}, BATCH_SIZE_CIRCUIT, cnn=True, gt_embeddings=test_embs, **flags)
                        disc.clear_cache()
                        gc.collect()
                        test_metrics = evaluate_circuit(disc, cnn_probe, seqs[test_indices], labels[test_indices], best_nodes, BATCH_SIZE_CIRCUIT, cnn=True, gt_embeddings=test_embs, **flags)
                        disc.clear_cache()
                        gc.collect()
                        test_spearman = test_metrics['spearman']
                        test_nmse = test_metrics['nmse']
                        recovered_ratio = 0.0
                        denom = clean_spearman_test - base_metrics_test['spearman']
                        if denom > 1e-6:
                            recovered_ratio = (test_spearman - base_metrics_test['spearman']) / denom
                        tqdm.write(f"[{name} {dms_name} {expt_name} F{i}] n={len(train_indices)} | Clean: {clean_spearman_test:.3f} | Max: {max_metrics_test['spearman']:.3f} | Recov Spearman: {test_spearman:.3f} | Recov NMSE: {test_nmse:.3f} | Nodes: {best_k}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Save
                        res = {
                            "DMS": dms_name,
                            "Method": name,
                            "Fold": i,
                            "n_train": len(train_indices),
                            "k": best_k,
                            "source": flags["source"],
                            "freeze_attention": flags["freeze_attention"],
                            "clean_spearman": float(clean_spearman_test),
                            "max_spearman": float(max_metrics_test['spearman']),
                            "max_nmse": float(max_metrics_test['nmse']),
                            "base_spearman": float(base_metrics_test['spearman']),
                            "base_nmse": float(base_metrics_test['nmse']),
                            "recovered_spearman": float(test_spearman),
                            "recovered_nmse": float(test_nmse),
                            "recovered_ratio": recovered_ratio,
                            "nodes": {str(l): list(s) for l, s in best_nodes.items()}
                        }                 
                        with open(out_json, "w") as f:
                            json.dump(res, f, indent=2)

                    # Cleanup Chunk Temp Files
                    if use_chunking:
                         layer_idx = args.layers - 1
                         # 1. Close Python Handles (Release memory)
                         if 'test_ds_wrap' in locals(): del test_ds_wrap
                         if 'val_ds_wrap' in locals(): del val_ds_wrap
                         # Check if the variables exist and are not None before deleting
                         if 'test_embs' in locals() and test_embs is not None: 
                            del test_embs
                         if 'val_embs_search' in locals() and val_embs_search is not None: 
                            del val_embs_search

                         # 2. Delete Files from Disk (Free storage)
                         # Delete Test Chunk
                         test_fname = f"{cache_key}_test_fold{i}_L{layer_idx}_mlp_seq.dat"
                         test_path = cache_root / test_fname
                         if test_path.exists():
                             try:
                                 os.remove(test_path)
                             except OSError as e:
                                 print(f"Error deleting test chunk {test_fname}: {e}")
                         # Delete Validation Chunk (The one re-computed for search)
                         val_fname = f"{cache_key}_val_fold{val_fold}_L{layer_idx}_mlp_seq.dat"
                         val_path = cache_root / val_fname
                         if val_path.exists():
                             try:
                                 os.remove(val_path)
                             except OSError as e:
                                 print(f"Error deleting val chunk {val_fname}: {e}")


            if not use_chunking and master_embeddings is not None:
                layer_idx = args.layers - 1
                # 1. Close the memmap handle
                del master_embeddings

                # 2. Delete the file
                master_fname = f"{cache_key}_all_L{layer_idx}_mlp_seq.dat"
                master_path = cache_root / master_fname
                
                if master_path.exists():
                    try:
                        print(f"Deleting Master File: {master_fname}")
                        os.remove(master_path)
                    except OSError as e:
                        print(f"Error deleting master file: {e}")
                        
if __name__ == "__main__":
    main()