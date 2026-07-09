"""
Main probe steering orchestration script.
OPTIMIZED: Batched attribution sampling.
"""
import argparse
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
np.random.seed(42)
torch.manual_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'function_circuit'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'circuit_utils'))

from function_utils import CNNProbe
from steering_utils import get_mutant_string, get_probe_input, get_scoring_model_path, get_full_model, infer_wildtype
from esm_activation import ESMInference

def parse_args():
    parser = argparse.ArgumentParser(description="Main Probe Steering Orchestration")
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory containing DMS CSVs")
    parser.add_argument("--output_dir", type=str, default="one_time_gb1_results", help="Output directory")
    parser.add_argument("--eval_models_dir", type=str, default="eval_models_35M", help="Directory containing held-out eval CNNs")
    parser.add_argument("--clt_ckpt", type=str, required=True, help="Path to CLT Checkpoint")
    parser.add_argument("--plt_ckpt", type=str, required=True, help="Path to PLT Checkpoint")
    parser.add_argument("--esm_weights", type=str, required=True, help="Path to ESM Weights")
    parser.add_argument("--circuit_base", type=str, required=True, help="Base path for circuit functions")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--trials", type=int, default=5, help="Number of sampling trials per fold")
    parser.add_argument("--configs", type=str, default="CLT_direct,CLT_sequential,CLT_sequential_no_frozen,PLT,PLT_no_frozen")
    parser.add_argument("--supp", type=int, default=10, help="Support size for circuit sampling")
    parser.add_argument("--max_mutations", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=5.0)
    parser.add_argument("--alpha_steps", type=int, default=25)
    return parser.parse_args()

def load_circuit(path):
    with open(path) as f:
        data = json.load(f)
    nodes = {int(k): v for k, v in data['nodes'].items()}
    node_str = data.get('layer_latent_string', "")
    return nodes, data.get('clean_spearman', 0), node_str

def get_base_circuit_path_multiples(circuit_base, config_map, config_name, dms_name, fold):
    _, _, _, circuit_dir = config_map[config_name]
    return os.path.join(circuit_base, circuit_dir, "multiples", dms_name, f"rand_multiples_fold{fold}.json")

def run_circuit_attribution_batch(circuit_json, config_map, config_name, wildtype, supp, esm_weights, trials, output_prefix, base_seed=42):
    """Run attribution ONCE, generate multiple trial JSONs."""
    ckpt, model_type, freeze_attention, _ = config_map[config_name]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "find_steering_circuit_attribution_sampler.py")
    scoring_model_path = get_scoring_model_path(circuit_json)

    cmd = [
        sys.executable, script_path,
        "--json_path", circuit_json,
        "--ckpt", ckpt,
        "--model_type", model_type,
        "--scoring_model", scoring_model_path,
        "--esm_weights", esm_weights,
        "--wt", wildtype,
        "--supp", str(supp),
        "--seed", str(base_seed),
        "--trials", str(trials),
        "--output_prefix", output_prefix
    ]

    if freeze_attention:
        cmd.append("--freeze_attention")

    print(f"      -> Running BATCH attribution (Trials={trials}, Seed={base_seed})...")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"WARNING: Attribution failed. Error: {result.stderr.decode('utf-8')}")
        return False
    return True

def compute_dynamic_random_baseline(inference, scoring_model, wt, mutated_seq, seed=None, min_position=None, max_position=None):
    from steering_utils import generate_random_mutant_sequence
    rand_seq, rand_mutant_str = generate_random_mutant_sequence(wt, mutated_seq, seed=seed, min_position=min_position, max_position=max_position)
    if rand_seq is None: return np.nan, None, None
    if rand_mutant_str == "WT": return np.nan, rand_seq, rand_mutant_str
    seq_rep = get_probe_input(inference, rand_seq)
    score = scoring_model(seq_rep).item()
    return score, rand_seq, rand_mutant_str

def process_dms_target(dms_name, wildtype, args, inference, model_configs):
    print(f"\n{'='*40}\nProcessing: {dms_name} (WT Len: {len(wildtype)})\n{'='*40}")
    folds = [int(f) for f in args.folds.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps).tolist()
    device = inference.device
    if dms_name == "SPG1_STRSG_Olson_2014":
        min_position = 228
        max_position = 282
    else:
        max_position = None
        min_position = None
    eval_model = None
    eval_path = os.path.join(args.eval_models_dir, f"{dms_name}.pt")
    if os.path.exists(eval_path):
        try:
            eval_model = CNNProbe(input_dim=inference.model.embed_dim).to(device)
            eval_model.load_state_dict(torch.load(eval_path, map_location=device))
            eval_model.eval()
        except: eval_model = None
    
    wt_eval_score = np.nan
    if eval_model:
        with torch.no_grad():
            wt_eval_score = eval_model(get_probe_input(inference, wildtype)).item()

    for config_name in configs:
        if config_name not in model_configs: continue
        print(f"\n    >>> Config: {config_name}")
        output_dir = os.path.join(args.output_dir, dms_name, config_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "probe_results.csv")
        
        if os.path.exists(csv_path):
            print(f"      [SKIPPING - Found {csv_path}]")
            continue
        
        ckpt, model_type, freeze_attention, _ = model_configs[config_name]
        try:
            model = get_full_model(model_type, ckpt, device, esm_weights_path=args.esm_weights)
        except Exception as e:
            print(f"      [Error] Model load failed: {e}")
            continue
        
        fold_results = []
        for fold in folds:
            base_circuit_path = get_base_circuit_path_multiples(args.circuit_base, model_configs, config_name, dms_name, fold)
            if not os.path.exists(base_circuit_path): continue
            
            _, spearman, _ = load_circuit(base_circuit_path) 
            if spearman < 0.1: continue

            # === BATCH ATTRIBUTION ===
            # Prefix: .../probe_fold0_steering
            # Generates: .../probe_fold0_steering_trial0.json, ..._trial1.json, etc.
            output_prefix = os.path.join(output_dir, f"probe_fold{fold}_steering")
            
            # Check if files exist
            expected_files = [f"{output_prefix}_trial{t}.json" for t in range(args.trials)]
            if not all(os.path.exists(f) for f in expected_files):
                # Run ONCE per fold
                success = run_circuit_attribution_batch(
                    base_circuit_path, model_configs, config_name, wildtype, 
                    args.supp, args.esm_weights, 
                    trials=args.trials,
                    output_prefix=output_prefix
                )
                if not success: continue

            # === LOOP TRIALS (LOAD & RUN) ===
            for trial in range(args.trials):
                steering_json = f"{output_prefix}_trial{trial}.json"
                
                try:
                    steering_circuit, _, layer_latent_string = load_circuit(steering_json)
                    probe_path = get_scoring_model_path(base_circuit_path)
                    
                    scoring_model = CNNProbe(input_dim=model.esm.embed_dim).to(device)
                    scoring_model.load_state_dict(torch.load(probe_path, map_location=device))
                    scoring_model.eval()
                    with torch.no_grad():
                        wt_score = scoring_model(get_probe_input(inference, wildtype)).item()
                except Exception as e:
                    print(f"      [Error] Load failed fold {fold} trial {trial}: {e}")
                    continue

                # Steer & Eval
                try:
                    with torch.no_grad():
                        if model_type == "clt_direct":
                            emb_batch, _, _, _, mask_batch = model.forward_steered(
                                wildtype, steering_circuit, before=False, alphas=alphas
                            )
                        else:
                            emb_batch, _, _, _, mask_batch = model.forward_steered(
                                wildtype, steering_circuit, before=False,
                                alphas=alphas, freeze_attention=freeze_attention
                            )
                    
                    steered_seqs, _ = model.get_sequences(emb_batch, mask_batch, wt=wildtype, max_mutations=args.max_mutations, min_position=min_position, max_position=max_position)
                    
                    for i, alpha in enumerate(alphas):
                        seq = steered_seqs[i]
                        mutant_str = get_mutant_string(wildtype, seq)
                        rep = get_probe_input(inference, seq)
                        score = scoring_model(rep).item()
                        
                        eval_score = np.nan
                        if eval_model: eval_score = eval_model(rep).item()
                        
                        # Random baseline seed
                        rand_seed = 42 + (fold * 10000) + (trial * 1000) + i
                        rand_score, rand_seq, rand_mut_str = compute_dynamic_random_baseline(
                            inference, scoring_model, wildtype, seq, seed=rand_seed, min_position=min_position, max_position=max_position
                        )
                        
                        rand_eval_score = np.nan
                        if eval_model and rand_seq:
                            rand_eval_score = eval_model(get_probe_input(inference, rand_seq)).item()

                        fold_results.append({
                            "fold": fold,
                            "trial": trial, 
                            "alpha": alpha,
                            "wt_probe_score": wt_score,
                            "probe_score": score,
                            "mutant": mutant_str,
                            "random_probe_score": rand_score,
                            "random_mutant": rand_mut_str,
                            "wt_eval_score": wt_eval_score,
                            "eval_score": eval_score,
                            "random_eval_score": rand_eval_score,
                            "layer:latent": layer_latent_string, 
                            "mutated_sequence": seq,
                        })

                except Exception as e:
                    print(f"      [Error] Steering loop error fold {fold} trial {trial}: {e}")
                    import traceback; traceback.print_exc()

        if fold_results:
            df = pd.DataFrame(fold_results)
            df.to_csv(csv_path, index=False)
            print(f"      -> Saved: {csv_path} ({len(df)} rows)")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*60}\nPROBE STEERING (Optimized Batch Sampling - {args.trials} Trials)\n{'='*60}")
    
    MODEL_CONFIGS = {
        "CLT_direct": (args.clt_ckpt, "clt_direct", None, "CLT_direct"),
        "CLT_sequential": (args.clt_ckpt, "clt", True, "CLT_sequential"),
        "CLT_sequential_no_frozen": (args.clt_ckpt, "clt", False, "CLT_sequential_no_frozen"),
        "PLT": (args.plt_ckpt, "plt", True, "PLT"),
        "PLT_no_frozen": (args.plt_ckpt, "plt", False, "PLT_no_frozen"),
    }

    print("Loading Shared ESMInference...")
    esm_weights_path = args.esm_weights
    # assuming the filename is of the form "esm2_t{num_layers}_{millions of params}M.pt"
    esm_filename = os.path.basename(esm_weights_path)
    num_layers = int(esm_filename.split("_")[1][1:])
    d_model = 320 if "8M" in esm_filename else 480 if "35M" in esm_filename else None
    if d_model is None:
        raise ValueError(f"Could not infer d_model from ESM filename: {esm_filename}")

    inference = ESMInference(
        device,
        esm_weights_path=esm_weights_path,
        num_layers=num_layers,
        d_model=d_model,
    )

    single_sub_dir = os.path.join(args.dms_dir, "cv_folds_single_substitutions")
    if not os.path.exists(single_sub_dir): return
    csv_files = [f for f in os.listdir(single_sub_dir) if f.endswith(".csv")]

    for filename in tqdm(csv_files, desc="DMS Datasets"):
        dms_name = os.path.splitext(filename)[0]
        try:
            df = pd.read_csv(os.path.join(single_sub_dir, filename))
            if df.empty: continue
            wildtype = infer_wildtype(df.iloc[0])
            process_dms_target(dms_name, wildtype, args, inference, MODEL_CONFIGS)
        except Exception as e:
            print(f"[Error] {filename}: {e}")

if __name__ == "__main__":
    main()