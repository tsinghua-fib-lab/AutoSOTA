"""
Main CAA steering orchestration script.
Processes DMS datasets with train/val/test splits aligned with folds.
Similar to run_sae_steering.py but for CAA steering.
"""
import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure prints flush immediately
import functools
print = functools.partial(print, flush=True)

# Set up path BEFORE any imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from steering.steering_utils import infer_wildtype
from esm_steering.steer_seq import main as steer_seq_main
from steering.steering_utils import get_probe_input
from esm_steering.caa_utils import get_esm_inference, score_sequences_mlp
from steering.scoring_utils import load_cnn, score_cnn, CNNProbe
from steering.gen_utils import ESMTokenizerWrapper
import torch

def get_indices(df, split_type, fold_id):
    """
    Get train/val/test indices matching CLT circuit discovery split strategy.
    For fold i: Test = fold i, Val = fold (i+1) % 5, Train = all other folds.
    """
    fold_col = f"fold_{split_type}"
    if fold_col not in df.columns:
        raise ValueError(f"Column {fold_col} not found in dataset")
    
    # Normalize fold values to 0-4 range (matching CLT circuit discovery)
    folds = df[fold_col].values
    fold_vals = folds - folds.min()
    
    # Split matching CLT circuit discovery (01_discover_circuits.py lines 222-230)
    test_mask = (fold_vals == fold_id)
    val_fold = (fold_id + 1) % 5
    val_mask = (fold_vals == val_fold)
    train_mask = ~(test_mask | val_mask)
    
    train_idx = df[train_mask].index.values
    val_idx = df[val_mask].index.values
    test_idx = df[test_mask].index.values
    
    # Confirm splits to user
    print(f"  Fold {fold_id}: Test Size={len(test_idx)}, Val Size={len(val_idx)}, Train Size={len(train_idx)} "
          f"(Test=fold{fold_id}, Val=fold{val_fold}, Train=other folds)")
    
    return train_idx, val_idx, test_idx

def parse_args():
    parser = argparse.ArgumentParser(description="Main CAA Steering Orchestration")
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory containing DMS CSVs")
    parser.add_argument("--output_dir", type=str, default="results_caa_steering", help="Output directory")
    parser.add_argument("--esm_weights", type=str, 
                       default="../models/esm2_t6_8M_UR50D.pt",
                       help="Path to ESM Weights")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated list of folds")
    parser.add_argument("--split_type", type=str, default="rand_multiples", 
                       help="Split type (e.g., rand_multiples, contiguous_5, modulo_5)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for sampling (matches CLT)")
    parser.add_argument("--trials", type=int, default=5, help="Number of sampling trials per fold")
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=21, help="Number of alpha steps")
    parser.add_argument("--pos_neg_prop", type=str, default=None, 
                       help="Proportion(s) (0-1) for percentile threshold, comma-separated (e.g., '0.1,0.25,0.5'). If 0.1, takes sequences >= 90th percentile of bin=1 (pos) and <= 10th percentile of bin=0 (neg). If None, uses all sequences with bin=1 (pos) and bin=0 (neg)")
    parser.add_argument("--eval_models_dir", type=str, default="eval_models", 
                       help="Directory containing eval CNN models (optional)")
    parser.add_argument("--dms_name_filter", type=str, default=None, 
                       help="Optional: Process only this specific DMS dataset (e.g., 'HIS7_YEAST_Pokusaeva_2019')")
    parser.add_argument("--disable_similarity_filter", action="store_true",
                       help="Disable cosine similarity filter in sequence decoding (use only mutation constraint)")
    parser.add_argument("--max_mutations", type=int, default=5,
                       help="Maximum number of mutations per sequence")
    return parser.parse_args()

def process_dms_target(dms_name, args, eval_esm_model=None, eval_alphabet=None):
    """Process a single DMS target across folds."""
    print(f"\n{'='*60}")
    print(f"Processing: {dms_name}")
    print(f"{'='*60}")
    
    folds = [int(f.strip()) for f in args.folds.split(",")]
    print(f"  Folds to process: {folds}")
    
    # Get main CSV to infer wildtype - use MULTIPLES to match probe training
    multiples_sub_dir = Path(args.dms_dir) / "cv_folds_multiples_substitutions"
    main_csv = multiples_sub_dir / f"{dms_name}.csv"
    print(f"  Looking for CSV: {main_csv}")
    
    if not main_csv.exists():
        print(f"  [SKIP] CSV not found: {main_csv}")
        return
    
    print(f"  Reading CSV...")
    try:
        df = pd.read_csv(main_csv)
        print(f"  CSV loaded: {len(df)} rows")
        if df.empty:
            print(f"  [SKIP] Empty CSV: {main_csv}")
            return
        print(f"  Inferring wildtype from first row...")
        wildtype = infer_wildtype(df.iloc[0])
        print(f"  Wildtype length: {len(wildtype)}")
    except Exception as e:
        print(f"  [ERROR] Failed to infer wildtype: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check for fold column
    fold_col = f"fold_{args.split_type}"
    print(f"  Checking for fold column: {fold_col}")
    if fold_col not in df.columns:
        print(f"  [SKIP] Fold column '{fold_col}' not found. Available: {[c for c in df.columns if 'fold' in c]}")
        return
    print(f"  Fold column found: {fold_col}")
    
    # Process each fold
    print(f"  Processing {len(folds)} folds...")
    
    for fold in tqdm(folds, desc=f"  {dms_name} Folds", leave=False):
        print(f"\n  >>> Fold {fold}")
        
        # Get test indices for this fold
        print(f"    Getting train/val/test indices...")
        train_idx, val_idx, test_idx = get_indices(df, args.split_type, fold)
        print(f"    Indices: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        if len(test_idx) == 0:
            print(f"    [SKIP] No test data for fold {fold}")
            continue
        
        if len(train_idx) < 2:
            print(f"    [SKIP] Insufficient training data: {len(train_idx)} samples")
            continue
        
        # Create temporary test CSV
        output_dir = Path(args.output_dir) / dms_name / args.split_type
        output_dir.mkdir(parents=True, exist_ok=True)
        test_csv = output_dir / f"test_fold{fold}.csv"
        test_df = df.iloc[test_idx].copy()
        test_df.to_csv(test_csv, index=False)

        # Output path for combined fold results
        combined_output = output_dir / f"{dms_name}_fold{fold}_combined_steering.csv"
        if combined_output.exists():
            print(f"    [SKIP] Combined results exist: {combined_output.name}")
            if test_csv.exists(): test_csv.unlink()
            continue

        # Parse pos_neg_prop values
        print(f"    Parsing pos_neg_prop values...")
        if args.pos_neg_prop is not None and args.pos_neg_prop.strip():
            prop_values = []
            for p in args.pos_neg_prop.split(","):
                p = p.strip()
                if p.lower() == "all" or p == "":
                    prop_values.append(None)
                else:
                    prop_values.append(float(p))
        else:
            prop_values = [None]
        
        # Accumulate all results for this fold
        fold_all_results = []
        
        # Process configurations
        total_configs = len(prop_values) * args.trials
        print(f"    Total configurations for Fold {fold}: {total_configs}")
        
        config_count = 0
        for prop in tqdm(prop_values, desc=f"      Props", leave=False):
            for trial in tqdm(range(args.trials), desc=f"        Trials", leave=False):
                config_count += 1
                prop_display = f"Prop={prop:.2f}" if prop is not None else "Prop=All"
                print(f"    [RUN {config_count}/{total_configs}] Trial {trial}, {prop_display}")
                
                # Create a simple args object for steer_seq_main
                class SteerArgs:
                    def __init__(self):
                        self.dms_id = str(test_csv)
                        self.dms_name = dms_name
                        self.alpha_min = args.alpha_min
                        self.alpha_max = args.alpha_max
                        self.k = args.k
                        self.split_type = args.split_type
                        self.seed = args.seed
                        # Use a temporary name for steer_seq_main internal saving (optional)
                        self.output_csv = str(output_dir / f"tmp_fold{fold}_T{trial}_steering.csv")
                        self.fold = fold
                        self.trial = trial
                        self.eval_models_dir = args.eval_models_dir
                        self.esm_weights = args.esm_weights
                        self.pos_neg_prop = prop
                        self.max_mutations = getattr(args, 'max_mutations', 5)
                        self.disable_similarity_filter = getattr(args, 'disable_similarity_filter', False)
                
                steer_args = SteerArgs()
                
                try:
                    current_steering_vector, results = steer_seq_main(steer_args, eval_esm_model=eval_esm_model, eval_alphabet=eval_alphabet)
                    
                    if results:
                        # Add proportion info to results if not already there
                        for res in results:
                            res['pos_neg_prop'] = prop if prop is not None else "all"
                        fold_all_results.extend(results)
                        
                        # Clean up individual trial CSV if created by steer_seq_main
                        trial_csv = Path(steer_args.output_csv)
                        if trial_csv.exists(): trial_csv.unlink()
                        
                except Exception as e:
                    print(f"      [ERROR] Config failed: {e}")
                    import traceback; traceback.print_exc()

        # Save aggregated results for the fold
        if fold_all_results:
            fold_df = pd.DataFrame(fold_all_results)
            fold_df.to_csv(combined_output, index=False)
            print(f"    [OK] Saved aggregated results for Fold {fold} to {combined_output.name} ({len(fold_df)} rows)")
        
        # Clean up temporary test CSV
        if test_csv.exists():
            test_csv.unlink()

def create_top50_probe_csv(dms_name, args, eval_esm_model, eval_alphabet, device, inference=None):
    """
    After all CAA steering for a protein, create a CSV with top 50 sequences by probe score.
    Matches CLT probe steering: scores each fold individually with that fold's probe,
    then selects top 50 across all folds, then scores those with eval model.
    """
    print(f"\n  Creating top 50 probe score CSV for {dms_name}...")
    
    folds = [int(f.strip()) for f in args.folds.split(",")]
    output_dir = Path(args.output_dir) / dms_name / args.split_type
    
    if eval_esm_model is None or eval_alphabet is None or inference is None:
        print(f"      [SKIP] ESM model/inference not available for scoring")
        return
    
    tokenizer = ESMTokenizerWrapper(eval_alphabet)
    
    # Step 1: Collect sequences from all folds and score with each fold's probe
    all_scored_sequences = []  # Will contain (seq, metadata, probe_score, fold)
    
    for fold in folds:
        print(f"    Processing fold {fold}...")
        
        # Load probe for this fold
        probe_path = Path(__file__).parent.parent / "function_circuit" / "probe" / "multiples" / dms_name / f"rand_multiples_fold{fold}_cnn.pt"
        
        if not probe_path.exists():
            print(f"      [SKIP] Probe not found: {probe_path}")
            continue
        
        try:
            # Load CNN probe for this fold
            probe_model = load_cnn(None, str(probe_path), device)
            
            # Collect all sequences from all steering runs for this fold
            fold_sequences = []
            fold_metadata = []
            
            # Find the combined CSV file for this fold
            csv_file = output_dir / f"{dms_name}_fold{fold}_combined_steering.csv"
            
            if not csv_file.exists():
                print(f"      [SKIP] Combined steering CSV not found for fold {fold}")
                continue
            
            print(f"      Loading sequences from {csv_file.name}")
            
            # Load all sequences from this fold
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    seq = row.get('mutated_sequence', '')
                    if pd.notna(seq) and isinstance(seq, str) and len(seq) > 0:
                        fold_sequences.append(seq)
                        fold_metadata.append({
                            'mutated_sequence': seq,
                            'source_file': csv_file.name,
                            'fold': row.get('fold', fold),
                            'trial': row.get('trial', np.nan),
                            'alpha': row.get('alpha', np.nan),
                            'mutant': row.get('mutant', np.nan),
                            'alpha': row.get('alpha', np.nan),
                            'mutant': row.get('mutant', np.nan),
                            'eval_score': row.get('eval_score', np.nan),  # Eval score from steering (already computed)
                            'wt_eval_score': row.get('wt_eval_score', np.nan),  # WT eval score from steering
                        })
            except Exception as e:
                print(f"      [WARN] Failed to load {csv_file.name}: {e}")
                continue
            
            if not fold_sequences:
                print(f"      [SKIP] No sequences found for fold {fold}")
                continue
            
            print(f"      Scoring {len(fold_sequences)} sequences with fold {fold} probe...")
            
            # Compute WT score for this specific fold's probe
            wt_probe_score_fold = np.nan
            try:
                # Need wildtype sequence
                if 'wildtype' not in locals():
                     multiples_sub_dir = Path(args.dms_dir) / "cv_folds_multiples_substitutions"
                     main_csv = multiples_sub_dir / f"{dms_name}.csv"
                     if main_csv.exists():
                         df_main = pd.read_csv(main_csv)
                         if len(df_main) > 0:
                             wildtype = infer_wildtype(df_main.iloc[0])
                
                if 'wildtype' in locals():
                     rep_wt = get_probe_input(inference, wildtype)
                     with torch.no_grad():
                         wt_score_val = probe_model(rep_wt).item()
                     wt_probe_score_fold = float(wt_score_val)
            except Exception as e:
                print(f"      [WARN] Could not compute WT probe score for fold {fold}: {e}")

            # Score all sequences from this fold with this fold's probe
            # Use get_probe_input loop to be consistent with run_probe_steering.py
            probe_scores = []
            for seq in tqdm(fold_sequences, desc="      Scoring", leave=False):
                try:
                    rep = get_probe_input(inference, seq)
                    with torch.no_grad():
                        score = probe_model(rep).item()
                    probe_scores.append(score)
                except Exception as e:
                    print(f"      [WARN] Scoring failed for sequence: {e}")
                    probe_scores.append(np.nan)
            
            probe_scores = np.array(probe_scores)
            
            if isinstance(probe_scores, np.ndarray):
                probe_scores = probe_scores.flatten()
            
            # Add probe scores and store for aggregation
            for i, (seq, metadata, score) in enumerate(zip(fold_sequences, fold_metadata, probe_scores)):
                metadata['probe_score'] = float(score)
                metadata['wt_probe_score'] = wt_probe_score_fold
                all_scored_sequences.append((seq, metadata, float(score), fold))
            
            # Print unique sequences across all CSVs for this fold
            fold_unique_seqs = pd.Series(fold_sequences).nunique()
            print(f"      Unique sequences across all CSVs for fold {fold}: {fold_unique_seqs} / {len(fold_sequences)} total")
            
        except Exception as e:
            print(f"      [ERROR] Failed to process fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_scored_sequences:
        print(f"      [SKIP] No sequences found across all folds")
        return
    
    print(f"\n    Aggregating {len(all_scored_sequences)} sequences from all folds...")
    
    # Step 2: Create DataFrame and sort by probe score (across all folds)
    all_metadata = [metadata for _, metadata, _, _ in all_scored_sequences]
    results_df = pd.DataFrame(all_metadata)
    results_df = results_df.sort_values(by='probe_score', ascending=False)
    
    # Debug: Check unique sequences before deduplication
    unique_before = results_df['mutated_sequence'].nunique()
    print(f"    Total sequences: {len(results_df)}, Unique sequences: {unique_before}")
    
    # Print all unique sequences across ALL CSVs with their probe scores and fold info
    print(f"\n    All unique sequences across ALL CSVs (sorted by probe_score):")
    unique_seqs_df = results_df.drop_duplicates(subset=['mutated_sequence'], keep='first').head(100)  # Show top 100
    for idx, row in unique_seqs_df.iterrows():
        seq = row['mutated_sequence']
        probe_score = row['probe_score']
        fold = row['fold']
        source_file = row['source_file']
        alpha = row['alpha']
        print(f"      Seq {idx+1}: fold={fold}, probe_score={probe_score:.4f}, alpha={alpha}, file={source_file}")
        print(f"        Sequence: {seq[:50]}..." if len(seq) > 50 else f"        Sequence: {seq}")
    
    # Step 3: Deduplicate sequences (keep first occurrence = highest probe score)
    results_df = results_df.drop_duplicates(subset=['mutated_sequence'], keep='first')
    
    # Debug: Check after deduplication
    unique_after = len(results_df)
    print(f"    After deduplication: {unique_after} unique sequences")
    
    # Step 4: Select top 50 unique sequences across all folds
    top50_df = results_df.head(50).copy()
    
    if len(top50_df) == 0:
        print(f"      [SKIP] No sequences to save")
        return
    
    print(f"    Selected top 50 sequences across all folds (by probe_score)")
    
    # Step 5: Use eval_score that was already computed during steering (matches CLT)
    # CLT: Selects top 50 by probe_score, then uses eval_score from steering (not re-scored)
    # The eval_score and wt_eval_score columns should already be in the metadata from steering CSV files
    
    # Ensure wt_eval_score column exists (use first non-nan value if available)
    if 'wt_eval_score' not in top50_df.columns:
        wt_eval_score = np.nan
        top50_df['wt_eval_score'] = wt_eval_score
    elif top50_df['wt_eval_score'].isna().all():
        # If all are nan, keep as nan
        pass
    else:
        # Use first non-nan value (should be same for all rows from same DMS)
        wt_eval_score = top50_df['wt_eval_score'].dropna().iloc[0]
        top50_df['wt_eval_score'] = wt_eval_score
    
    # wt_probe_score is now computed per-fold and stored in metadata
    if 'wt_probe_score' not in top50_df.columns:
         top50_df['wt_probe_score'] = np.nan
    
    # Save top 50 CSV (one file per DMS, not per fold)
    top50_output = output_dir / f"{dms_name}_top50_probe.csv"
    top50_df.to_csv(top50_output, index=False)
    print(f"      [OK] Saved top 50 sequences (across all folds) to {top50_output.name}")

def main():
    args = parse_args()
    
    print(f"{'='*60}")
    print(f"CAA STEERING ORCHESTRATION")
    print(f"{'='*60}")
    print(f"  DMS Directory: {args.dms_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  ESM Weights:      {args.esm_weights}")
    print(f"  Split Type:       {args.split_type}")
    print(f"  Folds:            {args.folds}")
    print(f"  Trials:           {args.trials}")
    print(f"  Alpha Range:      [{args.alpha_min}, {args.alpha_max}] ({args.k} steps)")
    if args.pos_neg_prop is not None and args.pos_neg_prop.strip():
        prop_values = []
        for p in args.pos_neg_prop.split(","):
            p = p.strip()
            if p.lower() == "all" or p == "":
                prop_values.append(None)
            else:
                prop_values.append(float(p))
        prop_str = ", ".join([f"{p:.2f}" if p is not None else "all" for p in prop_values])
        print(f"  Pos/Neg Prop:     {prop_str} (comma-separated, >= (100-X)th percentile bin=1, <= Xth percentile bin=0, or 'all')")
    else:
        print(f"  Pos/Neg Prop:     All (bin=1 for pos, bin=0 for neg)")
    print(f"  Eval Models Dir:  {args.eval_models_dir}")
    print(f"{'='*60}\n")
    
    # Initialize ESM model once for all runs (shared for CAA steering and eval scoring)
    print("Initializing ESM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize CAA steering ESM model (sets global in caa_utils)
    esm_inference = get_esm_inference(esm_weights_path=args.esm_weights)
    
    # Reuse the same model for eval scoring (extract from ESMInference)
    eval_esm_model = None
    eval_alphabet = None
    if args.eval_models_dir:
        # Use the same model instance - just extract it from ESMInference
        eval_esm_model = esm_inference.model
        eval_alphabet = esm_inference.alphabet
        print("Eval scoring will use the same ESM model instance.\n")
    
    print(f"ESM model initialized on {device}\n")
    
    multiples_sub_dir = Path(args.dms_dir) / "cv_folds_multiples_substitutions"
    if not multiples_sub_dir.exists():
        print(f"[ERROR] DMS directory not found: {multiples_sub_dir}")
        return
    
    csv_files = [f for f in os.listdir(multiples_sub_dir) if f.endswith(".csv")]
    
    # Filter by dms_name if specified
    if args.dms_name_filter:
        csv_files = [f for f in csv_files if os.path.splitext(f)[0] == args.dms_name_filter]
        if not csv_files:
            print(f"[ERROR] Dataset '{args.dms_name_filter}' not found in {multiples_sub_dir}")
            return
        print(f"Filtered to dataset: {args.dms_name_filter}\n")
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {multiples_sub_dir}")
        return
    
    print(f"Found {len(csv_files)} DMS datasets\n")
    
    # Process each dataset
    print(f"Starting to process {len(csv_files)} datasets...")
    for filename in tqdm(csv_files, desc="DMS Datasets"):
        dms_name = os.path.splitext(filename)[0]
        print(f"\n{'='*60}")
        print(f"Starting dataset: {dms_name} (file: {filename})")
        print(f"{'='*60}")
        try:
            process_dms_target(dms_name, args, eval_esm_model=eval_esm_model, eval_alphabet=eval_alphabet)
            
            # After all CAA steering, create top 50 CSV by probe score
            create_top50_probe_csv(dms_name, args, eval_esm_model, eval_alphabet, device, inference=esm_inference)
            
            print(f"Completed dataset: {dms_name}")
        except Exception as e:
            print(f"[ERROR] Failed to process {dms_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Pipeline Complete.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
