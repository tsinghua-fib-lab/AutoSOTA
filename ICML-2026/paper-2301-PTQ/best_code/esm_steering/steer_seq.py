"""
CAA Steering for a single sequence.
Similar to sae_steer_latents.py but for CAA steering vectors.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Add repo root to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from esm_steering.caa_utils import get_caa_vector, steer_caa_sequence, get_esm_inference, score_sequences_mlp
from steering.steering_utils import get_probe_input

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch size
# T: Sequence length
# V: model vocab dimension
# ──────────────────────────────────────────────────────────────────────────────


# Copy functions from steering_utils to avoid import issues
def get_mutant_string(wt_seq, gen_seq):
    """Compares wildtype and generated sequences to return a mutation string."""
    if len(wt_seq) != len(gen_seq):
        return "LengthMismatch"
    mutations = []
    for i, (wt_char, gen_char) in enumerate(zip(wt_seq, gen_seq)):
        if wt_char != gen_char:
            mutations.append(f"{wt_char}{i+1}{gen_char}")
    if not mutations:
        return "WT"
    return ":".join(mutations)

def infer_wildtype(row):
    """Get wildtype sequence by reverting mutations in DMS."""
    mutant_str = row['mutant']
    seq = list(row['mutated_sequence'])
    parts = mutant_str.split(':') if ':' in mutant_str else [mutant_str]
    for m in parts:
        if len(m) < 2:
            continue
        wt_aa = m[0]
        idx_str = "".join([c for c in m if c.isdigit()])
        if not idx_str:
            continue
        idx = int(idx_str) - 1
        if 0 <= idx < len(seq):
            seq[idx] = wt_aa
    return ''.join(seq)

# Try to import evaluation model utilities (optional)
try:
    from steering.scoring_utils import load_cnn, score_cnn
    from steering.gen_utils import load_esm_model, ESMTokenizerWrapper
    EVAL_MODEL_AVAILABLE = True
except ImportError:
    EVAL_MODEL_AVAILABLE = False

def decode_logits_to_sequence(logits_BTV, alphabet, wt_seq, wt_logits=None, max_mutations=5, similarity_threshold=0.90, use_similarity_filter=True, min_position=None, max_position=None):
    """
    Decode logits to sequence using cosine similarity and mutation constraint.
    Similar to steer_sequence_latent in sae_steering_utils.py.
    
    Args:
        logits: (B, T, V) tensor of steered logits
        alphabet: ESM alphabet
        wt_seq: Wildtype sequence for reference length
        wt_logits: (B, T, V) tensor of wildtype logits (optional, computed if None)
        max_mutations: Maximum number of mutations allowed (default 5)
        similarity_threshold: Cosine similarity threshold for keeping original AA (default 0.90)
        use_similarity_filter: If False, skip similarity filter and use only mutation constraint (default True)
        min_position: Minimum position for mutations, 1-indexed (default None)
        max_position: Maximum position for mutations, 1-indexed (default None)
    
    Returns:
        sequences: List of decoded sequences
    """
    # Get device from logits or use default
    if isinstance(logits_BTV, torch.Tensor):
        device = logits_BTV.device
    elif isinstance(logits_BTV, np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits_BTV = torch.tensor(logits_BTV, device=device, dtype=torch.float32)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits_BTV = torch.tensor(logits_BTV, device=device, dtype=torch.float32)

    if min_position is not None and max_position is not None:
        assert min_position > 0 and max_position > 0, "min_position and max_position must be positive"
        assert min_position <= max_position, "min_position must be less than max_position"
        min_position -= 1
        max_position -= 1
        logits_BTV = logits_BTV[:, min_position:max_position+1, :] # (B, T', V) but we still use T
        window_active = True
    else:
        window_active = False

    
    # Get wildtype logits if not provided
    if wt_logits is None:
        # Tokenize wildtype and get logits
        esm_inference = get_esm_inference()
        wt_tokens = [alphabet.cls_idx] + [alphabet.get_idx(aa) for aa in wt_seq] + [alphabet.eos_idx]
        batch_tokens = torch.tensor([wt_tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            import torch.nn as nn
            model = esm_inference.model.module if isinstance(esm_inference.model, nn.DataParallel) else esm_inference.model
            wt_outputs = model(batch_tokens)
            if isinstance(wt_outputs, dict):
                wt_logits_1TV = wt_outputs["logits"][:, 1:-1, :]  # Remove CLS/EOS
            else:
                wt_logits_1TV = wt_outputs.logits[:, 1:-1, :]  # Remove CLS/EOS
    else:
        wt_logits_1TV = wt_logits
    
    # Convert wt_logits_1TV to tensor if needed
    if isinstance(wt_logits_1TV, np.ndarray):
        wt_logits_1TV = torch.tensor(wt_logits_1TV, device=device)
    elif wt_logits_1TV.device != device:
        wt_logits_1TV = wt_logits_1TV.to(device)
    
    # Ensure logits and wt_logits have same batch size
    B = logits_BTV.shape[0]
    wt_logits_BTV = wt_logits_1TV
    if wt_logits_1TV.shape[0] == 1 and B > 1:
        wt_logits_BTV = wt_logits_1TV.expand(B, -1, -1)
    
    if window_active:
        wt_logits_BTV = wt_logits_BTV[:, min_position:max_position+1, :]
    
    # Compute cosine similarity (only if using similarity filter)
    if use_similarity_filter:
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarities = cos(logits_BTV, wt_logits_BTV)  # (B, T)
    else:
        print(f"    Similarity filter DISABLED - using only mutation constraint")
    
    # Get original token IDs from wildtype
    wt_token_ids = []
    for aa in wt_seq:
        wt_token_ids.append(alphabet.get_idx(aa))
    original_ids = torch.tensor([wt_token_ids] * B, dtype=torch.long, device=device)

    if window_active:
        original_ids = original_ids[:, min_position:max_position+1]
    
    # Get new predicted token IDs
    steered_token_ids_BT = torch.argmax(logits_BTV, dim=-1)  # (B, T)
    
    # Debug: Check how many positions would mutate
    if B > 1:
        mutations_first = (steered_token_ids_BT[0] != original_ids[0]).sum().item()
        mutations_last = (steered_token_ids_BT[-1] != original_ids[-1]).sum().item()
        print(f"    Positions that would mutate (first): {mutations_first}/{len(wt_seq)}, (last): {mutations_last}/{len(wt_seq)}")
    
    # Create mask where logits are too similar (keep original)
    # If similarity filter is disabled, allow all positions to potentially mutate
    if use_similarity_filter:
        is_similar_mask = (similarities >= similarity_threshold)  # (B, T)
        # Where similar, use original; where different, use steered
        final_token_ids = torch.where(
            is_similar_mask,
            original_ids,
            steered_token_ids_BT
        )
    else:
        # Skip similarity filter - use steered predictions directly
        final_token_ids = steered_token_ids_BT
    
    # Constrain to max_mutations: select top positions by difference
    sequences = []
    for batch_idx in range(B):
        seq_token_ids = final_token_ids[batch_idx]  # (T,)
        orig_seq_token_ids = original_ids[batch_idx]  # (T,)
        
        # Find positions where we would mutate
        mutation_mask = (seq_token_ids != orig_seq_token_ids)
        num_mutations = mutation_mask.sum().item()
        
        if num_mutations > max_mutations:
            # Compute logit differences to select top mutations
            steered_probs = torch.softmax(logits_BTV[batch_idx], dim=-1)  # (T, V)
            wt_probs = torch.softmax(wt_logits_BTV[batch_idx], dim=-1)  # (T, V)
            
            # Get probability of predicted token for each position
            steered_token_probs = torch.gather(steered_probs, 1, steered_token_ids_BT[batch_idx].unsqueeze(1)).squeeze(1)
            wt_token_probs = torch.gather(wt_probs, 1, orig_seq_token_ids.unsqueeze(1)).squeeze(1)
            
            # Difference in probability (higher = more confident mutation)
            prob_diff = steered_token_probs - wt_token_probs
            
            # Only consider positions that would mutate
            prob_diff = prob_diff * mutation_mask.float()
            
            # Select top max_mutations positions
            _, top_indices = torch.topk(prob_diff, max_mutations)
            
            # Debug: Print top mutation positions for first and last batch
            if batch_idx == 0 and B > 1:
                top_first = top_indices.cpu().numpy()
                print(f"    Top {max_mutations} mutation positions (first batch): {top_first}")
            
            # Create new sequence: mutate only at top positions
            constrained_seq = orig_seq_token_ids.clone()
            constrained_seq[top_indices] = steered_token_ids_BT[batch_idx][top_indices]
            seq_token_ids = constrained_seq
            
            # Debug: Check if sequences are different
            if batch_idx == 0 and B > 1:
                # Check what the last batch's top indices would be
                prob_diff_last = prob_diff.clone()  # This is for batch 0, need to compute for last
                # Actually, we'll check after processing all batches
        
        # Convert token IDs to amino acid sequence
        aa_seq = []
        for tok_id in seq_token_ids.cpu().numpy():
            tok = alphabet.get_tok(tok_id)
            # Only include single-character amino acids
            if len(tok) == 1 and tok.isalpha():
                aa_seq.append(tok)
        window_seq = ''.join(aa_seq)

        if window_active:
            seq = wt_seq[:min_position] + window_seq + wt_seq[max_position+1:]
        else:
            seq = window_seq
        
        # Truncate to wildtype length if needed
        if len(seq) > len(wt_seq):
            seq = seq[:len(wt_seq)]
        # Pad if needed (shouldn't happen, but safety check)
        if len(seq) < len(wt_seq):
            seq = seq + wt_seq[len(seq):]
        
        sequences.append(seq)
    
    return sequences

def main(args, eval_esm_model=None, eval_alphabet=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ESM inference should already be initialized globally in run_caa_steering.py
    # Just get the reference (won't reload if already initialized)
    esm_inference = get_esm_inference()
    
    # Load DMS data from CSV and infer wildtype sequence
    dms_df = pd.read_csv(args.dms_id)
    if len(dms_df) == 0:
        raise ValueError("DMS dataframe is empty. Cannot infer wildtype sequence.")

    # Infer wildtype from the first row
    wt_sequence = infer_wildtype(dms_df.iloc[0])
    print(f"WT Sequence: {wt_sequence[:10]}... ({len(wt_sequence)} AA)")
    
    # Load evaluation model if available (matches CLT probe steering)
    eval_model = None
    wt_eval_score = np.nan
    tokenizer = None
    esm_model = None
    
    if EVAL_MODEL_AVAILABLE and args.eval_models_dir:
        script_dir = Path(__file__).parent.parent
        eval_path = script_dir / "steering" / args.eval_models_dir / f"{args.dms_name}.pt"
        if eval_path.exists():
            try:
                eval_model = load_cnn(None, str(eval_path), device)
                # Use pre-loaded ESM model if provided, otherwise load it
                if eval_esm_model is not None and eval_alphabet is not None:
                    esm_model = eval_esm_model
                    alphabet = eval_alphabet
                else:
                    # Load ESM model and tokenizer for eval scoring (fallback if not pre-loaded)
                    # Don't wrap in DataParallel here - get_layer_activations will handle it
                    weights = args.esm_weights
                    esm_filename = os.path.basename(weights)
                    num_layers = int(esm_filename.split("_")[1][1:])
                    d_model = 320 if "8M" in esm_filename else 480 if "35M" in esm_filename else None
                    esm_model, alphabet = load_esm_model(args.esm_weights, device, num_layers=num_layers, d_model=d_model)
                if isinstance(esm_model, torch.nn.DataParallel):
                    esm_model = esm_model.module
                tokenizer = ESMTokenizerWrapper(alphabet)
                
                # Use get_probe_input to match run_probe_steering.py
                rep = get_probe_input(esm_inference, wt_sequence)
                with torch.no_grad():
                    wt_eval_score = eval_model(rep).item()
                print(f"Loaded eval model, WT eval score: {wt_eval_score:.3f}")
            except Exception as e:
                print(f"Warning: Could not load eval model: {e}")
                import traceback
                traceback.print_exc()
    
    # Use provided dms_name or infer from dms_id
    if args.dms_name:
        dms_name = args.dms_name
    elif ".csv" in args.dms_id:
        dms_path = Path(args.dms_id)
        dms_name = dms_path.stem
        # Remove "test_fold{fold}" prefix if present
        if dms_name.startswith("test_fold"):
            parent_name = dms_path.parent.name
            if parent_name and parent_name != "rand_multiples":
                dms_name = parent_name
            else:
                grandparent = dms_path.parent.parent.name
                if grandparent:
                    dms_name = grandparent
    else:
        dms_name = args.dms_id
    
    # Load training data to compute CAA vector
    # The training CSV should be in the same directory structure
    script_dir = Path(__file__).parent.parent
    train_csv_path = script_dir / "function_circuit" / "DMS" / "cv_folds_multiples_substitutions" / f"{dms_name}.csv"
    
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    
    train_df = pd.read_csv(train_csv_path)
    fold_col = f"fold_{args.split_type}"
    if fold_col not in train_df.columns:
        raise ValueError(f"Fold column '{fold_col}' not found in training data")
    
    # Get train indices (excluding only test fold - include val fold in training)
    folds = train_df[fold_col].values
    fold_vals = folds - folds.min()
    
    test_mask = (fold_vals == args.fold)
    train_mask = ~test_mask  # Include val fold in training
    train_idx = train_df[train_mask].index.values
    
    if len(train_idx) < 2:
        raise ValueError(f"Insufficient training data: {len(train_idx)} samples")
    
    # Get positive and negative sequences from training data
    # Filter by dms_score_bin: 1 for positive, 0 for negative
    train_subset = train_df.iloc[train_idx].copy()
    
    # Filter sequences that match wildtype length
    train_subset = train_subset[train_subset['mutated_sequence'].str.len() == len(wt_sequence)]
    
    # Get sequences with bin=1 (positive) and bin=0 (negative)
    if 'DMS_score_bin' not in train_subset.columns:
        raise ValueError("DMS_score_bin column not found in training data. Required for CAA steering.")
    
    pos_candidates = train_subset[train_subset['DMS_score_bin'] == 1].copy()
    neg_candidates = train_subset[train_subset['DMS_score_bin'] == 0].copy()
    
    if len(pos_candidates) == 0:
        raise ValueError("No sequences with DMS_score_bin=1 found in training data")
    if len(neg_candidates) == 0:
        raise ValueError("No sequences with DMS_score_bin=0 found in training data")
    
    # Verify assumption: bin=1 should have higher scores than bin=0
    pos_mean_score = pos_candidates['DMS_score'].mean()
    neg_mean_score = neg_candidates['DMS_score'].mean()
    print(f"    Verification: bin=1 mean score: {pos_mean_score:.3f}, bin=0 mean score: {neg_mean_score:.3f}")
    if pos_mean_score < neg_mean_score:
        print(f"    WARNING: bin=1 has LOWER mean score than bin=0! This may indicate reversed binning.")
        print(f"    Proceeding anyway, but verify your data convention.")
    
    # Select based on percentile threshold or all if proportion is None
    if args.pos_neg_prop is not None:
        if not (0 < args.pos_neg_prop <= 1):
            raise ValueError(f"pos_neg_prop must be between 0 and 1, got {args.pos_neg_prop}")
        
        # Calculate percentile thresholds
        # For positive: take sequences >= (100 - pos_neg_prop*100)th percentile
        # For negative: take sequences <= (pos_neg_prop*100)th percentile
        pos_percentile = (1 - args.pos_neg_prop) * 100  # e.g., 0.1 -> 90th percentile
        neg_percentile = args.pos_neg_prop * 100  # e.g., 0.1 -> 10th percentile
        
        pos_threshold = np.percentile(pos_candidates['DMS_score'].values, pos_percentile)
        neg_threshold = np.percentile(neg_candidates['DMS_score'].values, neg_percentile)
        
        # Filter sequences above/below threshold
        pos_seqs = pos_candidates[pos_candidates['DMS_score'] >= pos_threshold]['mutated_sequence'].tolist()
        neg_seqs = neg_candidates[neg_candidates['DMS_score'] <= neg_threshold]['mutated_sequence'].tolist()
        
        print(f"Positive sequences: {len(pos_seqs)} (DMS_score >= {pos_threshold:.3f}, {pos_percentile:.1f}th percentile of bin=1)")
        print(f"Negative sequences: {len(neg_seqs)} (DMS_score <= {neg_threshold:.3f}, {neg_percentile:.1f}th percentile of bin=0)")
    else:
        # Use all sequences with the appropriate bins
        pos_seqs = pos_candidates['mutated_sequence'].tolist()
        neg_seqs = neg_candidates['mutated_sequence'].tolist()
        
        print(f"Positive sequences: {len(pos_seqs)} (all with DMS_score_bin=1)")
        print(f"Negative sequences: {len(neg_seqs)} (all with DMS_score_bin=0)")
    
    # Validate sequences before computing CAA vector
    if len(pos_seqs) == 0:
        raise ValueError(f"No positive sequences selected. Check pos_neg_prop={args.pos_neg_prop} and data filtering.")
    if len(neg_seqs) == 0:
        raise ValueError(f"No negative sequences selected. Check pos_neg_prop={args.pos_neg_prop} and data filtering.")
    
    # Ensure all sequences are strings and have correct length
    pos_seqs = [str(seq) for seq in pos_seqs if isinstance(seq, str) and len(seq) == len(wt_sequence)]
    neg_seqs = [str(seq) for seq in neg_seqs if isinstance(seq, str) and len(seq) == len(wt_sequence)]
    
    if len(pos_seqs) == 0:
        raise ValueError(f"After filtering for correct length ({len(wt_sequence)}), no positive sequences remain.")
    if len(neg_seqs) == 0:
        raise ValueError(f"After filtering for correct length ({len(wt_sequence)}), no negative sequences remain.")
    
    print(f"Using {len(pos_seqs)} positive and {len(neg_seqs)} negative sequences (all length {len(wt_sequence)})")
    
    # Randomly sample 10% of sequences for each trial (to increase diversity)
    sampling_prop = 0.1
    # Use trial and seed for reproducible but unique sampling per trial
    rng = np.random.RandomState(args.seed + args.trial)
    
    num_pos_sample = max(1, int(len(pos_seqs) * sampling_prop))
    num_neg_sample = max(1, int(len(neg_seqs) * sampling_prop))
    
    pos_indices = rng.choice(len(pos_seqs), size=num_pos_sample, replace=False)
    neg_indices = rng.choice(len(neg_seqs), size=num_neg_sample, replace=False)
    
    sampled_pos_seqs = [pos_seqs[i] for i in pos_indices]
    sampled_neg_seqs = [neg_seqs[i] for i in neg_indices]
    
    print(f"    Trial {args.trial}: Randomly sampled {len(sampled_pos_seqs)} pos and {len(sampled_neg_seqs)} neg sequences (10% of {len(pos_seqs)}/{len(neg_seqs)}) for CAA vector")
    
    # Use sampled sequences for CAA vector
    pos_seqs = sampled_pos_seqs
    neg_seqs = sampled_neg_seqs
    
    try:
        steering_vector = get_caa_vector(pos_seqs, neg_seqs)
    except Exception as e:
        raise RuntimeError(f"Failed to compute CAA vector: {e}") from e
    print(f"Steering vector shape: {steering_vector.shape}")
    
    # Alpha values
    multipliers = np.linspace(args.alpha_min, args.alpha_max, args.k) # (B)
    
    # Get wildtype logits for cosine similarity comparison
    print(f"Getting wildtype logits for comparison...")
    esm_inference = get_esm_inference()
    wt_tokens = [esm_inference.alphabet.cls_idx] + [esm_inference.alphabet.get_idx(aa) for aa in wt_sequence] + [esm_inference.alphabet.eos_idx]
    batch_tokens = torch.tensor([wt_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        import torch.nn as nn
        #model = esm_inference.model.module if isinstance(esm_inference.model, nn.DataParallel) else esm_inference.model
        wt_outputs = esm_inference.model(batch_tokens)
        if isinstance(wt_outputs, dict):
            wt_logits_1TV = wt_outputs["logits"][:, 1:-1, :]  # Remove CLS/EOS
        else:
            wt_logits_1TV = wt_outputs.logits[:, 1:-1, :]  # Remove CLS/EOS
        # Ensure wt_logits are on the correct device
        if wt_logits_1TV.device != device:
            wt_logits_1TV = wt_logits_1TV.to(device)
    
    # Steer all sequences at once (batched)
    print(f"Steering {len(multipliers)} alpha values in batch...")
    print(f"  Alpha range: [{multipliers.min():.3f}, {multipliers.max():.3f}]")
    try:
        # Steer all sequences at once
        logits_BTV = steer_caa_sequence(wt_sequence, multipliers.tolist(), steering_vector)
        
        # Debug: Check if logits are different for different alphas
        if logits_BTV.shape[0] > 1:
            logit_diff = torch.norm(logits_BTV[0] - logits_BTV[-1]).item()
            print(f"  Logit difference (first vs last alpha): {logit_diff:.6f}")
        
        # Decode all sequences at once with cosine similarity and mutation constraint
        max_mutations = getattr(args, 'max_mutations', 5)
        use_similarity_filter = not getattr(args, 'disable_similarity_filter', False)
        if args.dms_name == "SPG1_STRSG_Olson_2014":
            min_position = 228
            max_position = 282
        else:
            min_position = None
            max_position = None
        # remove CLS/EOS, they weren't steered anyways
        logits_BTV = logits_BTV[:, 1:-1, :]
        steered_seqs = decode_logits_to_sequence(
            logits_BTV, esm_inference.alphabet, wt_sequence, wt_logits=wt_logits_1TV, 
            max_mutations=max_mutations, use_similarity_filter=use_similarity_filter, min_position=min_position, max_position=max_position
        )
        
        # Debug: Check unique sequences after decoding
        unique_decoded = len(set(steered_seqs))
        print(f"  Unique sequences after decoding: {unique_decoded} / {len(steered_seqs)}")
    except Exception as e:
        print(f"Warning: Batch steering failed: {e}")
        steered_seqs = [None] * len(multipliers)
    
    # Process each steered sequence
    results = []
    for i, (mult, steered_seq) in enumerate(zip(multipliers, steered_seqs)):
        try:
            if steered_seq is None or len(steered_seq) != len(wt_sequence):
                results.append({
                    "fold": args.fold,
                    "trial": args.trial,
                    "alpha": mult,
                    "mutant": np.nan,
                    "mutated_sequence": steered_seq if steered_seq else "",
                    # "DMS_score": np.nan,      <-- Removed as requested
                    # "DMS_score_bin": np.nan,  <-- Removed as requested
                    "wt_eval_score": wt_eval_score,
                    "eval_score": np.nan
                })
                continue
            
            # Revert mutations outside allowed positions (if needed)
            steered_seq_list = list(steered_seq)
            final_seq = ''.join(steered_seq_list)
            
            # Get mutant string
            mutant_str = get_mutant_string(wt_sequence, final_seq)
            if mutant_str == "WT" or mutant_str == "LengthMismatch":
                mutant_str = np.nan
            
            # Check if steered sequence exists in test set and get DMS score
            dms_score = np.nan
            dms_score_bin = np.nan
            test_match = dms_df[dms_df['mutated_sequence'] == final_seq]
            if not test_match.empty:
                dms_score = test_match.iloc[0]['DMS_score']
                if 'DMS_score_bin' in test_match.columns:
                    dms_score_bin = test_match.iloc[0]['DMS_score_bin']
            
            # Score with eval model if available (matches CLT probe steering)
            eval_score = np.nan
            if eval_model:
                try:
                    rep = get_probe_input(esm_inference, final_seq)
                    with torch.no_grad():
                        eval_score = eval_model(rep).item()
                except Exception as e:
                    print(f"Warning: Eval scoring failed for alpha={mult}: {e}")
                    pass
            
            results.append({
                "fold": args.fold,
                "trial": args.trial,
                "alpha": mult,
                "mutant": mutant_str,
                "mutated_sequence": final_seq,
                # "DMS_score": dms_score,          <-- Removed as requested
                # "DMS_score_bin": dms_score_bin,  <-- Removed as requested
                "wt_eval_score": wt_eval_score,
                "eval_score": eval_score
            })
        except Exception as e:
            print(f"Warning: Processing failed for alpha={mult}: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="alpha")
        
        # Count unique sequences in this CSV
        if 'mutated_sequence' in results_df.columns:
            unique_seqs = results_df['mutated_sequence'].nunique()
            total_seqs = len(results_df)
            print(f"  Unique sequences in this CSV: {unique_seqs} / {total_seqs} total")
        
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv} ({len(results_df)} rows)")
        
        # Return steering vector and results
        return steering_vector, results
    else:
        print("Warning: No results to save")
        return None, None

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Steer CAA Vector and Score")
    parser.add_argument("--dms_id", type=str, required=True, help="DMS ID or CSV path (test fold CSV)")
    parser.add_argument("--dms_name", type=str, default=None, help="DMS name (inferred from dms_id if not provided)")
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=21, help="Number of steps in sweep")
    parser.add_argument("--pos_neg_prop", type=float, default=None, 
                       help="Proportion (0-1) for percentile threshold. If 0.1, takes sequences >= 90th percentile of bin=1 (pos) and <= 10th percentile of bin=0 (neg). If None, uses all sequences with bin=1 (pos) and bin=0 (neg)")
    parser.add_argument("--split_type", type=str, default="rand_multiples", help="Split type for folds")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data splitting")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--trial", type=int, default=0, help="Trial number")
    parser.add_argument("--eval_models_dir", type=str, default="eval_models", 
                       help="Directory containing eval CNN models (optional)")
    parser.add_argument("--esm_weights", type=str, 
                       default="../models/esm2_t6_8M_UR50D.pt",
                       help="Path to ESM model weights (for eval model scoring)")
    parser.add_argument("--max_mutations", type=int, default=5,
                       help="Maximum number of mutations allowed in generated sequences (default: 5)")
    parser.add_argument("--disable_similarity_filter", action="store_true",
                       help="Disable cosine similarity filter (use only mutation constraint)")
    return parser

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
