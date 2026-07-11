import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfaidx import Fasta
import torch
import torch.nn.functional as F
import os
import pyBigWig
import json
import sys
sys.path.append("../gamba")
from tqdm import tqdm


from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS


def get_latest_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint in the directory"""
    ckpt_path = None
    if last_step == -1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path

def load_model(config_path, checkpoint_path):
    """Load the model from a checkpoint"""
    # Load model config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set up tokenizer
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    
    # Create the model
    model, _ = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item()
    )
    
    # Get model hyperparameters
    d_model = config.get("d_model", 512)
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    
    # Set up the model
    model = JambagambaModel(
        model, d_model=d_model, nhead=nhead, n_layers=n_layers, 
        padding_id=0, dim_feedfoward=dim_feedforward
    )
    
    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_path, "model_optimizer.pt"), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create collator
    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config.get("pad_to_multiple_of", None),
        test=True,
    )
    
    return model, collator, tokenizer, device

def get_sequence_window(genome, chromosome, position, window_size=2048):
    """Get a sequence window with the mutation at the end position"""
    target_pos = window_size - 1  # Position 2047 (last position in 0-indexed sequence)
    start = position - target_pos  # Start position to have mutation at end
    end = start + window_size  # End position
    
    # Get the reference sequence
    sequence = genome[chromosome][start:end].seq.upper()
    
    # Check if we got the full window length
    if len(sequence) != window_size:
        raise ValueError(f"Could not extract full sequence window of length {window_size}")
    
    return sequence, start, target_pos

def get_reverse_complement(sequence):
    """Get the reverse complement of a DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

def process_variants(genome, bw, model, collator, tokenizer, device, batch_size=32):
    """
    Process variants from a DataFrame
    
    This function:
    1. Places each mutation at position 2047 (end of sequence)
    2. Extracts model predictions for both reference and alternate alleles
    3. Accounts for sequence padding in the collator (+1 for start token)
    4. Calculates predictions for both forward and reverse complement strands
    
    Returns:
    - ref_logits_data: List of average logits at the mutation position for reference sequences
    - alt_logits_data: List of average logits at the mutation position for mutated sequences  
    - conservation_scores: List of average predicted conservation scores at the mutation position
    - true_conservation: List of true conservation scores at the mutation position
    - labels: List of variant labels (0=benign, 1=pathogenic)
    """
    valid_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    df = pd.read_parquet("hf://datasets/songlab/clinvar/test.parquet")
    # Lists to store results
    ref_logits_data = []
    alt_logits_data = []
    conservation_scores = []
    true_conservation_scores = []  # New list for true conservation scores
    labels = []
    non_matching_refs = []
    
    # Process variants in batches
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Lists for forward sequences
        batch_forward_sequences = []
        batch_forward_scores = []
        # Lists for reverse complement sequences
        batch_reverse_sequences = []
        batch_reverse_scores = []
        
        batch_refs = []
        batch_alts = []
        batch_labels = []
        batch_positions = []
        valid_indices = []
        
        # Prepare batch data
        for idx, row in batch_df.iterrows():
            chromosome = "chr" + str(row['chrom'])
            if chromosome not in valid_chromosomes:
                continue
                
            # Get 1-indexed genomic position
            position = int(row['pos'])
            
            # Convert to 0-indexed position for sequence access
            position = position - 1
            label = row['label']
            ref = row['ref']
            alt = row['alt']
            
            # Skip non-SNVs
            if len(ref) != 1 or len(alt) != 1:
                continue
                
            try:
                # Get sequence with mutation at the end
                forward_sequence, start, target_pos = get_sequence_window(genome, chromosome, position)
                
                # Check sequence length and reference allele
                if len(forward_sequence) != 2048:
                    print(f"Skipping sequence of length {len(forward_sequence)}")
                    continue
                    
                if forward_sequence[target_pos] != ref:
                    print(f"Reference allele mismatch: {forward_sequence[target_pos]} vs {ref}")
                    non_matching_refs.append((chromosome, position, ref, alt))
                    continue
                
                # Generate reverse complement sequence
                reverse_sequence = get_reverse_complement(forward_sequence)
                
                # Create reverse complement for alt/ref
                rev_ref = get_reverse_complement(ref)
                rev_alt = get_reverse_complement(alt)
                
                # Tokenize sequences
                forward_tokens = tokenizer.tokenizeMSA(forward_sequence)
                reverse_tokens = tokenizer.tokenizeMSA(reverse_sequence)
                
                # Get conservation scores
                vals = np.zeros(2048, dtype=np.float64)
                intervals = bw.intervals(chromosome, start, start + 2048)
                
                if intervals is not None:
                    for interval_start, interval_end, value in intervals:
                        offset_start = max(0, interval_start - start)
                        offset_end = min(2048, interval_end - start)
                        vals[offset_start:offset_end] = value
                
                forward_scores = np.round(vals, 2)
                # Reverse the conservation scores for the reverse complement
                reverse_scores = np.round(vals[::-1], 2)
                
                # Add to batch
                batch_forward_sequences.append(forward_tokens)
                batch_forward_scores.append(forward_scores)
                
                batch_reverse_sequences.append(reverse_tokens)
                batch_reverse_scores.append(reverse_scores)
                
                batch_refs.append((ref, rev_ref))
                batch_alts.append((alt, rev_alt))
                batch_labels.append(label)
                batch_positions.append(target_pos)
                valid_indices.append(len(batch_forward_sequences) - 1)
                
            except Exception as e:
                print(f"Error processing variant: {e}")
                continue
        
        if not batch_forward_sequences:
            continue
        
        # Process forward strand
        forward_results = process_strand(
            batch_forward_sequences, 
            batch_forward_scores,
            batch_refs, 
            batch_alts,
            batch_positions,
            valid_indices,
            model, 
            collator, 
            tokenizer, 
            device,
            strand="forward"
        )
        
        # Process reverse strand
        reverse_results = process_strand(
            batch_reverse_sequences, 
            batch_reverse_scores,
            batch_refs, 
            batch_alts,
            batch_positions,
            valid_indices,
            model, 
            collator, 
            tokenizer, 
            device,
            strand="reverse"
        )
        
        # Combine results from both strands
        for i, idx in enumerate(valid_indices):
            fwd_ref, fwd_alt, fwd_cons = forward_results[i]
            rev_ref, rev_alt, rev_cons = reverse_results[i]
            
            # Average the probabilities from both strands
            avg_ref_prob = (fwd_ref + rev_ref) / 2
            avg_alt_prob = (fwd_alt + rev_alt) / 2
            
            ref_logits_data.append(avg_ref_prob)
            alt_logits_data.append(avg_alt_prob)
            
                            # Average conservation scores if available
            if fwd_cons is not None and rev_cons is not None:
                avg_cons = (fwd_cons + rev_cons) / 2
                conservation_scores.append(avg_cons)
            
            # Get true conservation score at the mutation position
            true_cons = batch_forward_scores[idx][batch_positions[idx]]
            true_conservation_scores.append(true_cons)
            
            labels.append(batch_labels[idx])
            
    print(f"Percentage of non-matching reference alleles: {len(non_matching_refs) / len(df) * 100:.2f}%")
    return ref_logits_data, alt_logits_data, conservation_scores, true_conservation_scores, labels


def plot_results(ref_probs, alt_probs, labels, output_dir, name, metric="loglikelihood"):
    """Plot the results of the analysis"""
    # Convert to numpy arrays
    ref_probs = np.array(ref_probs)
    alt_probs = np.array(alt_probs)
    labels = np.array(labels)
    
    # Calculate log-likelihood ratios
    if metric == "loglikelihood":
        # Calculate log-likelihood ratio: log(p_alt / p_ref)
        log_ratios = np.log(alt_probs) - np.log(ref_probs)
    else:
        # Just use the direct probability ratio: p_alt / p_ref
        log_ratios = alt_probs / ref_probs
    
    # Separate benign and pathogenic
    benign_ratios = log_ratios[labels == 0]
    pathogenic_ratios = log_ratios[labels == 1]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(benign_ratios, bins=50, alpha=0.5, label=f'Benign (n={len(benign_ratios)})', color='blue')
    plt.hist(pathogenic_ratios, bins=50, alpha=0.5, label=f'Pathogenic (n={len(pathogenic_ratios)})', color='red')
    
    # Set labels and title
    if metric == "loglikelihood":
        plt.xlabel('Log-Likelihood Ratio (log(p_alt / p_ref))')
    else:
        plt.xlabel('Probability Ratio (p_alt / p_ref)')
    plt.ylabel('Frequency')
    plt.title(f'{metric.title()} Ratio Analysis for {name}')
    plt.legend(loc='upper right')
    
    # Save figure
    output_file = os.path.join(output_dir, f"{name}_{metric}_ratio_analysis.png")
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    
    # Calculate separation metrics
    auc = calculate_auc(log_ratios, labels)
    print(f"AUC for {metric}: {auc:.4f}")
    
    return auc

def plot_conservation_scores(conservation_scores, labels, output_dir, name, score_type="Predicted"):
    """Plot the conservation score analysis"""
    # Convert to numpy arrays
    conservation_scores = np.array(conservation_scores)
    labels = np.array(labels)
    
    # Separate benign and pathogenic
    benign_scores = conservation_scores[labels == 0]
    pathogenic_scores = conservation_scores[labels == 1]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(benign_scores, bins=50, alpha=0.5, label=f'Benign (n={len(benign_scores)})', color='blue')
    plt.hist(pathogenic_scores, bins=50, alpha=0.5, label=f'Pathogenic (n={len(pathogenic_scores)})', color='red')
    
    # Set labels and title
    plt.xlabel(f'{score_type} Conservation Score')
    plt.ylabel('Frequency')
    plt.title(f'{score_type} Conservation Score Analysis for {name}')
    plt.legend(loc='upper right')
    
    # Save figure
    output_file = os.path.join(output_dir, f"{name}_{score_type.lower()}_conservation_score_analysis.png")
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    
    # Calculate separation metrics
    auc = calculate_auc(conservation_scores, labels)
    print(f"AUC for {score_type.lower()} conservation score: {auc:.4f}")
    
    return auc

def plot_conservation_comparison(predicted_scores, true_scores, labels, output_dir, name):
    """Plot comparison between predicted and true conservation scores"""
    # Convert to numpy arrays
    predicted_scores = np.array(predicted_scores)
    true_scores = np.array(true_scores)
    labels = np.array(labels)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points by class
    benign_mask = labels == 0
    pathogenic_mask = labels == 1
    
    plt.scatter(true_scores[benign_mask], predicted_scores[benign_mask], 
                alpha=0.5, label='Benign', color='blue', s=10)
    plt.scatter(true_scores[pathogenic_mask], predicted_scores[pathogenic_mask], 
                alpha=0.5, label='Pathogenic', color='red', s=10)
    
    # Add diagonal line for reference
    max_val = max(np.max(predicted_scores), np.max(true_scores))
    min_val = min(np.min(predicted_scores), np.min(true_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Calculate correlation
    overall_corr = np.corrcoef(predicted_scores, true_scores)[0, 1]
    benign_corr = np.corrcoef(predicted_scores[benign_mask], true_scores[benign_mask])[0, 1]
    pathogenic_corr = np.corrcoef(predicted_scores[pathogenic_mask], true_scores[pathogenic_mask])[0, 1]
    
    plt.title(f'Predicted vs True Conservation Scores (r = {overall_corr:.3f})')
    plt.xlabel('True Conservation Score')
    plt.ylabel('Predicted Conservation Score')
    plt.legend()
    
    # Add correlation text
    plt.annotate(f'Overall Correlation: {overall_corr:.3f}\n'
                 f'Benign Correlation: {benign_corr:.3f}\n'
                 f'Pathogenic Correlation: {pathogenic_corr:.3f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                 va='top')
    
    # Save figure
    output_file = os.path.join(output_dir, f"{name}_conservation_comparison.png")
    plt.savefig(output_file)
    print(f"Saved comparison plot to {output_file}")
    
    return overall_corr

def calculate_auc(values, labels):
    """Calculate AUC for the prediction values"""
    from sklearn.metrics import roc_auc_score
    # If the AUC is below 0.5, invert the predictions
    try:
        auc = roc_auc_score(labels, values)
        if auc < 0.5:
            auc = roc_auc_score(labels, -values)
    except:
        auc = 0.5  # Default in case of error
    return auc

def process_strand(batch_sequences, batch_scores, batch_refs, batch_alts, 
                  batch_positions, valid_indices, model, collator, tokenizer, device, strand="forward"):
    """
    Process a single strand (forward or reverse complement)
    
    Returns:
    - List of tuples (ref_prob, alt_prob, conservation_score) for each variant
    """
    results = []
    
    # Prepare model inputs
    batch_inputs = list(zip(batch_sequences, batch_scores))
    collated = collator(batch_inputs)
    
    # Run model
    with torch.no_grad():
        output = model(collated[0].to(device), collated[1].to(device))
    
    # Extract logits at mutation position
    seq_logits = output["seq_logits"]
    
    # Check if conservation scores are available
    has_conservation = "scaling_logits" in output
    if has_conservation:
        conservation_logits = output["scaling_logits"]
    
    # Process each sequence in the batch
    for i, idx in enumerate(valid_indices):
        # Get the appropriate reference and alternate alleles based on strand
        ref_pair = batch_refs[idx]
        alt_pair = batch_alts[idx]
        
        if strand == "forward":
            ref_base = ref_pair[0]  # Forward ref
            alt_base = alt_pair[0]  # Forward alt
        else:
            ref_base = ref_pair[1]  # Reverse ref
            alt_base = alt_pair[1]  # Reverse alt
        
        # Get sequence at target position
        ref_token = batch_sequences[idx][batch_positions[idx]]
        
        # Convert bases to token indices
        alt_token = tokenizer.tokenizeMSA(alt_base)[0]
        
        # Get logits at mutation position (add 1 for the start token padding)
        orig_position = batch_positions[idx]
        model_position = orig_position + 1  # Add 1 to account for start token padding
        position_logits = seq_logits[i, model_position, :]
        
        # Get softmax probabilities
        position_probs = F.softmax(position_logits, dim=-1)
        
        # Get probabilities for reference and alternate alleles
        ref_prob = position_probs[ref_token].item()
        alt_prob = position_probs[alt_token].item()
        
        # Get conservation score if available
        conservation_score = None
        if has_conservation:
            try:
                #Get only mean at position (model outputs mean not mean and log_var)
                mean = conservation_logits[i, model_position,0].item()
                conservation_score = mean

                # Get mean and variance at mutation position
                # mean = conservation_logits[i, model_position, 0].item()
                # log_var = conservation_logits[i, model_position, 1].item()
                # variance = np.exp(log_var)


                
                # # Sample from the predicted distribution
                # sample = np.random.normal(loc=mean, scale=np.sqrt(variance))
                
                # # Store prediction
                # conservation_score = sample
            except Exception as e:
                print(f"Error sampling conservation score: {e}")
        
        results.append((ref_prob, alt_prob, conservation_score))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze ClinVar mutations at end position")
    parser.add_argument('--csv_file', type=str, default ="/home/mica/gamba/data_processing/data/hg38_noncoding_mutations/clin_var_GPNMSA.csv", help='Path to the CSV file with noncoding variants')
    parser.add_argument('--genome_fasta', type=str,  default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/VEP/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str,  default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for model evaluation')
    
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get name from CSV file
    name = os.path.basename(args.csv_file).split("_")[0]

    
    # Load genome and conservation scores
    print(f"Loading genome from {args.genome_fasta}")
    genome = Fasta(args.genome_fasta)
    
    print(f"Loading conservation scores from {args.big_wig}")
    bw = pyBigWig.open(args.big_wig)
    
    # Get checkpoint path
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/")
    #checkpoint_path = get_latest_checkpoint_path(ckpt_dir, 78000)
    #checkpoint_path = "/home/mica/gamba/dcps/dcp_4000_reweighted_cons"
    checkpoint_path = "/home/mica/gamba/dcps/103600_only_MSE"
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, collator, tokenizer, device = load_model(args.config_fpath, checkpoint_path)
    
    # Process variants
    print("Processing variants with forward and reverse complement averaging")
    ref_probs, alt_probs, conservation_scores, true_conservation_scores, labels = process_variants(
        genome, bw, model, collator, tokenizer, device, args.batch_size
    )
    
    # Save results
    results = {
        'ref_probs': ref_probs,
        'alt_probs': alt_probs,
        'conservation_scores': conservation_scores,
        'true_conservation_scores': true_conservation_scores,
        'labels': labels
    }
    torch.save(results, os.path.join(args.output_dir, f"{name}_results_bidirectional.pt"))
    
    # Plot log-likelihood analysis
    print("Plotting log-likelihood analysis")
    ll_auc = plot_results(ref_probs, alt_probs, labels, args.output_dir, f"{name}_bidirectional", "loglikelihood")
    
    # Plot probability ratio analysis
    print("Plotting probability ratio analysis")
    prob_auc = plot_results(ref_probs, alt_probs, labels, args.output_dir, f"{name}_bidirectional", "probability")
    
    # Plot conservation score analysis if available
    if conservation_scores:
        print("Plotting predicted conservation score analysis")
        pred_cons_auc = plot_conservation_scores(conservation_scores, labels, args.output_dir, f"{name}_bidirectional", "Predicted")
        
        print("Plotting true conservation score analysis")
        true_cons_auc = plot_conservation_scores(true_conservation_scores, labels, args.output_dir, f"{name}_bidirectional", "True")
        
        print("Plotting conservation score comparison")
        corr = plot_conservation_comparison(conservation_scores, true_conservation_scores, labels, args.output_dir, f"{name}_bidirectional")
        
        print(f"\nSummary for {name} (bidirectional):")
        print(f"Log-likelihood AUC: {ll_auc:.4f}")
        print(f"Probability ratio AUC: {prob_auc:.4f}")
        print(f"Predicted conservation score AUC: {pred_cons_auc:.4f}")
        print(f"True conservation score AUC: {true_cons_auc:.4f}")
        print(f"Conservation score correlation: {corr:.4f}")
    else:
        print(f"\nSummary for {name} (bidirectional):")
        print(f"Log-likelihood AUC: {ll_auc:.4f}")
        print(f"Probability ratio AUC: {prob_auc:.4f}")
        print("Conservation scores not available")


if __name__ == "__main__":
    main()