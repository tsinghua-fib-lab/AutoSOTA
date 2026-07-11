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
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel,JambaGambaNOALMModel
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
    # model = JambaGambaNoConsModel(
    #     model, d_model=d_model, nhead=nhead, n_layers=n_layers, 
    #     padding_id=0, dim_feedfoward=dim_feedforward
    # )
    # model = JambaGambaNOALMModel(
    #     model, d_model=d_model, nhead=nhead, n_layers=n_layers, 
    #     padding_id=0, dim_feedfoward=dim_feedforward
    # )
    # Load the model checkpoint
    #checkpoint = torch.load(os.path.join(checkpoint_path, "model_optimizer.pt"), weights_only=True)
    checkpoint = torch.load(
        os.path.join(checkpoint_path, "model_optimizer.pt"),
        map_location=lambda storage, loc: storage.cuda(0),
        weights_only=True
    )

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

# def get_sequence_window(genome, chromosome, position, window_size=2048):
#     """Get a sequence window with the mutation at the end position"""
#     target_pos = window_size - 1  # Position 2047 (last position in 0-indexed sequence)
#     start = position - target_pos  # Start position to have mutation at end
#     end = start + window_size  # End position
    
#     # Get the reference sequence
#     sequence = genome[chromosome][start:end].seq.upper()
    
#     # Check if we got the full window length
#     if len(sequence) != window_size:
#         raise ValueError(f"Could not extract full sequence window of length {window_size}")
    
#     return sequence, start, target_pos


def get_sequence_window(genome, chromosome, position, window_size=2048):
    """
    Get a sequence window with the mutation centered in the sequence.
    
    Args:
        genome: Indexed genome (e.g., from pyfaidx)
        chromosome: Chromosome name (e.g., "chr1")
        position: 0-based genomic coordinate of the mutation
        window_size: Total window size to extract

    Returns:
        sequence (str): The reference sequence
        start (int): The start genomic coordinate of the window
        target_pos (int): Position of the mutation within the returned sequence
    """
    target_pos = window_size // 2  # center the mutation
    start = position - target_pos
    end = start + window_size

    # Extract sequence
    sequence = genome[chromosome][start:end].seq.upper()

    # Validate
    if len(sequence) != window_size:
        raise ValueError(f"Could not extract full sequence window of length {window_size}")

    return sequence, start, target_pos


def process_variants(genome, bw, model, collator, tokenizer, device, batch_size=32):
    from Bio.Seq import Seq
    import torch.nn.functional as F

    valid_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    #df = pd.read_parquet("hf://datasets/songlab/clinvar/test.parquet")
    df = pd.read_parquet("hf://datasets/songlab/omim/test.parquet")

    ref_logits_data, alt_logits_data, conservation_scores = [], [], []
    labels, true_conservation_scores, non_matching_refs = [], [], []

    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        fwd_inputs, rev_inputs = [], []
        fwd_info, rev_info = [], []

        for idx, row in batch_df.iterrows():
            chromosome = f"chr{row['chrom']}"
            if chromosome not in valid_chromosomes: continue

            position = int(row['pos']) - 1
            label, ref, alt = row['label'], row['ref'], row['alt']
            if len(ref) != 1 or len(alt) != 1: continue

            try:
                seq, start, target_pos = get_sequence_window(genome, chromosome, position)
                if len(seq) != 2048 or seq[target_pos] != ref:
                    if len(seq) != 2048: continue
                    non_matching_refs.append((chromosome, position, ref, alt))
                    continue

                tokens_fwd = tokenizer.tokenizeMSA(seq)
                scores_fwd = np.zeros(2048)
                intervals = bw.intervals(chromosome, start, start + 2048)
                if intervals is not None:
                    for s, e, val in intervals:
                        scores_fwd[max(0, s - start):min(2048, e - start)] = val
                scores_fwd = np.round(scores_fwd, 2)

                seq_rev = str(Seq(seq).reverse_complement())
                tokens_rev = tokenizer.tokenizeMSA(seq_rev)
                scores_rev = scores_fwd[::-1].copy()

                fwd_inputs.append((tokens_fwd, scores_fwd))
                rev_inputs.append((tokens_rev, scores_rev))
                fwd_info.append((tokenizer.tokenizeMSA(ref)[0], tokenizer.tokenizeMSA(alt)[0], target_pos, scores_fwd[target_pos], label))
                rev_info.append((tokenizer.tokenizeMSA(ref)[0], tokenizer.tokenizeMSA(alt)[0], 2047 - target_pos, scores_rev[2047 - target_pos], label))

            except Exception:
                continue

        if not fwd_inputs: continue

        def run_model(batch_inputs):
            collated = collator(batch_inputs)
            with torch.no_grad():
                output = model(collated[0].to(device), collated[1].to(device))
            return output.get("seq_logits", None), output.get("scaling_logits", None)

        logits_fwd, cons_fwd = run_model(fwd_inputs)
        logits_rev, cons_rev = run_model(rev_inputs)
        no_seq = False
        if logits_fwd is None or logits_rev is None:
            no_seq = True

        for i in range(len(fwd_inputs)):
            try:
                ref_tok, alt_tok, pos_fwd, true_score, label = fwd_info[i]
                pos_rev = rev_info[i][2]
                if not no_seq: 
                    # Ensure both seq_logits are present for this sample
                    if logits_fwd[i] is None or logits_rev[i] is None:
                        continue

                    p_ref_fwd = F.softmax(logits_fwd[i, pos_fwd + 1], dim=-1)[ref_tok].item()
                    p_alt_fwd = F.softmax(logits_fwd[i, pos_fwd + 1], dim=-1)[alt_tok].item()
                    p_ref_rev = F.softmax(logits_rev[i, pos_rev + 1], dim=-1)[ref_tok].item()
                    p_alt_rev = F.softmax(logits_rev[i, pos_rev + 1], dim=-1)[alt_tok].item()

                    log_lik_ref = 0.5 * (np.log(p_ref_fwd) + np.log(p_ref_rev))
                    log_lik_alt = 0.5 * (np.log(p_alt_fwd) + np.log(p_alt_rev))
                    ref_logits_data.append(np.exp(log_lik_ref))
                    alt_logits_data.append(np.exp(log_lik_alt))
                
                # Conservation check
                if cons_fwd is not None and cons_rev is not None:
                    c_fwd = cons_fwd[i, pos_fwd + 1, 0].item()
                    c_rev = cons_rev[i, pos_rev + 1, 0].item()
                    conservation_scores.append(0.5 * (c_fwd + c_rev))
                    true_conservation_scores.append(true_score)

                labels.append(label)

            except Exception as e:
                print(f"Skipping one sample due to error: {e}")
                continue

    print(f"Percentage of non-matching reference alleles: {len(non_matching_refs) / len(df) * 100:.2f}%")
    return ref_logits_data, alt_logits_data, conservation_scores, labels, true_conservation_scores

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

def plot_conservation_scores(conservation_scores, labels, output_dir, name):
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
    plt.xlabel('Predicted Conservation Score')
    plt.ylabel('Frequency')
    plt.title(f'Conservation Score Analysis for {name}')
    plt.legend(loc='upper right')
    
    # Save figure
    output_file = os.path.join(output_dir, f"{name}_conservation_score_analysis.png")
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    
    # Calculate separation metrics
    auc = calculate_auc(conservation_scores, labels)
    print(f"AUC for conservation score: {auc:.4f}")
    
    return auc

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

def main():
    parser = argparse.ArgumentParser(description="Analyze ClinVar mutations at end position")
    parser.add_argument('--csv_file', type=str, default ="/home/mica/gamba/data_processing/data/hg38_noncoding_mutations/clin_var_GPNMSA.csv", help='Path to the CSV file with noncoding variants')
    parser.add_argument('--genome_fasta', type=str,  default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/VEP/OMIM/dual_gamba/', help='Path to the output file')
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
    # ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/")
    # checkpoint_path = get_latest_checkpoint_path(ckpt_dir, 78000)
    #checkpoint_path = "/home/mica/gamba/clean_dcps/dcp_noALM56000"
    #checkpoint_path = "/home/mica/gamba/clean_dcps/CCP/dcp_44000"
    checkpoint_path = "/home/mica/gamba/clean_dcps/focal_loss/dcp_22000"
    #checkpoint_path = "/home/mica/gamba/clean_dcps/CCP/dcp_noALM44000"
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, collator, tokenizer, device = load_model(args.config_fpath, checkpoint_path)
    
    # Process variants
    print("Processing variants")
    ref_probs, alt_probs, conservation_scores, labels, true_conservation_scores = process_variants(
        genome, bw, model, collator, tokenizer, device, args.batch_size
    )
    
    # Save results
    results = {
        'ref_probs': ref_probs,
        'alt_probs': alt_probs,
        'conservation_scores': conservation_scores,
        'labels': labels,
        'true_conservation_scores': true_conservation_scores,
    }
    torch.save(results, os.path.join(args.output_dir, f"{name}_results.pt"))
    if ref_probs:
        # Plot log-likelihood analysis
        print("Plotting log-likelihood analysis")
        ll_auc = plot_results(ref_probs, alt_probs, labels, args.output_dir, name, "loglikelihood")
        
        # Plot probability ratio analysis
        print("Plotting probability ratio analysis")
        prob_auc = plot_results(ref_probs, alt_probs, labels, args.output_dir, name, "probability")
        
    # Plot conservation score analysis if available
    if conservation_scores:
        print("Plotting conservation score analysis")
        cons_auc = plot_conservation_scores(conservation_scores, labels, args.output_dir, name)
        print(f"\nSummary for {name}:")
        print(f"Conservation score AUC: {cons_auc:.4f}")
    elif ref_probs:
            print(f"Log-likelihood AUC: {ll_auc:.4f}")
            print(f"Probability ratio AUC: {prob_auc:.4f}")
    else:
        print("Log-likelihood and probability ratios not available")
        print("Conservation scores not available")

    if conservation_scores:
        from scipy.stats import pearsonr, spearmanr

        pred = np.array(conservation_scores)
        true = np.array(true_conservation_scores)

        pearson_corr, _ = pearsonr(pred, true)
        spearman_corr, _ = spearmanr(pred, true)

        print(f"Pearson correlation (pred vs. true conservation): {pearson_corr:.4f}")
        print(f"Spearman correlation (pred vs. true conservation): {spearman_corr:.4f}")


if __name__ == "__main__":
    main()