import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os
import sys
sys.path.append("../gamba")
from typing import Optional, Sequence, Tuple, Type


from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_latest_dcp_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint path."""
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

def load_model(checkpoint_dir, config_fpath, last_step=52000, device=None):
    """Load the model from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Get the latest checkpoint path
    ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, last_step=last_step)
    #ckpt_path = '/home/mica/gamba/dcps/dcp_25000_only_MSE'
    ckpt_path = '/home/mica/gamba/dcps/dcp_132250_only_MSE'
    
    # Load configuration
    with open(config_fpath, "r") as f:
        config = json.load(f)
    
    # Setup tokenizer and task
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    
    logging.info(f"Task: {task}, Model: {config['model_type']}, Config: {config['model_config']}")
    
    # Create model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )
    
    # Get model parameters from config
    d_model = config.get("d_model", 512)
    nhead = config.get("n_head", 8)
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)
    
    # Set up the full model
    model = JambagambaModel(
        model, d_model=d_model, nhead=nhead, n_layers=n_layers,
        padding_id=padding_id, dim_feedfoward=dim_feedforward
    )
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model, tokenizer

def sample_regions(genome_fasta, bigwig_file, num_regions=100, region_length=2048, chromosomes=None):
    """Sample random regions from the genome."""
    # Open the genome fasta file
    genome = Fasta(genome_fasta)
    
    # Open the bigwig file
    bw = pyBigWig.open(bigwig_file)
    
    # Get list of chromosomes
    if chromosomes is None:
        chromosomes = list(genome.keys())
    else:
        # Ensure all specified chromosomes exist in the genome
        chromosomes = [c for c in chromosomes if c in genome.keys()]
        #only allow chromosomes: 2,22,3, 16
        chromosomes = ['chr2', 'chr22', 'chr3', 'chr16']
    
    sampled_regions = []
    
    logging.info(f"Sampling {num_regions} regions of length {region_length}...")
    
    with tqdm(total=num_regions) as pbar:
        while len(sampled_regions) < num_regions:
            # Choose a random chromosome
            chrom = random.choice(chromosomes)
            
            # Get chromosome length
            chrom_length = len(genome[chrom])
            
            if chrom_length <= region_length:
                continue
            
            # Choose a random start position
            start = random.randint(0, chrom_length - region_length)
            end = start + region_length
            
            # Get the sequence and scores
            try:
                sequence = genome[chrom][start:end].seq
                scores = bw.values(chrom, start, end)
                
                # Check if sequence and scores are valid
                if len(sequence) == region_length and not np.any(np.isnan(scores)):
                    sampled_regions.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'sequence': sequence,
                        'scores': scores
                    })
                    pbar.update(1)
            except Exception as e:
                logging.warning(f"Error processing {chrom}:{start}-{end}: {e}")
    
    bw.close()
    return sampled_regions

def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None):
    """Run predictions on sampled regions with batching for speed."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create collator
    collator = gLMCollator(tokenizer=tokenizer)
    
    all_predictions = []
    all_true_scores = []
    
    logging.info(f"Running predictions on {len(regions)} regions with batch size {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(regions), batch_size)):
        batch_regions = regions[i:i+batch_size]
        batch_inputs = []
        
        # Prepare batch
        for region in batch_regions:
            # Tokenize sequence
            sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            
            # Store for collation
            batch_inputs.append((sequence_tokens, scores))
            
            # Store true scores
            all_true_scores.append(scores)
        
        # Skip empty batches (shouldn't happen but just in case)
        if not batch_inputs:
            continue
            
        # Collate batch
        collated = collator(batch_inputs)
        
        # Run the model
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        
        # Process each item in the batch
        scaling_logits = output["scaling_logits"]
        
        # For each item in the batch
        for j in range(scaling_logits.size(0)):
            # Get scaling logits for this sequence
            seq_logits = scaling_logits[j]
            
            # Extract means and variances
            means = seq_logits[:, 0].cpu().numpy()
            # variances = seq_logits[:, 1].exp().cpu().numpy()  # Convert log-variance to variance
            
            # # Sample from the predicted distribution
            # samples = np.random.normal(loc=means, scale=np.sqrt(variances), size=means.shape)

            #just use mean instead of sampling
            samples = means
            
            # Store predictions
            all_predictions.append(samples)
    
    # Ensure all arrays have the same length before concatenating
    min_length = min(len(arr) for arr in all_true_scores + all_predictions)
    
    # Trim arrays to the minimum length
    all_predictions = [arr[:min_length] for arr in all_predictions]
    all_true_scores = [arr[:min_length] for arr in all_true_scores]
    
    # Flatten the lists
    all_predictions = np.concatenate(all_predictions)
    all_true_scores = np.concatenate(all_true_scores)
    
    return all_predictions, all_true_scores

def compute_score_distribution(scores, num_bins=1000, min_val=None, max_val=None):
    """Compute distribution of scores."""
    # Filter out NaN values
    valid_scores = scores[~np.isnan(scores)]
    
    # Determine min and max values if not provided
    if min_val is None:
        min_val = np.min(valid_scores)
    if max_val is None:
        max_val = np.max(valid_scores)
    
    # Create histogram
    hist, bin_edges = np.histogram(valid_scores, bins=num_bins, range=(min_val, max_val))
    
    return hist, bin_edges

def plot_score_distributions(true_scores, predicted_scores, save_path, num_bins=1000):
    """Plot distributions of true and predicted scores."""
    # Compute min and max across both distributions for consistent binning
    all_scores = np.concatenate([
        true_scores[~np.isnan(true_scores)],
        predicted_scores[~np.isnan(predicted_scores)]
    ])
    min_val = np.min(all_scores)
    max_val = np.max(all_scores)
    
    # Compute histograms
    true_hist, true_bin_edges = compute_score_distribution(
        true_scores, num_bins=num_bins, min_val=min_val, max_val=max_val
    )
    pred_hist, pred_bin_edges = compute_score_distribution(
        predicted_scores, num_bins=num_bins, min_val=min_val, max_val=max_val
    )
    
    # Get bin centers
    true_bin_centers = (true_bin_edges[:-1] + true_bin_edges[1:]) / 2
    pred_bin_centers = (pred_bin_edges[:-1] + pred_bin_edges[1:]) / 2
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot distributions
    axes[0].plot(true_bin_centers, true_hist, label='True PhyloP Scores', color='blue')
    axes[0].plot(pred_bin_centers, pred_hist, label='Predicted Scores', color='red')
    axes[0].set_title('Distribution of True vs Predicted Conservation Scores')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot log scale for better visibility of tails
    axes[1].plot(true_bin_centers, true_hist, label='True PhyloP Scores', color='blue')
    axes[1].plot(pred_bin_centers, pred_hist, label='Predicted Scores', color='red')
    axes[1].set_title('Distribution of True vs Predicted Conservation Scores (Log Scale)')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add zero line for reference
    for ax in axes:
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add statistics
    true_mean = np.nanmean(true_scores)
    true_std = np.nanstd(true_scores)
    pred_mean = np.nanmean(predicted_scores)
    pred_std = np.nanstd(predicted_scores)
    
    stats_text = (
        f"True scores: mean={true_mean:.4f}, std={true_std:.4f}\n"
        f"Predicted scores: mean={pred_mean:.4f}, std={pred_std:.4f}"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"Saved distribution plot to {save_path}")
    
    return true_hist, true_bin_edges, pred_hist, pred_bin_edges

def plot_score_comparison(true_scores, predicted_scores, save_path):
    """Create a 2D histogram/heatmap comparing true vs predicted scores."""
    # Ensure arrays have exactly the same shape
    if true_scores.shape != predicted_scores.shape:
        logging.warning(f"Shape mismatch: true_scores {true_scores.shape}, predicted_scores {predicted_scores.shape}")
        min_len = min(len(true_scores), len(predicted_scores))
        true_scores = true_scores[:min_len]
        predicted_scores = predicted_scores[:min_len]
        logging.info(f"Trimmed arrays to length {min_len}")
    
    # Remove NaN values
    mask = ~(np.isnan(true_scores) | np.isnan(predicted_scores))
    filtered_true = true_scores[mask]
    filtered_pred = predicted_scores[mask]
    
    logging.info(f"Using {len(filtered_true)} non-NaN values for comparison plot")
    
    # Calculate correlation
    correlation = np.corrcoef(filtered_true, filtered_pred)[0, 1]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create 2D histogram
    h = plt.hist2d(filtered_true, filtered_pred, bins=100, cmap='viridis', norm='log')
    
    # Add diagonal line (perfect prediction)
    min_val = min(filtered_true.min(), filtered_pred.min())
    max_val = max(filtered_true.max(), filtered_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Add correlation coefficient
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes, 
             bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    
    plt.colorbar(h[3], label='Count (log scale)')
    plt.xlabel('True PhyloP Score')
    plt.ylabel('Predicted Score')
    plt.title('True vs Predicted Conservation Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"Saved comparison plot to {save_path}")

def sample_and_plot_empirical_distribution(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    num_regions=100,
    region_length=2048,
    num_bins=1000,
    chromosomes=None,
    last_step=52000,
    batch_size=8
):
    """
    Sample regions, predict scores, and plot distributions.
    
    Args:
        genome_fasta: Path to genome FASTA file
        bigwig_file: Path to phyloP bigWig file
        checkpoint_dir: Directory containing model checkpoints
        config_fpath: Path to model config JSON
        output_dir: Directory to save outputs
        num_regions: Number of regions to sample
        region_length: Length of each region
        num_bins: Number of bins for histograms
        chromosomes: List of chromosomes to sample from (None for all)
        last_step: Checkpoint step to use
        batch_size: Batch size for model predictions
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log basic info
    logging.info(f"Starting distribution sampling with {num_regions} regions, batch size {batch_size}")
    logging.info(f"Using genome: {genome_fasta}")
    logging.info(f"Using scores: {bigwig_file}")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model, tokenizer = load_model(checkpoint_dir, config_fpath, last_step=last_step, device=device)
    
    # Sample regions
    regions = sample_regions(
        genome_fasta, bigwig_file, num_regions=num_regions, 
        region_length=region_length, chromosomes=chromosomes
    )
    
    # Run predictions with batching
    predicted_scores, true_scores = predict_scores_batched(
        model, tokenizer, regions, batch_size=batch_size, device=device
    )
    
    # Log shapes for debugging
    logging.info(f"Predicted scores shape: {predicted_scores.shape}")
    logging.info(f"True scores shape: {true_scores.shape}")
    
    # Check and ensure shapes match
    if predicted_scores.shape != true_scores.shape:
        min_len = min(len(predicted_scores), len(true_scores))
        predicted_scores = predicted_scores[:min_len]
        true_scores = true_scores[:min_len]
        logging.warning(f"Trimmed score arrays to matching length {min_len}")
    
    # Save the raw scores
    np.savez(
        output_dir / "score_distributions.npz",
        true_scores=true_scores,
        predicted_scores=predicted_scores
    )
    
    # Plot distributions
    true_hist, true_bins, pred_hist, pred_bins = plot_score_distributions(
        true_scores, predicted_scores, 
        save_path=output_dir / "score_distributions.png",
        num_bins=num_bins
    )
    
    # Plot true vs predicted comparison
    plot_score_comparison(
        true_scores, predicted_scores,
        save_path=output_dir / "score_comparison.png"
    )
    
    # Save distribution data
    np.savez(
        output_dir / "histogram_data.npz",
        true_hist=true_hist,
        true_bins=true_bins,
        pred_hist=pred_hist,
        pred_bins=pred_bins
    )
    
    logging.info("Completed empirical distribution sampling and plotting")

def main():
    parser = argparse.ArgumentParser(
        description="Sample and plot empirical distribution of predicted conservation scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/empirical_distribution",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "/tmp/"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default='/home/mica/gamba/configs/jamba-small-240mammalian.json',
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=5000,
        help="Number of regions to sample",
    )
    parser.add_argument(
        "--region_length",
        type=int,
        default=2048,
        help="Length of each sampled region",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=1000,
        help="Number of bins for histograms",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        help="List of chromosomes to sample from (default: all)",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=78000,
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Batch size for model predictions",
    )
    
    args = parser.parse_args()
    
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Starting script with {args.num_regions} regions, batch size {args.batch_size}")
    
    try:
        sample_and_plot_empirical_distribution(
            args.genome_fasta,
            args.bigwig_file,
            args.checkpoint_dir,
            args.config_fpath,
            args.output_dir,
            num_regions=args.num_regions,
            region_length=args.region_length,
            num_bins=args.num_bins,
            chromosomes=args.chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size
        )
        logging.info("Script completed successfully")
    except Exception as e:
        logging.error(f"Error in sample_and_plot_empirical_distribution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()