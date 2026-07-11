import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyBigWig
import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta
import random
import sys
sys.path.append('/home/mica/gamba')
from evodiff.utils import Tokenizer
import sequence_models.constants as constants
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sequence_models.utils import warmup
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)


def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> str:
    """Get the path to the latest checkpoint."""
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

def load_model(checkpoint_path, config_path, model_type):
    """Load the model from checkpoint."""
    import json
    ckpt_path = checkpoint_path 
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["task"] = config["task"].lower().strip()
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    
    print(f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}")
    
    # Create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item()
    )
    
    # Get model parameters from config
    d_model = config.get("d_model", 512)
    nhead = config.get("n_head", 8)
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)
    
    # Initialize JambagambaModel
    if model_type == "JambaGambaNOALMModel":
        model = JambaGambaNOALMModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, 
            padding_id=padding_id, dim_feedfoward=dim_feedforward
        )
    else:
        model = JambagambaModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, 
            padding_id=padding_id, dim_feedfoward=dim_feedforward
        )
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), 
                           map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Initialize optimizer and scheduler
    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.0)
    )
    lr_func = warmup(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    sd = torch.load(
        os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu")
    )
    scheduler.load_state_dict(sd["scheduler_state_dict"])
    
    return model, tokenizer, config

def generate_test_samples(bigwig_file, genome_fasta, num_samples=1000, seq_length=2048):
    """Generate random non-overlapping test samples."""
    # Open the bigwig file and genome fasta
    bw = pyBigWig.open(bigwig_file)
    genome = Fasta(genome_fasta)
    
    # Get chromosome sizes
    chroms = [(chrom, size) for chrom, size in bw.chroms().items() 
              if chrom in genome.keys() and not chrom.startswith("chr_")]
    
    #subset to only chromsomes 16,3,2,22
    chroms = [(chrom, size) for chrom, size in chroms if chrom in ['chr16', 'chr3', 'chr2', 'chr22']]
    #subset to ANY CHROM BUT 16,3,2,22
    #chroms = [(chrom, size) for chrom, size in chroms if chrom not in ['chr16', 'chr3', 'chr2', 'chr22']]
    samples = []
    pbar = tqdm(total=num_samples, desc="Generating test samples")
    
    while len(samples) < num_samples:
        # Randomly select a chromosome
        chrom, size = random.choice(chroms)
        
        # Ensure we stay within chromosome bounds
        max_start = size - seq_length - 1
        if max_start <= 0:
            continue
        
        # Generate a random start position
        start = random.randint(0, max_start)
        end = start + seq_length
        
        # Get the sequence and conservation scores
        try:
            sequence = genome[chrom][start:end].seq
            scores = np.array(bw.values(chrom, start, end))
            
            # Check if sequence has valid bases and scores
            if (len(sequence) == seq_length and 
                len(scores) == seq_length and 
                'N' not in sequence.upper() and 
                not np.isnan(scores).any()):
                
                samples.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'sequence': sequence,
                    'scores': scores
                })
                pbar.update(1)
        except Exception as e:
            print(f"Error with {chrom}:{start}-{end}: {e}")
            continue
    
    pbar.close()
    bw.close()
    
    return samples

def predict_conservation_scores(model, tokenizer, samples, batch_size=48, device=None):
    """Run model predictions on samples."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Create collator
    collator = gLMCollator(tokenizer=tokenizer, test=True)
    
    all_predictions = []
    all_true_scores = []
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Predicting scores"):
        batch_samples = samples[i:i+batch_size]
        batch_data = []
        
        for sample in batch_samples:
            # Tokenize sequence
            sequence_tokens = tokenizer.tokenizeMSA(sample['sequence'])

            # # only get first half of sequence
            # # Tokenize the sequence
            # sequence_tokens = tokenizer.tokenizeMSA(sample['sequence'][:1024])
            
            # # Pad sequence_tokens to match scores length (2050)
            # # Create a small test input with just the padding character
            # pad_token_id = tokenizer.tokenizeMSA(constants.MSA_PAD)[0]  # Get the ID of the padding token
            # sequence_tokens = np.pad(
            #     sequence_tokens, 
            #     (0, 2048 - len(sequence_tokens)),
            #     constant_values=pad_token_id
            # )
    
            batch_data.append((sequence_tokens, sample['scores']))
            all_true_scores.append(sample['scores'])
        
        # Collate batch
        collated = collator(batch_data)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(collated[0].to(device), collated[1].to(device))
        
        # Extract predictions (means)
        scaling_logits = outputs["scaling_logits"]
        
        # Process each prediction in the batch
        for j in range(scaling_logits.size(0)):
            # Extract means
            means = scaling_logits[j, :, 0].cpu().numpy()
            variances = scaling_logits[j, :, 1].exp().cpu().numpy()
            all_predictions.append((means, variances))
    
    return all_predictions, all_true_scores

def calculate_positional_correlations(predictions, true_scores, seq_length=2048):
    """Calculate correlation at each position across all samples."""
    # Initialize arrays to store data for each position
    position_corrs = np.zeros(seq_length)
    
    # For each position, calculate correlation across all samples
    for pos in range(seq_length):
        pos_preds = np.array([pred[0][pos] for pred in predictions])
        pos_true = np.array([score[pos] for score in true_scores])
        
        # Calculate correlation
        valid_indices = ~np.isnan(pos_preds) & ~np.isnan(pos_true)
        if np.sum(valid_indices) > 1:  # Need at least 2 points for correlation
            position_corrs[pos] = np.corrcoef(pos_preds[valid_indices], 
                                             pos_true[valid_indices])[0, 1]
        else:
            position_corrs[pos] = np.nan
    
    return position_corrs

def plot_positional_correlations(position_corrs, window_size=20, output_path=None):
    """Plot positional correlations with optional smoothing."""
    # Create smoothed version for visualization
    smoothed_corrs = np.convolve(
        np.nan_to_num(position_corrs), 
        np.ones(window_size)/window_size, 
        mode='valid'
    )
    
    # Create x-axis for smoothed data
    x_smoothed = np.arange(len(smoothed_corrs)) + window_size//2
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot raw correlations with transparency
    plt.plot(position_corrs, alpha=0.3, color='blue', label='Raw Correlations')
    
    # Plot smoothed correlations
    plt.plot(x_smoothed, smoothed_corrs, linewidth=2, color='red', 
             label=f'Smoothed ({window_size}-point window)')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add visualization of average correlation in different regions
    regions = [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
    for start, end in regions:
        # Check if there's any valid data in this region
        region_data = position_corrs[start:min(end, len(position_corrs))]
        valid_in_region = np.sum(~np.isnan(region_data))
        
        if valid_in_region > 0:
            region_avg = np.nanmean(region_data)
            plt.axvspan(start, end, alpha=0.1, 
                       label=f'Pos {start}-{end}: {region_avg:.3f}')
        else:
            plt.axvspan(start, end, alpha=0.1, 
                       label=f'Pos {start}-{end}: NaN')
    
    plt.xlabel('Position in Sequence')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation Between Predicted and True Conservation Scores by Position')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save or show the plot
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return plt

def main():
    parser = argparse.ArgumentParser(
        description="Analyze how correlation between predicted and true conservation scores changes with sequence position"
    )
    
    parser.add_argument(
        "--bigwig_file", type=str, 
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores"
    )
    
    parser.add_argument(
        "--genome_fasta", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file"
    )
    
    parser.add_argument(
        "--config_path", type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json",
        help="Path to the model configuration file"
    )
    
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/home/mica/gamba/clean_dcps/dcp_56000",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000,
        help="Number of samples to analyze (default: 1000)"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=48,
        help="Batch size for model inference (default: 48)"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="/home/mica/gamba/data_processing/data/correlations",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--sequence_length", type=int, default=2048,
        help="Length of sequences to sample (default: 2048)"
    )
    
    parser.add_argument(
        "--smoothing_window", type=int, default=50,
        help="Window size for smoothing correlations (default: 50)"
    )
    
    args = parser.parse_args()
    #based on checkpoint get model_type and step
    #if checkpoint_path contains noALM, model type is JambaGambaNoALM
    if "noALM" in args.checkpoint_path:
        model_type = "JambaGambaNOALMModel"
    else:
        model_type = "JambagambaModel"
    print(f"Using model type: {model_type}")
    # Create output directory if it doesn't exist
    output_dir = f"{args.output_dir}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)


    
    # Load model
    print("Loading model...")
    model, tokenizer, config = load_model(
        args.checkpoint_path, args.config_path, model_type
    )
    
    # Generate test samples
    print(f"Generating {args.num_samples} test samples...")
    import random
    samples = generate_test_samples(
        args.bigwig_file, args.genome_fasta, 
        num_samples=args.num_samples, 
        seq_length=args.sequence_length
    )
    
    # Run predictions
    print("Running predictions...")
    predictions, true_scores = predict_conservation_scores(
        model, tokenizer, samples, batch_size=args.batch_size
    )
    
    # Calculate positional correlations
    print("Calculating positional correlations...")
    position_corrs = calculate_positional_correlations(
        predictions, true_scores, seq_length=args.sequence_length
    )
    
    # Save correlation data
    corr_path = os.path.join(output_dir, "positional_correlations.npy")
    np.save(corr_path, position_corrs)
    print(f"Saved correlation data to {corr_path}")
    
    # Plot results
    print("Generating plot...")
    plot_path = os.path.join(output_dir, "positional_correlations.png")
    plot_positional_correlations(
        position_corrs, window_size=args.smoothing_window, output_path=plot_path
    )
    
    # Calculate summary statistics
    quarters = args.sequence_length // 4
    quarter_stats = []
    for i in range(4):
        start, end = i * quarters, (i + 1) * quarters
        avg_corr = np.nanmean(position_corrs[start:end])
        quarter_stats.append((start, end, avg_corr))
    
    # Print summary
    print("\nSummary of positional correlations:")
    print("Position Range | Average Correlation")
    print("-" * 40)
    for start, end, avg in quarter_stats:
        print(f"{start:5d} - {end:5d} | {avg:.4f}")
    
    overall_avg = np.nanmean(position_corrs)
    print("-" * 40)
    print(f"Overall        | {overall_avg:.4f}")

if __name__ == "__main__":
    main()