import argparse
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import torch
import os
import sys
sys.path.append("../gamba")
from Bio.Seq import Seq
import pyBigWig

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS

import json
import matplotlib.pyplot as plt
import random

class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]
        
def parse_promoter_bed(bed_file):
    """Parse the specialized promoter BED file format."""
    columns = [
        "chrom", "start", "end", "name", "score", "strand", 
        "core_start", "core_end"
    ]
    df = pd.read_csv(bed_file, sep="\s+", header=None, names=columns)
    return df

def generate_random_regions_matched_GC(promoter_df, genome, num_random=None, max_context_size=1748, tolerance=0.01):
        """
        Generate random regions matching the chromosomes, lengths, and GC content of input promoters.
        
        Args:
            promoter_df: DataFrame with promoter information
            genome: Genome object from pyfaidx
            num_random: Number of random regions to generate (defaults to match promoter count)
            max_context_size: Maximum context size (model max length - generation length)
            tolerance: Allowed GC content difference (in percentage) between promoters and random regions
            
        Returns:
            DataFrame: Random regions DataFrame
        """
        if num_random is None:
            num_random = len(promoter_df)
            
        random_regions = []
        
        # Calculate GC content for each promoter
        promoter_gc_content = []
        for _, row in promoter_df.iterrows():
            chrom = row["chrom"]
            start = row["start"]
            end = row["end"]
            try:
                seq = str(genome[chrom][start:end].seq.upper())
                gc_content = calculate_gc_content(seq)
                promoter_gc_content.append(gc_content)
            except Exception as e:
                print(f"Error calculating GC content for promoter {row['name']}: {e}")
                promoter_gc_content.append(0.0)
        
        for i in range(num_random):
            # Get a random row from the promoter dataframe to match chromosome and GC content
            idx = random.randint(0, len(promoter_df) - 1)
            original_row = promoter_df.iloc[idx]
            
            chrom = original_row["chrom"]
            target_gc_content = promoter_gc_content[idx]
            
            # Use the exact same promoter length for better comparison
            promoter_length = original_row["end"] - original_row["start"]
            
            # Get chromosome length
            chrom_length = len(genome[chrom])
            
            # For very large promoters, handle specially
            if promoter_length > max_context_size:
                promoter_length = max_context_size
            
            # Generate random regions until GC content matches within tolerance
            for _ in range(100):  # Limit attempts to avoid infinite loops
                # Generate random start position, ensuring it's not too close to the end
                max_start = chrom_length - promoter_length - 500
                if max_start <= 0:
                    print(f"Chromosome {chrom} is too short for random region, skipping")
                    break
                    
                random_start = random.randint(1, max_start)
                random_end = random_start + promoter_length
                
                # Extract sequence and calculate GC content
                try:
                    seq = str(genome[chrom][random_start:random_end].seq.upper())
                    gc_content = calculate_gc_content(seq)
                except Exception as e:
                    print(f"Error extracting random sequence for {chrom}:{random_start}-{random_end}: {e}")
                    continue
                
                # Check if GC content matches within tolerance
                if abs(gc_content - target_gc_content) <= tolerance:
                    # Create a new row for the random region
                    random_row = {
                        "chrom": chrom,
                        "start": random_start,
                        "end": random_end,
                        "name": f"random_{chrom}_{random_start}_{random_end}",
                        "score": 0,
                        "strand": "+",  # Use + strand for simplicity
                        "core_start": random_start,
                        "core_end": random_end,
                        "is_random": True  # Flag to indicate this is a random region
                    }
                    
                    random_regions.append(random_row)
                    break
        
        random_df = pd.DataFrame(random_regions)
        # Ensure the is_random column is explicitly present
        random_df["is_random"] = True
        
        print(f"Generated {len(random_df)} random regions with matched GC content")
        return random_df


def generate_random_regions(promoter_df, genome, num_random=None, max_context_size=1748):
    """
    Generate random regions matching the chromosomes and lengths of input promoters.
    Optimized to be consistent with the context extraction for promoters.
    
    Args:
        promoter_df: DataFrame with promoter information
        genome: Genome object from pyfaidx
        num_random: Number of random regions to generate (defaults to match promoter count)
        max_context_size: Maximum context size (model max length - generation length)
        
    Returns:
        DataFrame: Random regions DataFrame
    """
    if num_random is None:
        num_random = len(promoter_df)
        
    random_regions = []
    
    # Calculate promoter length statistics for more realistic random generation
    promoter_lengths = [row["end"] - row["start"] for _, row in promoter_df.iterrows()]
    avg_promoter_length = sum(promoter_lengths) / len(promoter_lengths) if promoter_lengths else 100
    
    for i in range(num_random):
        # Get a random row from the promoter dataframe to match chromosome
        idx = random.randint(0, len(promoter_df) - 1)
        original_row = promoter_df.iloc[idx]
        
        chrom = original_row["chrom"]
        
        # Use the exact same promoter length for better comparison
        promoter_length = original_row["end"] - original_row["start"]
        
        # Get chromosome length
        chrom_length = len(genome[chrom])
        
        # For very large promoters, handle specially
        if promoter_length > max_context_size:
            # For long promoters, just take a section with the max allowed size
            promoter_length = max_context_size
        
        # Generate random start position, ensuring it's not too close to the end
        # Allow at least 500bp downstream for potential generation
        max_start = chrom_length - promoter_length - 500
        if max_start <= 0:
            print(f"Chromosome {chrom} is too short for random region, skipping")
            continue
            
        random_start = random.randint(1, max_start)
        random_end = random_start + promoter_length
        
        # Create a new row for the random region
        random_row = {
            "chrom": chrom,
            "start": random_start,
            "end": random_end,
            "name": f"random_{chrom}_{random_start}_{random_end}",
            "score": 0,
            "strand": "+",  # Use + strand for simplicity
            "core_start": random_start,
            "core_end": random_end,
            "is_random": True  # Flag to indicate this is a random region
        }
        
        # Check if we should try to avoid known promoter regions
        # Simple approach: make sure we're at least 1000bp away from any known promoter
        is_overlapping = False
        min_distance = 1000
        
        for _, promoter_row in promoter_df.iterrows():
            if promoter_row["chrom"] != chrom:
                continue
                
            # Check if random region overlaps or is too close to a known promoter
            if (random_start >= promoter_row["start"] - min_distance and 
                random_start <= promoter_row["end"] + min_distance):
                is_overlapping = True
                break
                
            if (random_end >= promoter_row["start"] - min_distance and 
                random_end <= promoter_row["end"] + min_distance):
                is_overlapping = True
                break
        
        # Skip this random region if it overlaps with a known promoter
        if is_overlapping:
            # Try again with a different random region
            i -= 1
            continue
        
        random_regions.append(random_row)
    
    random_df = pd.DataFrame(random_regions)
    # Ensure the is_random column is explicitly present
    random_df["is_random"] = True
    
    print(f"Generated {len(random_df)} random regions")
    return random_df


# New imports to add at the top of the file
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
try:
    from sklearn.metrics import roc_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, AUC calculation will be skipped")

# New function to calculate GC content
def calculate_gc_content(sequence):
    """Calculate the GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    
    return (gc_count / total) * 100.0 if total > 0 else 0.0

# New function to compare GC content between promoter and random regions
def compare_gc_content(promoter_sequences, random_sequences, output_dir):
    """
    Compare GC content between promoter and random regions.
    
    Args:
        promoter_sequences: List of promoter sequences
        random_sequences: List of random sequences
        output_dir: Directory to save results
        
    Returns:
        tuple: (mean_diff, p_value_mw, p_value_ttest)
    """
    promoter_gc = [calculate_gc_content(seq) for seq in promoter_sequences]
    random_gc = [calculate_gc_content(seq) for seq in random_sequences]
    
    # Calculate statistics
    mean_promoter_gc = np.mean(promoter_gc)
    mean_random_gc = np.mean(random_gc)
    mean_diff = mean_promoter_gc - mean_random_gc
    
    # Statistical tests
    _, p_value_mw = mannwhitneyu(promoter_gc, random_gc)
    _, p_value_ttest = ttest_ind(promoter_gc, random_gc)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(promoter_gc, alpha=0.7, bins=20, label=f'Promoter Regions (Mean: {mean_promoter_gc:.2f}%)')
    plt.hist(random_gc, alpha=0.7, bins=20, label=f'Random Regions (Mean: {mean_random_gc:.2f}%)')
    
    plt.xlabel('GC Content (%)')
    plt.ylabel('Frequency')
    plt.title('GC Content Distribution: Promoters vs Random Regions')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gc_content_histogram.png"))
    plt.close()
    
    # Print summary
    print("\n=== GC CONTENT ANALYSIS ===")
    print(f"Promoter regions mean GC content: {mean_promoter_gc:.2f}%")
    print(f"Random regions mean GC content: {mean_random_gc:.2f}%")
    print(f"Difference: {mean_diff:.2f}%")
    print(f"Mann-Whitney U test p-value: {p_value_mw:.4f}")
    print(f"T-test p-value: {p_value_ttest:.4f}")
    
    # Evaluate separability
    if p_value_mw < 0.05 or p_value_ttest < 0.05:
        print("The GC content distributions are significantly different (p < 0.05)")
        
        # Simple classifier accuracy (using mean as threshold)
        # Calculate the midpoint between promoter and random GC content means to use as a threshold
        threshold = (mean_promoter_gc + mean_random_gc) / 2
        correct_promoter = sum(1 for gc in promoter_gc if (gc > threshold if mean_diff > 0 else gc < threshold))
        correct_random = sum(1 for gc in random_gc if (gc < threshold if mean_diff > 0 else gc > threshold))
        
        accuracy = (correct_promoter + correct_random) / (len(promoter_gc) + len(random_gc))
        print(f"Simple classifier accuracy using GC content: {accuracy:.2%}")
    else:
        print("The GC content distributions are not significantly different (p >= 0.05)")
    
    return mean_diff, p_value_mw, p_value_ttest

def rank_all_sequences_by_conservation(results, output_dir):
    """
    Rank all sequences by conservation score regardless of promoter/random status.
    
    Args:
        results: Analysis results
        output_dir: Directory to save results
        
    Returns:
        DataFrame: Combined ranked dataframe
    """
    # Create a combined DataFrame (no filtering by combined_score)
    all_df = pd.DataFrame([{
        "name": r["name"],
        "chrom": r["chrom"],
        "is_promoter": not r.get("is_random", False),
        "start_codon_score": r.get("start_codon_score", 0),
        "next_codons_score": r.get("next_codons_score", 0),
        "combined_score": r.get("combined_score", 0),
        "codon_region": r.get("codon_region", "")
    } for r in results])
    
    # Rank by combined score
    all_df = all_df.sort_values(by="combined_score", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    all_df.to_csv(os.path.join(output_dir, "all_sequences_ranked.csv"), index=False)
    
    # Create rank distribution visualization
    plt.figure(figsize=(10, 6))
    
    # Get ranks for each group
    promoter_ranks = all_df[all_df["is_promoter"]].index + 1
    random_ranks = all_df[~all_df["is_promoter"]].index + 1
    
    plt.hist(promoter_ranks, alpha=0.7, bins=20, label='Promoter Regions')
    plt.hist(random_ranks, alpha=0.7, bins=20, label='Random Regions')
    
    plt.xlabel('Rank (1 is highest conservation)')
    plt.ylabel('Frequency')
    plt.title('Rank Distribution: Promoters vs Random Regions')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rank_distribution.png"))
    plt.close()
    
    # Perform statistical tests
    promoter_scores = all_df[all_df["is_promoter"]]["combined_score"].values
    random_scores = all_df[~all_df["is_promoter"]]["combined_score"].values
    
    if len(promoter_scores) > 0 and len(random_scores) > 0:
        _, p_value = mannwhitneyu(promoter_scores, random_scores, alternative='greater')
        
        print("\n=== RANKING ANALYSIS ===")
        print(f"Wilcoxon rank-sum test p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Promoter regions have significantly higher conservation scores (p < 0.05)")
        else:
            print("No significant difference in conservation scores (p >= 0.05)")
        
        # ROC and AUC calculation
        if HAS_SKLEARN:
            # Prepare data for ROC
            true_labels = all_df["is_promoter"].astype(int).values
            scores = all_df["combined_score"].values
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(true_labels, scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC: Conservation Score as Promoter Classifier')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "conservation_roc_curve.png"))
            plt.close()
            
            print(f"AUC (Area Under ROC Curve): {roc_auc:.4f}")
            
            # Interpret AUC
            if roc_auc > 0.9:
                print("Excellent classifier performance (AUC > 0.9)")
            elif roc_auc > 0.8:
                print("Good classifier performance (AUC > 0.8)")
            elif roc_auc > 0.7:
                print("Fair classifier performance (AUC > 0.7)")
            elif roc_auc > 0.6:
                print("Poor classifier performance (AUC > 0.6)")
            else:
                print("Failed classifier performance (AUC ≤ 0.6)")
    
    return all_df

# Updated extract_context_sequences function
def extract_context_sequences(df, genome, bw, generation_length=300, max_model_length=2048):
    """
    Extract context sequences optimized for generation.
    Maximizes context by:
    1. Calculating appropriate context size based on generation length and model capacity
    2. For + strand, including the full promoter and as much upstream sequence as fits
    3. Handles cases where the promoter itself is too large
    
    Args:
        df: DataFrame with promoter information
        genome: Genome object from pyfaidx
        bw: BigWig file for conservation scores
        generation_length: Length of sequence to generate (default: 300bp)
        max_model_length: Maximum length the model can handle (default: 2048)
        
    Returns:
        tuple: (contexts, conservation_contexts, context_info)
    """
    contexts = []
    conservation_contexts = []
    context_info = []
    
    # Calculate maximum context size
    max_context_size = max_model_length - generation_length
    
    for _, row in df.iterrows():
        chrom = row["chrom"]
        promoter_start = row["start"]
        promoter_end = row["end"]
        strand = row["strand"]
        
        # Skip reverse strand sequences
        if strand == "-":
            print(f"Skipping {row['name']} - reverse strand sequences not supported")
            continue
        
        # For + strand sequences
        # Calculate promoter length
        promoter_length = promoter_end - promoter_start
        
        # If promoter is too large, truncate it
        if promoter_length > max_context_size:
            print(f"Promoter {row['name']} is too large ({promoter_length}bp), truncating")
            context_start = promoter_start
            context_end = promoter_start + max_context_size
            generation_start_pos = max_context_size  # Generate right after the context
        else:
            # Calculate how much upstream sequence we can include
            upstream_bp = max_context_size - promoter_length
            
            # Set context boundaries
            context_start = max(0, promoter_start - upstream_bp)
            context_end = promoter_end
            generation_start_pos = context_end - context_start  # Generate right after the promoter
        
        # Extract context sequence
        try:
            seq = genome[chrom][context_start:context_end].seq.upper()
            target_length = context_end - context_start
            
            # Get conservation scores
            vals = np.zeros(target_length, dtype=np.float64)
            intervals = bw.intervals(chrom, context_start, context_end)

            if intervals is not None:
                for interval_start, interval_end, value in intervals:
                    relative_start = interval_start - context_start
                    relative_end = interval_end - context_start
                    if 0 <= relative_start < target_length and 0 <= relative_end <= target_length:
                        vals[relative_start:relative_end] = value
                
            contexts.append(str(seq))
            conservation_contexts.append(vals)
            
            # Store context info with promoter length info
            # Make sure to properly capture the is_random flag
            is_random = False
            if "is_random" in row and row["is_random"]:
                is_random = True
                
            context_info.append({
                "name": row["name"],
                "chrom": chrom,
                "context_start": context_start,
                "context_end": context_end,
                "strand": strand,
                "promoter_start": promoter_start,
                "promoter_end": promoter_end,
                "context_length": len(seq),
                "promoter_length": promoter_end - promoter_start,
                "generation_start_pos": generation_start_pos,
                "is_random": is_random
            })
                
        except Exception as e:
            print(f"Error extracting sequence for {row['name']}: {e}")
    
    return contexts, conservation_contexts, context_info

# Updated generate_extended_sequences_with_conservation function
def generate_extended_sequences_with_conservation(model, tokenized_contexts, conservation_contexts, 
                                                device, tokenizer, collator, context_info, 
                                                generation_length=300):
    """
    Generate extended sequences with fixed generation length and predicted conservation.
    
    Args:
        model: The model to use for generation
        tokenized_contexts: Tokenized context sequences
        conservation_contexts: Conservation scores for context sequences
        device: Device to run the model on
        tokenizer: Tokenizer for encoding/decoding sequences
        collator: Collator for batching inputs
        context_info: Information about the context sequences
        generation_length: Length of sequence to generate (default: 300bp)
        
    Returns:
        tuple: (generated_sequences, generated_conservations)
    """
    model.eval()
    generated_sequences = []
    generated_conservations = []

    dataset = SequenceDataset(tokenized_contexts, conservation_contexts)
    dataloader = DataLoader(dataset, batch_size=30, collate_fn=collator)
    
    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch  # inputs shape: [batch, 2, seq_len]
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            # Get original tokens from input
            original_tokens = inputs[i][0].cpu().tolist()
            conservation_scores = inputs[i][1].cpu().tolist()
            
            # Start with original tokens and scores
            generated = original_tokens.copy()
            gen_conservation_scores = conservation_scores.copy()
            
            # Generate exactly generation_length tokens
            for j in range(generation_length):
                # Convert to tensor
                seq_tensor = torch.tensor([generated], device=device)
                conservation_tensor = torch.tensor([gen_conservation_scores], device=device)

                # Stack seq and conservation tensor and add a batch dim of 1
                seq_tensor = seq_tensor.unsqueeze(0) # Add batch dimension
                conservation_tensor = conservation_tensor.unsqueeze(0)  # Add batch dimension
                input_tensor = torch.cat([seq_tensor, conservation_tensor], dim=1)  # Concatenate along the second dimension

                input_tensor = input_tensor.cpu()
                input_tensor = collator(input_tensor)
                
                with torch.no_grad():
                    # Forward pass
                    outputs = model(input_tensor[0].to(device), input_tensor[1].to(device))
                    
                    # Get outputs["seq_logits"]
                    logits = outputs["seq_logits"]
                    # logits shape: [batch_size, seq_len, vocab_size]
                    conservation_logits = outputs["scaling_logits"]
                    
                    # Get probabilities
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    
                    # Sample next token only from DNA_ALPHABET_PLUS
                    valid_indices = list(range(len(DNA_ALPHABET_PLUS)))
                    filtered_probs = probs[valid_indices]
                    filtered_probs /= filtered_probs.sum()  # Normalize probabilities
                    next_token = int(valid_indices[torch.multinomial(filtered_probs, 1).item()])
                    
                    # Get conservation prediction (mean value)
                    next_score = conservation_logits[0, -1, 0].item()
                    gen_conservation_scores.append(next_score)
                    
                    # Append to the generated sequence
                    generated.append(next_token)
            
            # Add to results
            generated_sequences.append(generated)
            generated_conservations.append(gen_conservation_scores)
            
            # Ensure everything in generated is an int
            generated = [int(t) for t in generated]
            
            # Get the generation start position from context_info
            context_len = len(original_tokens)
            generation_start = context_len
            
            # Decode and print a preview of the generated part (not the context)
            decoded = tokenizer.untokenize(generated)
            generated_part = decoded[generation_start:generation_start+100]
            #print(f"Generated sequence preview (sample {i}, first 100bp): {generated_part}...")
        
    return generated_sequences, generated_conservations

def analyze_start_codon_and_next_three(generated_seq, generated_cons, context_info, 
                                      tokenizer, threshold=0.7):
    """Analyze conservation of start codon and next three codons in generated sequences."""
    print(f"Analyzing {len(generated_seq)} sequences for conserved start codons")
    results = []
    
    for i in range(len(generated_seq)):
        # Decode generated sequence
        seq_tokens = [int(t) for t in generated_seq[i]]
        sequence = tokenizer.untokenize(seq_tokens)
        conservation = generated_cons[i]
        info = context_info[i]
        
        # Calculate promoter end position in generated sequence
        context_length = info["context_length"]
        
        result = {
            "name": info["name"],
            "chrom": info["chrom"],
            "strand": info["strand"],
            "context_start": info["context_start"],
            "context_end": info["context_end"],
            "sequence_length": len(sequence),
            "is_random": info.get("is_random", False)
        }
        
        # Find first conserved start codon in generated sequence after promoter
        gen_start_pos, gen_start_score = find_conserved_start_codon(
            sequence, conservation, start_pos=context_length, threshold=threshold
        )
        
        if gen_start_pos is not None:
            result["start_pos"] = gen_start_pos
            result["start_codon_score"] = gen_start_score
            
            # Get conservation of next three codons in generated sequence
            next_codons_pos = gen_start_pos + 3
            next_codons_conserved, next_codons_score = is_conserved_codon(
                next_codons_pos, conservation, threshold, num_positions=9  # 3 codons = 9 bp
            )
            
            result["next_codons_conserved"] = next_codons_conserved
            result["next_codons_score"] = next_codons_score
            
            # Calculate combined score (start codon + next three codons)
            combined_score = (gen_start_score + next_codons_score) / 2
            result["combined_score"] = combined_score
            
            # Extract the sequence
            codon_region = sequence[gen_start_pos:gen_start_pos+12]  # Start + 3 codons
            result["codon_region"] = codon_region
        else:
            # Add zero scores for regions without a conserved start codon
            result["start_codon_score"] = 0.0
            result["next_codons_score"] = 0.0
            result["combined_score"] = 0.0
            result["codon_region"] = ""
            result["next_codons_conserved"] = False
        
        results.append(result)
    
    return results

def plot_conservation_comparison(results, output_dir):
    """Plot comparison of start codon conservation between promoter-derived and random sequences."""
    # Separate results for promoters and random regions (no filtering)
    promoter_results = [r for r in results if not r.get("is_random", False)]
    random_results = [r for r in results if r.get("is_random", False)]
    
    if not promoter_results or not random_results:
        print("Not enough data for plotting conservation comparison.")
        return
    
    # Get scores
    promoter_start_scores = [r.get("start_codon_score", 0) for r in promoter_results]
    random_start_scores = [r.get("start_codon_score", 0) for r in random_results]
    
    promoter_next_scores = [r.get("next_codons_score", 0) for r in promoter_results]
    random_next_scores = [r.get("next_codons_score", 0) for r in random_results]
    
    promoter_combined_scores = [r.get("combined_score", 0) for r in promoter_results]
    random_combined_scores = [r.get("combined_score", 0) for r in random_results]
    
    # Create boxplot comparison
    plt.figure(figsize=(12, 8))
    
    # Create positions for the boxplots
    positions = [1, 2, 4, 5, 7, 8]
    data = [promoter_start_scores, random_start_scores, 
            promoter_next_scores, random_next_scores, 
            promoter_combined_scores, random_combined_scores]
    
    # Create boxplots
    boxplot = plt.boxplot(data, positions=positions, patch_artist=True)
    
    # Set colors
    colors = ['lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add labels
    plt.xticks([1.5, 4.5, 7.5], ['Start Codon', 'Next Three Codons', 'Combined'])
    plt.ylabel('Conservation Score')
    plt.title('Conservation Score Comparison: Promoters vs Random Regions')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Promoter Regions'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Random Regions')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conservation_comparison_boxplot.png"))
    plt.close()
    
    # Create histogram comparison for combined scores
    plt.figure(figsize=(12, 6))
    
    plt.hist(promoter_combined_scores, alpha=0.7, bins=20, label='Promoter Regions')
    plt.hist(random_combined_scores, alpha=0.7, bins=20, label='Random Regions')
    
    plt.xlabel('Combined Conservation Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Combined Conservation Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conservation_score_histogram.png"))
    plt.close()

def create_ranked_lists(results, output_dir):
    """Create ranked lists of conservation scores for promoters and random regions."""
    # Separate results for promoters and random regions (no filtering by combined_score)
    promoter_results = [r for r in results if not r.get("is_random", False)]
    random_results = [r for r in results if r.get("is_random", False)]
    
    # Sort by combined score (descending)
    promoter_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    random_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    
    # Create DataFrames for ranked lists
    promoter_df = pd.DataFrame([{
        "name": r["name"],
        "chrom": r["chrom"],
        "start_codon_score": r.get("start_codon_score", 0),
        "next_codons_score": r.get("next_codons_score", 0),
        "combined_score": r.get("combined_score", 0),
        "codon_region": r.get("codon_region", "")
    } for r in promoter_results])
    
    random_df = pd.DataFrame([{
        "name": r["name"],
        "chrom": r["chrom"],
        "start_codon_score": r.get("start_codon_score", 0),
        "next_codons_score": r.get("next_codons_score", 0),
        "combined_score": r.get("combined_score", 0),
        "codon_region": r.get("codon_region", "")
    } for r in random_results])
    
    # Save to CSV
    promoter_df.to_csv(os.path.join(output_dir, "promoter_conservation_ranked.csv"), index=False)
    random_df.to_csv(os.path.join(output_dir, "random_conservation_ranked.csv"), index=False)
    
    # Calculate differentiation metrics
    if promoter_results and random_results:
        # Calculate ROC curve data
        all_scores = [(r.get("combined_score", 0), 1) for r in promoter_results] + [(r.get("combined_score", 0), 0) for r in random_results]
        all_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate true positive and false positive rates
        total_pos = len(promoter_results)
        total_neg = len(random_results)
        tp, fp = 0, 0
        tpr_list, fpr_list = [], []
        
        for score, is_promoter in all_scores:
            if is_promoter:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / total_pos
            fpr = fp / total_neg
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr_list, tpr_list, label=f'Promoter vs Random')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Promoters vs Random Regions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
    
    return promoter_df, random_df


def print_summary_statistics(results):
    """Print summary statistics for conservation analysis."""
    # Separate results for promoters and random regions
    promoter_results = [r for r in results if not r.get("is_random", False)]
    random_results = [r for r in results if r.get("is_random", False)]
    
    # Count how many have conserved start codons (score > 0)
    promoter_conserved = [r for r in promoter_results if r.get("start_codon_score", 0) > 0]
    random_conserved = [r for r in random_results if r.get("start_codon_score", 0) > 0]
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total promoter regions analyzed: {len(promoter_results)}")
    print(f"Total random regions analyzed: {len(random_results)}")
    
    if promoter_results:
        print(f"\nPromoter regions with conserved start codons: {len(promoter_conserved)} ({len(promoter_conserved)/len(promoter_results)*100:.1f}%)")
    else:
        print("\nPromoter regions with conserved start codons: 0 (0.0%)")
        
    if random_results:
        print(f"Random regions with conserved start codons: {len(random_conserved)} ({len(random_conserved)/len(random_results)*100:.1f}%)")
    else:
        print("Random regions with conserved start codons: 0 (0.0%)")
    
    # Calculate average scores for all sequences (including those with 0 scores)
    if promoter_results:
        avg_promoter_start = np.mean([r.get("start_codon_score", 0) for r in promoter_results])
        avg_promoter_next = np.mean([r.get("next_codons_score", 0) for r in promoter_results])
        avg_promoter_combined = np.mean([r.get("combined_score", 0) for r in promoter_results])
        
        print(f"\nPromoter regions average scores (all sequences):")
        print(f"  - Start codon conservation: {avg_promoter_start:.4f}")
        print(f"  - Next three codons conservation: {avg_promoter_next:.4f}")
        print(f"  - Combined score: {avg_promoter_combined:.4f}")
    
    if random_results:
        avg_random_start = np.mean([r.get("start_codon_score", 0) for r in random_results])
        avg_random_next = np.mean([r.get("next_codons_score", 0) for r in random_results])
        avg_random_combined = np.mean([r.get("combined_score", 0) for r in random_results])
        
        print(f"\nRandom regions average scores (all sequences):")
        print(f"  - Start codon conservation: {avg_random_start:.4f}")
        print(f"  - Next three codons conservation: {avg_random_next:.4f}")
        print(f"  - Combined score: {avg_random_combined:.4f}")
    
    # Also calculate average scores only for sequences with conserved start codons
    if promoter_conserved:
        avg_promoter_conserved_start = np.mean([r.get("start_codon_score", 0) for r in promoter_conserved])
        avg_promoter_conserved_next = np.mean([r.get("next_codons_score", 0) for r in promoter_conserved])
        avg_promoter_conserved_combined = np.mean([r.get("combined_score", 0) for r in promoter_conserved])
        
        print(f"\nPromoter regions with conserved start codons average scores:")
        print(f"  - Start codon conservation: {avg_promoter_conserved_start:.4f}")
        print(f"  - Next three codons conservation: {avg_promoter_conserved_next:.4f}")
        print(f"  - Combined score: {avg_promoter_conserved_combined:.4f}")
    
    if random_conserved:
        avg_random_conserved_start = np.mean([r.get("start_codon_score", 0) for r in random_conserved])
        avg_random_conserved_next = np.mean([r.get("next_codons_score", 0) for r in random_conserved])
        avg_random_conserved_combined = np.mean([r.get("combined_score", 0) for r in random_conserved])
        
        print(f"\nRandom regions with conserved start codons average scores:")
        print(f"  - Start codon conservation: {avg_random_conserved_start:.4f}")
        print(f"  - Next three codons conservation: {avg_random_conserved_next:.4f}")
        print(f"  - Combined score: {avg_random_conserved_combined:.4f}")
    
    # Calculate differentiation statistics with all sequences included
    if promoter_results and random_results:
        promoter_scores = [r.get("combined_score", 0) for r in promoter_results]
        random_scores = [r.get("combined_score", 0) for r in random_results]
        
        score_diff = np.mean(promoter_scores) - np.mean(random_scores)
        print(f"\nDifference in average combined score (all sequences): {score_diff:.4f}")
        
        # Calculate Mann-Whitney U test for statistical significance
        try:
            u_stat, p_value = mannwhitneyu(promoter_scores, random_scores, alternative='greater')
            print(f"Mann-Whitney U test (all sequences): U={u_stat:.4f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print("The difference is statistically significant (p < 0.05)")
            else:
                print("The difference is not statistically significant (p >= 0.05)")
                
            # Calculate percentage of promoters with higher score than random region median
            random_median = np.median(random_scores)
            higher_than_median = sum(1 for score in promoter_scores if score > random_median)
            percent_higher = (higher_than_median / len(promoter_scores)) * 100
            print(f"Percentage of promoters with score higher than random median: {percent_higher:.1f}%")
            
        except ImportError:
            print("SciPy not available for statistical testing")
        except Exception as e:
            print(f"Error in statistical testing: {e}")


def tokenize_sequences(sequences, tokenizer):
    """Tokenize sequences for the model."""
    tokenized_sequences = []
    for seq in sequences:
        tokens = tokenizer.tokenizeMSA(seq)
        tokenized_sequences.append(tokens)
    return tokenized_sequences


def is_conserved_codon(pos, conservation_scores, threshold=0.7, num_positions=3):
    """Check if a codon or multi-codon region is conserved."""
    if pos + num_positions > len(conservation_scores):
        return False, 0.0
    
    # Get conservation scores for the region
    region_scores = conservation_scores[pos:pos+num_positions]
    avg_score = np.mean(region_scores)
    
    return avg_score >= threshold, avg_score

def find_conserved_start_codon(sequence, conservation_scores, start_pos=0, threshold=0.7):
    """Find the first conserved start codon (ATG) in a sequence."""
    for pos in range(start_pos, len(sequence)-2):
        if sequence[pos:pos+3] == "ATG":
            is_conserved, score = is_conserved_codon(pos, conservation_scores, threshold)
            if is_conserved:
                return pos, score
    return None, 0.0


def scramble_sequences(promoter_df, genome):
    """
    Generate scrambled versions of promoter sequences.
    
    Args:
        promoter_df: DataFrame with promoter information
        genome: Genome object from pyfaidx
        
    Returns:
        DataFrame: Scrambled promoter regions DataFrame
    """
    scrambled_regions = []
    
    for _, row in promoter_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        
        try:
            # Extract original sequence
            seq = str(genome[chrom][start:end].seq.upper())
            
            # Scramble the sequence (shuffle characters)
            seq_list = list(seq)
            random.shuffle(seq_list)
            scrambled_seq = ''.join(seq_list)
            
            # Create a new row for the scrambled region
            scrambled_row = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "name": f"scrambled_{row['name']}",
                "score": 0,
                "strand": row["strand"],
                "core_start": row.get("core_start", start),
                "core_end": row.get("core_end", end),
                "is_random": True,    # This is our negative class
                "original_seq": seq,
                "scrambled_seq": scrambled_seq
            }
            
            scrambled_regions.append(scrambled_row)
        except Exception as e:
            print(f"Error processing sequence for {row['name']}: {e}")
    
    scrambled_df = pd.DataFrame(scrambled_regions)
    print(f"Generated {len(scrambled_df)} scrambled sequences from promoters")
    return scrambled_df


def main():
    parser = argparse.ArgumentParser(description="Analyze start codon and next three codons conservation in promoter vs random/scrambled regions")
    parser.add_argument('--promoter_bed', type=str, default='/home/mica/gamba/data_processing/data/promoters/promoters.bed', help='Path to the promoter BED file')
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--config_fpath', default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the model config file')
    parser.add_argument('--bigwig_path', default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigwig file')
    parser.add_argument('--checkpoint_path', default='/home/mica/gamba/dcps/dcp_132250_only_MSE',  help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/promoters/', help='Path to save results')
    parser.add_argument('--upstream_bp', type=int, default=500, help='Number of bp to include upstream of promoter')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum length of generated sequences')
    parser.add_argument('--num_sequences', type=int, default=500, help='Number of promoter sequences to process')
    parser.add_argument('--conservation_threshold', type=float, default=0.7, help='Threshold for determining conserved codons')
    parser.add_argument('--generation_length', type=int, default=300, help='Length of sequence to generate')
    parser.add_argument('--negative_class', type=str, default='scrambled', choices=['random', 'scrambled'], help='Type of negative class to use (random genomic regions or scrambled promoter sequences)')
    args = parser.parse_args()

    bw = pyBigWig.open(args.bigwig_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data 
    print("Loading genome and BED file...")
    genome = Fasta(args.genome_fasta)
    promoter_df = parse_promoter_bed(args.promoter_bed)

    # Subset to specific chromosomes
    promoter_df = promoter_df[promoter_df["chrom"].isin(["chr16", "chr22", "chr2", "chr3"])]
    print(f"Loaded {len(promoter_df)} promoters from the BED file.")

    # Drop any reverse strand sequences
    promoter_df = promoter_df[promoter_df["strand"] == "+"]
    print(f"Filtered to {len(promoter_df)} forward strand promoters.")
    
    # Limit the number of sequences to process if specified
    if args.num_sequences > 0 and args.num_sequences < len(promoter_df):
        promoter_df = promoter_df.sample(args.num_sequences, random_state=42).reset_index(drop=True)
    
    # Generate negative class based on user selection
    if args.negative_class == 'random':
        print("Generating random genomic regions with matched GC...")
        negative_df = generate_random_regions_matched_GC(promoter_df, genome)
        negative_label = "random"
    else:  # args.negative_class == 'scrambled'
        print("Generating scrambled promoter sequences...")
        negative_df = scramble_sequences(promoter_df, genome)
        negative_label = "scrambled"

    # Make sure promoter_df has is_random set to False
    promoter_df["is_random"] = False

    # Extract sequences for GC content analysis
    print("Extracting sequences for GC content analysis...")
    promoter_sequences = []
    negative_sequences = []
    
    for _, row in promoter_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        try:
            seq = str(genome[chrom][start:end].seq.upper())
            promoter_sequences.append(seq)
        except Exception as e:
            print(f"Error extracting promoter sequence for {row['name']}: {e}")
    
    # For scrambled sequences, we already have the sequences
    if args.negative_class == 'scrambled':
        negative_sequences = negative_df["scrambled_seq"].tolist()
    else:
        # For random regions, extract from genome
        for _, row in negative_df.iterrows():
            chrom = row["chrom"]
            start = row["start"]
            end = row["end"]
            try:
                seq = str(genome[chrom][start:end].seq.upper())
                negative_sequences.append(seq)
            except Exception as e:
                print(f"Error extracting {negative_label} sequence for {row['name']}: {e}")
    
    # Perform GC content analysis
    print("Analyzing GC content...")
    compare_gc_content(promoter_sequences, negative_sequences, args.output_dir)
    
    # Combine dataframes for processing
    all_df = pd.concat([promoter_df, negative_df], ignore_index=True)
    
    # Verify the negative class
    print(f"Combined dataframe has {sum(all_df['is_random'])} random regions and {len(all_df) - sum(all_df['is_random'])} promoter regions")
    
    # Extract context sequences with updated function
    print("Extracting context sequences...")
    contexts, conservation_contexts, context_info = extract_context_sequences(
        all_df, genome, bw, args.generation_length, args.max_length
    )

    print(f"After extraction: {len(contexts)} contexts obtained")

    # Load model configuration
    with open(args.config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    epochs = config["epochs"]
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())

    ckpt_path = args.checkpoint_path
    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # Create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )

    # Get d_model, n_head, n_layers, dim_feedforward and padding_id from the config
    d_model = config.get("d_model", 512)
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)

    # Set up the model load from last checkpoint
    model = JambagambaModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )
    
    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
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

    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create the collator
    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )
    
    # Tokenize contexts
    print("Tokenizing context sequences...")
    tokenized_contexts = tokenize_sequences(contexts, tokenizer)

    # After tokenization
    print(f"After tokenization: {len(tokenized_contexts)} sequences tokenized")

    
    # Generate extended sequences and conservation scores
    print(f"Generating extended sequences up to {args.max_length} bp with predicted conservation...")
    generated_sequences, generated_conservations = generate_extended_sequences_with_conservation(
        model, tokenized_contexts, conservation_contexts, device, tokenizer, collator, args.max_length
    )

    # After generation
    print(f"After generation: {len(generated_sequences)} sequences generated")

    
    # Analyze start codon and next three codons conservation
    print("Analyzing start codon and next three codons conservation...")
    results = analyze_start_codon_and_next_three(
        generated_sequences, generated_conservations, context_info, 
        tokenizer, args.conservation_threshold
    )
    
    # Create ranked lists
    print("Creating ranked lists of conservation scores...")
    promoter_df, random_df = create_ranked_lists(results, args.output_dir)

    # Create unified ranked list and perform statistical analysis
    print("Ranking all sequences by conservation score...")
    all_ranked_df = rank_all_sequences_by_conservation(results, args.output_dir)
    
    # Plot conservation comparison
    print("Plotting conservation score comparisons...")
    plot_conservation_comparison(results, args.output_dir)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

    # Print top 10 sequences from the overall ranking
    if all_ranked_df is not None and len(all_ranked_df) > 0:
        print("\n=== TOP 10 SEQUENCES OVERALL ===")
        for i, row in all_ranked_df.head(10).iterrows():
            region_type = "Promoter" if row["is_promoter"] else "Random"
            print(f"{i+1}. {row['name']} ({region_type}, Combined score: {row['combined_score']:.4f})")
            print(f"   Start codon + 3 codons: {row['codon_region']}")

if __name__ == "__main__":
    main()


#median 5' UTR length is 150bp https://bmcresnotes.biomedcentral.com/articles/10.1186/1756-0500-4-312

#coding vs noncoding high phyloP segments and see if they can be separated in the representation space, look for high phyloP scores in the region and cluster them in the representatio space