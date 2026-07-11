import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfaidx import Fasta
import torch
import sys
import os
sys.path.append("../gamba")
from typing import Optional, Sequence, Tuple, Type
import umap

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNOALMModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json

class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]
    
def safe_mean(arr, axis=None):
    """Calculate mean while handling empty arrays and NaN values."""
    if len(arr) == 0 or (isinstance(arr, np.ndarray) and arr.size == 0):
        return np.nan
    return np.nanmean(arr, axis=axis)

def get_representations(model, dataloader, device, original_spans):
    model.eval()
    representations = []
    pred_conservations = []
    true_conservations = []
    variances = []
    sequence_conservation_profiles = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, labels = batch  # inputs shape: [batch, 2, seq_len]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Pass both inputs and labels to the model
            output = model(inputs, labels)
            
            batch_representations = output["representation"].cpu().numpy()
            scaling_logits = output["scaling_logits"].cpu().numpy()
            sequence_data = inputs[:, 0].cpu().numpy()  # Sequences
            true_scores = inputs[:, 1].cpu().numpy()    # Conservation scores
            
            
            # batch_pred_conservation = scaling_logits[..., 0]
            # batch_log_var = scaling_logits[..., 1]
            batch_pred_conservation = scaling_logits
            batch_log_var = np.ones_like(scaling_logits)
            batch_variances = np.exp(batch_log_var)
            
            for idx in range(len(batch_representations)):
                span_idx = batch_idx * dataloader.batch_size + idx
                if span_idx < len(original_spans):
                    start, end = original_spans[span_idx]
                    
                    if start < end:
                        pred_cons_slice = batch_pred_conservation[idx, start:end]
                        true_cons_slice = true_scores[idx, start:end]
                        var_slice = batch_variances[idx, start:end]
                        repr_slice = batch_representations[idx, start:end]
                        
                        pred_cons_mean = np.mean(pred_cons_slice)
                        true_cons_mean = np.mean(true_cons_slice)
                        var_mean = np.mean(var_slice)
                        repr_mean = np.mean(repr_slice, axis=0)
                        
                        representations.append(repr_mean)
                        pred_conservations.append(pred_cons_mean)
                        true_conservations.append(true_cons_mean)
                        variances.append(var_mean)
                        
                        full_profile = {
                            'predicted': batch_pred_conservation[idx],
                            'true': true_scores[idx],
                            'start': start,
                            'end': end
                        }
                        
                        print(f"\nSample {idx}:")
                        print(f"Region length: {end-start}")
                        print(f"Predicted Conservation (region only): {pred_cons_mean:.3f}")
                        print(f"True Conservation (region only): {true_cons_mean:.3f}")
                        print(f"Variance: {var_mean:.3f}")
            
            del inputs, labels, output
            torch.cuda.empty_cache()
    
    return (np.array(representations), 
            np.array(pred_conservations), 
            np.array(true_conservations), 
            np.array(variances), 
            sequence_conservation_profiles)

def plot_conservation_profiles(profiles, output_path, dataset_name):
    """Create visualization of conservation profiles."""
    plt.figure(figsize=(20, 10))  # Reduced height since we removed middle plot
    
    # Plot 1: Average profiles with confidence intervals
    plt.subplot(2, 1, 1)
    seq_length = len(profiles[0]['predicted'])
    x = np.arange(seq_length)
    
    # Process data
    all_pred = np.array([p['predicted'] for p in profiles])
    all_true = np.array([p['true'][1:-1] if len(p['true']) > seq_length else p['true'] 
                        for p in profiles])
    
    # Calculate means and confidence intervals
    pred_mean = np.nanmean(all_pred, axis=0)
    pred_ci = np.nanstd(all_pred, axis=0) * 1.96 / np.sqrt(len(profiles))
    true_mean = np.nanmean(all_true, axis=0)
    true_ci = np.nanstd(all_true, axis=0) * 1.96 / np.sqrt(len(profiles))
    
    # Plot means with confidence intervals
    plt.plot(x, pred_mean, label='Predicted', color='blue', linewidth=2)
    plt.fill_between(x, pred_mean-pred_ci, pred_mean+pred_ci, color='blue', alpha=0.2)
    plt.plot(x, true_mean, label='True', color='red', linewidth=2)
    plt.fill_between(x, true_mean-true_ci, true_mean+true_ci, color='red', alpha=0.2)
    
    # Add ROI indication
    start_mean = int(np.mean([p['start'] for p in profiles]))
    end_mean = int(np.mean([p['end'] for p in profiles]))
    plt.axvspan(start_mean, end_mean, color='gray', alpha=0.1, label='Typical ROI')
    
    plt.title(f'Average Conservation Profile - {dataset_name}')
    plt.ylabel('Conservation Score')
    plt.legend()
    
    # Plot 2: Distribution of scores
    plt.subplot(2, 1, 2)
    
    # Collect scores
    roi_pred = []
    roi_true = []
    non_roi_pred = []
    non_roi_true = []
    
    for p in profiles:
        start, end = p['start'], p['end']
        mask = np.zeros(seq_length, dtype=bool)
        mask[start:end] = True
        
        true_vals = p['true'][1:-1] if len(p['true']) > seq_length else p['true']
        
        roi_pred.extend(p['predicted'][mask])
        roi_true.extend(true_vals[mask])
        non_roi_pred.extend(p['predicted'][~mask])
        non_roi_true.extend(true_vals[~mask])
    
    # Plot distributions
    plt.hist(roi_pred, bins=50, alpha=0.5, color='blue', 
             label='Predicted (ROI)', density=True)
    plt.hist(roi_true, bins=50, alpha=0.5, color='red', 
             label='True (ROI)', density=True)
    plt.hist(non_roi_pred, bins=50, alpha=0.3, color='lightblue', 
             label='Predicted (non-ROI)', density=True, linestyle='--')
    plt.hist(non_roi_true, bins=50, alpha=0.3, color='pink', 
             label='True (non-ROI)', density=True, linestyle='--')
    
    plt.title('Distribution of Conservation Scores')
    plt.xlabel('Conservation Score')
    plt.ylabel('Density')
    plt.legend()
    
    # Add summary statistics as text
    stats_text = (
        f'ROI Stats:\n'
        f'  Pred: {np.mean(roi_pred):.3f}±{np.std(roi_pred):.3f}\n'
        f'  True: {np.mean(roi_true):.3f}±{np.std(roi_true):.3f}\n'
        f'Non-ROI Stats:\n'
        f'  Pred: {np.mean(non_roi_pred):.3f}±{np.std(non_roi_pred):.3f}\n'
        f'  True: {np.mean(non_roi_true):.3f}±{np.std(non_roi_true):.3f}'
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_individual_examples(profiles, output_path, dataset_name, seed=42):
    """Create visualization of conservation profiles for 3 random examples."""
    np.random.seed(seed)
    
    # Randomly select 3 examples
    example_indices = np.random.choice(len(profiles), size=3, replace=False)
    selected_profiles = [profiles[i] for i in example_indices]
    
    plt.figure(figsize=(20, 15))
    
    # Plot each example in its own subplot
    for idx, p in enumerate(selected_profiles):
        plt.subplot(3, 1, idx + 1)
        seq_length = len(p['predicted'])
        x = np.arange(seq_length)
        
        # Get true values with proper handling of length
        true_vals = p['true'][1:-1] if len(p['true']) > seq_length else p['true']
        
        # Plot predicted and true values for this example
        plt.plot(x, p['predicted'], color='blue', linestyle='-', 
                label='Predicted', alpha=0.7)
        plt.plot(x, true_vals, color='red', linestyle='-', 
                label='True', alpha=0.7)
        
        # Add ROI indication for this example
        plt.axvspan(p['start'], p['end'], color='gray', alpha=0.1, 
                   label='Region of Interest')
        
        plt.title(f'{dataset_name} Example {idx+1} (ROI: {p["start"]}-{p["end"]})')
        plt.ylabel('Conservation Score')
        plt.xlabel('Position')
        plt.legend()
        
        # Add stats text for this example
        roi_mask = np.zeros(seq_length, dtype=bool)
        roi_mask[p['start']:p['end']] = True
        
        stats_text = (
            f'ROI Stats:\n'
            f'  Pred: {np.mean(p["predicted"][roi_mask]):.3f}±{np.std(p["predicted"][roi_mask]):.3f}\n'
            f'  True: {np.mean(true_vals[roi_mask]):.3f}±{np.std(true_vals[roi_mask]):.3f}\n'
            f'Non-ROI Stats:\n'
            f'  Pred: {np.mean(p["predicted"][~roi_mask]):.3f}±{np.std(p["predicted"][~roi_mask]):.3f}\n'
            f'  True: {np.mean(true_vals[~roi_mask]):.3f}±{np.std(true_vals[~roi_mask]):.3f}'
        )
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def print_detailed_stats(profiles, dataset_name):
    """Print detailed statistics about conservation scores."""
    roi_pred = []
    roi_true = []
    non_roi_pred = []
    non_roi_true = []
    
    seq_length = len(profiles[0]['predicted'])
    
    for p in profiles:
        start, end = p['start'], p['end']
        mask = np.zeros(seq_length, dtype=bool)
        mask[start:end] = True
        
        true_vals = p['true'][1:-1] if len(p['true']) > seq_length else p['true']
        
        roi_pred.extend(p['predicted'][mask])
        roi_true.extend(true_vals[mask])
        non_roi_pred.extend(p['predicted'][~mask])
        non_roi_true.extend(true_vals[~mask])
    
    print(f"\nDetailed Statistics for {dataset_name}:")
    print("\nROI Regions:")
    print(f"  Predicted: {np.mean(roi_pred):.3f} ± {np.std(roi_pred):.3f}")
    print(f"  True:      {np.mean(roi_true):.3f} ± {np.std(roi_true):.3f}")
    print("\nNon-ROI Regions:")
    print(f"  Predicted: {np.mean(non_roi_pred):.3f} ± {np.std(non_roi_pred):.3f}")
    print(f"  True:      {np.mean(non_roi_true):.3f} ± {np.std(non_roi_true):.3f}")


def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
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

def load_bed_file(bed_file):
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'label', 'info'])
    return bed_df

def process_bed_file(bed_df, genome, chrom_sizes, bw, tokenizer):
    sequences = []
    scores_list = []
    original_spans = []
    target_length = 2048
    
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    chrom_sizes_df = pd.read_csv(chrom_sizes, sep='\t', header=None, names=['chrom', 'size'])
    
    skipped = 0
    for index, row in bed_df.iterrows():
        chromosome = row['chrom']
        if chromosome not in valid_chromosomes:
            skipped += 1
            continue
            
        chrom_size = chrom_sizes_df[chrom_sizes_df['chrom'] == chromosome]['size'].iloc[0]
        start = row['start']
        end = row['end']
        
        if start > end:
            start, end = end, start
            
        original_length = end - start
        pad_length = target_length - original_length
        
        # Calculate padded_start while ensuring we don't go beyond chromosome boundaries
        padded_start = max(0, start - pad_length)
        actual_start_offset = start - padded_start
        
        # Check if we'll exceed chromosome bounds
        if padded_start + target_length > chrom_size:
            skipped += 1
            continue
            
        # Track where original sequence starts in padded input
        original_spans.append((actual_start_offset, actual_start_offset + original_length))
        
        try:
            # Get sequence with padding
            sequence = genome[chromosome][padded_start:padded_start + target_length].seq
            sequence_tokens = tokenizer.tokenizeMSA(sequence)
            
            # Get conservation scores
            vals = np.zeros(target_length, dtype=np.float64)
            intervals = bw.intervals(chromosome, padded_start, padded_start + target_length)
            
            if intervals is not None:
                for interval_start, interval_end, value in intervals:
                    relative_start = interval_start - padded_start
                    relative_end = interval_end - padded_start
                    if 0 <= relative_start < target_length and 0 <= relative_end <= target_length:
                        vals[relative_start:relative_end] = value
                        
            scores = np.round(vals, 2)
            sequences.append(sequence_tokens)
            scores_list.append(scores)
            
        except Exception as e:
            print(f"Error processing {chromosome}:{start}-{end}: {str(e)}")
            skipped += 1
            continue
    
    print(f"Processed {len(sequences)} sequences, skipped {skipped} sequences")
    return sequences, scores_list, original_spans


def generate_random_regions(n_regions, genome_sizes_file, region_size=500):
    """Generate random genomic regions that are properly spaced."""
    chrom_sizes = pd.read_csv(genome_sizes_file, sep='\t', header=None, names=['chrom', 'size'])
    valid_chroms = [f'chr{i}' for i in range(1,23)] + ['chrX']
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(valid_chroms)]
    
    random_regions = []
    target_length = 2048  # Match the target length used in process_bed_file
    
    while len(random_regions) < n_regions:
        chrom = np.random.choice(valid_chroms)
        chrom_size = chrom_sizes[chrom_sizes['chrom'] == chrom]['size'].iloc[0]
        
        # Ensure we have enough space for padding
        max_pos = chrom_size - target_length
        if max_pos <= 0:
            continue
            
        start = np.random.randint(0, max_pos)
        end = start + region_size
        
        # Check if we have enough space for padding
        if start >= 0 and end <= chrom_size:
            random_regions.append([chrom, start, end])
    
    df = pd.DataFrame(random_regions, columns=['chrom', 'start', 'end'])
    df = df.sort_values(['chrom', 'start']).reset_index(drop=True)
    return df

def process_dataset(bed_path_or_df, genome, chrom_sizes, bw, tokenizer, 
                   model, device, collator, output_path, dataset_name, is_random=False):
    """Process a single dataset and return representations and scores."""
    if os.path.exists(output_path):
        print(f"Loading existing representations from {output_path}")
        data = np.load(output_path, allow_pickle=True)
        
        # Try to reconstruct profiles if we have all needed data
        if all(k in data for k in ['sequences', 'scores', 'spans']):
            print(f"Reconstructing profiles for {dataset_name} from saved data...")
            profiles = reconstruct_profiles_from_saved(
                output_path, data['sequences'], data['scores'], data['spans']
            )
        else:
            profiles = []
            
        return (data['repr'], data['pred_cons'], data['true_cons'], 
                data['var'], profiles)
    
    print(f"Processing sequences from {dataset_name}...")
    
    if isinstance(bed_path_or_df, pd.DataFrame):
        bed_df = bed_path_or_df
    else:
        bed_df = pd.read_csv(bed_path_or_df, sep='\t', header=None, 
                            names=['chrom', 'start', 'end', 'label', 'info'])
    
    sequences, cons_scores, spans = process_bed_file(bed_df, genome, chrom_sizes, bw, tokenizer)
    
    if not sequences:
        raise ValueError(f"No valid sequences processed for {dataset_name}")
        
    dataset = SequenceDataset(sequences, cons_scores)
    loader = DataLoader(dataset, batch_size=20, collate_fn=collator)
    representations, pred_cons, true_cons, variances, profiles = get_representations(
        model, loader, device, spans
    )
    
    if (np.all(np.isnan(pred_cons)) or np.all(np.isnan(true_cons)) 
            or np.all(np.isnan(variances))):
        print(f"Warning: All values are NaN for {dataset_name}")
    
    # Save everything needed to reconstruct profiles
    np.savez_compressed(output_path,
                       repr=representations,
                       pred_cons=pred_cons,
                       true_cons=true_cons,
                       var=variances,
                       sequences=np.array([p['predicted'] for p in profiles]),
                       scores=np.array([p['true'] for p in profiles]),
                       spans=np.array([(p['start'], p['end']) for p in profiles]))
    
    print(f"Saved {len(representations)} representations to {output_path}")
    return representations, pred_cons, true_cons, variances, profiles

def generate_flanking_regions(bed_df, output_dir):
    """Generate flanking regions 500bp upstream of each UCNE region and save to bed file."""
    flanking_regions = []
    
    for _, row in bed_df.iterrows():
        # Original region info
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        length = end - start
        
        # Generate flanking region 500bp upstream
        #flanking_start = start - 500  # Shift up by 500bp 
        #lets try shift by 2kb
        flanking_start = start - 2000
        flanking_end = flanking_start + length  # Keep same length as original
        
        # Add label and info columns to match bed format
        flanking_regions.append([chrom, flanking_start, flanking_end, "flanking", "flanking_region"])
    
    # Create flanking bed file
    flanking_df = pd.DataFrame(flanking_regions, columns=['chrom', 'start', 'end', 'label', 'info'])
    flanking_bed_path = os.path.join(output_dir, "flanking_regions.bed")
    flanking_df.to_csv(flanking_bed_path, sep='\t', header=False, index=False)
    
    return flanking_bed_path



def ensure_2d(array):
    """Ensure array is 2D."""
    if len(array.shape) == 1:
        return array.reshape(1, -1)
    return array

def visualize_embeddings(repr1, repr2, labels1, labels2, output_path):
    """Create and save UMAP visualization of embeddings."""
    print(f"Shapes - Group 1: {repr1.shape}, Group 2: {repr2.shape}")

     # Get output directory from output_path
    output_dir = os.path.dirname(output_path)
    
    # Analyze neighborhood overlap
    overlap_stats = analyze_neighbourhood_overlap(repr1, repr2, labels1, labels2, 
                                              k=1, output_dir=output_dir)
    print(f"\nNeighbourhood Overlap Analysis:")
    print(f"{labels1} points with {labels1} neighbours: {overlap_stats['class1']:.1f}%")
    print(f"{labels2} points with {labels2} neighbours: {overlap_stats['class2']:.1f}%")
    print(f"Overall neighbourhood preservation: {overlap_stats['overall']:.1f}%")
    print(f"Overall accuracy: {overlap_stats['accuracy']:.1f}%")
    
    # Ensure 2D arrays
    repr1 = ensure_2d(repr1)
    repr2 = ensure_2d(repr2)
    
    # Combine data
    combined_repr = np.concatenate([repr1, repr2])
    labels = [labels1] * len(repr1) + [labels2] * len(repr2)
    
    # Generate UMAP embeddings
    embeddings = umap.UMAP().fit_transform(combined_repr)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = {'UCNE': 'blue', f"{labels2}": 'orange'}
    
    for label in set(labels):
        mask = np.array(labels) == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   color=colors[label], label=label, s=5, alpha=0.2)
    
    plt.legend()
    plt.title('UMAP of Sequence Embeddings')
    plt.savefig(output_path)
    #replace .png in output path with .svg
    svg_output_path = output_path.replace('.png', '.svg')
    plt.savefig(svg_output_path, format='svg') 
    plt.close()


def reconstruct_profiles_from_saved(output_path, sequences, scores, spans):
    """Reconstruct profile data from saved sequences and scores."""
    profiles = []
    for i in range(len(spans)):
        start, end = spans[i]
        profile = {
            'predicted': sequences[i],  # The full sequence predictions
            'true': scores[i],         # The full sequence true scores
            'start': start,
            'end': end
        }
        profiles.append(profile)
    return profiles

def analyze_neighbourhood_overlap(repr1: np.ndarray, 
                                repr2: np.ndarray, 
                                labels1: str,
                                labels2: str,
                                k: int = 1,
                                output_dir: str = None) -> dict:
    """
    Analyze overlap between two classes using k-nearest neighbours.
    
    Args:
        repr1: Representations from first class (e.g., UCNE)
        repr2: Representations from second class (e.g., Random/Flanking)
        labels1: Name of first class for plotting
        labels2: Name of second class for plotting
        k: Number of nearest neighbours to consider
        output_dir: Directory to save confusion matrix plot
        
    Returns:
        Dictionary with overlap statistics
    """
    from sklearn.neighbors import NearestNeighbors
    import seaborn as sns
    
    # Combine representations and create labels
    combined_repr = np.concatenate([repr1, repr2])
    labels = np.array([0] * len(repr1) + [1] * len(repr2))
    
    # Fit nearest neighbours
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 because point is its own nearest neighbour
    nn.fit(combined_repr)
    
    # Get nearest neighbours for all points
    distances, indices = nn.kneighbors(combined_repr)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((2, 2))
    
    # Calculate statistics for each class and fill confusion matrix
    stats = {}
    for class_idx, class_name in enumerate(['class1', 'class2']):
        class_mask = labels == class_idx
        class_points = np.where(class_mask)[0]
        
        # For each point in class, check its neighbors
        same_class_neighbors = 0
        total_neighbors = 0
        
        for point_idx in class_points:
            # Get neighbors (excluding the point itself)
            neighbor_indices = indices[point_idx][1:] 
            neighbor_labels = labels[neighbor_indices]
            
            # For k=1, we're only looking at the closest neighbor
            predicted_class = neighbor_labels[0]
            confusion_matrix[class_idx, predicted_class] += 1
            
            # Count neighbors of same class
            same_class_neighbors += np.sum(neighbor_labels == class_idx)
            total_neighbors += len(neighbor_indices)
        
        # Calculate percentage
        pct_same_class = (same_class_neighbors / total_neighbors) * 100
        stats[class_name] = pct_same_class
    
    # Calculate overall statistics
    stats['overall'] = (stats['class1'] + stats['class2']) / 2
    
    # Create confusion matrix plot
    if output_dir:
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='g',
                   xticklabels=[labels1, labels2],
                   yticklabels=[labels1, labels2],
                   cmap='Blues')
        plt.title(f'Nearest Neighbor Confusion Matrix\nk={k}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Save plot
        confusion_matrix_path = f"{output_dir}/confusion_matrix_{labels2}_k{k}.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # Add additional metrics to stats
        total = confusion_matrix.sum()
        stats['true_positive'] = confusion_matrix[0,0]
        stats['false_positive'] = confusion_matrix[1,0]
        stats['false_negative'] = confusion_matrix[0,1]
        stats['true_negative'] = confusion_matrix[1,1]
        stats['accuracy'] = (confusion_matrix[0,0] + confusion_matrix[1,1]) / total * 100
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Get representations of sequences")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes', help='Path to the chromosome sizes file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/conserved_elements/CCP/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--bed_file1', type=str, default ='/home/mica/gamba/data_processing/data/conserved_elements/filteredunseen_hg38UCNE_coordinates.bed', help='First BED file')
    parser.add_argument('--bed_file2', type=str, default='', help='Second BED file (optional)')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation even if cached results exist')
    parser.add_argument('--flanking', action='store_true', help='Generate flanking regions instead of random')
    parser.add_argument('--checkpoint_num', type=int, default=44000, help='Checkpoint number to load')

    args = parser.parse_args()
    
    # Load data
    bed1 = load_bed_file(args.bed_file1)
    checkpoint_num = args.checkpoint_num

    COMPARISON_TYPE = "exons" if args.bed_file2 else ("flanking" if args.flanking else "random")
    #args.output_dir = args.output_dir +f"dcp_{checkpoint_num}_noALM_results/"
    args.output_dir = args.output_dir +f"dcp_{checkpoint_num}_results/"
    os.makedirs(args.output_dir, exist_ok=True)
    if args.bed_file2:
        bed2 = load_bed_file(args.bed_file2)
    if args.flanking:
        bed2_path = generate_flanking_regions(bed1, args.output_dir)
        bed2 = load_bed_file(bed2_path)
    else:
        bed2 = generate_random_regions(len(bed1), args.chrom_sizes)
    
    genome = Fasta(args.genome_fasta)
    bw = pyBigWig.open(args.big_wig)
    #ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    #ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, checkpoint_num)
    #ckpt_path = "/home/mica/gamba/dcps/dcp_34000"
    #ckpt_path = "/home/mica/gamba/dcps/dcp_4000_reweighted_cons"
    #ckpt_path ="/home/mica/gamba/dcps/dcp_14000_reweighted_cons"
    #ckpt_path="/home/mica/gamba/dcps/dcp_132250_only_MSE"
    ckpt_path="/home/mica/gamba/clean_dcps/CCP/dcp_44000"
    #ckpt_path="/home/mica/gamba/clean_dcps/CCP/dcp_noALM44000"

    # Load model configuration
    with open(args.config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    epochs = config["epochs"]
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())
    

    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )

    #get d_model, n_head, n_layers, dim_feedforward and padding_id from the config
    d_model = config.get("d_model", 512) #512/2
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)


    #set up the model load from last checkpoint
    model = JambagambaModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )
    # model = JambaGambaNOALMModel(
    #         model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
    #     )
    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    

    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
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


    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )
    # Generate keywords from bed files
    bed1_name = os.path.splitext(os.path.basename(args.bed_file1))[0]
    #bed1_filename is: hg19_UCNE_coordinates
    bed1_filename = f"representations_{bed1_name}.npz"
    if args.bed_file2:
        bed2_name = os.path.splitext(os.path.basename(args.bed_file2))[0]
        bed2_filename = f"representations_{bed2_name}.npz"
    else:
        bed2_filename = f"representations_{bed1_name}_{COMPARISON_TYPE}.npz"
    
    bed1_repr_path = os.path.join(args.output_dir, bed1_filename)
    bed2_repr_path = os.path.join(args.output_dir, bed2_filename)

    print(f"bed1_repr_path: {bed1_repr_path}, bed2_repr_path: {bed2_repr_path}")
    print(f"bed1_filename: {bed1_filename}, bed2_filename: {bed2_filename}")
    
    # Remove existing files if force_recompute is True
    if args.force_recompute:
        if os.path.exists(bed1_repr_path):
            os.remove(bed1_repr_path)
        if os.path.exists(bed2_repr_path):
            os.remove(bed2_repr_path)
    
    # Process both datasets
    repr1, pred_cons1, true_cons1, var1, profiles1 = process_dataset(
        bed1, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed1_repr_path, "UCNE dataset"
    )
    
    repr2, pred_cons2, true_cons2, var2, profiles2 = process_dataset(
        bed2, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed2_repr_path, f"{COMPARISON_TYPE} dataset"
    )
    
    # Create conservation profile plots if we have profiles or --force_replot
    if (profiles1 and profiles2): #or args.force_replot:
        cons_plot_path1 = f'{args.output_dir}/conservation_profiles_UCNE_{bed1_filename}_{bed2_filename}.png'
        cons_plot_path2 = f'{args.output_dir}/conservation_profiles_{COMPARISON_TYPE}_{bed1_filename}_{bed2_filename}.png'
        cons_indivplot_path1 = f'{args.output_dir}/indiv_conservation_profiles_UCNE_{bed1_filename}_{bed2_filename}.png'
        cons_indivplot_path2 = f'{args.output_dir}/indiv_conservation_profiles_{COMPARISON_TYPE}_{bed1_filename}_{bed2_filename}.png'
        print("Creating conservation profile plots...")
        plot_conservation_profiles(profiles1, cons_plot_path1, 'UCNE')
        plot_conservation_profiles(profiles2, cons_plot_path2, f"{COMPARISON_TYPE}")
        plot_individual_examples(profiles1, cons_indivplot_path1, "UCNE")
        plot_individual_examples(profiles2, cons_indivplot_path2,  f"{COMPARISON_TYPE}")
    else:
        print("\nSkipping conservation profile plots (no profile data available).")
        print("Use --force_recompute to regenerate all data or --force_replot to regenerate plots.")
    
    # Create UMAP visualization
    umap_plot_path = f'{args.output_dir}/umap_plot_{bed1_filename}_{bed2_filename}.png'
    visualize_embeddings(repr1, repr2, 'UCNE', f"{COMPARISON_TYPE}", umap_plot_path)
    
    # Print statistics
    print_detailed_stats(profiles1, 'UCNE')
    print_detailed_stats(profiles2, f"{COMPARISON_TYPE}")


if __name__ == "__main__":
    main()