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
from sequence_models.constants import START, STOP, MSA_PAD
import umap
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import CaduceusForMaskedLM, CaduceusConservation

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMMLMCollator
from gamba.model import create_model, JambaGambaNoConsModel
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
    sequence_profiles= []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            sequence_input = batch[0][:, 0, :].long()       # (B, T)
            scaling = batch[0][:, 1, :].float()             # (B, T)
            sequence_labels = batch[1][:, 0, :].long()      # (B, T)
            scale_lbls = batch[1][:, 1, :].float()          # (B, T)

            model_kwargs = {
                "input_ids": sequence_input.to(device),
                "labels": sequence_labels.to(device),
            }
             # If model supports conservation prediction, pass conservation labels too
            if hasattr(model, "conservation_head"):
                model_kwargs["conservation_labels"] = scale_lbls.to(device)
            
            # Pass both inputs and labels to the model
            output = model(**model_kwargs)
            
            batch_representations = output["representation"].cpu().numpy()
           
            sequence_data = sequence_input.cpu().numpy()
            true_scores = scaling.cpu().numpy()
            
            
            
            for idx in range(len(batch_representations)):
                span_idx = batch_idx * dataloader.batch_size + idx
                if span_idx < len(original_spans):
                    start, end = original_spans[span_idx]
                    
                    if start < end:
                        repr_slice = batch_representations[idx, start:end]
                 
                        repr_mean = np.mean(repr_slice, axis=0)
                        
                        representations.append(repr_mean)
                       
                    
                        
                        full_profile = {
                            'true': true_scores[idx],
                            'start': start,
                            'end': end
                        }
                        sequence_profiles.append(full_profile)

        
            torch.cuda.empty_cache()
    
    return (np.array(representations), sequence_profiles)



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
        # Center the ROI within the padded sequence (for MLM)
        half_pad = pad_length // 2
        padded_start = max(0, start - half_pad)
        actual_start_offset = start - padded_start

        # Ensure padded region doesn't overflow chromosome end
        if padded_start + target_length > chrom_size:
            padded_start = chrom_size - target_length
            actual_start_offset = start - padded_start

        if padded_start < 0:
            skipped += 1
            continue

        
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
            
        return (data['repr'], profiles)
    
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
    representations, profiles = get_representations(
        model, loader, device, spans
    )
    
    
    # Save everything needed to reconstruct profiles
    np.savez_compressed(output_path,
                       repr=representations,
                       scores=np.array([p['true'] for p in profiles]),
                       spans=np.array([(p['start'], p['end']) for p in profiles]))
    
    print(f"Saved {len(representations)} representations to {output_path}")
    return representations, profiles

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
        flanking_start = start - 500  # Shift up by 500bp
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
                   color=colors[label], label=label, s=5, alpha=0.45)
    
    plt.legend()
    plt.title('UMAP of Sequence Embeddings')
    plt.savefig(output_path)
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

def build_complement_map(vocab_size):
    """
    Build a complement map tensor of length `vocab_size`, where:
    A=0, T=1, G=2, C=3 are reversed: A <-> T, G <-> C
    All other tokens map to themselves
    """
    complement_map = torch.arange(vocab_size)
    complement_map[0] = 1  # A → T
    complement_map[1] = 0  # T → A
    complement_map[2] = 3  # G → C
    complement_map[3] = 2  # C → G
    return complement_map



def main():
    parser = argparse.ArgumentParser(description="Get representations of sequences")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes', help='Path to the chromosome sizes file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/caduceus_conserved_elements/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--bed_file1', type=str, default ='/home/mica/gamba/data_processing/data/conserved_elements/filteredunseen_hg38UCNE_coordinates.bed', help='First BED file')
    parser.add_argument('--bed_file2', type=str, default='', help='Second BED file (optional)')
    #parser.add_argument('--bed_file2', type=str, default='/home/mica/gamba/data_processing/data/UCSC coordinates/unseen_exons_chr2_chr22_chr16_chr3.bed', help='Second BED file (optional)')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation even if cached results exist')
    parser.add_argument('--flanking', action='store_true', help='Generate flanking regions instead of random')
    parser.add_argument('--checkpoint_num', type=int, default=44000, help='Checkpoint number to load')

    args = parser.parse_args()
    
    # Load data
    bed1 = load_bed_file(args.bed_file1)
    checkpoint_num = args.checkpoint_num

    COMPARISON_TYPE = "exons" if args.bed_file2 else ("flanking" if args.flanking else "random")
    args.output_dir = args.output_dir +f"dcp_nocons_{checkpoint_num}_results/"
    #args.output_dir = args.output_dir +f"dcp_cons_{checkpoint_num}_results/"
    #ensure exists:
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
    #ckpt_path= '/home/mica/gamba/dcps/dcp_nocons_50000'
    #ckpt_path="/home/mica/gamba/clean_caduceus_dcps/dcp_56000"
    #ckpt_path="/home/mica/gamba/clean_caduceus_dcps/dcp_consONLYcaduceus_44000"
    ckpt_path="/home/mica/gamba/clean_caduceus_dcps/dcp_caduceus_44000"
    
    # Pull current config
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    pad_token_id = tokenizer.tokenizeMSA([MSA_PAD])[0]
    config = CaduceusConfig(
        d_model=256,
        n_layer=8,
        vocab_size=len(DNA_ALPHABET_PLUS)
    )
    model = CaduceusForMaskedLM(config)
    #model = CaduceusConservation(config)
    model.config.pad_token_id = pad_token_id

    block = None  # Not applicable for huggingface model


    

    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location="cuda:0")

    model.load_state_dict(checkpoint["model_state_dict"])


    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = gLMMLMCollator(
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
    repr1, profiles = process_dataset(
        bed1, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed1_repr_path, "UCNE dataset"
    )
    
    repr2, profiles = process_dataset(
        bed2, genome, args.chrom_sizes, bw, tokenizer,
        model, device, collator, bed2_repr_path, f"{COMPARISON_TYPE} dataset"
    )
    
    # Create UMAP visualization
    umap_plot_path = f'{args.output_dir}/umap_plot_NOCONS_{bed1_filename}_{bed2_filename}.png'
    visualize_embeddings(repr1, repr2, 'UCNE', f"{COMPARISON_TYPE} NOCons", umap_plot_path)
    


if __name__ == "__main__":
    main()