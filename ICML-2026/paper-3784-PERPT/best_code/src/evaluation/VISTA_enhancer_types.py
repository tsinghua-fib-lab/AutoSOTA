import torch
import numpy as np
from pyfaidx import Fasta
import pandas as pd
from typing import Dict, List, Set, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../gamba")
from typing import Optional, Sequence, Tuple, Type
import pyBigWig
import torch
import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import umap

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import re


from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json
import argparse


class SequenceDataset(Dataset):
    def __init__(self, sequences, coordinates):
        self.sequences = sequences
        self.coordinates = coordinates

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.coordinates[idx]


def load_enhancer_data(tsv_path: str) -> pd.DataFrame:
    """Load enhancer data from TSV file."""
    print(f"Reading data from {tsv_path}")
    
    # Read the TSV file
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(df)} total records")
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return pd.DataFrame()
    
    # Print column names to debug
    print("Columns in file:", df.columns.tolist())
    
    # Handle tissue sets with more robust error checking
    df['Tissues'] = df['Tissues'].fillna('').astype(str)
    df['tissue_set'] = df['Tissues'].apply(
        lambda x: set(x.split(', ')) if x and x.strip() else set()
    )
    
    # Filter for positive expression with error checking
    if 'Expression' not in df.columns:
        print("Error: 'Expression' column not found in data")
        return pd.DataFrame()
        
    positive_df = df[df['Expression'] == 'positive'].copy()
    print(f"Found {len(positive_df)} records with positive expression")
    
    # Parse coordinates
    def parse_coordinates(coord_str):
        try:
            if pd.isna(coord_str):
                return None, None, None
            # Try both formats: chr:start-end and just the coordinates
            match = re.match(r'(?:chr)?(\d+):(\d+)-(\d+)', str(coord_str))
            if match:
                return f"chr{match.group(1)}", int(match.group(2)), int(match.group(3))
            return None, None, None
        except Exception as e:
            print(f"Error parsing coordinates {coord_str}: {e}")
            return None, None, None
    
    # Apply coordinate parsing
    coords = positive_df['Element Coordinates'].apply(parse_coordinates)
    positive_df['chrom'] = coords.apply(lambda x: x[0])
    positive_df['start'] = coords.apply(lambda x: x[1])
    positive_df['end'] = coords.apply(lambda x: x[2])
    
    # Filter out rows with invalid coordinates
    valid_coords = (
        positive_df['chrom'].notna() & 
        positive_df['start'].notna() & 
        positive_df['end'].notna()
    )
    valid_df = positive_df[valid_coords].copy()
    
    # Convert coordinates to integers
    valid_df['start'] = valid_df['start'].astype(int)
    valid_df['end'] = valid_df['end'].astype(int)
    
    print(f"Final dataset: {len(valid_df)} valid enhancers with positive expression")
    print("Sample of processed data:")
    print(valid_df[['Element ID', 'chrom', 'start', 'end', 'Tissues']].head())
    
    return valid_df

def process_sequences(tissue_groups: Dict[str, pd.DataFrame],
                     genome: Fasta,
                     tokenizer,
                     bw,
                     target_length: int = 2048) -> Dict[str, Tuple[List, List, List[Tuple[int, int]]]]:
    """Process sequences for each tissue group."""
    sequences_by_group = {}
    
    for tissue_pattern, group_df in tissue_groups.items():
        print(f"\nProcessing tissue pattern: {tissue_pattern}")
        sequences = []
        scores_list = []
        spans = []
        skipped = 0
        
        for idx, row in group_df.iterrows():
            try:
                chrom = row['chrom']
                start = int(row['start'])
                end = int(row['end'])
                
                if start >= end:
                    print(f"Skipping invalid coordinates: start >= end")
                    skipped += 1
                    continue
                    
                original_length = end - start
                pad_length = target_length - original_length
                padded_start = max(0, start - pad_length)
                actual_start_offset = start - padded_start
                
                # Get sequence
                sequence = genome[chrom][padded_start:padded_start + target_length].seq
                sequence_tokens = tokenizer.tokenizeMSA(sequence)
                
                # Get conservation scores
                vals = np.zeros(target_length, dtype=np.float64)
                intervals = bw.intervals(chrom, padded_start, padded_start + target_length)
                
                if intervals is not None:
                    for interval_start, interval_end, value in intervals:
                        relative_start = interval_start - padded_start
                        relative_end = interval_end - padded_start
                        if 0 <= relative_start < target_length and 0 <= relative_end <= target_length:
                            vals[relative_start:relative_end] = value
                
                scores = np.round(vals, 2)
                sequences.append(sequence_tokens)
                scores_list.append(scores)
                spans.append((actual_start_offset, 
                            actual_start_offset + original_length))
                
            except Exception as e:
                print(f"Error processing sequence {row.get('Element ID', 'unknown')}: {str(e)}")
                skipped += 1
                continue
        
        if sequences:
            sequences_by_group[tissue_pattern] = (sequences, scores_list, spans)
            print(f"Processed {len(sequences)} sequences, skipped {skipped} sequences")
        else:
            print(f"No valid sequences for pattern: {tissue_pattern}")
    
    return sequences_by_group


def get_representations(model, sequences_by_group: Dict[str, Tuple[List, List, List]], 
                       device,
                       collator,
                       batch_size: int = 32) -> Dict[str, List]:
    """Get model representations for each tissue group with improved error handling."""
    model.eval()
    all_results = {}
    
    for tissue_pattern, (sequences, scores, spans) in sequences_by_group.items():
        print(f"\nProcessing representations for {tissue_pattern}")
        
        # Create dataset and dataloader
        dataset = SequenceDataset(sequences, scores)
        loader = DataLoader(dataset, 
                          batch_size=batch_size,
                          collate_fn=collator)
        
        group_representations = []
        group_pred_cons = []
        group_true_cons = []
        group_variances = []
        group_profiles = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                output = model(inputs, labels)
                batch_representations = output["representation"].cpu().numpy()
                scaling_logits = output["scaling_logits"].cpu().numpy()
                sequence_data = inputs[:, 0].cpu().numpy()
                true_scores = inputs[:, 1].cpu().numpy()
                
                batch_pred_conservation = scaling_logits[..., 0]
                batch_log_var = scaling_logits[..., 1]
                batch_variances = np.exp(batch_log_var)
                
                for idx in range(len(batch_representations)):
                    span_idx = batch_idx * batch_size + idx
                    if span_idx < len(spans):
                        start, end = spans[span_idx]
                        
                        if start < end:
                            pred_cons_slice = batch_pred_conservation[idx, start:end]
                            true_cons_slice = true_scores[idx, start:end]
                            var_slice = batch_variances[idx, start:end]
                            repr_slice = batch_representations[idx, start:end]
                            
                            # Check for valid data before computing means
                            if len(pred_cons_slice) > 0 and len(true_cons_slice) > 0 and len(var_slice) > 0 and len(repr_slice) > 0:
                                pred_cons_mean = np.nanmean(pred_cons_slice)
                                true_cons_mean = np.nanmean(true_cons_slice)
                                var_mean = np.nanmean(var_slice)
                                repr_mean = np.nanmean(repr_slice, axis=0)
                                
                                # Only add if we have valid means
                                if not np.any(np.isnan(repr_mean)):
                                    group_representations.append(repr_mean)
                                    group_pred_cons.append(pred_cons_mean)
                                    group_true_cons.append(true_cons_mean)
                                    group_variances.append(var_mean)
                                    
                                    profile = {
                                        'predicted': batch_pred_conservation[idx],
                                        'true': true_scores[idx],
                                        'start': start,
                                        'end': end
                                    }
                                    group_profiles.append(profile)
                
                del inputs, labels, output
                torch.cuda.empty_cache()
        
        if len(group_representations) > 0:
            all_results[tissue_pattern] = {
                'representations': np.array(group_representations),
                'pred_cons': np.array(group_pred_cons),
                'true_cons': np.array(group_true_cons),
                'variances': np.array(group_variances),
                'profiles': group_profiles
            }
            print(f"Generated {len(group_representations)} valid representations")
        else:
            print(f"Warning: No valid representations generated for {tissue_pattern}")
    
    return all_results

def create_umap_visualization(representations_by_group: Dict[str, np.ndarray],
                            output_path: str,
                            min_samples: int = 5):
    """Create UMAP visualization of tissue pattern representations."""
    import umap
    
    # Filter groups with sufficient samples and valid data
    valid_groups = {}
    for k, v in representations_by_group.items():
        if isinstance(v, dict) and 'representations' in v:
            repr_array = v['representations']
            if len(repr_array) >= min_samples and not np.any(np.isnan(repr_array)):
                valid_groups[k] = repr_array
    
    if len(valid_groups) < 2:
        print("Not enough groups with sufficient samples for UMAP visualization")
        return
    
    # Combine all representations
    all_repr = []
    all_labels = []
    
    for tissue_pattern, repr_array in valid_groups.items():
        if len(repr_array.shape) == 2:  # Check if the array is 2D
            all_repr.append(repr_array)
            all_labels.extend([tissue_pattern] * len(repr_array))
    
    if not all_repr:
        print("No valid representations found for UMAP")
        return
        
    try:
        combined_repr = np.vstack(all_repr)  # Use vstack instead of concatenate
        
        # Create UMAP embedding
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(combined_repr)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot each group
        unique_patterns = list(valid_groups.keys())
        for i, pattern in enumerate(unique_patterns):
            mask = np.array(all_labels) == pattern
            plt.scatter(embedding[mask, 0], 
                       embedding[mask, 1],
                       label=pattern[:30] + '...' if len(pattern) > 30 else pattern,
                       alpha=0.6, 
                       s=10)
        
        plt.title('UMAP of Tissue Expression Patterns')
        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"UMAP visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error during UMAP visualization: {str(e)}")
        print(f"Debug info: {[arr.shape for arr in all_repr]}")


def analyze_tissue_combinations(df: pd.DataFrame) -> Dict:
    """Analyze tissue combinations and create meaningful groups."""
    # Get all unique tissues
    all_tissues = set()
    for tissue_set in df['tissue_set']:
        all_tissues.update(tissue_set)
    print(f"Total unique tissues: {len(all_tissues)}")
    print("Unique tissues:", sorted(all_tissues))

    # Count frequency of each tissue
    tissue_counts = {tissue: 0 for tissue in all_tissues}
    for tissue_set in df['tissue_set']:
        for tissue in tissue_set:
            tissue_counts[tissue] += 1
    
    # Sort tissues by frequency
    sorted_tissues = sorted(tissue_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTissue frequencies:")
    for tissue, count in sorted_tissues:
        print(f"{tissue}: {count}")

    # Create co-occurrence matrix
    tissue_list = sorted(all_tissues)
    tissue_to_idx = {tissue: i for i, tissue in enumerate(tissue_list)}
    cooccurrence = np.zeros((len(all_tissues), len(all_tissues)))
    
    for tissue_set in df['tissue_set']:
        for t1 in tissue_set:
            for t2 in tissue_set:
                i, j = tissue_to_idx[t1], tissue_to_idx[t2]
                cooccurrence[i, j] += 1

    # Define anatomical groups
    anatomical_groups = {
        'Neural': {'Neural tube', 'Hindbrain', 'Midbrain', 'Forebrain', 'Cranial nerve'},
        'Sensory': {'Nose', 'Trigeminal V (ganglion,cranial)'},
        'Musculoskeletal': {'Limb', 'Somite'},
        'Other': {'Heart', 'Tail', 'Other'}
    }

    # Assign anatomical groups
    df['broad_groups'] = df['tissue_set'].apply(
        lambda x: ','.join(sorted(set(
            group_name 
            for group_name, tissues in anatomical_groups.items()
            if any(tissue in x for tissue in tissues)
        )))
    )

    # Convert sets to frozensets for hashing
    df['tissue_frozenset'] = df['tissue_set'].apply(frozenset)
    
    # Calculate Jaccard similarity groups
    def calculate_jaccard(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    # Group by Jaccard similarity
    similarity_threshold = 0.5
    unique_patterns = df['tissue_frozenset'].unique()
    similarity_groups = []
    used_patterns = set()

    for pattern1 in unique_patterns:
        if pattern1 in used_patterns:
            continue
        
        current_group = {pattern1}
        for pattern2 in unique_patterns:
            if pattern2 not in used_patterns and pattern1 != pattern2:
                if calculate_jaccard(pattern1, pattern2) >= similarity_threshold:
                    current_group.add(pattern2)
        
        if current_group:
            similarity_groups.append(current_group)
            used_patterns.update(current_group)

    # Create pattern to group mapping
    pattern_to_group = {}
    for i, group in enumerate(similarity_groups):
        for pattern in group:
            pattern_to_group[pattern] = f"Similarity_Group_{i+1}"

    # Add similarity groups to dataframe
    df['similarity_group'] = df['tissue_frozenset'].apply(
        lambda x: pattern_to_group.get(x, 'Other')
    )

    return {
        'tissue_counts': tissue_counts,
        'cooccurrence': cooccurrence,
        'tissue_to_idx': tissue_to_idx,
        'tissue_list': tissue_list,
        'anatomical_groups': df.groupby('broad_groups').size().to_dict(),
        'similarity_groups': df.groupby('similarity_group').size().to_dict(),
        'grouped_df': df
    }


def visualize_tissue_relationships(cooccurrence: np.ndarray, 
                                 tissue_to_idx: Dict[str, int],
                                 tissue_list: List[str],
                                 output_dir: str):
    """Visualize tissue co-occurrence patterns."""
    plt.figure(figsize=(15, 12))
    
    # Create co-occurrence heatmap
    sns.heatmap(cooccurrence, 
                xticklabels=tissue_list,
                yticklabels=tissue_list,
                cmap='YlOrRd')
    
    plt.title('Tissue Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'tissue_cooccurrence.png'))
    plt.close()

def plot_group_sizes(group_counts: Dict[str, int], 
                    title: str, 
                    output_path: str):
    """Plot group sizes as a bar chart."""
    plt.figure(figsize=(12, 6))
    
    groups = list(group_counts.keys())
    counts = list(group_counts.values())
    
    sns.barplot(x=counts, y=groups)
    plt.title(title)
    plt.xlabel('Number of sequences')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def update_main_with_grouping(args, analysis_results):
    """Update the main output directory and create visualizations."""
    # Save tissue analysis visualizations
    visualize_tissue_relationships(
        analysis_results['cooccurrence'],
        analysis_results['tissue_to_idx'],
        analysis_results['tissue_list'],
        args.output_dir
    )
    
    # Plot group sizes
    if args.grouping == 'anatomical':
        plot_group_sizes(
            analysis_results['anatomical_groups'],
            'Anatomical Group Sizes',
            os.path.join(args.output_dir, 'anatomical_group_sizes.png')
        )
    else:
        plot_group_sizes(
            analysis_results['similarity_groups'],
            'Similarity Group Sizes',
            os.path.join(args.output_dir, 'similarity_group_sizes.png')
        )
    
    # Return the grouped dataframe and selected grouping column
    group_column = 'broad_groups' if args.grouping == 'anatomical' else 'similarity_group'
    return analysis_results['grouped_df'], group_column

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

def analyze_tissue_groups(representations_by_group: Dict[str, Dict], 
                         k: int = 1, 
                         output_dir: Optional[str] = None) -> dict:
    """Analyze separation between tissue expression groups using kNN with improved error handling."""
    # Extract and validate representations
    valid_representations = {}
    for group, data in representations_by_group.items():
        if 'representations' in data and len(data['representations']) > 0:
            repr_array = data['representations']
            if not np.any(np.isnan(repr_array)):
                valid_representations[group] = repr_array
    
    # Filter out groups with too few samples
    min_samples = k + 1
    valid_groups = {key: repr_array for key, repr_array in valid_representations.items()
                   if len(repr_array) >= min_samples}
    
    if len(valid_groups) < 2:
        print("Not enough groups with sufficient samples for kNN analysis")
        return {'accuracy': 0, 'class_sizes': {}, 'confusion_matrix': None}
    
    class_names = list(valid_groups.keys())
    all_repr = []
    labels = []
    
    for idx, (tissue_pattern, repr_array) in enumerate(valid_groups.items()):
        all_repr.append(repr_array)
        labels.extend([idx] * len(repr_array))
    
    try:
        combined_repr = np.concatenate(all_repr)
        labels = np.array(labels)
        
        # Fit nearest neighbours
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(combined_repr)
        distances, indices = nn.kneighbors(combined_repr)
        
        # Calculate confusion matrix and statistics
        n_classes = len(class_names)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for class_idx, class_name in enumerate(class_names):
            class_mask = labels == class_idx
            class_points = np.where(class_mask)[0]
            
            for point_idx in class_points:
                neighbor_indices = indices[point_idx][1:]
                neighbor_labels = labels[neighbor_indices]
                predicted_class = np.bincount(neighbor_labels).argmax()
                confusion_matrix[class_idx, predicted_class] += 1
        
        # Convert to percentages (normalize by row)
        confusion_matrix_pct = confusion_matrix.copy()
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix_pct = np.divide(confusion_matrix, row_sums, 
                                       where=row_sums!=0) * 100
        
        # Create visualization if output directory provided
        if output_dir:
            plt.figure(figsize=(20, 16))
            
            # Create the heatmap with percentages
            sns.heatmap(confusion_matrix_pct, 
                       annot=True, 
                       fmt='.1f',  # Show one decimal place
                       xticklabels=class_names,
                       yticklabels=class_names,
                       cmap='Blues')
            
            plt.title(f'Tissue Expression Pattern Confusion Matrix (k={k})\nValues show % of class total')
            plt.xlabel('Predicted Pattern')
            plt.ylabel('True Pattern')
            
            # Rotate axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add the raw counts as text annotations
            for i in range(n_classes):
                for j in range(n_classes):
                    if confusion_matrix[i, j] > 0:
                        raw_count = int(confusion_matrix[i, j])
                        plt.text(j + 0.5, i + 0.7, f'n={raw_count}', 
                                ha='center', va='center',
                                color='darkgrey', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tissue_confusion_matrix_k{k}.png'), 
                       bbox_inches='tight', 
                       dpi=300)
            plt.close()
        
        # Calculate accuracy using the raw counts
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100

        #print out the stats
        print(f"Overall accuracy: {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix_pct,  # Return percentages
            'raw_counts': confusion_matrix,  # Also return raw counts
            'class_sizes': {name: len(repr_array) 
                           for name, repr_array in valid_groups.items()}
        }
        
    except (ValueError, np.AxisError) as e:
        print(f"Error during analysis: {str(e)}")
        return {'accuracy': 0, 'class_sizes': {}, 'confusion_matrix': None}

def main():
    parser = argparse.ArgumentParser(description="Get representations of VISTA enhancer types")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes', help='Path to the chromosome sizes file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/VISTA_enhancers/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--enhancer_file', type=str, default ='/home/mica/gamba/data_processing/data/VISTA_enhancers/experiments.tsv', help='BED file for UCNEs')
    parser.add_argument('--checkpoint_num', type=int, default=44000, help='Checkpoint number to load')
    parser.add_argument('--grouping', type=str, choices=['anatomical', 'jaccard'], 
                       default='anatomical', help='Tissue grouping strategy')
    parser.add_argument('--min_group_size', type=int, default=10,
                       help='Minimum number of sequences per group')
   
    args = parser.parse_args()
    checkpoint_num = args.checkpoint_num

    args.output_dir = os.path.join(args.output_dir, f"dcp_{checkpoint_num}_results", f"grouping_{args.grouping}")
    #if output_dir doesn't exist, make it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    #ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, checkpoint_num)
    ckpt_path = "/home/mica/gamba/clean_dcps/CCP/dcp_44000"

    bw = pyBigWig.open(args.big_wig)

    
    # Initialize model and other components (similar to original code)
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
    d_model = config.get("d_model", 512) #576/2
    nhead = config.get("n_head", 8)  
    n_layers = config.get("n_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model)
    padding_id = config.get("padding_id", 0)


    #set up the model load from last checkpoint
    model = JambagambaModel(
            model, d_model=d_model, nhead=nhead, n_layers=n_layers, padding_id=0, dim_feedfoward=dim_feedforward
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        os.path.join(ckpt_path, "scheduler.pt"), map_location=device
    )
    scheduler.load_state_dict(sd["scheduler_state_dict"])

    # Move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )

    # Load genome data
    genome = Fasta(args.genome_fasta)

    # Load and analyze enhancer data
    print("Loading enhancer data...")
    enhancer_df = load_enhancer_data(args.enhancer_file)
    
     
    print("\nAnalyzing tissue patterns...")
    analysis_results = analyze_tissue_combinations(enhancer_df)  # Changed function call
    
    # Update the main output directory with the analysis results
    enhancer_df = analysis_results['grouped_df']
    group_column = 'broad_groups' if args.grouping == 'anatomical' else 'similarity_group'

    # Save tissue analysis visualizations
    visualize_tissue_relationships(
        analysis_results['cooccurrence'],
        analysis_results['tissue_to_idx'],
        analysis_results['tissue_list'],
        args.output_dir
    )
    
    # Plot group sizes
    if args.grouping == 'anatomical':
        plot_group_sizes(
            analysis_results['anatomical_groups'],
            'Anatomical Group Sizes',
            os.path.join(args.output_dir, 'anatomical_group_sizes.png')
        )
    else:
        plot_group_sizes(
            analysis_results['similarity_groups'],
            'Similarity Group Sizes',
            os.path.join(args.output_dir, 'similarity_group_sizes.png')
        )
    
    # Use grouped dataframe for analysis
    enhancer_df = analysis_results['grouped_df']
    group_column = 'broad_groups' if args.grouping == 'anatomical' else 'similarity_group'
    
    print(f"\nGrouping by {args.grouping} patterns...")
    tissue_groups = {
        group: group_df for group, group_df in enhancer_df.groupby(group_column)
        if len(group_df) >= args.min_group_size
    }
    
    print(f"\nFound {len(tissue_groups)} groups with ≥{args.min_group_size} sequences:")
    for group, df in tissue_groups.items():
        print(f"{group}: {len(df)} sequences")
    
    print("\nProcessing sequences...")
    sequences_by_group = process_sequences(tissue_groups, genome, tokenizer, bw)
    
    print("\nGetting representations...")
    results_by_group = get_representations(
        model, sequences_by_group, device, collator
    )
    
    # Analyze and visualize
    print("\nAnalyzing tissue patterns...")
    stats = analyze_tissue_groups(
        results_by_group, k=1, output_dir=args.output_dir
    )
    
    print("\nCreating UMAP visualization...")
    create_umap_visualization(
        results_by_group,
        os.path.join(args.output_dir, f'tissue_umap_{args.grouping}.png')
    )
    
    # Save results
    print("\nSaving results...")
    stats_file = os.path.join(args.output_dir, f'tissue_analysis_stats_{args.grouping}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Grouping strategy: {args.grouping}\n")
        f.write(f"Minimum group size: {args.min_group_size}\n\n")
        f.write(f"Overall accuracy: {stats['accuracy']:.2f}%\n\n")
        f.write("Group sizes:\n")
        for class_name, size in stats['class_sizes'].items():
            f.write(f"{class_name}: {size}\n")
    
    print(f"\nResults saved to: {args.output_dir}")
    bw.close()

if __name__ == "__main__":
    main()
