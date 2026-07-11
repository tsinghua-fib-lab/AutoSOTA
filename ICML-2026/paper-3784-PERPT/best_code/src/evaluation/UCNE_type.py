import argparse
import pandas as pd
import numpy as np
from pyfaidx import Fasta
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


from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json


# Create dataset and dataloader
class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]
    
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

def analyze_multiclass_neighbourhood(representations_dict: dict, k: int = 1, output_dir: str = None) -> dict:
    """
    Analyze overlap between multiple classes using k-nearest neighbours.
    
    Args:
        representations_dict: Dictionary of representations for each class
        k: Number of nearest neighbours to consider
        output_dir: Directory to save confusion matrix plot
    """
    from sklearn.neighbors import NearestNeighbors
    import seaborn as sns
    
    # Get class names and combine representations
    class_names = list(representations_dict.keys())
    all_repr = []
    labels = []
    
    for idx, (class_name, repr_array) in enumerate(representations_dict.items()):
        all_repr.append(repr_array)
        labels.extend([idx] * len(repr_array))
    
    combined_repr = np.concatenate(all_repr)
    labels = np.array(labels)
    
    # Fit nearest neighbours
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(combined_repr)
    
    # Get nearest neighbours for all points
    distances, indices = nn.kneighbors(combined_repr)
    
    # Initialize confusion matrix
    n_classes = len(class_names)
    confusion_matrix = np.zeros((n_classes, n_classes))
    
    # Calculate statistics for each class and fill confusion matrix
    stats = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        class_points = np.where(class_mask)[0]
        
        # For each point in class, check its neighbors
        class_neighbor_counts = np.zeros(n_classes)
        
        for point_idx in class_points:
            # Get neighbors (excluding the point itself)
            neighbor_indices = indices[point_idx][1:]
            neighbor_labels = labels[neighbor_indices]
            
            # For confusion matrix, use most common neighbor class
            predicted_class = np.bincount(neighbor_labels).argmax()
            confusion_matrix[class_idx, predicted_class] += 1
            
            # Count neighbors of each class
            for neighbor_class in range(n_classes):
                class_neighbor_counts[neighbor_class] += np.sum(neighbor_labels == neighbor_class)
        
        # Calculate percentages for each class
        total_neighbors = len(class_points) * k
        class_percentages = (class_neighbor_counts / total_neighbors) * 100
        stats[class_name] = {
            other_class: pct 
            for other_class, pct in zip(class_names, class_percentages)
        }
    
    # Create confusion matrix plot
    if output_dir:
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='.0f',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cmap='Blues')
        plt.title(f'Nearest Neighbor Confusion Matrix (k={k})')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Save plot
        confusion_matrix_path = os.path.join(output_dir, f'UCNE_confusion_matrix_k{k}.png')
        plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Calculate overall accuracy
        stats['accuracy'] = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100
        
    return stats

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
                   
def read_metadata_files(base_dir):
    """Read chromosome-specific metadata files."""
    chroms = ['2', '3', '16', '22']
    meta_data = {}
    
    for chrom in chroms:
        file_path = os.path.join(base_dir, f'chr{chrom}_UCNE_meta.txt')
        try:
            df = pd.read_csv(file_path, sep='\t')
            meta_data[f'chr{chrom}'] = df
        except Exception as e:
            print(f"Error reading chr{chrom} metadata: {e}")
    
    return meta_data

def read_and_annotate_coordinates(coord_file, meta_data):
    """Read UCNE coordinates and annotate with types from metadata."""
    coords = pd.read_csv(coord_file, sep='\t',
                        names=['chrom', 'start', 'end', 'name', 'info'])
    
    # Create a new column for type annotation
    coords['type'] = 'unknown'
    
    # Create a mapping of UCNE name to type from all metadata files
    type_mapping = {}
    for meta_df in meta_data.values():
        type_mapping.update(dict(zip(meta_df['UCNE name'], meta_df['Type'])))
    
    # Update types based on UCNE names
    coords['type'] = coords['name'].map(type_mapping)
    
    return coords

def read_metadata_files(base_dir):
    """Read chromosome-specific metadata files."""
    chroms = ['2', '3', '16', '22']
    meta_data = {}
    
    for chrom in chroms:
        file_path = os.path.join(base_dir, f'chr{chrom}_UCNE_meta.txt')
        try:
            df = pd.read_csv(file_path, sep='\t')
            # Extract UCNE name and type
            meta_data[f'chr{chrom}'] = df[['UCNE name', 'Type']].copy()
        except Exception as e:
            print(f"Error reading chr{chrom} metadata: {e}")
    
    return meta_data

def add_upstream_context(coords_df, context_size=2048):
    """Add upstream context to coordinates."""
    extended_coords = coords_df.copy()
    extended_coords['original_start'] = extended_coords['start']
    extended_coords['original_end'] = extended_coords['end']
    extended_coords['start'] = extended_coords['start'] - context_size
    
    # Ensure start positions don't go below 0
    extended_coords['start'] = extended_coords['start'].clip(lower=0)
    
    return extended_coords

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

def create_umap_visualization(embeddings, types, title, output_path):
    """Create and save UMAP visualization."""
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    unique_types = np.unique(types)
    for t in unique_types:
        mask = types == t
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   label=t, alpha=0.6, s=10)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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
            
            
            batch_pred_conservation = scaling_logits[..., 0]
            batch_log_var = scaling_logits[..., 1]
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
                        sequence_conservation_profiles.append(full_profile)
                        
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

def main():
    parser = argparse.ArgumentParser(description="Get representations of UCNE sequence types")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes', help='Path to the chromosome sizes file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_dir', type=str, default='/home/mica/gamba/data_processing/data/conserved_elements/', help='Path to the output file')
    parser.add_argument('--meta_dir', type=str, default='/home/mica/gamba/data_processing/data/conserved_elements/meta_dir', help='Path to the metadata directory')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--bed_file', type=str, default ='/home/mica/gamba/data_processing/data/conserved_elements/hg38_UCNE_coordinates.bed', help='BED file for UCNEs')
    parser.add_argument('--checkpoint_num', type=int, default=132250, help='Checkpoint number to load')
   
    args = parser.parse_args()
    checkpoint_num = args.checkpoint_num

    args.output_dir = args.output_dir +f"dcp_{checkpoint_num}_results/all_UCNE/"
    
    # Read metadata and coordinates
    meta_data = read_metadata_files(args.meta_dir)
    coords = read_and_annotate_coordinates(
        args.bed_file,
        meta_data
    )
    genome = Fasta(args.genome_fasta)
    bw = pyBigWig.open(args.big_wig)
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    #ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, checkpoint_num)
    #ckpt_path = "/home/mica/gamba/dcps/dcp_34000"
    #ckpt_path = '/home/mica/gamba/dcps/dcp_89000_only_MSE'
    ckpt_path="/home/mica/gamba/dcps/dcp_132250_only_MSE"
    
    # Add upstream context
    extended_coords = add_upstream_context(coords)
    
    # Save annotated coordinates
    output_path = os.path.join(args.output_dir, 'annotated_UCNE_coordinates.bed')
    extended_coords.to_csv(output_path, sep='\t', index=False)
    
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

    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        test=True,
    )

    # Drop NaN types
    coords = coords.dropna(subset=['type'])
    
    
    # Process each UCNE type separately
    unique_types = coords['type'].unique()
    all_representations = {}
    all_conservations = {}
    
    for ucne_type in unique_types:
        print(f"\nProcessing {ucne_type} UCNEs...")
        
        # Filter coordinates for this type
        type_coords = coords[coords['type'] == ucne_type]
        
        # Create output path for this type
        output_path = os.path.join(args.output_dir, f'representations_{ucne_type}.npz')
        
        try:
            representations, pred_cons, true_cons, variances, profiles = process_dataset(
                type_coords,  # passing DataFrame directly
                genome,
                args.chrom_sizes,
                bw,
                tokenizer,
                model,
                device,
                collator,
                output_path,
                f"UCNE_{ucne_type}"
            )
            
            all_representations[ucne_type] = representations
            all_conservations[ucne_type] = {
                'pred': pred_cons,
                'true': true_cons,
                'var': variances,
                'profiles': profiles
            }
            
        except Exception as e:
            print(f"Error processing {ucne_type}: {str(e)}")
            continue
    
    # Create combined UMAP visualization
    if all_representations:
        # Create UMAP visualization as before
        all_reps = np.concatenate([reps for reps in all_representations.values()])
        all_types = np.concatenate([[ucne_type] * len(reps) 
                                  for ucne_type, reps in all_representations.items()])
        
        create_umap_visualization(
            all_reps,
            all_types,
            "UMAP of All UCNE Types",
            os.path.join(args.output_dir, 'umap_all_types.png')
        )
        
        # Analyze neighbourhood overlap
        print("\nAnalyzing nearest neighbor relationships between UCNE types...")
        stats = analyze_multiclass_neighbourhood(
            all_representations,
            k=1,  # You can adjust k as needed
            output_dir=args.output_dir
        )
        
        # Print summary statistics
        print("\nNearest Neighbor Analysis Results:")
        print(f"Overall accuracy: {stats['accuracy']:.2f}%")
        print("\nClass-wise neighbor distributions:")
        for class_name, class_stats in stats.items():
            if class_name != 'accuracy':
                print(f"\n{class_name}:")
                for neighbor_class, pct in class_stats.items():
                    print(f"  {neighbor_class}: {pct:.1f}%")
    
    # Close resources
    bw.close()

if __name__ == "__main__":
    main()