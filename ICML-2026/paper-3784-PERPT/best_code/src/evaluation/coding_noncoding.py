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
import seaborn as sns

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

class SequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.scores[idx]

def get_representations(model, dataloader, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for batch in dataloader:
            sequences, scores = batch
            sequences = sequences.to(device)
            scores = scores.to(device)
            output = model(sequences, scores)
            batch_representations = output["representation"].cpu().numpy()
            # Take the mean of each sequence representation along the sequence length dimension (axis 1)
            batch_representations = np.mean(batch_representations, axis=1)
            representations.extend(batch_representations)
            del sequences, scores, output
            torch.cuda.empty_cache()
    return np.array(representations)

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
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'name', 'label', 'strand'])
    return bed_df

def process_bed_file(bed_df, genome, bw, tokenizer, label):
    sequences = []
    scores_list = []
    labels = []
    #valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    valid_chromosomes = "chr2 chr22".split()

    for index, row in bed_df.iterrows():
        #get chromosomes start and end by tab
        chromosome = row['chrom']
        start = row['start']
        end = row['end']

        #print(f"Processing {label} sequence {index} on chromosome {chromosome} from {start} to {end}")

        #cut sequence to 3000bp max:
        if end - start > 3000:
            end=start+3000

        if chromosome not in valid_chromosomes:
            continue
    
        # Get the reference sequence
        ref_sequence = genome[chromosome][start:end].seq

        # Tokenize the reference sequence
        ref_sequence_tokens = tokenizer.tokenizeMSA(ref_sequence)

        # Initialize vals with zeros
        vals = np.zeros(end - start, dtype=np.float64)

        # Get the conservation scores from the bigwig file
        intervals = bw.intervals(chromosome, start, end)

        # Check if intervals is None
        if intervals is None:
            # print(f"Chromosome: {chromosome}, Start: {start}, End: {end}")
            # print("Error: intervals is None")
            continue
        else:
            for interval_start, interval_end, value in intervals:
                vals[interval_start - start : interval_end - start] = value

        # Round scores to 2 decimal places
        scores = np.round(vals, 2)

        sequences.append(ref_sequence_tokens)
        scores_list.append(scores)
        labels.append(label)

    print(f"Processed {len(sequences)} sequences for {label}")
    return sequences, scores_list, labels

def main():
    parser = argparse.ArgumentParser(description="Process exon and intron sequences and get representations")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_file', type=str, default='/home/mica/gamba/data_processing/data/exons_introns/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--exon_bed_file', type=str, default='/home/mica/gamba/UCSC coordinates/UCSC_3UTR_exons.bed', help='Path to the BED file with exon sequences')
    parser.add_argument('--intron_bed_file', type=str, default='/home/mica/gamba/UCSC coordinates/UCSC_5UTR_exons.bed', help='Path to the BED file with intron sequences')
    args = parser.parse_args()

    #check randomly initialized model

    # Load BED files
    exon_bed_df = load_bed_file(args.exon_bed_file)
    intron_bed_df = load_bed_file(args.intron_bed_file)

    if "3UTR" in args.exon_bed_file:
        exon_label = '3UTR'
        intron_label = '5UTR'
    elif "5UTR" in args.exon_bed_file:
        exon_label = '5UTR'
        intron_label = '3UTR'
    else:
        exon_label = 'exon'
        intron_label = 'intron'

    rep_name = "representations.npz"
    label_name = "labels.csv"
    type_label= "exon_intron"
    #check if files say UTR, and if so add to name: "5UTR_3UTR_representations.npz"
    plot_title = 'Heatmap of Exon and Intron Representations'
    if "UTR" in args.exon_bed_file:
        rep_name = "UTR_representations.npz"
        label_name = "UTR_labels.csv"
        plot_title = 'Heatmap of 5UTR and 3UTR Representations'
        type_label = "5UTR_3UTR"
        


    #print number of exons and introns
    print(f"Number of exons: {len(exon_bed_df)}")
    print(f"Number of introns: {len(intron_bed_df)}")

    # Load genome
    genome = Fasta(args.genome_fasta)

    # Load bigwig file
    bw = pyBigWig.open(args.big_wig)

    # Get checkpoint path with step=5400
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 80000)

    # Load model configuration
    with open(args.config_fpath, "r") as f:
        config = json.load(f)
    config["task"] = config["task"].lower().strip()
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    task = TaskType(config["task"].lower().strip())

    print(
        f"Task: {task}, Model: {config['model_type']}, Dataset: {config['dataset']}, Model Config: {config['model_config']}"
    )
    # Create the model
    model, block = create_model(
        task, config["model_type"], config["model_config"], tokenizer.mask_id.item(), 
    )

    # Get d_model, n_head, n_layers, dim_feedforward and padding_id from the config
    d_model = config.get("d_model", 576) #576/2
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
        model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0)
    )
    lr_func = warmup(config["warmup_steps"])
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

    # Check if sequences are already saved to file representations and labels file
    if os.path.exists(os.path.join(args.output_file, rep_name)) and os.path.exists(os.path.join(args.output_file, label_name)):
        print("Loading representations and labels from file")
        # Load representations and labels
        representations_file = os.path.join(args.output_file, rep_name)
        labels_file = os.path.join(args.output_file, label_name)

        representations = np.load(representations_file)['representations']
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['label'].values
    else:
        print("Processing sequences")
        # Process exon sequences
        exon_sequences, exon_scores, exon_labels = process_bed_file(exon_bed_df, genome, bw, tokenizer, exon_label)
        exon_dataset = SequenceDataset(exon_sequences, exon_scores)
        exon_dataloader = DataLoader(exon_dataset, batch_size=20, collate_fn=collator)
        exon_representations = get_representations(model, exon_dataloader, device)

        # Process intron sequences
        intron_sequences, intron_scores, intron_labels = process_bed_file(intron_bed_df, genome, bw, tokenizer, intron_label)
        intron_dataset = SequenceDataset(intron_sequences, intron_scores)
        intron_dataloader = DataLoader(intron_dataset, batch_size=20, collate_fn=collator)
        intron_representations = get_representations(model, intron_dataloader, device)

        # Combine representations and labels
        representations = np.concatenate((exon_representations, intron_representations), axis=0)
        labels = exon_labels + intron_labels

        # Save representations to a .npz file
        np.savez(os.path.join(args.output_file, rep_name), representations=representations)

        # Save labels to a CSV file
        labels_df = pd.DataFrame({'label': labels})
        labels_df.to_csv(os.path.join(args.output_file, label_name), index=False)
        print(f"Representations saved to {args.output_file}/{rep_name}")
        print(f"Labels saved to {args.output_file}/{label_name}")

    #create heatmap of representations
    # Create heatmap of representations
    # Create heatmap of exon representations
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(exon_representations, cmap='viridis', xticklabels=False, yticklabels=False)
    # plt.title('Heatmap of Exon Representations')
    # plt.savefig(os.path.join(args.output_file, "heatmap_exon_plot.png"))
    # plt.show()
    # print(f"Heatmap of exon representations saved to {args.output_file}/heatmap_exon_plot.png")

    # # Create heatmap of intron representations
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(intron_representations, cmap='viridis', xticklabels=False, yticklabels=False)
    # plt.title('Heatmap of Intron Representations')
    # plt.savefig(os.path.join(args.output_file, "heatmap_intron_plot.png"))
    # plt.show()
    # print(f"Heatmap of intron representations saved to {args.output_file}/heatmap_intron_plot.png")

    # Combine exon and intron representations for comparison
    #subset to 10000 samples from each class
    exon_indices = [i for i, l in enumerate(labels) if l == exon_label]
    intron_indices = [i for i, l in enumerate(labels) if l == intron_label]


    np.random.shuffle(exon_indices)
    np.random.shuffle(intron_indices)
    exon_indices = exon_indices[:10000]
    intron_indices = intron_indices[:10000]
    indices = exon_indices + intron_indices
    representations = representations[indices]
    labels = [labels[i] for i in indices ]

    # Create heatmap of combined representations
    plt.figure(figsize=(10, 8))
    sns.heatmap(representations, cmap='viridis', xticklabels=False, yticklabels=labels)
    plt.title(plot_title)
    plt.savefig(os.path.join(args.output_file, "heatmap_combined_plot.png"))
    plt.show()
    print(f"Heatmap of combined exon and intron representations saved to {args.output_file}/heatmap_combined_plot.png")


    # Create UMAP plot
    umap_model = umap.UMAP()
    umap_embeddings = umap_model.fit_transform(representations)
    plt.figure(figsize=(10, 6))

    # Plot exon and intron sequences
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        indices = np.array(indices)
        if np.any(indices >= umap_embeddings.shape[0]):
            print(f"Warning: Some indices are out of bounds for label {label}")
            indices = indices[indices < umap_embeddings.shape[0]]
        color = 'green' if label == exon_label else 'blue'
        plt.scatter(umap_embeddings[indices, 0], umap_embeddings[indices, 1], label=label, s=5, color=color)

    plt.legend()
    plt.title(plot_title)
    plt.savefig(os.path.join(args.output_file, f"umap_plot_{type_label}.png"))
    plt.show()
    print(f"UMAP plot saved to {args.output_file}")

if __name__ == "__main__":
    main()