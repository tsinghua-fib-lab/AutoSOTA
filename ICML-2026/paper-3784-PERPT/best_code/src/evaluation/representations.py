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
            # take the mean of each sequence representation along the sequence length dimension (axis 1)
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
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end'])
    return bed_df

def process_bed_file(bed_df, genome, bw, tokenizer, label):
    sequences = []
    scores_list = []
    labels = []
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()


    for index, row in bed_df.iterrows():
        chromosome = row['chrom']
        if chromosome not in valid_chromosomes:
            continue
        start = row['start']
        end = row['end']

        #check if seq length > 2048, skip
        if end - start > 2048:
            #just take the first 2048 bp
            end = start + 2048

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
            print("Error: intervals is None")
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
    parser = argparse.ArgumentParser(description="Process enhancer sequences and get representations")
    parser.add_argument('--genome_fasta', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_file', type=str, default='/home/mica/gamba/data_processing/data/enhancers/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str, default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    args = parser.parse_args()

    # Load BED files
    chef_bed_file = '/home/mica/gamba/data_processing/data/enhancers/chef_hg38.bed'
    clef_bed_file = '/home/mica/gamba/data_processing/data/enhancers/clef_hg38.bed'
    chef_bed_df = load_bed_file(chef_bed_file)
    clef_bed_df = load_bed_file(clef_bed_file)
    #add mshef and cmef
    mshef_bed_file = '/home/mica/gamba/data_processing/data/enhancers/mshef_hg38.bed'
    cmef_bed_file = '/home/mica/gamba/data_processing/data/enhancers/cmef_hg38.bed'
    mshef_bed_df = load_bed_file(mshef_bed_file)
    cmef_bed_df = load_bed_file(cmef_bed_file)

    # Load genome
    genome = Fasta(args.genome_fasta)

    # Load bigwig file
    bw = pyBigWig.open(args.big_wig)


    #get checkpoint path with step=5400
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 18000)


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
    d_model = config.get("d_model", 576) #576/2
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

    #check if sequences are alrady saved to file representations and labels file
    if os.path.exists(os.path.join(args.output_file, "representations.npz")) and os.path.exists(os.path.join(args.output_file, "labels.csv")):
        print("Loading representations and labels from file")
        # Load representations and labels
        
        # Load representations and labels
        representations_file = os.path.join(args.output_file, "representations.npz")
        labels_file = os.path.join(args.output_file, "labels.csv")

        representations = np.load(representations_file)['representations']
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['label'].values
    else:
        print("Processing sequences")
        # Process chef sequences (Processed 1735 sequences for chef)
        chef_sequences, chef_scores, chef_labels = process_bed_file(chef_bed_df, genome, bw, tokenizer, 'chef')
        chef_dataset = SequenceDataset(chef_sequences, chef_scores)
        chef_dataloader = DataLoader(chef_dataset, batch_size=20, collate_fn=collator)
        chef_representations = get_representations(model, chef_dataloader, device)

        # Process clef sequences (Processed 14097 sequences for clef)
        clef_sequences, clef_scores, clef_labels = process_bed_file(clef_bed_df, genome, bw, tokenizer, 'clef')
        # Subset to 5000 sequences
        clef_sequences = clef_sequences[:10000]
        clef_dataset = SequenceDataset(clef_sequences, clef_scores)
        clef_dataloader = DataLoader(clef_dataset, batch_size=15, collate_fn=collator)
        clef_representations = get_representations(model, clef_dataloader, device)

        #print shape of chef and clef representations
        print(f"Chef representations shape: {chef_representations.shape}")
        print(f"Clef representations shape: {clef_representations.shape}")

        #add processing for mshef and cmef:
        # Process mshef sequences (Processed 1735 sequences for chef)
        mshef_sequences, mshef_scores, mshef_labels = process_bed_file(mshef_bed_df, genome, bw, tokenizer, 'mshef')
        mshef_dataset = SequenceDataset(mshef_sequences, mshef_scores)
        mshef_dataloader = DataLoader(mshef_dataset, batch_size=20, collate_fn=collator)
        mshef_representations = get_representations(model, mshef_dataloader, device)

        # Process clef sequences (Processed 14097 sequences for clef)
        cmef_sequences, cmef_scores, cmef_labels = process_bed_file(cmef_bed_df, genome, bw, tokenizer, 'cmef')
        # Subset to 5000 sequences
        cmef_sequences = cmef_sequences[:10000]
        cmef_dataset = SequenceDataset(cmef_sequences, cmef_scores)
        cmef_dataloader = DataLoader(cmef_dataset, batch_size=15, collate_fn=collator)
        cmef_representations = get_representations(model, cmef_dataloader, device)


        # Combine representations and labels
        representations = np.concatenate((chef_representations, clef_representations, mshef_representations, cmef_representations), axis=0)
        labels = chef_labels + clef_labels + mshef_labels + cmef_labels

        # Save representations to a .npz file
        np.savez(os.path.join(args.output_file, "representations.npz"), representations=representations)

        # Save labels to a CSV file
        labels_df = pd.DataFrame({'label': labels})
        labels_df.to_csv(os.path.join(args.output_file, "labels.csv"), index=False)
        print(f"Representations saved to {args.output_file}/representations.npz")
        print(f"Labels saved to {args.output_file}/labels.csv")
    # Create UMAP plot
    umap_model = umap.UMAP()
    umap_embeddings = umap_model.fit_transform(representations)
    plt.figure(figsize=(10, 6))

    # Plot clef first
    for label in ['clef']:
        indices = [i for i, l in enumerate(labels) if l == label]
        indices = np.array(indices)
        if np.any(indices >= umap_embeddings.shape[0]):
            print(f"Warning: Some indices are out of bounds for label {label}")
            indices = indices[indices < umap_embeddings.shape[0]]
        plt.scatter(umap_embeddings[indices, 0], umap_embeddings[indices, 1], label=label, s=10)

    # Plot other labels, with mshef and cmef in gray
    for label in set(labels):
        if label == 'clef':
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        indices = np.array(indices)
        if np.any(indices >= umap_embeddings.shape[0]):
            print(f"Warning: Some indices are out of bounds for label {label}")
            indices = indices[indices < umap_embeddings.shape[0]]
        color = 'gray' if label in ['mshef', 'cmef'] else None
        plt.scatter(umap_embeddings[indices, 0], umap_embeddings[indices, 1], label=label, s=10, color=color)

    plt.legend()
    plt.title('UMAP of mean seq embeddings')
    plt.savefig(os.path.join(args.output_file, "umap_plot.png"))
    plt.show()
    print(f"UMAP plot saved to {args.output_file}")



if __name__ == "__main__":
    main()