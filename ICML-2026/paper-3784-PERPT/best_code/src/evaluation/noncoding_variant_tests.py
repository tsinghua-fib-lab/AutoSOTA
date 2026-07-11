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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import pyBigWig
import json
import random


def generate_random_sequences_conservation(genome_fasta, chrom_sizes, big_wig, output_file, num_sequences=10000, sequence_length=2000):
    # Load the genome
    genome = Fasta(genome_fasta)
    
    # Load the bigWig file
    bw = pyBigWig.open(big_wig)
    #read chrom sizes bed file
    chrom_sizes_df = pd.read_csv(chrom_sizes, sep="\t", header=None)
    chrom_sizes_df.columns = ["chrom", "start", "size"]
    
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    
    random_conservation_scores = []

    for _ in range(num_sequences):
        chromosome = random.choice(valid_chromosomes)
        #get chrom_length from chrom_sizes_df
        chrom_length = chrom_sizes_df[chrom_sizes_df["chrom"] == chromosome]["size"].values[0]
        print('chrom_length:', chrom_length)
        start = random.randint(0, chrom_length - sequence_length)
        end = start + sequence_length
        
        # Get the random sequence
        random_sequence = genome[chromosome][start:end].seq
        
        # Get the conservation scores for the random sequence
        intervals = bw.intervals(chromosome, start, end)
        if intervals is None:
            print(f"Error: intervals is None for random sequence {chromosome}:{start}-{end}")
            continue
        
        vals = np.zeros(end - start, dtype=np.float64)
        for interval_start, interval_end, value in intervals:
            vals[interval_start - start : interval_end - start] = value
        
        random_conservation_score = np.mean(vals)
        random_conservation_scores.append(random_conservation_score)
    
    #print total number of random sequences
    print("Total number of random sequences:", len(random_conservation_scores))
    # Save the random conservation scores to a CSV file
    random_conservation_df = pd.DataFrame({'conservation_score': random_conservation_scores})
    random_conservation_df.to_csv(output_file, index=False)
    print(f"Random conservation scores saved to {output_file}")
    
    return random_conservation_scores

def check_conservation_separability(csv_file, genome_fasta, big_wig, cutoff_conservation_score):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Load the genome
    genome = Fasta(genome_fasta)
    
    # Load the bigWig file
    bw = pyBigWig.open(big_wig)
    
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    
    conservation_scores = []
    indices_to_keep = []

    for index, row in df.iterrows():
        chromosome = "chr" + row['Lifted_Chr']
        if chromosome not in valid_chromosomes:
            continue
        position = row['Lifted_Position']
        ref = row['Ref']
        alt = row['Alt']
        
        # Get the reference sequence
        start = position - 1010
        end = position + 1011
        ref_sequence = genome[chromosome][start:end].seq
        
        # Check if the reference allele matches the sequence at the specified position
        if ref_sequence[position - start -1 ] != ref:
            print(f"Reference allele does not match at {chromosome}:{position}. Expected {ref}, found {ref_sequence[position - start -1 ]}")
            continue
        
        # Modify the sequence to have the alternate allele instead of the reference allele
        mutated_sequence = ref_sequence[:1010] + alt + ref_sequence[1011:]
        
        # Get the conservation scores for the mutated sequence
        intervals = bw.intervals(chromosome, start, end)
        if intervals is None:
            print(f"Error: intervals is None for {chromosome}:{start}-{end}")
            continue
        
        vals = np.zeros(end - start, dtype=np.float64)
        for interval_start, interval_end, value in intervals:
            vals[interval_start - start : interval_end - start] = value
        
        conservation_score = np.mean(vals)
        conservation_scores.append(conservation_score)

        #print conservation score
        print("conservation score:", conservation_score)
        
        # Decide whether to keep the example based on conservation score
        if conservation_score > cutoff_conservation_score:
            indices_to_keep.append(index)
    
    # Filter the DataFrame to keep only the examples with higher conservation scores
    filtered_df = df.loc[indices_to_keep]
    
    print(f"Original number of examples: {len(df)}")
    print(f"Number of examples with higher conservation scores: {len(filtered_df)}")
    
    return filtered_df

def calculate_perplexity(sequence_tokens, scores, model, collator, device, mutation_pos):
    collated = collator([(sequence_tokens, scores)])
    with torch.no_grad():
        output = model(collated[0].to(device), collated[1].to(device))
    logits = output["seq_logits"]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Calculate perplexity within 10bp of the mutation position
    start = max(0, mutation_pos - 10)
    end = min(len(sequence_tokens), mutation_pos + 10)
    perplexity = torch.exp(-torch.sum(probs[:, start:end, :] * torch.log(probs[:, start:end, :]), dim=-1)).mean().item()
    return perplexity

def process_variants(df, bigwig_file, genome_fasta, model, collator, tokenizer, device):
    genome = Fasta(genome_fasta)
    ref_perplexities = []
    mutated_perplexities = []

    bw = pyBigWig.open(bigwig_file)

    for index, row in df.iterrows():
        chromosome = row['Chr']
        #if chrY, skip
        if chromosome == 'chrY':
            continue
        position = row['Position']
        ref = row['Ref']
        alt = row['Alt']
        start = position - 1024
        end = position + 1024

        # Get the reference sequence
        ref_sequence = genome[chromosome][start:end].seq

        # Tokenize the reference sequence
        ref_sequence_tokens = tokenizer.tokenizeMSA(ref_sequence)

         # initialize vals with zeros
        vals = np.zeros(end - start, dtype=np.float64)

        # get the conservation scores from the bigwig file
        intervals = bw.intervals(chromosome, start, end)

        # Check if intervals is None
        if intervals is None:
            print("Error: intervals is None")
        else:
            for interval_start, interval_end, value in intervals:
                vals[interval_start - start : interval_end - start] = value

        # round scores to 2 decimal places
        scores = np.round(vals, 2)

        #print length of scores
        #print("the length of scores is:", len(scores))
        #print lengyh of seq
        #print("the length of sequence is:", len(ref_sequence))

        # Calculate perplexity for the reference sequence around pos
        mutation_pos = 1024
        ref_perplexity = calculate_perplexity(ref_sequence_tokens, scores, model, collator, device, mutation_pos)


        # Create the mutated sequence
        mutated_sequence = list(ref_sequence)
        if len(alt) > 1:  # Check if the mutation is multiple nucleotides
            # Insert the mutation and adjust the sequence length to remain 2048
            print("INSERTION MUTATION:", alt)
            mutated_sequence = mutated_sequence[:1024] + list(alt) + mutated_sequence[1024 + len(alt):]
            if len(mutated_sequence) > 2048:
                mutated_sequence = mutated_sequence[:2048]
        else:
            mutated_sequence[1024] = alt
        mutated_sequence = "".join(mutated_sequence)

        # Tokenize the mutated sequence
        mutated_sequence_tokens = tokenizer.tokenizeMSA(mutated_sequence)

        # Calculate perplexity for the mutated sequence
        mutated_perplexity = calculate_perplexity(mutated_sequence_tokens, scores, model, collator, device, mutation_pos)
        # Append perplexities
        ref_perplexities.append(ref_perplexity)
        mutated_perplexities.append(mutated_perplexity)

    return ref_perplexities, mutated_perplexities

def plot_perplexity_diffs(ref_perplexities, mutated_perplexities, output_file, mutation_file):
    plt.figure(figsize=(10, 6))
    plt.scatter(ref_perplexities, mutated_perplexities, color='blue', alpha=0.7)
    plt.plot([min(ref_perplexities), max(ref_perplexities)], [min(ref_perplexities), max(ref_perplexities)], color='red', linestyle='--')
    plt.xlabel('Reference Perplexity')
    plt.ylabel('Mutated Perplexity')
    plt.title(f'Perplexity of Mutated vs Reference Sequence for {mutation_file}')
    #print number of mutated perplexities above reference perplexities
    print("Number of mutated perplexities above reference perplexities:", sum([1 for i in range(len(ref_perplexities)) if mutated_perplexities[i] > ref_perplexities[i]]))
    #print as a percentage
    print("Percentage of mutated perplexities above reference perplexities:", sum([1 for i in range(len(ref_perplexities)) if mutated_perplexities[i] > ref_perplexities[i]])/len(ref_perplexities))
    plt.savefig(output_file)


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


def main():
    parser = argparse.ArgumentParser(description="Process noncoding variants and calculate perplexity differences")
    parser.add_argument('--csv_file', type=str, default ="/home/mica/gamba/data_processing/data/hg38_noncoding_mutations/promoter_mutations_hg38.csv", help='Path to the CSV file with noncoding variants')
    parser.add_argument('--genome_fasta', type=str,  default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_file', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str,  default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    parser.add_argument('--chrom_sizes', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.bed', help='Path to the chromosome sizes bed file')
    args = parser.parse_args()

    #strip name of mutation file of specific mutation (i.e. whats before _mutations):
    mutation_file = args.csv_file.split("/")[-1].split("_mutations")[0]
    print(f"Processing {mutation_file} mutations")
    #change output file name to append mutation file name
    output_file = os.path.join(args.output_file, f"{mutation_file}_perplexity_diffs.png")

    #get checkpoint path with step=5400
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 80000)

    #check to see if the sequences in the file are separable by conservation value (i.e. if these sequences have higher conservation than a random sequence of same length)
    #if not, then the conservation values are not useful for this task

    # Generate or load random sequences conservation scores
    random_conservation_file = os.path.join(args.output_file, "random_conservation_scores.csv")
    if os.path.exists(random_conservation_file):
        print("Loading random conservation scores from file")
        random_conservation_df = pd.read_csv(random_conservation_file)
        random_conservation_scores = random_conservation_df['conservation_score'].values
    else:
        print("Generating random sequences and calculating conservation scores")
        random_conservation_scores = generate_random_sequences_conservation(args.genome_fasta, args.chrom_sizes, args.big_wig, random_conservation_file)

    # Calculate the cutoff conservation score
    cutoff_conservation_score = np.mean(random_conservation_scores)
    print(f"Cutoff conservation score: {cutoff_conservation_score}")

    # Check to see if the sequences in the file are separable by conservation value
    df = check_conservation_separability(args.csv_file, args.genome_fasta, args.big_wig, cutoff_conservation_score)

    if len(df) ==0:
        print("Sequences are not separable by conservation value")
        return
    
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
        pad_to_multiple_of=config.get("pad_to_multiple_of", None),
        test=True,
    )

    # Change this to use the df with higher conservation scores
    ref_perplexities, mutated_perplexities = process_variants(df, args.big_wig, args.genome_fasta, model, collator, tokenizer, device)
    #print len of ref_perplexities
    print("the length of ref_perplexities is:", len(ref_perplexities))
    #Number of mutated perplexities
    print("the length of mutated_perplexities is:", len(mutated_perplexities))
    print("total number of mutations processed:", len(ref_perplexities) + len(mutated_perplexities))
    plot_perplexity_diffs(ref_perplexities, mutated_perplexities, output_file, mutation_file)

if __name__ == "__main__":
    main()