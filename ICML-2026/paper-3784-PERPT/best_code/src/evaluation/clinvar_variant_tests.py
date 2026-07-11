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

#wobble effect on the third codon QUANTIFY THIS EFFECT ON THE TEST SET
#mutation in the ORF (open reading frame), third position is more benign than first and second position
#hold out regions in test dataset with low sequence similarity to the training set
#separate coding and non-coding regions

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

def reverse_complement(sequence: torch.Tensor) -> torch.Tensor:
    """
    Reverse complement DNA sequences where:
    G=0, A=1, T=2, C=3
    """
    complement_map = torch.tensor([3, 2, 1, 0], device=sequence.device)  # C, T, A, G
    # apply complement and reverse
    reverse_comp = complement_map[sequence.flip(dims=[-1])]
    return reverse_comp

def calculate_log_likelihood(sequence_tokens, scores, model, collator, device, mutation_pos):
    collated = collator([(sequence_tokens, scores)])
    sequence, scaling = collated[0].unbind(dim=1)
    seq_lbls, scale_lbs = collated[1].unbind(dim=1)
    
    with torch.no_grad():
        output = model(collated[0].to(device), collated[1].to(device))
    
    logits = output["seq_logits"]
    log_probs = torch.nn.functional.log_softmax(logits[:, mutation_pos, :], dim=-1)
    
    if mutation_pos >= seq_lbls.shape[1]:
        raise IndexError(f"mutation_pos ({mutation_pos}) is out of bounds for seq_lbls with size {seq_lbls.shape[1]}")
    
    indices = seq_lbls[:, mutation_pos].long()
    
    if (indices < 0).any() or (indices >= log_probs.size(1)).any():
        raise ValueError(f"Invalid indices found: {indices}")
    
    log_probs = log_probs.cpu()
    log_likelihood = log_probs.gather(1, indices.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.item()

def calculate_average_log_likelihood(sequence_tokens, scores, model, collator, device, mutation_pos):
    log_likelihood = calculate_log_likelihood(sequence_tokens, scores, model, collator, device, mutation_pos)
    
    if 4 in sequence_tokens:
        print("Sequence contains 'N' characters, skipping reverse complement calculation")
        return log_likelihood

    sequence_tensor = torch.tensor(sequence_tokens, dtype=torch.long)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    reverse_complement_sequence = reverse_complement(sequence_tensor).tolist()
    reverse_complement_scores = scores_tensor.flip(dims=[0]).tolist()
    
    reverse_log_likelihood = calculate_log_likelihood(
        reverse_complement_sequence, reverse_complement_scores, model, collator, device, mutation_pos
    )
    
    return (log_likelihood + reverse_log_likelihood) / 2


def process_variants(name, bigwig_file, genome_fasta, model, collator, tokenizer, device):
    genome = Fasta(genome_fasta)

    df = pd.read_parquet("hf://datasets/songlab/clinvar/test.parquet")

    bw = pyBigWig.open(bigwig_file)
    common_log_likelihood_ratios = []
    pathogenic_log_likelihood_ratios = []
    valid_chromosomes = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX".split()
    non_matching =0
    total = len(df)
    print(f"Total number of variants: {total}")
    #df = df[df['label'].isin(['Common', 'Pathogenic'])]
    #get first 100
    df = df.head(100)
    print(f"Total number of subsetted variants: {len(df)}")
    for index, row in df.iterrows():
        #Lifted_Chr,Lifted_Position
        chromosome = "chr" + str(row['chrom'])
        if chromosome not in valid_chromosomes:
            print(f"Chromosome {chromosome} is not in valid chromosomes")
            continue
        #make sure chromosome in valid chromosomes
        position = int(row['pos'])
        label = row['label']
        ref = row['ref']
        alt = row['alt']
        # chromosome = "chr" + row['chrom']
        # position = row['pos']
        # ref = row['ref']
        # alt = row['alt']
        # label=row['label']
        start = position - 1024
        end = position + 1024

        # Get the reference sequence
        ref_sequence = genome[chromosome][start:end].seq
        print(f"reference sequence length: {len(ref_sequence)}")

        # Check if the reference allele matches the sequence at the specified position
        if ref_sequence[position - start -1 ] != ref:
            non_matching += 1
            print(f"Reference allele does not match at {chromosome}:{position}. Expected {ref}, found {ref_sequence[position - start -1 ]}")

        #HERE KEEP COUNTER FOR ONLY 100 OF COMMON & PATHOGENIC USING common_log_likelihood_ratios
        if label == 'Common' and len(common_log_likelihood_ratios) >= 100:
            continue
        if label == 'Pathogenic' and len(pathogenic_log_likelihood_ratios) >= 100:
            continue
            
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

        # Ensure mutation_pos is within bounds
        mutation_pos = 1024
        if mutation_pos >= len(ref_sequence_tokens):
            print(f"Mutation position {mutation_pos} is out of bounds for sequence length {len(ref_sequence_tokens)}")
            continue

        # Calculate average log-likelihood for the reference sequence at the mutation position
        ref_log_likelihood = calculate_average_log_likelihood(ref_sequence_tokens, scores, model, collator, device, mutation_pos)

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

        # Ensure mutation_pos is within bounds for the mutated sequence
        if mutation_pos >= len(mutated_sequence_tokens):
            print(f"Mutation position {mutation_pos} is out of bounds for mutated sequence length {len(mutated_sequence_tokens)}")
            continue
        
        print("Calculating log-likelihood ratio for", chromosome, position, ref, ">", alt)
         # Calculate average log-likelihood for the mutated sequence at the mutation position
        mutated_log_likelihood = calculate_average_log_likelihood(mutated_sequence_tokens, scores, model, collator, device, mutation_pos)

        # Calculate log-likelihood ratio
        log_likelihood_ratio = mutated_log_likelihood - ref_log_likelihood
        
        # Append to the appropriate list 
        if label == 0:
            common_log_likelihood_ratios.append(log_likelihood_ratio)
        elif label == 1:
            pathogenic_log_likelihood_ratios.append(log_likelihood_ratio)

    print("length of common_log_likelihood_ratios:", len(common_log_likelihood_ratios))
    print("length of pathogenic_log_likelihood_ratios:", len(pathogenic_log_likelihood_ratios))
    print(f"the total number of non-matching reference alleles is {non_matching} out of {total}, or {non_matching/total*100}%")
    return common_log_likelihood_ratios, pathogenic_log_likelihood_ratios

def plot_log_likelihood_ratios(common_log_likelihood_ratios, pathogenic_log_likelihood_ratios, output_file, mutation_file):
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for benign scores
    plt.hist(common_log_likelihood_ratios, bins=50, alpha=0.5, label='Common', color='blue')
    
    # Plot histogram for pathogenic scores
    plt.hist(pathogenic_log_likelihood_ratios, bins=50, alpha=0.5, label='Pathogenic', color='red')
    
    plt.xlabel('Log-Likelihood Ratio')
    plt.ylabel('Frequency')
    plt.title(f'Log-Likelihood Ratio for {mutation_file} Variants')
    plt.legend(loc='upper right')
    plt.savefig(output_file)
    print(f"File saved to {output_file}")


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
    parser.add_argument('--name', type=str, default ="clinvar", help='type of clinvar variant to use')
    parser.add_argument('--genome_fasta', type=str,  default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa', help='Path to the genome FASTA file')
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    parser.add_argument('--output_file', type=str, default='/home/mica/gamba/data_processing/data/VEP/', help='Path to the output file')
    parser.add_argument('--config_fpath', type=str,  default='/home/mica/gamba/configs/jamba-small-240mammalian.json', help='Path to the config file')
    args = parser.parse_args()

    #strip name of mutation file of specific mutation (i.e. whats before _mutations):
    mutation_file = args.name
    print(f"Processing {mutation_file} mutations")
    #change output file name to append mutation file name
    output_file = os.path.join(args.output_file, f"{mutation_file}_loglikelihood.png")

    #get checkpoint path with step=5400
    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, 78000)

    
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

    common_log_likelihood_ratios, pathogenic_log_likelihood_ratios = process_variants(args.name, args.big_wig, args.genome_fasta, model, collator, tokenizer, device)
    plot_log_likelihood_ratios(common_log_likelihood_ratios, pathogenic_log_likelihood_ratios, output_file, mutation_file)

    #we want to get log-likelihood ratio between alternate and reference allele and do predictions from both strands were averaged.
    # and do:    get log-likelihood ratio between alternate and reference position and do predictions from the sequence and the reverse complement and  average the log-lieklihood.
if __name__ == "__main__":
    main()