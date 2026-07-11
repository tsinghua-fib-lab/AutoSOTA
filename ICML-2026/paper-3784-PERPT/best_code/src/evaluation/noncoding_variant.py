import argparse
import datetime
import functools
import json
import os
import random
import glob
import logomaker
from typing import Optional, Sequence, Tuple, Type
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import wandb
import pyBigWig
import pandas as pd
import os
import argparse
import numpy as np
from pyfaidx import Fasta
import json
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr, warmup

import torch.nn.functional as F 
from evodiff.utils import Tokenizer
# import gamba using sys.append
import sys

sys.path.append("../gamba")
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
import logging
_logger = logging.getLogger(__name__)
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel

import matplotlib.pyplot as plt
import seaborn as sns

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

def sequence_output(
    bigwig_file: str,
    genome_fasta: str,
    chromosome: str,
    start: int,
    end: int, 
    save_path: str,
    checkpoint: str,
    config_fpath: str,
    masked: bool = False,
    mutation_pos: int = None,
    mutation: str = None
):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    if masked:
        last_step=16000
    else:
        last_step=54000

    # get the latest checkpoint path
    ckpt_path = get_latest_dcp_checkpoint_path(checkpoint, last_step=last_step)

    with open(config_fpath, "r") as f:
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
    
    print("Using standard checkpoint loading...", flush=True)
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

    # get the sequence from the genome
    sequence = genome[chromosome][start:end].seq

    #print("sequence:", sequence)
    print("position 1161:1163 in the sequence is:", sequence[1160:1163])

    # get the conservation scores from the bigwig file
    scores = bw.values(chromosome, start, end)

    # round scores to 2 decimal places
    scores = np.round(scores, 2)
    true_scores = scores.copy()

    print("conservation scores at position 1160:1163 are:", scores[1160:1163])

     print(f"start and end: {start} and {end}")

    # Tokenize the unmutated sequence
    sequence_tokens = tokenizer.tokenizeMSA(sequence)

    print("the first 10 TOKENIZED chars of the sequence are:", sequence_tokens[:10])
    print("the first 10 conservation scores are:", scores[:10])

    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config.get("pad_to_multiple_of", None),
    )

    # move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set the model to eval mode
    model.eval()

    # Function to calculate perplexity for the whole sequence
    def calculate_perplexity(sequence_tokens, scores):
        collated = collator([(sequence_tokens, scores)])
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        logits = output["seq_logits"]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs), dim=-1)).mean().item()
        return perplexity

    # Function to calculate perplexity at a specific position
    def calculate_perplexity_at_position(sequence_tokens, scores, position):
        collated = collator([(sequence_tokens, scores)])
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        logits = output["seq_logits"][:, position, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs), dim=-1)).item()
        return perplexity

    # 1. Unmutated sequence perplexity
    unmutated_perplexity = calculate_perplexity(sequence_tokens, scores)
    #print(f"Perplexity of unmutated sequence at position: {unmutated_perplexity}")

    # 2. Sequence with the specified mutation
    if mutation_pos is not None and mutation is not None:
        mutated_sequence = list(sequence)
        print(f"mutated sequence at position {mutation_pos} is {mutated_sequence[mutation_pos]}")
        mutated_sequence[mutation_pos] = mutation
        mutated_sequence_tokens = tokenizer.tokenizeMSA("".join(mutated_sequence))
        mutated_perplexity = calculate_perplexity(mutated_sequence_tokens, scores)
        #print(f"Perplexity at specified mutated position {mutation_pos}: {mutated_perplexity}")

    # 3. Sequence with a random base pair mutation
    random_pos = random.randint(0, len(sequence) - 1)
    random_mutation = random.choice([nuc for nuc in "ATGC" if nuc != sequence[random_pos]])
    random_mutated_sequence = list(sequence)
    random_mutated_sequence[random_pos] = random_mutation
    random_mutated_sequence_tokens = tokenizer.tokenizeMSA("".join(random_mutated_sequence))
    random_mutation_perplexity = calculate_perplexity(random_mutated_sequence_tokens, scores)##)
    #print(f"Perplexity at random mutated position {random_pos}: {random_mutation_perplexity}")

    # Calculate average perplexity across positions in the sequence
    average_perplexity = calculate_perplexity(sequence_tokens, scores)
    print(f"Average perplexity across positions in the sequence: {average_perplexity}")

    # Calculate perplexity at position 1160
    perplexity_at_1160 = calculate_perplexity_at_position(sequence_tokens, scores, 1160)
    print(f"Perplexity at position 1160: {perplexity_at_1160}")

    # Function to calculate the representation of the sequence
    # Function to calculate the representation of the sequence
    def get_representation(sequence_tokens, scores):
        collated = collator([(sequence_tokens, scores)])
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        representation = output["representation"].cpu().numpy() 
        print("representation.shape:", representation.shape)
        return representation.squeeze()

    # Function to calculate cosine distance
    def cosine_distance(a, b):
        return 1 - cosine_similarity(a, b)

    unmutated_representation = get_representation(sequence_tokens, scores)
    mutated_representation = get_representation(mutated_sequence_tokens, scores)
    random_mutated_representation = get_representation(random_mutated_sequence_tokens, scores)

   
    cosine_distance_specific_to_normal = cosine_distance(unmutated_representation, mutated_representation).mean()
    cosine_distance_random_to_normal = cosine_distance(unmutated_representation, random_mutated_representation).mean()
    # random_distances = []
    # for _ in range(100):
    #     random_pos = random.randint(0, len(sequence) - 1)
    #     random_mutation = random.choice([nuc for nuc in "ATGC" if nuc != sequence[random_pos]])
    #     random_mutated_sequence = list(sequence)
    #     random_mutated_sequence[random_pos] = random_mutation
    #     random_mutated_sequence_tokens = tokenizer.tokenizeMSA("".join(random_mutated_sequence))
    #     random_mutation_representation = get_representation(random_mutated_sequence_tokens, scores)
    #     distance = cosine_distance(unmutated_representation, random_mutation_representation).mean()
    #     random_distances.append(distance)

    
    #average_cosine_distance_random_to_normal = np.mean(random_distances)

    #print(f"Cosine distance from random mutation to normal (100 average): {average_cosine_distance_random_to_normal}")
    print(f"Cosine distance from random mutation to normal: {cosine_distance_random_to_normal}")
    print(f"Cosine distance from specific mutation to normal: {cosine_distance_specific_to_normal}")


    # Check accuracy of the model at predicting the conservation score at position 1160
    collated = collator([(sequence_tokens, scores)])
    with torch.no_grad():
        output = model(collated[0].to(device), collated[1].to(device))
    predicted_scores = output["scaling_logits"].cpu().numpy()
    predicted_score_at_1160 = predicted_scores[0, 1160, 0]  # Assuming the first dimension is batch size
    true_score_at_1160 = true_scores[1160]

    print(f"Predicted conservation score at position 1160: {predicted_score_at_1160}")
    print(f"True conservation score at position 1160: {true_score_at_1160}")

    return unmutated_perplexity, mutated_perplexity, random_mutation_perplexity, average_perplexity, perplexity_at_1160

def save_predicted_scores_as_bigwig(
    chrom: str,
    start: int,
    predicted_scores,
    output_path: str,
    chrom_sizes_path: str,
):
    # load chromosome sizes from a file (e.g., hg38.chrom.sizes)
    chrom_sizes = {}
    with open(chrom_sizes_path, "r") as f:
        for line in f:
            chrom_name, size = line.strip().split()
            chrom_sizes[chrom_name] = int(size)

    # bigWig file
    bw = pyBigWig.open(output_path, "w")
    bw.addHeader([(chrom, chrom_sizes[chrom])])
    print("predicted_scores:", predicted_scores)

    positions = np.arange(start, start + len(predicted_scores))
    chrom_ends = positions + 1  # half-open intervals [start, end)
    bw.addEntries([chrom] * len(predicted_scores), positions, ends=chrom_ends, values=predicted_scores)

    # close bigwig
    bw.close()
    print(f"Saved bigWig file to {output_path}")

def main():
    # process command line arguments
    parser = argparse.ArgumentParser(
        description="Generate data files for training, testing, and validation sets"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/results.pdf",
        help="Path to save the results figure",
    )
    parser.add_argument(
        "--chromosome",
        type=str,
        default="chr1",
        help="Chromosome to analyze",
    )
    parser.add_argument(
        "--start_pos",
        type=int,
        default=16513122, 
        help="Chromosome start pos to analyze",
    )
    parser.add_argument(
        "--end_pos",
        type=int,
        default=16515170, 
        help="Chromosome end pos to analyze",
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Path to the chromosome sizes file",
    )
    parser.add_argument(
        "--masked",
        type=str,
        default=False,
        help="Run the model with masked conservation scores",
    )
    parser.add_argument(
        "--mutation_pos",
        type=int,
        default=1160,
        help="Position to introduce the mutation",
    )
    parser.add_argument(
        "--mutation",
        type=str,
        default="G",
        help="Mutation to introduce",
    )

    #U1 noncoding variant at: 16,514,285 to 16,514,122
    # the gene is in the reverse complement so when we say 3rd position its third position from the end
    # 16,514,285-2 = 16,514,283
    # the last three nucleotides are: TAT and the mutation will be GAT
    #we'll turn the third last position here to a G
    # 16,514,285- 16,514,122 = 163 bp long
    # the mutation will be at position 163-2 = 161
    #or in the sequence that is 16,514,122 + 161 = 16,514,283
    #let's take 1000bp upstream from 16,514,122 = 16,513,122
    #end at position 16,513,122 + 2048 = 16,515,170
    # the mutation is now 16,513,122 + 1000 + 161 = 16,514,283
    # or posiition 1161 in the sequence


    args = parser.parse_args()

    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 

    unmutated_perplexity, mutated_perplexity, random_mutation_perplexity, average_perplexity, perplexity_at_1160 = sequence_output(
        args.bigwig_file,
        args.genome_fasta,
        args.chromosome,
        args.start_pos,
        args.end_pos,
        args.save_path,
        ckpt_dir,
        '/home/mica/gamba/configs/jamba-small-240mammalian.json',
        args.masked,
        args.mutation_pos,
        args.mutation
    )
    

    print(f"Perplexity of sequence with mutated position {args.mutation_pos}: {mutated_perplexity}")
    print(f"Perplexity of sequence with random mutated position: {random_mutation_perplexity}")
    print(f"Perplexity of unmutated sequence: {unmutated_perplexity}")
    print(f"Average perplexity across positions in the sequence: {average_perplexity}")
    print(f"Perplexity at position 1160: {perplexity_at_1160}")


if __name__ == "__main__":
    main()