import argparse
import datetime
import functools
import json
import os
import random
import glob
import logomaker
from typing import Optional, Sequence, Tuple, Type


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


#want to load chr16:23,683,829-23,683,929 as a test sequence from the human genome
#load the sequence and tokenize it
#then run the model on it
#then plot the sequence and conservation scores

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
    masked: bool = False
):
    # open the bigwig file
    bw = pyBigWig.open(bigwig_file)

    # open the genome fasta file
    genome = Fasta(genome_fasta)

    if masked:
        last_step=16000
    else:
        last_step=52000 

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

    print("sequence:", sequence)

    print(f"start and end: {start} and {end}")

    # get the conservation scores from the bigwig file
    scores = bw.values(chromosome, start, end)

    # round scores to 2 decimal places
    scores = np.round(scores, 2)
    true_scores = scores.copy()

    # tokenize the sequence
    sequence_tokens = tokenizer.tokenizeMSA(sequence)

    print("the first 10 TOKENIZED chars of the sequence are:", sequence_tokens[:10])
    print("the first 10 conservation scores are:", scores[:10])

    collator = gLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config.get("pad_to_multiple_of", None),
    )


    #move device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set the model to eval mode
    model.eval()
    
    if masked:
        #lets set scores to all be  -100 mask
        masked_scores = np.full_like(scores, -100)
        print("MASKED VERSION RUNNING!!!")
        #put sequence through gLM collator
        collated = collator([(sequence_tokens, masked_scores)])
        #need to run the model on the sequence, one forward pass per token
        print("collated[0].shape:", collated[0].shape)
        print("collated[1].shape:", collated[1].shape)
        # Initialize the output sequence with the input sequence
        sequence_input = collated[0][:, 0, :].clone()
        conservation_input = collated[0][:, 1, :].clone()
        sequence_target = collated[1][:, 0, :].clone()
        conservation_target = collated[1][:, 1, :].clone()


        print("sequence_input:", sequence_input.shape)
        print("conservation_input:", conservation_input.shape)
        print("sequence_target:", sequence_target.shape)
        print("conservation_target:", conservation_target.shape)

        
        # Run the forward pass and build the sequence token by token, excluding start and stop & padding tokens
        for i in range(2048):
            print("we are on token:", i, "of length:", sequence_input.shape[1])
            # Run the model's forward pass on the current sequence
            # put the sequence_input, conservation_input, targets back together
            # Concatenate the inputs and targets along the appropriate dimension
            inputs = torch.stack([sequence_input, conservation_input], dim=1)
            targets = torch.stack([sequence_target, conservation_target], dim=1)

            # Move the inputs and targets to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Run the model with the concatenated inputs and targets
            with torch.no_grad():
                output = model(inputs, targets)
            
            # Get the logits for the current token position
            seq_logits = output["seq_logits"][:, i, :]
            scaling_logits = output["scaling_logits"][:, i, :]
            
            # Sample a token from the logits
            sampled_token = torch.multinomial(torch.nn.functional.softmax(seq_logits, dim=-1), 1)
            
            # Extract means and variances
            means = scaling_logits[:, 0].cpu().numpy()
            variances = scaling_logits[:, 1].exp().cpu().numpy()  # Convert log-variance to variance

            # Sample from the predicted distribution for each position
            sampled_score = np.random.normal(loc=means, scale=np.sqrt(variances), size=means.shape)

            print("sampled token:", sampled_token.shape)
            print("sampled score:", sampled_score.shape)
            
            # Update the output sequence with the sampled token
            sequence_input[:, i] = sampled_token.squeeze()

            # Update the output scores with the sampled score
            conservation_input[:, i] = torch.tensor(sampled_score, device=conservation_input.device)

        # Convert the output sequence and scores to numpy arrays
        output_sequence_np = sequence_input.cpu().detach().numpy()
        output_scores_np = conservation_input.cpu().detach().numpy()
        means = output_scores_np.squeeze(0)
        print("shape of means:", means.shape)

    else:
        collated = collator([(sequence_tokens, scores)])
        # run the model on the sequence
        with torch.no_grad():
            output = model(collated[0].to(device), collated[1].to(device))
        #get the logit outputs from the model
        seq_logits = output["seq_logits"]
        scaling_logits = output["scaling_logits"]
        # Remove batch dimension and trim to target length by removing tokens evenly from each end to get to 2048
        scaling_logits = scaling_logits.squeeze(0)
        if scaling_logits.size(0) > 2048:
            scaling_logits = scaling_logits[:2048]  # Limits to 2048 positions if tensor is larger
        elif scaling_logits.size(0) < 2048:
            raise ValueError("The sequence length is less than 2048 positions.")
        print("Shape of scaling_logits after squeeze and trim:", scaling_logits.shape)  # Should be (2048, 2)

        # Extract means and variances
        means = scaling_logits[:, 0].cpu().numpy()
        variances = scaling_logits[:, 1].exp().cpu().numpy()  # Convert log-variance to variance

        # Sample from the predicted distribution for each position
        samples = np.random.normal(loc=means, scale=np.sqrt(variances), size=len(means))

    #cut scores to the first 2048:
    true_scores = true_scores[:2048]
    means= means[:2048]
    print("length of true scores:", len(true_scores))
    print("length of means:", len(means))

    # Calculate the correlation
    correlation = np.corrcoef(true_scores, means)[0, 1]
    print(f"Correlation between true and predicted scores: {correlation}")


    fig, ax = plt.subplots(figsize=(10, 8))

    # Predicted scores with variance shading
    ax.plot(means, label="Predicted Mean Scores", color="blue")

    # True conservation scores as green dots
    ax.plot(true_scores, 'go', label="True Conservation Scores", markersize=2)

    
    ax.fill_between(
        range(len(means)),
        means - np.sqrt(variances),
        means + np.sqrt(variances),
        color="gray",
        alpha=0.3,
        label="Predicted Std Dev",
    )

    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Score")
    ax.set_title("True Conservation Scores and Predicted Distribution by Position")
    ax.legend()

    # Set x-axis labels to positions in sequence from start to end
    ax.set_xticks(range(0, len(true_scores), 200))
    ax.set_xticklabels(range(start, end, 200))

    plt.tight_layout()
    if masked:
        plt.savefig(f"{save_path}_masked_true_vs_predicted_with_highlights.svg", format="svg")
        plt.savefig(f"{save_path}_masked_true_vs_predicted_with_highlights.png")
    else:
        plt.savefig(f"{save_path}_true_vs_predicted_with_highlights.svg", format="svg")
        plt.savefig(f"{save_path}_true_vs_predicted_with_highlights.png")
    plt.show()

    
    
    return means

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
        description="Make bigwig files"
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
        default="chr16",
        help="Chromosome to analyze",
    )
    # parser.add_argument(
    #     "--start_pos",
    #     type=int,
    #     default=23678000, 
    #     help="Chromosome start pos to analyze",
    # )
    # parser.add_argument(
    #     "--end_pos",
    #     type=int,
    #     default=23680048, 
    #     help="Chromosome end pos to analyze",
    # )
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

    args = parser.parse_args()

    ckpt_dir = os.getenv("AMLT_OUTPUT_DIR", "/tmp/") 

    predicted_scores = sequence_output(
        args.bigwig_file,
        args.genome_fasta,
        args.chromosome,
        args.start_pos,
        args.end_pos,
        args.save_path,
        ckpt_dir,
        '/home/mica/gamba/configs/jamba-small-240mammalian.json',
        args.masked
    )

    save_predicted_scores_as_bigwig(
        args.chromosome,
        args.start_pos,
        predicted_scores,
        "/home/mica/gamba/data_processing/data/240-mammalian/predicted_scores.bw",
        args.chrom_sizes,
    )


if __name__ == "__main__":
    main()





