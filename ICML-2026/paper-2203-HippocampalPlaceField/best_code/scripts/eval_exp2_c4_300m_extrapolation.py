import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
import random
import numpy as np
import wandb
import subprocess
import sys
from torch.amp import autocast
from transformers import AutoTokenizer

# Imports from OLMo
from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo
from datasets import load_from_disk


def create_dataloader(tokenized_ds, seq_len, batch_size, shuffle=False, desc="Grouping"):
    block_size = seq_len + 1
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    
    grouped_ds = tokenized_ds.map(group_texts, batched=True, num_proc=8, desc=desc)
    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        x = data[:, :-1].contiguous() 
        y = data[:, 1:].contiguous()  
        return x, y
        
    return DataLoader(grouped_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def run_evaluation(model, dataloader, vocab_size, device):
    model.eval()
    total_val_loss = 0.0
    total_val_tokens = 0
    
    with torch.no_grad():
        for vx, vy in dataloader:
            vx, vy = vx.to(device), vy.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=vx)
                loss = nn.functional.cross_entropy(
                    outputs.logits.view(-1, vocab_size), 
                    vy.view(-1), 
                    reduction='sum'
                )
            total_val_loss += loss.item()
            total_val_tokens += vy.numel() 
            
    avg_val_loss = total_val_loss / total_val_tokens if total_val_tokens > 0 else 0.0
    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 1e9 
    return avg_val_loss, val_ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arxiv_path", type=str, default="/data/xxxxxxxxx/03-proj/PE/arxiv_data/arxiv_validation")
    args = parser.parse_args()

    lengths = [512, 1024, 2048, 4096]
    results = {}

    for L in lengths:
        cfg = ModelConfig(..., max_sequence_length=L, yarn_enabled=True, 
                          yarn_max_position_embeddings=512, yarn_target_max_position_embeddings=L)
        model = OLMo(cfg).cuda()
        model.load_state_dict(torch.load(args.ckpt))
        
        loader = create_dataloader(load_from_disk(args.arxiv_path), L, batch_size=1)
        _, ppl = run_evaluation(model, loader, vocab_size, "cuda")
        results[L] = ppl
        print(f"Length {L} | Arxiv PPL: {ppl:.4f}")