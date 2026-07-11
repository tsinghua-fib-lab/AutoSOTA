#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
from pathlib import Path
from scipy import stats
sys.path.append("../gamba")
from torch.nn import MSELoss, CrossEntropyLoss
from sequence_models.constants import MSA_PAD, START, STOP
from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator, gLMMLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)

from Bio.Seq import Seq
#import Counter
from collections import Counter

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import os
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn


def get_latest_dcp_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint path."""
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

import torch.nn as nn

def vishniakov_init(module):
    # Linear / Embedding: N(0, 0.02), bias=0
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    # LayerNorm: gamma=1, beta=0
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def load_model(
    checkpoint_dir,
    config_fpath,
    last_step=44000,
    model_type="gamba",
    training_task="dual",
    device=None
):
    """Load Gamba or Caduceus model depending on model_type and training_task."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    if model_type == "gamba":
        # Resolve checkpoint path
        if training_task == "dual":
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_{last_step}")
        elif training_task == "seq_only":
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_nocons_{last_step}")
        elif training_task == "cons_only":
            ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, f"noALM{last_step}")

        # Load config
        with open(config_fpath, "r") as f:
            config = json.load(f)

        task = TaskType(config["task"].lower().strip())

        logging.info(f"Gamba Model | Task: {task}, Type: {config['model_type']}")

        # Create base model
        base_model, _ = create_model(
            task,
            config["model_type"],
            config["model_config"],
            tokenizer.mask_id.item(),
        )

        # Set up Gamba model wrapper
        d_model = config.get("d_model", 512)
        nhead = config.get("n_head", 8)
        n_layers = config.get("n_layers", 6)
        dim_feedforward = config.get("dim_feedforward", d_model)
        padding_id = config.get("padding_id", 0)

        if training_task == "dual":
            model = JambagambaModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )
        elif training_task == "seq_only":
            model = JambaGambaNoConsModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )
        else:
            model = JambaGambaNOALMModel(
                base_model,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                padding_id=padding_id,
                dim_feedfoward=dim_feedforward,
            )

        # Load checkpoint
        if last_step==0:
            logging.info("Loading Gamba with random initialization")
            model.apply(vishniakov_init) 
        else:
            logging.info(f"Loading Gamba trained on {training_task} from {ckpt_path}")
            checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            

        model.to(device)
        model.eval()
        return model, tokenizer

    elif model_type == "caduceus":

        config = CaduceusConfig(
            d_model=256,
            n_layer=8,
            vocab_size=len(DNA_ALPHABET_PLUS)
        )

        if training_task == "dual":
            model = CaduceusConservationForMaskedLM(config)
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_conscaduceus_{last_step}")
        elif training_task == "seq_only":
            model = CaduceusForMaskedLM(config)
            ckpt_path = os.path.join(checkpoint_dir, f"dcp_caduceus_{last_step}")
        elif training_task == "cons_only":
            model = CaduceusConservation(config)
            ckpt_path = get_latest_dcp_checkpoint_path(checkpoint_dir, f"consONLYcaduceus_{last_step}")

        if last_step==0:
            logging.info("Loading Caduceus with random initialization")
            model.apply(vishniakov_init) 
        else:
            logging.info(f"Loading Caduceus checkpoint from {ckpt_path}")
            checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)
        model.eval()
        return model, tokenizer

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None,
                            model_type="gamba", training_task="dual"):
    """Extract full hidden state representations for each region without masking.
    Filters and appends region_info only after the batch has been processed successfully.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_hidden_states = []
    all_region_info = []

    logging.info(f"Extracting representations for {len(regions)} regions with batch size {batch_size}...")

    if model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

    for i in tqdm(range(0, len(regions), batch_size), desc="Batch encoding"):
        batch_regions = regions[i:i + batch_size]
        batch_inputs = []
        batch_region_info = []

        # ---- Build batch-local info ----
        for region in batch_regions:
            seq_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']

            # Skip completely empty or None sequences
            if seq_tokens is None or len(seq_tokens) == 0:
                logging.warning(f"Skipping region {region['chrom']}:{region['start']}-{region['end']} due to empty sequence.")
                continue
            fs = region.get('feature_start_in_window', 0)
            fe = region.get('feature_end_in_window', len(scores))

            batch_inputs.append((seq_tokens, scores))
            batch_region_info.append({
                'chrom': region['chrom'],
                'start': region['start'],
                'end': region['end'],
                'feature_id': region.get('feature_id', 'unknown'),
                'mean_score': region.get('mean_score', 0.0),
                'feature_start_in_window': fs,
                'feature_end_in_window': fe
            })

        if not batch_inputs:
            continue  # skip empty batch

        # ---- Run model for this batch ----
        if model_type == "gamba":
            collated = collator(batch_inputs)
            with torch.no_grad():
                outputs = model(collated[0].to(device), collated[1].to(device))

            if "representation" in outputs:
                hs = outputs["representation"]  # (B, T, D)
                all_hidden_states.extend(hs.cpu().numpy())
                all_region_info.extend(batch_region_info)
            else:
                # still extend both lists so they stay in sync
                for _ in range(len(batch_inputs)):
                    all_hidden_states.append(
                        np.full((collated[0].shape[1], model.config.d_model), np.nan)
                    )
                all_region_info.extend(batch_region_info)

        elif model_type == "caduceus":
            feature_spans = [(r["feature_start_in_window"], r["feature_end_in_window"])
                             for r in batch_region_info]
            batch = collator(batch_inputs, region=feature_spans)
            with torch.no_grad():
                sequence_input = batch[0][:, 0, :].long()  # (B, T)
                outputs = model(input_ids=sequence_input.to(device),
                                output_hidden_states=True)

            if "hidden_states" in outputs:
                hs_all = outputs["hidden_states"]
                hs = hs_all[-1]  # final layer (B, T, D)
                for j in range(hs.shape[0]):
                    all_hidden_states.append(hs[j].cpu().numpy())
                all_region_info.extend(batch_region_info)
            else:
                for _ in range(len(batch_inputs)):
                    all_hidden_states.append(
                        np.full((sequence_input.shape[1], model.config.d_model), np.nan)
                    )
                all_region_info.extend(batch_region_info)

    return all_hidden_states, all_region_info


# def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None, model_type="gamba", training_task="dual"):
#     """Extract full hidden state representations for each region without masking."""
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     all_hidden_states = []
#     region_info = []

#     logging.info(f"Extracting representations for {len(regions)} regions with batch size {batch_size}...")

#     if model_type == "gamba":
#         collator = gLMCollator(tokenizer=tokenizer, test=True)
#     else:
#         collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

#     for i in tqdm(range(0, len(regions), batch_size), desc="Batch encoding"):
#         batch_regions = regions[i:i + batch_size]
#         batch_inputs = []
#         batch_region_info = []

#         for region in batch_regions:
#             sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
#             scores = region['scores']
#             fs = region.get('feature_start_in_window', 0)
#             fe = region.get('feature_end_in_window', len(scores))

#             batch_inputs.append((sequence_tokens, scores))
#             batch_region_info.append({
#                 'chrom': region['chrom'],
#                 'start': region['start'],
#                 'end': region['end'],
#                 'feature_id': region.get('feature_id', 'unknown'),
#                 'mean_score': region.get('mean_score', 0.0),
#                 'feature_start_in_window': fs,
#                 'feature_end_in_window': fe
#             })
#             region_info.append(batch_region_info[-1])

#         if not batch_inputs:
#             continue

#         if model_type == "gamba":
#             collated = collator(batch_inputs)
#             with torch.no_grad():
#                 outputs = model(collated[0].to(device), collated[1].to(device))

#             if "representation" in outputs:
#                 hs = outputs["representation"]  # (B, T, D)
#                 all_hidden_states.extend(hs.cpu().numpy())  # Directly append the hidden states
#             else:
#                 for _ in range(len(batch_inputs)):
#                     all_hidden_states.append(np.full((collated[0].shape[1], model.config.d_model), np.nan))

#         elif model_type == "caduceus":
#             feature_spans = [(r["feature_start_in_window"], r["feature_end_in_window"]) for r in batch_region_info]
#             batch = collator(batch_inputs, region=feature_spans)
#             with torch.no_grad():
#                 sequence_input = batch[0][:, 0, :].long()  # (B, T)
#                 model_kwargs = {
#                     "input_ids": sequence_input.to(device),
#                     "output_hidden_states": True
#                 }
#                 outputs = model(**model_kwargs)

#             if "hidden_states" in outputs:
#                 hs_all = outputs["hidden_states"]  # list of (B, T, D)
#                 hs = hs_all[-1]  # final layer (B, T, D)

#                 for j in range(hs.shape[0]):
#                     all_hidden_states.append(hs[j].cpu().numpy())

#             else:
#                 for _ in range(len(batch_inputs)):
#                     all_hidden_states.append(np.full((sequence_input.shape[1], model.config.d_model), np.nan))

#     return all_hidden_states, region_info

