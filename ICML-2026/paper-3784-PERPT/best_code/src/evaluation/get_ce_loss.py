import argparse
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append("../gamba")
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel
from evodiff.utils import Tokenizer
import numpy as np
import os
from torch.utils.data import Dataset
import random

class OppositeRegionDataset(Dataset):
    def __init__(self, sequence_path, conservation_path, max_len=2048, num_sequences=10000, filter_N=True, seed=None):
        self.sequence_path = sequence_path
        self.conservation_path = conservation_path
        self.max_len = max_len
        self.num_sequences = num_sequences
        self.filter_N = filter_N

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load full 1D arrays via memory mapping
        self.sequence_array = np.load(sequence_path, mmap_mode='r')
        self.conservation_array = np.load(conservation_path, mmap_mode='r')
        self.chrom_size = len(self.sequence_array)

        # Sample valid positions
        self.indices = self._sample_valid_positions()

    def _sample_valid_positions(self):
        indices = []
        attempts = 0
        while len(indices) < self.num_sequences and attempts < self.num_sequences * 10:
            start = np.random.randint(0, self.chrom_size - self.max_len)
            seq_window = self.sequence_array[start : start + self.max_len]
            if self.filter_N and np.count_nonzero(seq_window == 4) > 0.1 * self.max_len:
                attempts += 1
                continue
            indices.append(start)
        if len(indices) < self.num_sequences:
            print(f"⚠️ Only sampled {len(indices)} valid windows (target was {self.num_sequences})")
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        seq = self.sequence_array[start : start + self.max_len]
        cons = self.conservation_array[start : start + self.max_len]
        cons = np.round(cons, 2)
        return seq, cons


def compute_ce_loss(model, loader, device):
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            seq_tensor, sc_tensor = batch
            seq_tensor = seq_tensor.to(device)
            sc_tensor = sc_tensor.to(device)

            outputs = model(seq_tensor, sc_tensor)
            ce_loss = outputs["cross_entropy_loss"].item()
            n_tokens = outputs.get("n_tokens", (sc_tensor[:, 0, :] != 0).sum().item())

            total_loss += ce_loss * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate CE loss on excluded data.")
    parser.add_argument("--seq_path", default="/home/mica/gamba/data_processing/data/240-mammalian/opposite_data_chr2_sequence.npy", type=str)
    parser.add_argument("--score_path", default="/home/mica/gamba/data_processing/data/240-mammalian/opposite_data_chr2_score.npy", type=str)
    parser.add_argument("--ckpt_dir", default="/home/mica/gamba/clean_dcps/", type=str)
    parser.add_argument("--config_path", default="/home/mica/gamba/configs/jamba-small-240mammalian.json", type=str)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    # Load config
    with open(args.config_path) as f:
        config = json.load(f)

    task = TaskType(config["task"].lower().strip())
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    # Init model
    model_core, _ = create_model(task, config["model_type"], config["model_config"], tokenizer.mask_id.item())
    model = JambagambaModel(
        model_core,
        d_model=config.get("d_model", 512),
        nhead=config.get("n_head", 8),
        n_layers=config.get("n_layers", 6),
        padding_id=config.get("padding_id", 0),
        dim_feedfoward=config.get("dim_feedforward", 512),
    )

    # Load checkpoint
    ckpt_path = "/home/mica/gamba/clean_dcps/dcp_56000"
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup collator and loader
    dataset = OppositeRegionDataset(args.seq_path, args.score_path, max_len=2048, num_sequences=10000)
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    collator = gLMCollator(tokenizer=tokenizer)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    avg_ce_loss = compute_ce_loss(model, loader, device)
    print(f"\n✅ Average CE Loss on {args.max_samples} excluded samples: {avg_ce_loss:.6f}")


if __name__ == "__main__":
    main()
