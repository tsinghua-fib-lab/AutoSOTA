import os
import multiprocessing
import polars as pl
import torch
import pytorch_lightning as pl_lightning
from torch.utils.data import Dataset, DataLoader
import numpy as np

def parse_fasta(file_path):
    """Parses a FASTA/A2M file into a list of dicts."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header:
                    sequences.append({
                        "id": current_header[1:].split()[0], 
                        "sequence": "".join(current_seq)
                    })
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_header:
            sequences.append({
                "id": current_header[1:].split()[0],
                "sequence": "".join(current_seq)
            })
    return sequences

class PolarsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        return {"Sequence": row["sequence"], "Entry": row["id"]}

class SequenceDataModule(pl_lightning.LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count() - 1
        base, ext = os.path.splitext(self.data_path)
        if ext in [".a2m", ".fasta", ".fa"]:
            self.real_data_path = base + ".parquet"
            self.needs_conversion = True
        else:
            self.real_data_path = self.data_path
            self.needs_conversion = False

    def prepare_data(self):
        """
        Checks if the input is .a2m or .fasta. If so, checks if a .parquet version exists.
        If not, converts it.
        """
        if self.needs_conversion:
            if not os.path.exists(self.real_data_path):
                print(f"Conversion needed: {self.data_path} -> {self.real_data_path}")
                data = parse_fasta(self.data_path)
                df = pl.DataFrame(data)
                df.write_parquet(self.real_data_path)
                print(f"Saved {len(df)} sequences to {self.real_data_path}")

    def setup(self, stage=None):
        df = pl.read_parquet(self.real_data_path)
        
        # Simple random split (80/10/10)
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        n = len(df)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        self.train_data = df.slice(0, n_train)
        self.val_data = df.slice(n_train, n_val)
        self.test_data = df.slice(n_train + n_val, n - n_train - n_val)

        print(f"Data Loaded: Train {len(self.train_data)}, Val {len(self.val_data)}, Test {len(self.test_data)}")

    def train_dataloader(self):
        return DataLoader(
            PolarsDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            PolarsDataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            PolarsDataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )