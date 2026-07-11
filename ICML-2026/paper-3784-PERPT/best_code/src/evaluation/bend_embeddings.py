import torch
import numpy as np
import os
import json
import tarfile
import argparse
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from pyfaidx import Fasta
import pyBigWig
import h5py
import math
import sys
import webdataset as wds
yaml = __import__('yaml')

sys.path.append("../gamba")
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel
from gamba.collators import gLMCollator
from evodiff.utils import Tokenizer

def multi_hot(labels, num_labels):
    encoded = np.eye(num_labels, dtype=np.int64)[labels].sum(axis=0)
    return encoded

class GambaChunkDataset(Dataset):
    def __init__(self, chunked_sequences, chunked_scores, tokenizer):
        self.chunked_sequences = chunked_sequences
        self.chunked_scores = chunked_scores
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.chunked_sequences)

    def __getitem__(self, idx):
        seq = self.chunked_sequences[idx]
        score = self.chunked_scores[idx]
        if isinstance(seq, str):
            seq = self.tokenizer.tokenizeMSA(seq)
        return seq, score

class GambaEmbedder:
    def __init__(self, config_path, ckpt_path, device="cuda"):
        self.device = torch.device(device)
        with open(config_path, "r") as f:
            config = json.load(f)

        self.tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
        task = TaskType(config["task"].lower().strip())

        model_core, _ = create_model(
            task, config["model_type"], config["model_config"], self.tokenizer.mask_id.item()
        )

        self.model = JambagambaModel(
            model_core,
            d_model=config.get("d_model", 512),
            nhead=config.get("n_head", 8),
            n_layers=config.get("n_layers", 6),
            dim_feedfoward=config.get("dim_feedforward", 512),
            padding_id=config.get("padding_id", 0)
        ).to(self.device).eval()

        # self.model = JambaGambaNoConsModel(
        #     model_core,
        #     d_model=config.get("d_model", 512),
        #     nhead=config.get("n_head", 8),
        #     n_layers=config.get("n_layers", 6),
        #     dim_feedfoward=config.get("dim_feedforward", 512),
        #     padding_id=config.get("padding_id", 0)
        # ).to(self.device).eval()

        ckpt = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.collator = gLMCollator(tokenizer=self.tokenizer, test=True)
        self.max_length = 2048

    def embed_and_save(self, chunked_data, output_dir, task, split_name, chunk_size=1000, batch_size=48):
        os.makedirs(output_dir, exist_ok=True)
        total_regions = len(chunked_data)
        num_chunks = math.ceil(total_regions / chunk_size)

        for i in range(num_chunks):
            tar_path = os.path.join(output_dir, f"{split_name}_{i}.tar.gz")
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_regions)
            print(f"Writing {tar_path} with regions {start} to {end - 1}")

            sink = wds.TarWriter(tar_path, compress=True)

            with torch.no_grad():
                for region_id, region_data in enumerate(tqdm(chunked_data[start:end], desc=f"{split_name} chunk {i}")):
                    global_id = f"sample_{start + region_id}"
                    sequences, scores, label = region_data['sequences'], region_data['scores'], region_data['label']

                    dataset = GambaChunkDataset(sequences, scores, tokenizer=self.tokenizer)
                    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collator)

                    all_embeddings = []
                    for batch in loader:
                        seqs, scrs = batch
                        seqs, scrs = seqs.to(self.device), scrs.to(self.device)
                        output = self.model(seqs, scrs)
                        emb = output["representation"].cpu().numpy()
                        print(f"[DEBUG] Raw embedding shape: {emb.shape}")  # e.g., (batch_size, seq_len+2, dim)
                        emb = emb[:, 1:-1] if emb.shape[1] == self.max_length + 2 else emb
                        print(f"[DEBUG] Cleaned embedding shape: {emb.shape}")  # Should be (batch_size, seq_len, dim)
                        all_embeddings.extend([e for e in emb])

                    if len(all_embeddings) == 0:
                        print(f"⚠️ Skipping {global_id}: empty embedding")
                        continue

                    full_embedding = np.concatenate(all_embeddings, axis=0).astype(np.float32)

                    if "variant_effects" in task:
                        snp_pos = region_data.get('snp_relative_pos', 2047)
                        print(f"[DEBUG] Extracting SNP position embedding at {snp_pos}")
                        full_embedding = full_embedding[snp_pos][np.newaxis, np.newaxis, :]  # shape (1, 1, dim)
                    else:
                        true_seq_len = region_data.get('true_seq_len', full_embedding.shape[0])
                        if full_embedding.shape[0] > true_seq_len:
                            print(f"[DEBUG] Trimming padded tokens: {full_embedding.shape[0]} → {true_seq_len}")
                            full_embedding = full_embedding[:true_seq_len]
                        full_embedding = full_embedding[np.newaxis, ...]  # shape (1, seq_len, dim)
                    label_np = np.array(label, dtype=np.float32)
                    print(f"[DEBUG] Full embedding shape for region {global_id}: {full_embedding.shape}")
                    print(f"[DEBUG] Label shape for region {global_id}: {label_np.shape}")
                    sink.write({
                        "__key__": global_id,
                        "input.npy": full_embedding,
                        "output.npy": label_np
                    })

            sink.close()
import sequence_models.constants as constants

def extract_sequences(bed_df, genome_fasta, bigwig_path, label_matrix, seq_len=2048, flank=0, label_depth=None):
    torch.manual_seed(42)
    np.random.seed(42)

    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_path)

    def reverse_complement(seq):
        complement = str.maketrans("ACGTN", "TGCAN")
        return seq.translate(complement)[::-1]

    chunked_data = []
    for idx, row in tqdm(bed_df.iterrows(), total=len(bed_df)):
        chrom = row['chromosome']
        #if chrom is an int, add "chr" prefix
        if isinstance(chrom, int):
            chrom = f"chr{chrom}"
        if flank >0:
            start = int(row['start']) - flank
            end = int(row['start']) + 1
        else:
            start = int(row['start']) 
            end = int(row['end']) #+ flank don't care about downstream context
        strand = row['strand'] if 'strand' in row else '+'
        if chrom not in genome:
            print(f"Skipping missing chrom: {chrom}")
            continue
        try:
            seq = genome[chrom][start:end].seq.upper()
            original_seq_len = len(seq)
            if strand == '-':
                seq = reverse_complement(seq)

            region_chunks, region_scores = [], []
            for i in range(0, len(seq), seq_len):
                chunk_seq = seq[i:i + seq_len]
                if len(chunk_seq) < seq_len:
                    pad_len = seq_len - len(chunk_seq)
                    chunk_seq += constants.MSA_PAD * pad_len  # pad with N's to reach 2048bp
                assert len(chunk_seq) == seq_len

                chunk_start = start + i
                chunk_end = chunk_start + seq_len

                vals = np.zeros(seq_len, dtype=np.float32)
                intervals = bw.intervals(chrom, chunk_start, chunk_end)
                if intervals is not None:
                    for iv_start, iv_end, val in intervals:
                        s = max(0, iv_start - chunk_start)
                        e = min(seq_len, iv_end - chunk_start)
                        vals[s:e] = val

                region_chunks.append(chunk_seq)
                region_scores.append(vals)


            if region_chunks:
                if label_matrix is not None:
                    label = label_matrix[idx]
                elif 'label' in row:
                    raw = row['label']
                    if isinstance(raw, str):
                        label = multi_hot(list(map(int, raw.split(','))), label_depth)
                    elif np.issubdtype(type(raw), np.integer):
                        label = multi_hot([raw], label_depth)
                    else: 
                        label = np.zeros(label_depth, dtype=np.float32)
                else:
                    label = np.zeros(label_depth, dtype=np.float32)

                chunked_data.append({
                    'sequences': region_chunks,
                    'scores': region_scores,
                    'label': label,
                    'true_seq_len': original_seq_len  # store actual length for slicing later
                })

        except Exception as e:
            print(f"Error at idx {idx} ({chrom}:{start}-{end}): {e}")
            continue

    return chunked_data

def load_label_depths(yaml_path="/home/mica/BEND/conf/datadims/dimensions/datadims.yaml"):
    with open(yaml_path, "r") as f:
        dims = yaml.safe_load(f)
    return dims["datadims"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--flank", type=int, default=0)
    parser.add_argument("--label_depths", type=str, default="/home/mica/BEND/conf/datadims/dimensions/datadims.yaml")
    args = parser.parse_args()

    root = "/media/data/BEND_backups/data"
    task = args.task
    output_dir = f"/media/data2/gamba_nocons_embed/{task}/gamba_nocons/"

    #if task contains variant_effects:
    if "variant_effects" in task:
        flank = 2047
        main_dir = "variant_effects"
    else:
        main_dir = task
        flank = args.flank

    label_dims = load_label_depths(args.label_depths)
    label_depth = label_dims.get(f"{main_dir}_label_dim")

    print("label_depth key:", f"{task}_label_dim")
    print("label_depth value:", label_depth)

    embedder = GambaEmbedder(
        config_path="/home/mica/gamba/configs/jamba-small-240mammalian.json",
        #ckpt_path="/home/mica/gamba/clean_dcps/dcp_nocons_56000"
        ckpt_path="/home/mica/gamba/clean_dcps/dcp_56000"
    )

    bed_path = f"{root}/{main_dir}/{task}.bed"
    hdf5_path = f"{root}/{main_dir}/{task}.hdf5"
    genome_fasta = "/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa"
    bigwig_path = "/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig"

    bed = pd.read_csv(bed_path, sep="\t")
    assert 'split' in bed.columns, "'split' column missing from BED file."
    print("Loaded BED file with shape:", bed.shape, "and columns:", bed.columns.tolist())

    for split in bed['split'].unique():
        print(f"Processing split: {split}")
        split_mask = bed['split'] == split
        bed_split = bed[split_mask].reset_index(drop=True)

        label_matrix = None
        if os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, "r") as h5:
                indices = np.where(split_mask.to_numpy())[0]
                label_matrix = h5["labels"][indices]
        chunked_data = extract_sequences(
            bed_split, genome_fasta, bigwig_path, label_matrix,
            seq_len=2048, flank=flank, label_depth=label_depth
        )
        print(f"Extracted {len(chunked_data)} regions for split {split}")
        embedder.embed_and_save(chunked_data, output_dir, task=task, split_name=split, chunk_size=1000)

if __name__ == "__main__":
    main()