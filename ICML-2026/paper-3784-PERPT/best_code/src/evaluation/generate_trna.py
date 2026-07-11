# save as generate_trna.py
import os
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from evodiff.utils import Tokenizer
import sys
import pyBigWig
sys.path.append('/home/mica/gamba')
from gamba.constants import DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel
import json
import numpy as np
import argparse

def extract_context_with_conservation(genome, bw, chrom, trna_start, context_bp):
    start = max(0, trna_start - context_bp)
    seq = str(genome[chrom][start:trna_start].seq.upper())
    print(f"Upstream context sequence for {chrom}:{trna_start} is {seq}")
    print(f"Length of upstream context sequence: {len(seq)}")

    cons_scores = np.zeros(len(seq), dtype=np.float32)
    try:
        intervals = bw.intervals(chrom, start, trna_start)
        if intervals:
            for interval_start, interval_end, value in intervals:
                rel_start = max(interval_start - start, 0)
                rel_end = min(interval_end - start, len(seq))
                cons_scores[rel_start:rel_end] = value
    except Exception as e:
        print(f"Failed to load conservation for {chrom}:{start}-{trna_start}: {e}")

    return seq, cons_scores.tolist()

def generate_sequence(model, tokenizer, collator, device, context_tokens, cons_scores, gen_len):
    model.eval()
    generated = list(context_tokens)
    generated_cons = list(cons_scores)
    for _ in range(gen_len):
        input_tensor = torch.tensor([generated], dtype=torch.long)
        cons_tensor = torch.tensor([generated_cons], dtype=torch.float32)
        input_batch = collator([(input_tensor.squeeze(0), cons_tensor.squeeze(0))])
        with torch.no_grad():
            outputs = model(input_batch[0].to(device), input_batch[1].to(device))
            logits = outputs["seq_logits"]
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            cons_pred = outputs.get("scaling_logits", None)
            cons_pred = cons_pred[0, -1, 0].item() if cons_pred is not None else 0.0
            generated.append(next_token)
            generated_cons.append(cons_pred)
    return generated[-gen_len:]  # the generated region only

def compute_accuracy(pred_seq, true_seq):
    match = sum(p == t for p, t in zip(pred_seq, true_seq))
    return match / len(true_seq)

def append_fasta(fasta_path: str, header: str, seq: str):
    with open(fasta_path, "a") as f:
        f.write(f">{header}\n")
        f.write(seq + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate tRNA sequences.")
    parser.add_argument("--tsv_path", type=str, default="/home/mica/gamba/data_processing/data/generation_data/trna.tsv")
    parser.add_argument("--genome_path", type=str, default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa")
    parser.add_argument("--config_path", type=str, default="/home/mica/gamba/configs/jamba-small-240mammalian.json")
    parser.add_argument("--no_cons", action='store_true', help="Ignore conservation if set.")
    parser.add_argument("--bigwig_path", type=str, default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    parser.add_argument("--n_prefix", type=int, default=44, help="Number of true bp to prepend to the context.")
    parser.add_argument("--context_bp", type=int, default=1000, help="Upstream context length.")
    parser.add_argument("--num_generations", type=int, default=100, help="Number of samples.")
    parser.add_argument("--fasta_path", type=str, default="/home/mica/gamba/data_processing/data/generation_data/trna_fasta.fa",
                        help="FASTA to append best sequences to.")

    args = parser.parse_args()
    tsv_path = args.tsv_path
    genome_path = args.genome_path
    config_path = args.config_path
    no_cons = args.no_cons
    bigwig_path = args.bigwig_path
    n_prefix = args.n_prefix
    context_bp = args.context_bp
    num_generations = args.num_generations
    fasta_path = args.fasta_path

    ckpt_path = "/home/mica/gamba/clean_dcps/dcp_nocons_56000" if no_cons else "/home/mica/gamba/clean_dcps/CCP/dcp_44000"
    output_path = f"/home/mica/gamba/data_processing/data/generation_data/trna_generation_output_{no_cons}_{n_prefix}.tsv"

    # === Load data ===
    df = pd.read_csv(tsv_path, sep="\t")
    trna_row = df[df["chrom"] == "chr22"].iloc[0]
    chrom, start, end = trna_row["chrom"], int(trna_row["chromStart"]), int(trna_row["chromEnd"])

    genome = Fasta(genome_path)
    bw = pyBigWig.open(bigwig_path)

    true_seq = str(genome[chrom][start:end].seq.upper())
    gen_len = len(true_seq)
    print(f"True sequence for {chrom}:{start}-{end} is {true_seq} of length {gen_len}")

    # === Extract context and prepend prefix truth ===
    context_seq, cons_scores = extract_context_with_conservation(genome, bw, chrom, start, context_bp)
    prefix_seq = true_seq[:n_prefix]
    prefix_cons = np.zeros(n_prefix, dtype=np.float32).tolist()

    full_context_seq = context_seq + prefix_seq
    full_cons_scores = cons_scores + prefix_cons

    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)
    context_tokens = tokenizer.tokenizeMSA(full_context_seq)
    true_tokens = tokenizer.tokenizeMSA(true_seq)

    # === Load model ===
    with open(config_path) as f:
        config = json.load(f)
    task = config["task"].lower().strip()
    model_base, _ = create_model(task, config["model_type"], config["model_config"], tokenizer.mask_id.item())
    if no_cons:
        model = JambaGambaNoConsModel(
            model_base,
            d_model=config.get("d_model", 512),
            nhead=config.get("n_head", 8),
            n_layers=config.get("n_layers", 6),
            padding_id=0,
            dim_feedfoward=config.get("dim_feedforward", 512)
        )
    else:
        model = JambagambaModel(
            model_base,
            d_model=config.get("d_model", 512),
            nhead=config.get("n_head", 8),
            n_layers=config.get("n_layers", 6),
            padding_id=0,
            dim_feedfoward=config.get("dim_feedforward", 512)
        )
    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = gLMCollator(tokenizer=tokenizer, test=True)

    # === Generation loop ===
    results = []
    best_acc = -1.0
    best_seq_full = ""

    gen_region_len = gen_len - n_prefix  # generate the remainder

    for _ in tqdm(range(num_generations), desc="Generating"):
        gen_tokens = generate_sequence(
            model, tokenizer, collator, device, context_tokens, full_cons_scores, gen_region_len
        )
        # Full sequence within the analysis window (prefix start → end)
        full_sequence_tokens = list(context_tokens) + gen_tokens
        full_untok = tokenizer.untokenize(full_sequence_tokens)

        # Keep only the window that starts at the prefix and spans gen_len bases
        window_seq = full_untok[-gen_len:]

        # Accuracy on generated region only
        acc = compute_accuracy(gen_tokens, true_tokens[n_prefix:])

        results.append((acc, window_seq))
        if acc > best_acc:
            best_acc = acc
            best_seq_full = window_seq  # includes the prefix + generated region

    # === Save results table ===
    out_df = pd.DataFrame(results, columns=["accuracy", "generated_window_seq"])
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nAverage accuracy: {out_df['accuracy'].mean():.4f}")
    print(f"Best accuracy: {best_acc:.4f}")

    # === Append best sequence to FASTA ===
    header = f"trna_{'nocons' if no_cons else 'cons'}_{n_prefix}"
    append_fasta(fasta_path, header, best_seq_full)
    print(f"Appended best sequence to {fasta_path} as >{header}")

if __name__ == "__main__":
    main()
