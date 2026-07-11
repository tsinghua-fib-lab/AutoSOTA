# save as src/evaluation/generate_trna.py
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
import pyBigWig
from tqdm import tqdm
sys.path.append('/home/mica/gamba')
from evodiff.utils import Tokenizer
from gamba.constants import DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel

CHR_KEEP = {"chr22","chr16","chr3","chr2"}

def rc(seq: str) -> str:
    tbl = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(tbl)[::-1]

def clamp(a, lo, hi): return max(lo, min(a, hi))

def read_true_seq(genome, chrom, start, end, strand):
    seq = str(genome[chrom][start:end].seq.upper())
    return rc(seq) if strand == "-" else seq

def extract_context_with_conservation(genome, bw, chrom, anchor, strand, ctx, chrom_len):
    if strand == "+":
        s = clamp(anchor - ctx, 0, chrom_len)
        e = anchor
        seq = str(genome[chrom][s:e].seq.upper())
        # already 5'->3' relative to gene
        cons = np.zeros(len(seq), dtype=np.float32)
        try:
            ivals = bw.intervals(chrom, s, e)
            if ivals:
                for a,b,v in ivals:
                    rs, re = max(a - s, 0), min(b - s, len(seq))
                    cons[rs:re] = v
        except Exception:
            pass
        return seq, cons.tolist()
    else:
        # minus strand: 5' context is genomic downstream [end : end+ctx], then RC to gene orientation
        s = anchor
        e = clamp(anchor + ctx, 0, chrom_len)
        seq_raw = str(genome[chrom][s:e].seq.upper())
        seq = rc(seq_raw)
        cons_arr = np.zeros(len(seq_raw), dtype=np.float32)
        try:
            ivals = bw.intervals(chrom, s, e)
            if ivals:
                for a,b,v in ivals:
                    rs, re = max(a - s, 0), min(b - s, len(seq_raw))
                    cons_arr[rs:re] = v
        except Exception:
            pass
        cons = cons_arr[::-1].tolist()  # align with RC sequence
        return seq, cons

def generate_sequence(model, tokenizer, collator, device, context_tokens, cons_scores, gen_len):
    model.eval()
    generated = list(context_tokens)
    generated_cons = list(cons_scores)
    for _ in range(gen_len):
        inp = torch.tensor([generated], dtype=torch.long)
        cons = torch.tensor([generated_cons], dtype=torch.float32)
        batch_tokens, batch_cons = collator([(inp.squeeze(0), cons.squeeze(0))])
        with torch.no_grad():
            out = model(batch_tokens.to(device), batch_cons.to(device))
            probs = torch.softmax(out["seq_logits"][0, -1], dim=-1)
            nt = torch.multinomial(probs, 1).item()
            # predicted conservation to feed next step if present
            cons_pred = out.get("scaling_logits", None)
            cp = cons_pred[0, -1, 0].item() if cons_pred is not None else 0.0
            generated.append(nt)
            generated_cons.append(cp)
    return generated[-gen_len:]

def compute_accuracy(pred_tok, true_tok):
    L = min(len(pred_tok), len(true_tok))
    if L == 0: return 0.0
    return sum(int(p == t) for p, t in zip(pred_tok[:L], true_tok[:L])) / L

def ensure_true_written(fasta_path, true_seq):
    if not os.path.exists(fasta_path):
        with open(fasta_path, "w") as f:
            f.write(">true\n")
            f.write(true_seq + "\n")

def append_fasta(fasta_path: str, header: str, seq: str):
    with open(fasta_path, "a") as f:
        f.write(f">{header}\n{seq}\n")

def load_model_and_tools(config_path, no_cons, device, tokenizer):
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
            dim_feedfoward=config.get("dim_feedforward", 512),
        )
        ckpt_path = "/home/mica/gamba/clean_dcps/dcp_nocons_56000"
    else:
        model = JambagambaModel(
            model_base,
            d_model=config.get("d_model", 512),
            nhead=config.get("n_head", 8),
            n_layers=config.get("n_layers", 6),
            padding_id=0,
            dim_feedfoward=config.get("dim_feedforward", 512),
        )
        ckpt_path = "/home/mica/gamba/clean_dcps/CCP/dcp_44000"

    checkpoint = torch.load(os.path.join(ckpt_path, "model_optimizer.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    collator = gLMCollator(tokenizer=tokenizer, test=True)
    return model, tokenizer, collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv_path", type=str, required=True)
    ap.add_argument("--genome_path", type=str, required=True)
    ap.add_argument("--config_path", type=str, required=True)
    ap.add_argument("--bigwig_path", type=str, required=True)
    ap.add_argument("--context_bp", type=int, default=1000)
    ap.add_argument("--num_generations", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="/home/mica/gamba/data_processing/data/generation_data/")
    ap.add_argument("--prefix_grid", type=str, default="45,50,55,60,65,70,75,80")  # percents
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.tsv_path, sep="\t")
    df = df[df["chrom"].isin(CHR_KEEP)].copy()

    genome = Fasta(args.genome_path)
    bw = pyBigWig.open(args.bigwig_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    # Preload both variants
    models = {
        "cons": load_model_and_tools(args.config_path, no_cons=False, device=device, tokenizer=tokenizer),
        "nocons": load_model_and_tools(args.config_path, no_cons=True, device=device, tokenizer=tokenizer),
    }

    results_rows = []
    prefix_percents = [int(x) for x in args.prefix_grid.split(",")]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="tRNAs"):
        chrom = row["chrom"]
        start = int(row["chromStart"])
        end   = int(row["chromEnd"])
        strand = row["strand"].strip()
        trna_id = str(row["name"])

        # True sequence in gene orientation
        true_seq = read_true_seq(genome, chrom, start, end, strand)
        gen_len = len(true_seq)
        if gen_len == 0: continue

        # Per-ID FASTA path
        fasta_path = os.path.join(args.out_dir, f"trna_fasta{trna_id}.fa")
        ensure_true_written(fasta_path, true_seq)

        # Context extraction anchor depends on strand
        anchor = start if strand == "+" else end
        chrom_len = len(genome[chrom])

        # Base context (5' relative to gene) + conservation
        ctx_seq, ctx_cons = extract_context_with_conservation(
            genome, bw, chrom, anchor, strand, args.context_bp, chrom_len
        )

        for p in prefix_percents:
            n_prefix = max(1, int(round(p * gen_len / 100.0)))
            n_prefix = min(n_prefix, gen_len)  # cap
            prefix_seq = true_seq[:n_prefix]
            prefix_cons = [0.0] * n_prefix  # no ground-truth cons fed

            full_context_seq = ctx_seq + prefix_seq
            full_cons_scores = ctx_cons + prefix_cons

            for variant, (model, tokenizer, collator) in models.items():
                context_tokens = tokenizer.tokenizeMSA(full_context_seq)
                true_tokens = tokenizer.tokenizeMSA(true_seq)

                gen_region_len = gen_len - n_prefix
                if gen_region_len <= 0:
                    # no generation needed; still record
                    best_seq = prefix_seq
                    best_acc = 1.0
                else:
                    best_acc = -1.0
                    best_seq = ""
                    for _ in range(args.num_generations):
                        gen_tokens = generate_sequence(
                            model, tokenizer, collator, device,
                            context_tokens, full_cons_scores, gen_region_len
                        )
                        full_tokens = list(context_tokens) + gen_tokens
                        full_untok = tokenizer.untokenize(full_tokens)
                        window_seq = full_untok[-gen_len:]
                        acc = compute_accuracy(gen_tokens, true_tokens[n_prefix:])
                        if acc > best_acc:
                            best_acc = acc
                            best_seq = window_seq

                header = f"{trna_id}|{variant}|p{p}|acc={best_acc:.3f}"
                append_fasta(fasta_path, header, best_seq)
                results_rows.append({
                    "trna_id": trna_id,
                    "chrom": chrom,
                    "strand": strand,
                    "len": gen_len,
                    "prefix_percent": p,
                    "n_prefix": n_prefix,
                    "variant": variant,
                    "best_acc": best_acc,
                    "fasta_path": fasta_path,
                })

    out_tsv = os.path.join(args.out_dir, "trna_generation_summary.tsv")
    pd.DataFrame(results_rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote summary: {out_tsv}")

if __name__ == "__main__":
    main()
