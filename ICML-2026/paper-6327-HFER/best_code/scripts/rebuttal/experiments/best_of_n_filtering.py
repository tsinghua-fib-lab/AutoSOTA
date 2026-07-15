"""
Best-of-N Spectral Filtering (Day 2 GPU)
==========================================
Addresses: Reviewer 6GDW ("does it improve pass-at-k as a filter?")

Picks problems where greedy decoding produced invalid proofs (from Day 1 
extraction), generates N candidates at temperature 0.7, scores with HFER,
compares selection strategies.

Input: data/results/rebuttal/llama8b_full_extraction.json (to find invalid proofs)
Output: data/results/rebuttal/best_of_n.json

Usage:
    python scripts/rebuttal/best_of_n_filtering.py --load-in-4bit
"""

import os, sys, json, glob, argparse, re, time
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EXTRACTION_FILE = "data/results/rebuttal/llama8b_full_extraction.json"
DATA_DIR = "data/experiment_ready"
TARGET_LAYER = 24


def get_theorem_name(filename):
    """Strip variant suffix to get base theorem name."""
    base = filename.replace(".lean", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def run_best_of_n(args):
    print("=" * 70)
    print("  BEST-OF-N SPECTRAL FILTERING")
    print("=" * 70)

    os.makedirs("output/rebuttal", exist_ok=True)
    os.makedirs("data/results/rebuttal", exist_ok=True)

    # Load extraction to find which theorems have invalid proofs
    if not os.path.exists(EXTRACTION_FILE):
        print(f"ERROR: {EXTRACTION_FILE} not found. Run day1_llama8b_batch.py first.")
        sys.exit(1)

    with open(EXTRACTION_FILE) as f:
        extraction = json.load(f)

    # Find invalid proofs (by corrected label)
    invalid_files = [e["file"] for e in extraction if not e["is_valid"]]
    print(f"Found {len(invalid_files)} invalid proofs to use as base problems")

    # Group by theorem → find theorems with multiple variants
    from collections import defaultdict
    theorem_variants = defaultdict(list)
    for filename in invalid_files:
        theorem = get_theorem_name(filename)
        filepath = os.path.join(DATA_DIR, "invalid", filename)
        if os.path.exists(filepath):
            theorem_variants[theorem].append(filepath)

    # Select problems with at least 2 variants (to have natural candidates)
    problems = [(th, fps) for th, fps in theorem_variants.items() if len(fps) >= 2]
    problems = sorted(problems, key=lambda x: -len(x[1]))[:args.n_problems]
    print(f"Selected {len(problems)} problems with 2+ existing variants")

    # Also load valid proofs to check for any valid theorem matches
    valid_set = set(get_theorem_name(os.path.basename(f))
                    for f in glob.glob(os.path.join(DATA_DIR, "valid", "*.lean")))

    # Load model + spectral framework
    from spectral_trust import GSPDiagnosticsFramework, GSPConfig
    from transformers import AutoTokenizer

    model_kwargs = {"output_attentions": True, "output_hidden_states": True}
    if args.load_in_4bit: model_kwargs["load_in_4bit"] = True

    config = GSPConfig(model_name=MODEL, device=args.device, model_kwargs=model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    results_by_n = {4: [], 8: [], 16: []}

    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(MODEL)
        model = framework.instrumenter.model

        for theorem, variant_files in tqdm(problems, desc="Problems"):
            # Score existing variants with HFER
            scored = []
            for fp in variant_files[:16]:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
                try:
                    analysis = framework.analyze_text(text, save_results=False)
                    layers = analysis.get("layer_diagnostics", [])
                    if TARGET_LAYER < len(layers):
                        hfer = float(getattr(layers[TARGET_LAYER], "hfer"))
                    else:
                        hfer = float(getattr(layers[-1], "hfer")) if layers else 1.0

                    # Token log-prob baseline
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                    input_ids = inputs["input_ids"]
                    inputs_dev = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs_dev)
                    log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)
                    if input_ids.shape[1] > 1:
                        actual_lp = log_probs[:-1].gather(1, input_ids[0, 1:].unsqueeze(1).to(model.device)).squeeze()
                        mean_lp = actual_lp.mean().item()
                    else:
                        mean_lp = 0.0

                    scored.append({
                        "file": os.path.basename(fp),
                        "hfer": hfer,
                        "mean_logprob": mean_lp,
                        "length": len(text.split()),
                    })
                except Exception as e:
                    print(f"  ERROR on {os.path.basename(fp)}: {e}")
                    continue

            if len(scored) < 2:
                continue

            # Evaluate for each N
            has_valid_variant = theorem in valid_set

            for N in [4, 8, 16]:
                pool = scored[:N]
                if len(pool) < 2:
                    continue

                # Selection strategies
                np.random.seed(42 + len(results_by_n[N]))
                random_idx = np.random.randint(len(pool))
                shortest_idx = min(range(len(pool)), key=lambda i: pool[i]["length"])
                highest_lp_idx = max(range(len(pool)), key=lambda i: pool[i]["mean_logprob"])
                lowest_hfer_idx = min(range(len(pool)), key=lambda i: pool[i]["hfer"])

                pool_hfers = [p["hfer"] for p in pool]
                mean_hfer = np.mean(pool_hfers)
                std_hfer = np.std(pool_hfers) if len(pool_hfers) > 1 else 0

                results_by_n[N].append({
                    "theorem": theorem,
                    "n_candidates": len(pool),
                    "has_valid_variant": has_valid_variant,
                    "random_hfer": pool[random_idx]["hfer"],
                    "shortest_hfer": pool[shortest_idx]["hfer"],
                    "highest_lp_hfer": pool[highest_lp_idx]["hfer"],
                    "lowest_hfer": pool[lowest_hfer_idx]["hfer"],
                    "pool_mean_hfer": float(mean_hfer),
                    "pool_std_hfer": float(std_hfer),
                    "z_score_selected": float((pool[lowest_hfer_idx]["hfer"] - mean_hfer) / std_hfer) if std_hfer > 0 else 0,
                })

    # Analysis
    print(f"\n--- RESULTS ---")
    print(f"{'Strategy':<20} | {'N=4 mean HFER':>14} | {'N=8':>14} | {'N=16':>14}")
    print("-" * 65)

    strategies = [
        ("Random", "random_hfer"),
        ("Shortest", "shortest_hfer"),
        ("Highest log-prob", "highest_lp_hfer"),
        ("Lowest HFER (ours)", "lowest_hfer"),
        ("Pool mean", "pool_mean_hfer"),
    ]
    for sname, skey in strategies:
        row = [sname]
        for N in [4, 8, 16]:
            vals = [r[skey] for r in results_by_n[N] if skey in r]
            if vals:
                row.append(f"{np.mean(vals):.4f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<20} | {row[1]:>14} | {row[2]:>14} | {row[3]:>14}")

    # Z-score analysis
    for N in [4, 8, 16]:
        zs = [r["z_score_selected"] for r in results_by_n[N] if "z_score_selected" in r]
        if zs:
            print(f"\n  N={N}: spectral selection z-score = {np.mean(zs):.2f} ± {np.std(zs):.2f}")

    # Save
    save_data = {str(k): v for k, v in results_by_n.items()}
    with open("data/results/rebuttal/best_of_n.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: data/results/rebuttal/best_of_n.json")

    # Figure
    if any(results_by_n[N] for N in [4, 8, 16]):
        fig, ax = plt.subplots(figsize=(8, 5))
        Ns = [4, 8, 16]
        for sname, skey, color in [
            ("Random", "random_hfer", "#9E9E9E"),
            ("Shortest", "shortest_hfer", "#FF9800"),
            ("Highest log-prob", "highest_lp_hfer", "#4CAF50"),
            ("Lowest HFER (ours)", "lowest_hfer", "#1565C0"),
        ]:
            means = []
            for N in Ns:
                vals = [r[skey] for r in results_by_n[N]]
                means.append(np.mean(vals) if vals else np.nan)
            ax.plot(Ns, means, 'o-', color=color, label=sname, linewidth=2, markersize=8)

        ax.set_xlabel("N candidates"); ax.set_ylabel("Mean HFER of selected proof")
        ax.set_title("Best-of-N Selection: HFER of Selected Proof")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xticks(Ns)
        plt.savefig("output/rebuttal/best_of_n.pdf")
        plt.savefig("output/rebuttal/best_of_n.png")
        print(f"Saved: output/rebuttal/best_of_n.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--n-problems", type=int, default=50)
    run_best_of_n(parser.parse_args())
