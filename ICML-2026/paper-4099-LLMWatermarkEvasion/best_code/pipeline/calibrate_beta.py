# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================
# Description: Per-model BIRA initial logit bias (β₀) calibration.
#   To initialize β, generate 50 paraphrases from the C4 dataset and gradually
#   decrease β (e.g., from -1 down to -12) until degeneration appears in at
#   least one of the 50 outputs. That value is used as the initial logit bias
#   β₀: it makes the attack as strong as possible while minimizing the risk of
#   degeneration. During the attack β is then adaptively relaxed per sample
#   (β += learning_rate) whenever an output degenerates.
#
#   Here each β is applied ONCE (max_iter=1) to the natural text and we count
#   how many of the 50 outputs degenerate (collapse into repetition).
#
#   Degeneration is model-dependent, NOT watermark-dependent (the suppression
#   targets high-self-information tokens of the input text regardless of any
#   watermark), so this calibration needs no watermark / detector.
#
#   It reuses BIRAAttack.rewrite() with max_iter=1, so the per-β behavior is
#   identical to one iteration of the real attack.
#
#   e.g.  python pipeline/calibrate_beta.py --backend hf \
#             --model_cfg_path ./model_config/llama3.1-8b.yaml --use_sampling \
#             --betas -1 -2 -3 -4 -5 -6 -7 -8 -9 -10
# ==========================================

import os
import sys
import json
import argparse

sys.path.append(".")
from attack_utils.attacks import BIRAAttack

from types import SimpleNamespace
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate BIRA's starting beta on natural text.")

    # BIRA backend: local HuggingFace model (hf) or API model (api).
    parser.add_argument("--backend", type=str, choices=["hf", "api"], default="hf")
    parser.add_argument("--model_cfg_path", default="./model_config/llama3.1-8b.yaml")
    parser.add_argument("--api_model_name", type=str, default="gpt4o-mini")

    # Natural (human-written) text to probe for degeneration.
    parser.add_argument("--dataset_path", type=str, default="./dataset/c4/processed_c4.json")
    parser.add_argument("--num_data", type=int, default=50)

    # The beta sweep. Default mirrors the original logit-value sweep.
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12])
    parser.add_argument("--percentile", type=int, default=50)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.125)

    parser.add_argument("--result_save_dir", type=str, default="./beta_calibration")

    return parser.parse_args()


def build_attack_args(cli):
    """Assemble the namespace BIRAAttack expects, fixing it to a single-shot
    (max_iter=1) BIRA run so each rewrite() applies one beta and reports
    whether the output degenerated."""
    return SimpleNamespace(
        attack_algorithms="BIRA",
        backend=cli.backend,
        model_cfg_path=cli.model_cfg_path,
        api_model_name=cli.api_model_name,
        percentile=cli.percentile,
        use_sampling=cli.use_sampling,
        learning_rate=cli.learning_rate,
        max_iter=1,          # single shot — no adaptive loop during calibration
        beta=cli.betas[0],   # overwritten per sweep step below
    )


def main():
    cli = parse_args()
    args = build_attack_args(cli)

    # Build the BIRA attack and load its model(s) once. We never call run(),
    # so no watermark/detector is loaded.
    attack = BIRAAttack(args)
    attack.load_models()

    # Load natural (human) text.
    with open(cli.dataset_path, "r") as f:
        lines = f.readlines()[:cli.num_data]
    natural_texts = [json.loads(line)["natural_text"] for line in lines]
    num_data = len(natural_texts)

    print(f"\nCalibrating beta on {num_data} natural texts | backend={cli.backend} | "
          f"model={attack.model_name} | percentile={cli.percentile}\n")

    # Sweep beta from weakest to strongest suppression so the onset of
    # degeneration is easy to read off (count climbs as |beta| grows).
    betas = sorted(cli.betas, reverse=True)  # e.g. -1, -2, ..., -10

    sweep = []
    for beta in betas:
        args.beta = beta
        num_degenerated = 0
        for text in tqdm(natural_texts, desc=f"beta={beta}"):
            _, meta = attack.rewrite(text)
            num_degenerated += int(meta["is_degenerated"])
        ratio = num_degenerated / num_data
        sweep.append({"beta": beta, "num_degenerated": num_degenerated,
                      "num_data": num_data, "degeneration_ratio": ratio})
        print(f"  beta={beta:>6}: degenerated {num_degenerated}/{num_data} ({ratio:.2%})")

    # Recommend the onset: the first beta (weakest suppression) at which any
    # degeneration appears.
    onset = next((s["beta"] for s in sweep if s["num_degenerated"] > 0), None)
    print("\n==================== beta calibration ====================")
    for s in sweep:
        mark = "  <-- onset" if s["beta"] == onset else ""
        print(f"  beta={s['beta']:>6}: {s['num_degenerated']:>3}/{s['num_data']} "
              f"({s['degeneration_ratio']:.2%}){mark}")
    if onset is not None:
        print(f"\nRecommended initial logit bias beta0 (first beta with >=1 degenerated output): {onset}")
    else:
        print("\nNo degeneration observed in the swept range — try stronger (more negative) betas.")
    print("==========================================================\n")

    os.makedirs(cli.result_save_dir, exist_ok=True)
    out_path = os.path.join(cli.result_save_dir, f"{attack.model_name}_beta_calibration.json")
    with open(out_path, "w") as f:
        json.dump({
            "backend": cli.backend,
            "model_name": attack.model_name,
            "percentile": cli.percentile,
            "num_data": num_data,
            "recommended_starting_beta": onset,
            "sweep": sweep,
        }, f, indent=4)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
