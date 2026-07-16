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
# Description: Single entry point for the watermark-removal attacks.
#   Dispatches --attack_algorithms to the matching Attack class and runs it.
#   e.g.  python pipeline/run_attack.py --attack_algorithms BIRA --backend hf
#         python pipeline/run_attack.py --attack_algorithms dipper-2
# ==========================================

import sys
import argparse

sys.path.append(".")
from attack_utils.attacks import ATTACK_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Run a watermark-removal attack.")

    parser.add_argument(
        "--attack_algorithms",
        type=str,
        choices=["BIRA", "vanilla_paraphrasing", "dipper-1", "dipper-2", "SIRA"],
        default="BIRA",
        help="Which attack to run (selects the Attack class).",
    )
    # BIRA backend: local HuggingFace model (hf) or API model (api).
    parser.add_argument("--backend", type=str, choices=["hf", "api"], default="hf")

    parser.add_argument("--model_cfg_path", default="./model_config/llama3.1-8b.yaml")
    parser.add_argument("--api_model_name", type=str, default="gpt4o-mini")
    # dipper has no rewriting LLM; --model_name is just the TPR output-path label.
    parser.add_argument("--model_name", type=str, default="dipper")

    parser.add_argument("--input_path", type=str, default="./watermarked_dataset")
    parser.add_argument("--result_save_dir", type=str, default="./experimental_results")

    # ---- TPR-at-fixed-FPR (detectability) evaluation args ----
    parser.add_argument("--dataset_path", type=str, default="./dataset/c4/processed_c4.json")
    parser.add_argument("--human_text_result_save_dir", type=str, default="./experimental_results_human_text")
    parser.add_argument("--labels", nargs="+", default=["TPR", "F1"])
    parser.add_argument("--rules", nargs="+", default=["target_fpr", "best"],
                        help="Detectability decision rules to evaluate: 'target_fpr' (swept over "
                             "--target_fprs) and/or 'best' (best-threshold F1).")
    parser.add_argument("--target_fprs", type=float, nargs="+", default=[0.01, 0.1],
                        help="Fixed FPRs at which to report TPR for the 'target_fpr' rule.")

    parser.add_argument("--num_data", type=int, default=500)

    # ---- BIRA params (ignored by other attacks) ----
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--percentile", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.125)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=800)

    # ---- SIRA params (ignored by other attacks) ----
    parser.add_argument("--paraphrase_model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--attack_model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--threshold", type=int, default=30, help="Self-information percentile threshold for SIRA blanking.")
    parser.add_argument("--max_new_tokens", type=int, default=1500)

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "SIR",
            "KGW",
            "Unigram",
            "UPV",
            "EWD",
            "DIP",
            "EXP",
        ],
        help="List of watermarking algorithms to attack.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    attack_cls = ATTACK_REGISTRY[args.attack_algorithms]
    attack_cls(args).run()


if __name__ == "__main__":
    main()
