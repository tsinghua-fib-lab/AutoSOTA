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

# =================================================================
# assess_detectability.py
# Description: Assess the detectability of a watermarking algorithm
# =================================================================

import torch
import sys
sys.path.append(".")

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.pipelines.detection import UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.dataset import C4Dataset
from evaluation.tpr import load_or_compute_human_text_results, evaluate_detectability, attach_detectability
from evaluation.paths import attack_result_path

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse

import yaml
from types import SimpleNamespace

import json

import sys
sys.path.append(".")


def attack_result_path_for(args, model_name, algorithm):
    return attack_result_path(
        args.attack_result_save_dir, args.attack_algorithms, algorithm, model_name,
        num_data=args.num_data, beta=args.beta, percentile=args.percentile,
    )

def extract_data(data, attack_algorithm):
    if attack_algorithm == "SIRA":
        watermarked_text = data['watermarked_text']
        attacked_text = data['SIRA_text']
    else:
        watermarked_text = data['watermarked_text']
        attacked_text = data['attacked_text']
    
    return watermarked_text, attacked_text
        
def main(args):
    with open(args.model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)
        model_cfg = SimpleNamespace(**model_cfg)

    algorithms = args.algorithms
    model_name = model_cfg.name.split("/")[-1]
    
    print(f"\nEvaluating on {algorithms}")

    detector_model = AutoModelForCausalLM.from_pretrained("/models/opt-1.3b").to(device)
    detector_tokenizer = AutoTokenizer.from_pretrained("/models/opt-1.3b")
    transformers_config = TransformersConfig(model=detector_model,
                                                tokenizer=detector_tokenizer,
                                                vocab_size=detector_model.config.vocab_size,
                                                device=device,
                                                max_new_tokens=230,
                                                do_sample=True,
                                                no_repeat_ngram_size=4)

    human_text_dataset = C4Dataset(args.dataset_path, max_samples=500)
    os.makedirs(args.human_text_result_save_dir, exist_ok=True)
    
    for algorithm in algorithms:
        print(f"\nEvaluating {algorithm}")

        my_watermark = AutoWatermark.load(f'{algorithm}', 
        algorithm_config=f'./config/{algorithm}.json',
        transformers_config=transformers_config)

        human_text_results = load_or_compute_human_text_results(my_watermark, human_text_dataset, args.human_text_result_save_dir, algorithm)

        dataset_path = attack_result_path_for(args, model_name, algorithm)

        if args.attack_algorithms == "no-attack":
            with open(dataset_path, 'r') as f:
                watermarked_dataset = json.load(f)[:500]
                watermarked_dataset = [item['watermarked_text'] for item in watermarked_dataset]
            
            if os.path.exists(os.path.join(args.no_attack_result_save_dir, f"{algorithm}.json")):
                print(f"\ncache exists for no-attack results")
                with open(os.path.join(args.no_attack_result_save_dir, f"{algorithm}.json"), 'r') as f:
                    watermarked_results = json.load(f)
            else:
                return_type = DetectionPipelineReturnType.SCORES

                no_attack_pipline = UnWatermarkedTextDetectionPipeline(dataset=watermarked_dataset, text_editor_list=[],
                                                    show_progress=True, return_type=return_type)
                watermarked_results = no_attack_pipline.evaluate(my_watermark)
                os.makedirs(args.no_attack_result_save_dir, exist_ok=True)
                with open(os.path.join(args.no_attack_result_save_dir, f"{algorithm}.json"), 'w') as f:
                    json.dump(watermarked_results, f, indent=4)
        else:
            with open(dataset_path, 'r') as f:
                attack_results = json.load(f)
            watermarked_results = [item['score'] for item in attack_results
                                   if isinstance(item, dict) and 'score' in item]

        detectability = evaluate_detectability(
            watermarked_results, human_text_results, args.labels,
            rules=args.rules, target_fprs=args.target_fprs,
        )
        # Embed the detectability metrics back into the attack-result JSON (same file).
        attack_results = attach_detectability(attack_results, detectability)
        with open(dataset_path, 'w') as f:
            json.dump(attack_results, f, indent=4)
        for name, metric in detectability.items():
            print(f"  {name}: {metric}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the text quality"
    )

    parser.add_argument(
        "--model_cfg_path",
        default="./model_config/llama3.1-8b.yaml",
    )

    parser.add_argument('--labels', nargs='+', default=['TPR', 'F1'])
    parser.add_argument('--rules', nargs='+', default=['target_fpr', 'best'],
                        help="Detectability decision rules to evaluate: 'target_fpr' (swept over "
                             "--target_fprs) and/or 'best' (best-threshold F1).")
    parser.add_argument('--target_fprs', type=float, nargs='+', default=[0.01, 0.1],
                        help="Fixed FPRs at which to report TPR for the 'target_fpr' rule.")
    
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument("--attack_result_save_dir", type=str)
    parser.add_argument("--human_text_result_save_dir", type=str)
    parser.add_argument("--no_attack_result_save_dir", type=str)

    parser.add_argument("--attack_algorithms",type=str, choices=['no-attack', "BIRA", "SIRA", "vanilla_paraphrasing", 'dipper-1', 'dipper-2'])

    parser.add_argument("--num_data", type=int, default=500)

    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--percentile", type=int, default=0)
    
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
        help="List of watermarking algorithms to use",
    )
    

    args = parser.parse_args()
    main(args)


