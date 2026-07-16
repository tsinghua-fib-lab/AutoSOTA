import argparse

import yaml
from types import SimpleNamespace

import json
import os

from tqdm import tqdm

from eval_pipelines import BertScorePipeline, PPLScorePipeline, LLMAsJudge, NLIEquivalenceScorePipeline, SelfBLEUScorePipeline
import re

import sys
sys.path.append(".")
from attack_utils.utils import HfModel, APIModel
from evaluation.paths import attack_result_path

def get_dataset_path(args, model_name, algorithm=None):
    return attack_result_path(
        args.result_save_dir, args.attack_algorithms, algorithm, model_name,
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
    if args.attack_algorithms in ["BIRA", "SIRA", "vanilla_paraphrasing"]:
        with open(args.model_cfg_path, "r") as f:
            model_cfg = yaml.safe_load(f)
            model_cfg = SimpleNamespace(**model_cfg)

    if args.metric == "s-bert":
        pipeline = BertScorePipeline()
    elif args.metric == "ppl":
        with open(args.ppl_eval_model_cfg_path, "r") as f:
            ppl_eval_model_cfg = yaml.safe_load(f)
            ppl_eval_model_cfg = SimpleNamespace(**ppl_eval_model_cfg)
        
        print(f"\nLoad evaluation model for PPL: {ppl_eval_model_cfg.name}\n")
        model = HfModel(model_cfg=ppl_eval_model_cfg)
        pipeline = PPLScorePipeline(model=model.model, tokenizer=model.tokenizer)
    elif args.metric == "nli":
        pipeline = NLIEquivalenceScorePipeline()
    elif args.metric == "gpt":
        pass
    elif args.metric == "llm_judge":
        model = APIModel(model_name=args.llm_judge_model_name)
        pipeline = LLMAsJudge(model=model, max_tokens=5)
    elif args.metric == "self_bleu":
        pipeline = SelfBLEUScorePipeline()
    else:
        NotImplementedError(f"{args.metric} is not implemented")
    
    print(f"\nattack_algorithm: {args.attack_algorithms}")
    print(f"\nEvaluate the text with {args.metric}")
    

    algorithms = args.algorithms
    if args.attack_algorithms in ["BIRA", "SIRA", "vanilla_paraphrasing"]:
        model_name = model_cfg.name.split("/")[-1]
    else:
        model_name = None
    
    print(f"\nEvaluating on {algorithms}")

    for algorithm in algorithms:
        print(f"\nEvaluating {algorithm}")
        dataset_path = get_dataset_path(args, model_name=model_name, algorithm=algorithm)

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # The file is `num_data` per-sample records followed by a few summary/metadata dicts
        # (e.g. avg_ASR, detectability). We score the leading records and pass the rest through,
        # so the only precondition is that the records we intend to score are actually present.
        assert len(dataset) >= args.num_data, \
            f"result file has {len(dataset)} entries, fewer than num_data={args.num_data}: {dataset_path}"

        total_results = []
        total_scores = 0

        if args.metric == "llm_judge":
            watermarked_datasets = []
            attacked_datasets = []
            
            for data_idx, data in enumerate(tqdm(dataset[:args.num_data])):
                watermarked_text, attacked_text = extract_data(data, attack_algorithm=args.attack_algorithms)
                watermarked_datasets.append(watermarked_text)
                attacked_datasets.append(attacked_text)
            
            scores = pipeline.evaluate(watermarked_datasets=watermarked_datasets, attacked_datasets=attacked_datasets)

            for data_idx, (data, score) in enumerate(zip(tqdm(dataset[:args.num_data]), scores)):
                score = int(re.search(r'\[(\d+)\]', score).group(1))
                
                data[f"{args.metric} score"] = score
                total_results.append(data)
                total_scores += score
            
            for data in dataset[args.num_data:]:
                total_results.append(data)
        elif args.metric == "nli":
            for data_idx, data in enumerate(tqdm(dataset[:args.num_data])):
                watermarked_text, attacked_text = extract_data(data, attack_algorithm=args.attack_algorithms)
                label, score = pipeline.evaluate(reference_text=watermarked_text, target_text=attacked_text)
                
                data[f"{args.metric} label"] = label
                data[f"{args.metric} score"] = score
                total_results.append(data)

                if label == "entailment":
                    total_scores += 1
                    
            for data in dataset[args.num_data:]:
                total_results.append(data)
        elif args.metric in ["s-bert", 'ppl', 'self_bleu']:
            for data_idx, data in enumerate(tqdm(dataset[:args.num_data])):
                watermarked_text, attacked_text = extract_data(data, attack_algorithm=args.attack_algorithms)
                score = pipeline.evaluate(reference_text=watermarked_text, target_text=attacked_text)
                data[f"{args.metric} score"] = score
                total_results.append(data)
                total_scores += score
            
            for data in dataset[args.num_data:]:
                total_results.append(data)

        avg_scores = total_scores / args.num_data
        print(f"\navg_scores: {avg_scores:.4f}")
        
        total_results.append({f"avg_{args.metric} score": avg_scores})

        assert len(total_results) == len(dataset) + 1, f"# of total results: {len(total_results)}"

        base, _ = os.path.splitext(dataset_path)
        save_path = f"{base}_text_quality_{args.metric}.json"

        with open(save_path, "w") as f:
            json.dump(total_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the text quality"
    )

    parser.add_argument(
        "--ppl_eval_model_cfg_path",
        default="./model_config/mistral-7b-v0.1.yaml",
    )

    parser.add_argument(
        "--model_cfg_path",
    )

    parser.add_argument("--metric", type=str, default="s-bert", choices=['s-bert', 'ppl', 'llm_judge', 'nli', 'self_bleu'])
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument("--result_save_dir", type=str, default="./experimental_results")

    parser.add_argument("--attack_algorithms",type=str, choices=["BIRA", "SIRA", "vanilla_paraphrasing", 'dipper-1', 'dipper-2'])
    
    parser.add_argument("--num_data", type=int)

    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--percentile", type=int, default=0)

    parser.add_argument("--llm_judge_model_name", type=str, default="gpt4o", choices=["gpt4o", "gpt4o-mini", "gpt5"])
    
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


