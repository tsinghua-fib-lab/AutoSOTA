#!/usr/bin/env python3
"""
Generate base model responses on a dataset without any intervention.
This script collects completions and toxicity scores for baseline evaluation.
"""

import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
from vllm import SamplingParams

from utils.model import get_prompt_embeds, init_model
from utils.toxicity import Toxicity
from utils.utils import numpy_serializer, print_toxicity_statistics, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


PERSPECTIVE_QPS = 'inf'


def main(args):
    # Set seed
    set_seed(args.seed)

    # Load dataset
    dataset_path = os.path.join("datasets", args.dataset + ".csv")
    df = pd.read_csv(dataset_path)

    prompts = df["prompt"].tolist()

    # Dynamically set the context length based on the max number of tokens in the prompt
    max_num_tokens = df["num_tokens_prompt"].max()
    context_len = max_num_tokens + args.max_tokens + 20  # leave a tolerance of 20 tokens

    # Initialize detoxify client
    toxicity_client = Toxicity(qps=PERSPECTIVE_QPS)

    # Initialize model
    model_args = {
        "model": args.model,
        "max_model_len": context_len,
        "enable_prompt_embeds": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "auto",
        "trust_remote_code": True,
    }

    tokenizer, embed_layer, llm = init_model(model_args)

    # Get the embeddings of the prompts
    p_bar = tqdm(prompts, desc="Getting embeddings")

    prompt_embeds = [{
            'prompt_embeds': get_prompt_embeds(p, tokenizer, embed_layer),
        } for p in p_bar]

    # Sampling parameters
    sampling_params = SamplingParams(
        n=args.K,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    completions = llm.generate(
        prompt_embeds,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    # Results path
    output_dir = os.path.join("responses",
                              "baselines",
                              args.model.split("/")[-1].lower(),
                              f"temp={args.temperature}-K={args.K}")
    os.makedirs(output_dir, exist_ok=True)

    # Compute the toxicity scores
    results = []

    p_bar = tqdm(enumerate(zip(prompts, completions)),
                 total=len(prompts),
                 desc="Saving results")

    for i, (inp, comp) in p_bar:
        comp_str = [o.text for o in comp.outputs]
        _, toxicity_scores = toxicity_client.predict(comp_str)

        entry = {
            'id': i,
            'prompt': inp,
            'num_tokens_prompt': df.iloc[i]['num_tokens_prompt'],
            'completions': comp_str,
            'toxicity_scores': toxicity_scores,
            'avg_toxicity': sum(toxicity_scores) / len(toxicity_scores),
            'max_toxicity': max(toxicity_scores)
        }
        results.append(entry)
    
    # Save results
    with open(os.path.join(output_dir, f"{args.dataset}.json"), 'w') as f:
        json.dump(results, f, indent=4, default=numpy_serializer)

    # Print the final results
    print_toxicity_statistics(results, args.dataset, args.model.split('/')[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate base model responses on a dataset"
    )
    parser.add_argument("--model",
                        type=str,
                        default="openai-community/gpt2-large",
                        help="Model name on Hugging Face Hub. "
                        "Model is expected to be a vLLM-compatible model.")

    parser.add_argument("--dataset",
                        type=str,
                        default="rtp",
                        help="Path to the preprocessed dataset. "
                        "Dataset is expected under './datasets' as a CSV file. "
                        "Prompts are expected to be in the 'prompt' column.")

    # Sampling parameters - literature standard values
    parser.add_argument("--max-tokens",
                        type=int,
                        default=20,
                        help="Maximum number of tokens to generate")

    parser.add_argument("--temperature",
                        type=float,
                        default=0.1,
                        help="Temperature for sampling")

    parser.add_argument("--top-p",
                        type=float,
                        default=1.0,
                        help="Top-p for sampling")

    parser.add_argument("--K",
                        type=int,
                        default=3,
                        help="Number of responses to generate per input")

    # Model parameters
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=2,
                        help="Tensor parallel size")

    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.8,
                        help="GPU memory utilization")

    # Reproducibility
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed for reproducibility")

    args = parser.parse_args()

    main(args)

