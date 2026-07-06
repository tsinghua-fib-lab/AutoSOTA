#!/usr/bin/env python3
"""
Generate detoxified responses using TIDE (Test-time Iterative Detoxification via Embeddings).
This script performs zeroth-order gradient descent on prompt embeddings to minimize output toxicity.
"""

import argparse
import json
import logging
import os
import sys

import torch.nn.functional as F
from tqdm import tqdm
from vllm import SamplingParams

from utils.model import (cleanup_vllm_and_cuda, decode_embedding,
                         get_prompt_embeds, get_vllm_text_output, init_model)
from utils.toxicity import Toxicity
from utils.utils import (find_or_create_config_dir,
                         get_responses_above_threshold, numpy_serializer,
                         print_toxicity_statistics, set_seed)
from utils.tide import backward, normalize_grad, project_cosine

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Suppress vLLM's verbose logging
logging.getLogger("vllm").setLevel(logging.WARNING)


PERSPECTIVE_QPS = 100

SAVE_INTERVAL = 1  # per prompt


def main(args):
    # Set seed
    set_seed(args.seed)

    # Load the baseline model records for the given dataset
    baseline_path = os.path.join("responses",
                                 "baselines",
                                 args.model.split("/")[-1].lower(),
                                 f"temp={args.temperature}-K={args.K}",
                                 f"{args.dataset}.json")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model records not found at {baseline_path}")

    baseline_records = json.load(open(baseline_path))
    subset_records = get_responses_above_threshold(baseline_records, args.toxic_th, mode=args.toxic_th_mode) if args.toxic_th != -1 else baseline_records

    # Dynamically set the context length based on the max number of tokens in the prompt
    max_num_tokens = max([entry['num_tokens_prompt'] for entry in subset_records])
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

    # Sampling parameters
    sampling_params = SamplingParams(
        n=1,  # only one response per prompt with TIDE
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Find or create config directory
    # Create configuration parameters dictionary
    config_params = {
        'K': args.K,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'toxic_th': args.toxic_th,
        'toxic_th_mode': args.toxic_th_mode,
        'early_stopping_th': args.early_stopping_th,
        'cosine_sim_th': args.cosine_sim_th,
        'mu': args.mu,
        'N': args.N,
        'stepsize': args.stepsize,
        'num_iter': args.num_iter
    }
    
    # Check if the dataset is already in the config directory
    base_dir = os.path.join("responses",
                            "tide",
                            args.model.split("/")[-1].lower())
    exists, config_dir = find_or_create_config_dir(base_dir, config_params)

    save_name = os.path.join(config_dir, f"{args.dataset}.json")

    # Load results if they exist
    if exists and os.path.exists(save_name):
        results = json.load(open(save_name))
        logger.info(f"↳ Loaded existing results with {len(results)} samples from {save_name}")
    else:   
        results = []
        if exists:
            logger.info(f"↳ Config directory exists but no results found for {args.dataset}. Starting from scratch.")
        else:
            logger.info(f"↳ No results found for {args.dataset}. Starting from scratch.")

    total_iterations = len(subset_records) * args.K
    
    logger.info(f"Starting TIDE optimization: {len(subset_records)} prompts × {args.K} trials = {total_iterations} total iterations")
    
    p_bar = tqdm(total=total_iterations,
                 desc=f"Running zeroth-order optimization ({len(subset_records)} prompts x {args.K} trials)",
                 leave=True,)
    
    for i, record in enumerate(subset_records):
        # Skip if the record already exists in the results
        if any(entry['id'] == record['id'] for entry in results):
            p_bar.update(args.K)
            continue
        
        final_toxicity_scores, final_cosine_sims, comp_str, num_total_iter = [], [], [], [0] * args.K
        for trial in range(args.K):
            current_embeds = get_prompt_embeds(record['prompt'], tokenizer, embed_layer).clone()
            original_embeds = current_embeds.clone()  # Store original for projection   

            # Start the zeroth-order optimization
            for _ in range(args.num_iter):
                output = llm.generate({'prompt_embeds': current_embeds},
                                       sampling_params=sampling_params,
                                       use_tqdm=False)
                completion = get_vllm_text_output(output)[0]
                toxicity = toxicity_client.predict(completion)[-1][0]

                grad = backward(llm, current_embeds, sampling_params, toxicity_client, mu=args.mu, N=args.N)
                grad = normalize_grad(grad)
                current_embeds = current_embeds - args.stepsize * grad

                # Project back onto cosine similarity ball to avoid semantic drift
                if args.cosine_sim_th is not None and args.cosine_sim_th > 0.0:
                    current_embeds = project_cosine(current_embeds, original_embeds, args.cosine_sim_th)

                cos_sim_after = F.cosine_similarity(current_embeds.flatten(), original_embeds.flatten(), dim=0)

                # Increment the total number of iterations
                num_total_iter[trial] += 1

                # Early stopping conditions
                if toxicity < args.early_stopping_th:
                    break
                    
            # Check the final toxicity score, cosine similarity, and the steered input prompt text
            final_toxicity_scores.append(toxicity)
            final_cosine_sims.append(cos_sim_after)
            comp_str.append(completion)

            p_bar.update(1)
            p_bar.set_postfix({"prompt": f"{i+1}/{len(subset_records)}", "trial": f"{trial+1}/{args.K}"})

        entry = {
            'id': record['id'],
            'prompt': record['prompt'],
            # Keep track of the final prompt after TIDE optimization
            'final_prompt': decode_embedding(current_embeds, embed_layer, tokenizer, metric='cosine'),
            'num_tokens_prompt': record['num_tokens_prompt'],
            'num_iter_steps': num_total_iter,
            'completions': comp_str,
            'toxicity_scores': final_toxicity_scores,
            'cosine_sims': final_cosine_sims,
            'avg_toxicity': sum(final_toxicity_scores) / len(final_toxicity_scores),
            'max_toxicity': max(final_toxicity_scores),
        }
        results.append(entry)

        if i % SAVE_INTERVAL == 0:
            with open(os.path.join(config_dir, f"{args.dataset}.json"), 'w') as f:
                json.dump(results, f, indent=4, default=numpy_serializer)
    
    p_bar.close()

    # Save results    
    logger.info(f"↳ Saving the final results to {config_dir}")
    
    with open(save_name, 'w') as f:
        json.dump(results, f, indent=4, default=numpy_serializer)

    # Print the final results
    print_toxicity_statistics(results, args.dataset, args.model.split('/')[-1])
    
    # Clean up vLLM and CUDA resources
    logger.info("Cleaning up GPU resources...")
    cleanup_vllm_and_cuda(llm=llm, embed_layer=embed_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate detoxified responses using TIDE"
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

    # Zeroth-order optimization parameters
    parser.add_argument("--toxic-th",
                        type=float,
                        default=0.9,
                        help="Toxicity threshold to filter the inputs. Inputs with >= this threshold will be optimized. "
                        "If None, all inputs will be optimized.")

    parser.add_argument("--early-stopping-th",
                        type=float,
                        default=0.5,
                        help="Early stopping threshold for the zeroth-order optimization.")

    parser.add_argument("--toxic-th-mode",
                        type=str,
                        default='mean',
                        help="Mode to use for the toxicity threshold. "
                        "If 'max', the maximum toxicity score will be used. "
                        "If 'mean', the mean toxicity score will be used.")

    parser.add_argument("--cosine-sim-th",
                        type=float,
                        default=0.0,
                        help="Cosine similarity threshold to project the embeddings back onto the original prompt.")

    parser.add_argument("--mu",
                        type=float,
                        default=0.1,
                        help="Perturbation scale for the zeroth-order optimization")

    parser.add_argument("--N",
                        type=int,
                        default=16,
                        help="Number of random noise initializations per optimization step")

    parser.add_argument("--stepsize",
                        type=float,
                        default=1.0,
                        help="Stepsize for the zeroth-order optimization")

    parser.add_argument("--num-iter",
                        type=int,
                        default=10,
                        help="Number of iterations for the zeroth-order optimization per batch")

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
                        default=0.25,
                        help="GPU memory utilization")

    # Reproducibility
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed for reproducibility")

    args = parser.parse_args()

    main(args)

