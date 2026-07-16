import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import pandas as pd
import pathlib
import time
import torch
from typing import Any, Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel

from monkey_query_utils import create_prompts_and_answers, exact_match, quasi_exact_match, is_equiv_chain_of_thought, final_number_exact_match

def generate_outputs_from_model(
    model_nickname: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset: str = "math",
    temperature: float = 1.0,
    max_tokens: int = 512,
    num_prompts_to_use: int = 100,
    total_num_samples_per_prompt: int = 10000,
    num_samples_per_sampling_call: int=128,
):
    policy_model_sampled_outputs_dir = os.path.join(
        "../../data/monkey_query/eval_results",
        dataset,
        model_nickname.split("/")[-1],
    )
    os.makedirs(policy_model_sampled_outputs_dir, exist_ok=True)

    data: Dict[str, List[str]] = create_prompts_and_answers(
        model_nickname=model_nickname,
        dataset=dataset,
        num_prompts_to_use=num_prompts_to_use,
    )
    sample_outputs_from_policy_model_and_write_to_disk(
        dataset=dataset,
        data=data,
        model_nickname=model_nickname,
        model_sampled_outputs_dir=policy_model_sampled_outputs_dir,
        total_num_samples_per_prompt=total_num_samples_per_prompt,
        num_samples_per_sampling_call=num_samples_per_sampling_call,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def sample_outputs_from_policy_model_and_write_to_disk(
    dataset: str,
    model_sampled_outputs_dir: str,
    data: Dict[str, List[str]],
    model_nickname: str,
    total_num_samples_per_prompt: int = 10000,
    num_samples_per_sampling_call: int = 128,
    temperature: float = 1.0,
    max_tokens: int = 512,
):
    kwargs = {
        "model": model_nickname,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.95,
        # "swap_space": 4,  # GiB
        # "swap_space": 24,  # GiB
        "enable_prefix_caching": True,
        "trust_remote_code": True
    }
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        num_gpus = len(cuda_visible.split(","))
        kwargs["tensor_parallel_size"] = num_gpus

    # Load the model.
    model = LLM(**kwargs)
    print(f"Loaded model {model_nickname}.")
    model_nickname_list = [model_nickname for _ in range(num_samples_per_sampling_call)]

    problems = data["problems"]
    problems_indices = data["problems_indices"]
    prompts = data["prompts"]
    # levels = data["levels"]
    # problem_types = data["problem_types"]
    solutions = data["solutions"]

    if dataset == "gsm":
        evaluate_fn = final_number_exact_match
    elif dataset in ["mmlu", "commonsense"]:
        evaluate_fn = exact_match
    elif dataset in ["med_qa", "legalbench", "bbq", "lsat_qa", "legal_support"]:
        evaluate_fn = quasi_exact_match
    elif dataset == "math":
        evaluate_fn = is_equiv_chain_of_thought
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")

    # for problem, prompt_idx, prompt, level, problem_type, solution in zip(
    for problem, prompt_idx, prompt, solution in zip(
        problems,
        problems_indices,
        prompts,
        # levels,
        # problem_types,
        solutions,
    ):
        repeated_problems = [problem for _ in range(num_samples_per_sampling_call)]
        repeated_prompts = [prompt for _ in range(num_samples_per_sampling_call)]
        repeated_prompts_indices = [prompt_idx for _ in range(num_samples_per_sampling_call)]
        # repeated_levels = [level for _ in range(num_samples_per_sampling_call)]
        # repeated_problem_types = [problem_type for _ in range(num_samples_per_sampling_call)]

        for batch_generation_idx in range(
            0,
            total_num_samples_per_prompt,
            num_samples_per_sampling_call,
        ):
            print(f"Prompt {prompt_idx}: Sampled {batch_generation_idx} / {total_num_samples_per_prompt}.")
            model_sampled_outputs_batch_filepath = (
                pathlib.Path(model_sampled_outputs_dir)
                / f"prompt={prompt_idx}"
                / f"batch_{str(batch_generation_idx).zfill(16)}.parquet"
            )

            # Skip if this file exists and is not empty.
            if (
                model_sampled_outputs_batch_filepath.exists()
                and model_sampled_outputs_batch_filepath.stat().st_size > 0
            ):
                continue

            # Otherwise, create the file and sample.
            model_sampled_outputs_batch_filepath.parent.mkdir(
                parents=True, exist_ok=True
            )
            model_sampled_outputs_batch_filepath.touch()

            model_sampling_params = SamplingParams(
                n=num_samples_per_sampling_call,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=batch_generation_idx,
            )

            try:
                requests_outputs: List[RequestOutput] = model.generate(
                    prompts=[prompt], sampling_params=model_sampling_params
                )
            except Exception as e:
                print(f"Error occurred while sampling from model: {e}")
                continue

            responses = [
                output.text
                for request_output in requests_outputs
                for output in request_output.outputs
            ]
            num_input_tokens = [
                len(requests_outputs[0].prompt_token_ids)
                for _ in range(num_samples_per_sampling_call)
            ]
            num_output_tokens = [
                len(output.token_ids)
                for request_output in requests_outputs
                for output in request_output.outputs
            ]
            responses_indices = [
                idx
                for idx in range(
                    batch_generation_idx,
                    batch_generation_idx + num_samples_per_sampling_call,
                )
            ]
            scores = [float(evaluate_fn(response, solution)) for response in responses]

            prompts_and_responses_df = pd.DataFrame(
                {
                    "Model Nickname": model_nickname_list,
                    "prompt_idx": repeated_prompts_indices,
                    "response_idx": responses_indices,
                    "score": scores,
                    "problem": repeated_problems,
                    "prompt": repeated_prompts,
                    "response": responses,
                    "num_input_tokens": num_input_tokens,
                    "num_output_tokens": num_output_tokens,
                    # "level": repeated_levels,
                    # "problem_type": repeated_problem_types,
                }
            )

            prompts_and_responses_df.to_parquet(
                model_sampled_outputs_batch_filepath,
                index=False,
            )
            print(f"Saved {model_sampled_outputs_batch_filepath}.")

            del responses
            del prompts_and_responses_df
            gc.collect()

        del repeated_prompts, repeated_prompts_indices

    print("Sampled outputs from model.")

    # Freeing up VLLM memory is harder than I thought!
    # See: https://github.com/vllm-project/vllm/issues/1908
    # Hit it with everything recommended!
    destroy_model_parallel()
    # del model.llm_engine.model_executor.driver_worker
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(7)
    print("Finished cleaning up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate outputs from a language model."
    )
    parser.add_argument(
        "--model_nickname",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm",
        # required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num_prompts_to_use", type=int, default=997, help="Number of prompts to use"
        # "--num_prompts_to_use", type=int, default=96, help="Number of prompts to use"
    )
    parser.add_argument(
        "--total_num_samples_per_prompt",
        type=int,
        default=10000,
        # default=500,
        help="Total number of samples per prompt",
    )
    parser.add_argument(
        "--num_samples_per_sampling_call",
        type=int,
        default=256,
        # default=50,
    )
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    generate_outputs_from_model(
        model_nickname=args.model_nickname,
        dataset=args.dataset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_prompts_to_use=args.num_prompts_to_use,
        total_num_samples_per_prompt=args.total_num_samples_per_prompt,
        num_samples_per_sampling_call=args.num_samples_per_sampling_call,
    )
    print("Finished sample_outputs_from_policy_model.py!")
