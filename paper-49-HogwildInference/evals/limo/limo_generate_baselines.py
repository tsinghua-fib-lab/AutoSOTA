import argparse
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import sys; sys.path.insert(0, "../utils"); sys.path.insert(0, "../..");

from formatting import MathFormatting
from generation import finalize_response_with_s1_finisher
from gpu_parallel import get_worker_rank, init_worker_logger
from task_queue import TaskQueue


def parse_args():
    parser = argparse.ArgumentParser(description="Eval baselines")
    parser.add_argument(
        "--model_name",
        type=str,
        default='Qwen/QwQ-32B',
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=(256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384),
        help="A list of budgets "
    )
    parser.add_argument(
        "--finisher_max_new_tokens",
        type=int,
        default=16,
        help="If there is no answer by a given budget, prompt the model to give the answer and give it this many tokens"
             " | set to 0 for naive baseline, positive values for tuned baseline (same as in lcb_generate_hogwild)"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Endpoint for a zmq task dispenser that dispenses task indices. Provide *either* this or start & end"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First task to be processed by script inclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last task to be processed by script exclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--eval_folder",
        type=str,
        default='.',
        help='Results will be written to "args.eval_folder/evals_data/limo/exp_name".'
    )
    parser.add_argument(
        "--dump_snapshot_freq",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Load model from_pretrained with this as device_map'
    )
    args = parser.parse_args()
    return args


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_result(
    response: str,
    prefix: str = "\\boxed{",
    suffix: str = "}",
    extract_result: Callable[[str], int] = lambda box: int("".join(x for x in box if x.isdigit()))
) -> Optional[int]:
    """Extract the rightmost entry between prefix and suffix"""
    while True:
        try:
            start = response.rindex(prefix)
            try:
                end = response.index(suffix, start)
                return extract_result(response[start + len(prefix):end])
            except ValueError:  # missing suffix or extract_result failed
                response = response[:start]
        except ValueError:
            return None


class ResultStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        result = get_result(decoded)
        return result is not None


def main():
    torch.set_grad_enabled(False)
    device = torch.device('cuda')
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    cots_directories_by_budget = {
        budget: os.path.join(
            args.eval_folder,
            f"evals_data/limo/{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}"
        ) 
        for budget in args.budgets
    }
    logger.info(f'Output directory: {cots_directories_by_budget.values()}')

    for cots_directory in cots_directories_by_budget.values():
        if not os.path.exists(cots_directory):
            os.makedirs(cots_directory, exist_ok=True)
            logger.info(f'Created directory {cots_directory}')
        else:
            logger.info(f'Directory {cots_directory} already exists')

    logger.info('Loading model and tokenizer')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map, revision=args.revision, torch_dtype='auto',
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    fmt = MathFormatting(tokenizer, extract_result=lambda box: int("".join(x for x in box if x.isdigit())))
    # ^-- fmt is only used for S1-like early finisher
    stopping_criteria = StoppingCriteriaList([ResultStopCriteria(tokenizer)])

    logger.info('Loading dataset')
    dataset = load_dataset("GAIR/LIMO", split="train")
    accuracy_numerator = accuracy_denominator = 0

    def _solve_task_and_save(idx: int):
        nonlocal accuracy_numerator, accuracy_denominator
        if os.path.exists(f'{cots_directories_by_budget[args.budgets[-1]]}/Task_{idx}.txt'):
            return  # already solved by previous attempt and saved in snapshot
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        prompt_str = system_prompt + str(dataset[idx]['question'])
        prompt_with_template_str = tokenizer.apply_chat_template(
            [dict(role='user', content=prompt_str)],
            add_generation_prompt=True, tokenize=False
        )
        prompt = torch.tensor(tokenizer.encode(prompt_with_template_str, add_special_tokens=False)).to(device)[None, :]
        fix_seed(args.seed)
        response_ids = model.generate(prompt, max_new_tokens=max(args.budgets), stopping_criteria=stopping_criteria,
                                      eos_token_id=tokenizer.eos_token_id)
        response_str = tokenizer.decode(response_ids[0])

        n_tokens_in_prompt = prompt.numel()
        for budget, cots_directory in cots_directories_by_budget.items():
            response_prefix = tokenizer.decode(response_ids[0, n_tokens_in_prompt: n_tokens_in_prompt + budget])
            if fmt.get_final_answer(response_prefix) is not None:
                response_for_budget = response_prefix  # found answer, no need for S1 prompt
            else:
                response_for_budget = finalize_response_with_s1_finisher(
                    response=response_prefix, model=model, tokenizer=tokenizer, fmt=fmt,
                    max_new_tokens=args.finisher_max_new_tokens
                )
            with open(f'{cots_directory}/Task_{idx}.txt', 'w') as file:
                file.write(response_for_budget)

        result = get_result(response_str)
        gt = int(dataset[idx]['answer'])
        accuracy_numerator += int(gt == result)
        accuracy_denominator += 1
        current_accuracy = (accuracy_numerator / accuracy_denominator)
        logger.info(f"{idx=}, {gt=}, {result=}| {current_accuracy=:.3f} (budget={max(args.budgets)})")

    if args.start is not None and args.end is not None:
        logger.info(f'Generating tasks [{args.start}; {args.end})')
        for idx in tqdm(range(args.start, args.end), desc=f'Process {rank}'):
            _solve_task_and_save(idx)
    elif args.queue is not None:
        logger.info(f'Generating tasks from {args.queue}')
        for idx in tqdm(TaskQueue.iterate_tasks_from_queue(endpoint=args.queue), desc=f"Process {rank}"):
            _solve_task_and_save(idx)
    else:
        raise NotImplementedError("Please specify either --queue or both --start and --end")
    logger.info(f'Process {rank} has finished.')


if __name__ == "__main__":
    main()
