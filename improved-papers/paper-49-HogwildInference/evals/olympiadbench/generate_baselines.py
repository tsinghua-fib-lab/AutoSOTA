import argparse
import os
import random
import json
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import StoppingCriteria

import sys; sys.path.insert(0, "../utils"); sys.path.insert(0, "../..");

from inference.judge import find_last_valid_expression
from gpu_parallel import get_worker_rank, init_worker_logger
from task_queue import TaskQueue
from generation import finalize_response_with_s1_finisher
from formatting import CommonFormatting
from inference.code.evaluators.llama_evaluator import LlamaEvaluator


class OlympiadBenchFormatting(CommonFormatting):
    s1_finisher_suffix = (f"\n\nWait, given the limited time, I have to give an answer right now. "
                          "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{")

    def get_final_answer(self, response: str) -> Optional[str]:
        return find_last_valid_expression(response, prefix="\\boxed{")


def parse_args():
    parser = argparse.ArgumentParser(description="Eval baselines")

    parser.add_argument(
        "--model_name",
        type=str,
        default='Qwen/QwQ-32B',
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Load model from_pretrained with this as device_map'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=(256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384),
        help="A list of budgets "
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
        "--dataset_path",
        type=str
    )

    parser.add_argument(
        "--finisher_max_new_tokens",
        type=int,
        default=64,
        help="If there is no answer by a given budget, prompt the model to give the answer and give it this many tokens"
             " | set to 0 for naive baseline, positive values for tuned baseline (same as in generate_hogwild)"
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
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    cots_directories_by_budget = {budget: os.path.join(
        args.eval_folder,
        f"evals_data/olympiadbench/{args.dataset_path.split('/')[-1].split('.')[0]}/{args.model_name.split('/')[-1]}/{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}"
    ) for budget in args.budgets}
    logger.info(f'Output directory: {cots_directories_by_budget.values()}')

    for cots_directory in cots_directories_by_budget.values():
        if not os.path.exists(cots_directory):
            os.makedirs(cots_directory, exist_ok=True)
            logger.info(f'Created directory {cots_directory}')
        else:
            logger.info(f'Directory {cots_directory} already exists')

    logger.info('Loading dataset')
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        json_dataset = json.load(f)
    tasks_solved = 0

    logger.info('Loading model and tokenizer')
    evaluator = LlamaEvaluator(
        model_name=args.model_name, budget=max(args.budgets), revision=args.revision,
        device_map=args.device_map, torch_dtype='auto'
    )
    evaluator.is_theorem_proving = 'TP' in args.dataset_path
    evaluator.is_math = 'math' in args.dataset_path
    evaluator.is_chinese = 'zh' in args.dataset_path
    fmt_s1_baseline = OlympiadBenchFormatting(evaluator.tokenizer)

    def _solve_task_and_save(idx: int):
        nonlocal tasks_solved
        fix_seed(args.seed)
        if os.path.exists(f'{cots_directories_by_budget[args.budgets[-1]]}/Task_{idx}.json'):
            return  # already solved by previous attempt and saved in snapshot

        question = json_dataset[idx]
        prompt = evaluator.make_prompt(question)
        if evaluator.is_math:
            input = evaluator.make_input(prompt, question['question'])
        else:
            if 'context' in question.keys() and question['context']:  # cannot be null
                input = evaluator.make_input(prompt, question['context'] + '\n' + question['question'])
            else:
                input = evaluator.make_input(prompt, question['question'])
        full_answer = evaluator.get_answer(input)
        full_response = input[0]['content'] + full_answer

        n_tokens_in_prompt = len(evaluator.tokenizer.encode(input[0]['content'], add_special_tokens=False))
        response_ids = evaluator.tokenizer(full_response, add_special_tokens=False, return_tensors='pt')['input_ids']

        for budget, cots_directory in cots_directories_by_budget.items():
            response_prefix = evaluator.tokenizer.decode(response_ids[0, n_tokens_in_prompt: n_tokens_in_prompt + budget])
            if args.finisher_max_new_tokens > 0:
                response_for_budget = finalize_response_with_s1_finisher(
                    response=response_prefix, model=evaluator.model, tokenizer=evaluator.tokenizer, fmt=fmt_s1_baseline,
                    max_new_tokens=args.finisher_max_new_tokens)

                if 'model_output' not in question.keys():
                    question['model_output'] = {f"{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}": {'raw_output': response_for_budget}}
                else:
                    question['model_output'][f"{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}"] = {'raw_output': response_for_budget}

                with open(f'{cots_directory}/Task_{idx}.json', 'w', encoding='utf-8') as f:
                    json.dump([question], f, ensure_ascii=False, indent=4)

        tasks_solved += 1
        logger.info(f"{idx=} | (budget={max(args.budgets)})")

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
