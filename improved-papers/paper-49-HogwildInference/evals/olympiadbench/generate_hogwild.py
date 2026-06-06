import argparse
import os
import json
import re
from typing import Optional

import peft
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys; sys.path.insert(0, "../utils"); sys.path.insert(0, "../..");

from inference.judge import find_last_valid_expression
from formatting import CommonFormatting, get_default_options_for_model
from generation import fix_seed, solve_task_2agents
from gpu_parallel import get_worker_rank, init_worker_logger
from task_queue import TaskQueue
from evals.olympiadbench.inference.code.evaluators.evaluator import Evaluator
from inference.code.evaluators.llama_evaluator import LlamaEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Eval threads")

    parser.add_argument(
        "--model_name",
        type=str,
        default='Qwen/QwQ-32B',
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--budgets",
        nargs="+",
        type=int,
        default=(256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384),
        help="A list of budgets "
    )
    parser.add_argument(
        "--finisher_max_new_tokens",
        type=int,
        default=64,
        help="If there is no answer by a given budget, prompt the model to give the answer and give it this many tokens"
             " | set to 0 for naive baseline, positive values for tuned baseline (same as in generate_hogwild)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help='Adapter to load and merge into the model'
    )
    parser.add_argument("--merge_adapter", action="store_true", help="call peft_model.merge_and_unload()")
    parser.add_argument(
        "--eval_folder",
        type=str,
        default='.',
        help='Results will be written to "args.eval_folder/evals_data/limo/exp_name".'
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Load model from_pretrained with this as device_map'
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
    return parser.parse_args()


class OlympiadBenchFormatting(CommonFormatting):
    s1_finisher_suffix = (f"\n\nWait, given the limited time, I have to give an answer right now. "
                          "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{")

    def get_final_answer(self, response: str) -> Optional[str]:
        return find_last_valid_expression(response, prefix="\\boxed{")


class PatchedEvaluator(LlamaEvaluator):
    def __init__(self, model_name, model, tokenizer, k=-1, budget=16384):
        Evaluator.__init__(self, model_name, k)  # instantiate the base evaluator that does not re-download the model
        self.model, self.tokenizer, self.budget = model, tokenizer, budget


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python3 {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    cots_directories_by_budget = {budget: os.path.join(
        args.eval_folder,
        f"evals_data/olympiadbench/{args.dataset_path.split('/')[-1].split('.')[0]}/{args.model_name.split('/')[-1]}/{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}-hogwild"
    ) for budget in args.budgets}
    logger.info(f'Output directory: {cots_directories_by_budget.values()}')

    for cots_directory in cots_directories_by_budget.values():
        if not os.path.exists(cots_directory):
            os.makedirs(cots_directory, exist_ok=True)
            logger.info(f'Created directory {cots_directory}')
        else:
            logger.info(f'Directory {cots_directory} already exists')

    logger.info('Loading model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map, revision=args.revision, torch_dtype='auto',
        low_cpu_mem_usage=True, trust_remote_code=True)
    if args.adapter_path:
        model = peft.PeftModel.from_pretrained(model, args.adapter_path)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.train(False)
    opts = get_default_options_for_model(model)
    logger.info(f"Using inferred formatting options {opts}")
    fmt = OlympiadBenchFormatting(tokenizer, **opts)  # answer can be a latex string with variables

    # instantiate OlympiadBench evaluator; go through the same rites as in the original benchmark except device/dtype
    model.generation_config = GenerationConfig.from_pretrained(args.model_name)  # < from original OlympiadBench
    tokenizer.pad_token_id = tokenizer.eos_token_id  # < from original OlympiadBench
    evaluator = PatchedEvaluator(model_name=args.model_name, model=model, tokenizer=tokenizer, budget=max(args.budgets))
    evaluator.is_theorem_proving = 'TP' in args.dataset_path
    evaluator.is_math = 'math' in args.dataset_path
    evaluator.is_chinese = 'zh' in args.dataset_path

    logger.info('Loading dataset')
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        json_dataset = json.load(f)
    tasks_solved = 0

    def _solve_task_and_save(idx: int):
        nonlocal tasks_solved
        if os.path.exists(f'{cots_directories_by_budget[args.budgets[-1]]}/Task_{idx}.json'):
            return  # already solved by previous attempt and saved in snapshot
        fix_seed(args.seed)
        question = json_dataset[idx]
        prompt = evaluator.make_prompt(question)
        if evaluator.is_math:
            input = evaluator.make_input(prompt, question['question'])
        else:
            if 'context' in question.keys() and question['context']:  # cannot be null
                input = evaluator.make_input(prompt, question['context'] + '\n' + question['question'])
            else:
                input = evaluator.make_input(prompt, question['question'])
        assert len(input) == 1
        problem = input[0]['content']
        reasoning_outputs = solve_task_2agents(
            problem=problem, model=model, tokenizer=tokenizer, max_steps=max(args.budgets),
            fmt=fmt, save_on_steps=args.budgets, finisher_max_new_tokens=args.finisher_max_new_tokens)
        for budget in args.budgets:
            question['model_output'] = {f"{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}-hogwild": {'raw_output': reasoning_outputs[budget]}}
            with open(f'{cots_directories_by_budget[budget]}/Task_{idx}.json', 'w', encoding='utf-8') as f:
                json.dump([question], f, ensure_ascii=False, indent=4)

        tasks_solved += 1
        logger.info(f"{idx=} | (budget={args.budgets[-1]})")

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
