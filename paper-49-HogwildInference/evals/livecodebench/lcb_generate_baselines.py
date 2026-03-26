import argparse
import json
import os

import peft
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys; sys.path.insert(0, "../utils"); sys.path.insert(0, "../..");

from formatting import CodeFormatting
from generation import fix_seed, finalize_response_with_s1_finisher
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
        default=42,
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=(256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192),
        help="A list of budgets"
    )
    parser.add_argument(
        "--lcb_input_file",
        type=str,
        default='./data/livecodebench_v5.jsonl',
    )
    parser.add_argument(
        "--finisher_max_new_tokens",
        type=int,
        default=0,
        help="If there is no answer by a given budget, prompt the model to give the answer and give it this many tokens"
             " | set to 0 for naive baseline, positive values for tuned baseline (same as in lcb_generate_hogwild)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help='Adapter to load and merge into the model'
    )
    parser.add_argument("--merge_adapter", action="store_true", help="call peft_model.merge_and_unload()")
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
        "--device_map",
        type=str,
        default="auto",
        help='Load model from_pretrained with this as device_map'
    )
    parser.add_argument(
        "--eval_folder",
        type=str,
        default='.',
        help='Results will be written to "args.eval_folder/evals_data/livecodebench/exp_name".'
    )
    parser.add_argument(
        "--dump_snapshot_freq",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    return args


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python3 {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    cots_directories_by_budget = {
        budget: os.path.join(
            args.eval_folder,
            f"evals_data/livecodebench/{args.model_name.split('/')[-1]}-seed-{args.seed}-budget-{budget}"
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    fmt = CodeFormatting(tokenizer)  # for s1-like finisher
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map, revision=args.revision, torch_dtype='auto',
        low_cpu_mem_usage=True, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.adapter_path:
        model = peft.PeftModel.from_pretrained(model, args.adapter_path)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.train(False)

    logger.info('Loading dataset')
    with open(args.lcb_input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    local_tasks_solved = 0

    def _solve_task_and_save(idx: int):
        nonlocal local_tasks_solved
        if os.path.exists(f'{cots_directories_by_budget[args.budgets[-1]]}/Task_{idx}.txt'):
            return  # already solved by previous attempt and saved in snapshot
        fix_seed(args.seed)
        inputs = tokenizer.apply_chat_template(
            conversation=[dict(role='user', content=dataset[idx]['prompt'])],
            tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True
        ).to(device)
        response_ids = model.generate(**inputs, max_new_tokens=max(args.budgets), eos_token_id=tokenizer.eos_token_id)
        local_tasks_solved += 1

        n_tokens_in_prompt = inputs['input_ids'].numel()
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
                file.write(json.dumps(dict(dataset[idx], gen=[response_for_budget])))

        logger.info(f"{idx=}, {local_tasks_solved=}")

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
