import argparse
import os
import json
from tqdm import tqdm
import torch
import logging
from src.dataset_utils import *
from src.models import create_model
from src.defense import *
from src.attack import *
from src.helper import get_log_name
import pandas as pd
import time
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Robust Dynamic RAG')

    # --- 基础设置 ---
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument('--model_name', type=str, default='mistral7b',
                        choices=['mistral7b', 'llama3b', 'gpt-4o', 'gpt-4o-mini', 'o1-mini',
                                 'deepseek7b', 'llama1b', 'tai_llama8b', 'tai_mistral7b',
                                 'deepseek-chat', 'deepseek-reasoner', 'grok-4-fast'],
                        help='LLM model name')
    parser.add_argument('--dataset_name', type=str, default='dynamic_serpapi', help='Dataset name')
    parser.add_argument('--model_dir', type=str, help='Directory for huggingface models')
    parser.add_argument('--rep', type=int, default=1, help='Repeat times')

    # --- 动态与检索设置 ---
    parser.add_argument('--top_k', type=int, default=50, help='Total retrieved documents to use')
    parser.add_argument('--initial_k', type=int, default=1, help='Number of documents in the initial batch')
    parser.add_argument('--dynamic_step_size', type=int, default=1, help='Number of documents added per step')

    # --- 防御方法 ---
    parser.add_argument('--defense_method', type=str, default='mincut',
                        choices=['none', 'mincut'],
                        help='The defense method to use')

    # --- MinCut 防御参数 ---
    parser.add_argument('--nli_model_path', type=str,
                        default="DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                        help='Path to NLI model')

    # --- 攻击设置 ---
    parser.add_argument('--attack_method', type=str, default='none',
                        choices=['none', 'Poison', 'PIA'],
                        help='Attack method')
    parser.add_argument('--attackpos', type=int, default=0, help='Position of attack in the full top-k list')
    parser.add_argument('--corruption_size', type=int, default=1, help='Number of poisoned documents')
    parser.add_argument('--attack_each_step', action='store_true', help='Attack during each dynamic step')

    # --- 其他 ---
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save_response', action='store_true', help='Save JSON results')
    parser.add_argument('--use_cache', action='store_true', help='Use LLM cache')
    parser.add_argument('--use_open_model_api', action='store_true', help='Use API for open models')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples for testing')

    return parser.parse_args()


def main():
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    args = parse_args()

    os.environ["SANDBOX_GRADER_MODEL"] = args.model_name
    LOG_NAME = get_log_name(args) + f"_dynamic_init{args.initial_k}_step{args.dynamic_step_size}"

    # ========== 0) Logger 初始化 ==========
    logging_level = logging.DEBUG if args.debug else logging.INFO
    os.makedirs('log', exist_ok=True)
    logging.basicConfig(format=':::::::::::::: %(message)s')
    logger = logging.getLogger('RRAG-main')
    logger.setLevel(level=logging_level)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.FileHandler(f"log/{LOG_NAME}.log"))
    logger.addHandler(logging.StreamHandler())
    logger.info(args)

    # ========== 1) 加载动态数据集 ==========
    dynamic_json_path = getattr(args, "dynamic_json_path", "data/poisoned_dynamic_dataset_500.json")
    data_tool = DynamicDataset(dynamic_json_path, args.top_k, logger)
    data_list = load_json(dynamic_json_path)
    if args.max_samples is not None:
        data_list = data_list[:args.max_samples]

    # ========== 2) 初始化 LLM ==========
    cache_path = None
    if args.use_cache:
        os.makedirs('cache/', exist_ok=True)
        cache_path = f'cache/{args.model_name}.z'

    llm = create_model(
        args.model_name,
        args.model_dir,
        args.use_open_model_api,
        cache_path=cache_path,
        max_output_tokens=512
    )

    # ========== 3) 初始化防御模型 ==========
    if args.defense_method == 'none':
        model = RRAG(llm)
    elif args.defense_method == 'mincut':
        model = DynamicMinCutRRAG(llm, nli_model_path=args.nli_model_path)
    else:
        raise ValueError(f"Invalid defense method: {args.defense_method}")

    # ========== 4) 初始化攻击者 ==========
    if args.attack_method == 'none':
        attacker = None
    elif args.attack_method == 'PIA':
        attacker = PIA(top_k=args.top_k, repeat=10,
                       poison_pos=args.attackpos, poison_num=args.corruption_size)
    elif args.attack_method == 'Poison':
        attacker = Poison(top_k=args.top_k, repeat=10,
                          poison_pos=args.attackpos, poison_num=args.corruption_size)
    else:
        raise ValueError("Invalid attack method")

    # ========== 5) 准备结果 CSV ==========
    os.makedirs('output', exist_ok=True)
    output_csv_file = f"./output/{LOG_NAME}.csv"

    fieldnames = [
        "rep_idx",
        "step_acc", "step_asr", "total_steps",
        "final_acc", "final_asr", "num_questions",
        "input_tokens", "output_tokens", "total_time_sec",
        "defense_method", "attack_method", "dataset_name", "step_size", "init_k"
    ]
    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    response_list = []

    # ========== 6) 主循环：rep 次重复实验 ==========
    for rep_idx in range(args.rep):
        corr_cnt = 0
        asr_cnt = 0
        input_tokens = 0
        output_tokens = 0
        total_time = 0

        total_step_correct = 0
        total_step_asr_success = 0
        total_steps = 0

        # ========== 7) 遍历样本 ==========
        for data_idx, raw_item in enumerate(tqdm(data_list)):
            logger.info(f'==== rep_idx #{rep_idx}; item: {data_idx} ====')
            data_item = data_tool.process_data_item(raw_item)

            # 初始攻击（如果不每步攻击）
            if attacker and not args.attack_each_step:
                data_item = attacker.attack(data_item)

            yearly_contexts = data_item.get("yearly_contexts", {}) or {}
            years_sorted = sorted([int(y) for y in yearly_contexts.keys()])
            total_years = len(years_sorted)

            # 动态状态变量
            current_ans = None
            priors = None

            llm.reset_token_count()
            start_time = time.perf_counter()

            year_ptr = 0
            end_ptr = min(total_years, int(args.initial_k))
            step_count = 0

            all_steps_correct = True
            all_steps_asr_success = True if attacker else False

            while year_ptr < total_years:
                batch_years = years_sorted[year_ptr:end_ptr]
                new_docs_batch = []

                # --- 获取该批次文档 ---
                for y in batch_years:
                    year_str = str(y)
                    year_data = yearly_contexts.get(year_str, {})
                    docs = year_data.get("docs", []) or []

                    for d in docs:
                        title = d.get("title", "").strip()
                        snippet = d.get("snippet", "").strip()
                        content = d.get("content", "").strip()
                        full_text = f"[Title] {title}\n[Snippet] {snippet}\n[Content] {content}".strip()
                        new_docs_batch.append({
                            "title": title,
                            "text": full_text,
                            "year": y,
                            "month": 0,
                            "id": d.get("id", ""),
                            "sorting_key": 0
                        })

                if not new_docs_batch:
                    year_ptr = end_ptr
                    end_ptr = min(total_years, end_ptr + int(args.dynamic_step_size))
                    continue

                step_count += 1
                logger.info(f"--- Step {step_count}: Adding docs for years {batch_years} ---")

                # --- 准备问题对象 ---
                latest_year = str(batch_years[-1])
                step_data_item = data_item.copy()
                original_question = data_item['question']
                modified_question = f"{original_question} in {latest_year}"

                step_data_item = data_item.copy()
                step_data_item["question"] = modified_question

                # 设置 Ground Truth
                latest_year_data = yearly_contexts.get(latest_year, {})
                step_data_item["answer"] = data_tool._coerce_list(latest_year_data.get("answer", []))
                step_data_item["incorrect_answer"] = data_tool._coerce_list(
                    latest_year_data.get("incorrect_answer", []))
                step_data_item["incorrect_context"] = data_tool._coerce_list(
                    latest_year_data.get("incorrect_context", []))

                # --- 生成当前步内容 & 攻击 ---
                step_data_item["topk_content"] = docs_to_topk_content(new_docs_batch, include_title=True)

                if args.attack_each_step and attacker:
                    step_data_item = attacker.attack(step_data_item)

                # --- 执行查询 ---
                if args.defense_method == 'none':
                    current_ans = model.query_undefended(step_data_item)
                elif args.defense_method == 'mincut':
                    current_ans, priors = model.dynamic_query(
                        step_data_item, previous_answer=current_ans, previous_priors=priors
                    )

                logger.info(f"Step {step_count} Answer: {current_ans}")

                # --- 评测 ---
                is_correct = data_tool.eval_response(current_ans, step_data_item)
                is_asr = data_tool.eval_response_asr(current_ans, step_data_item) if attacker else 0

                logger.info(f"Step {step_count} correct: {is_correct}, asr: {is_asr}")

                total_steps += 1
                total_step_correct += int(is_correct)
                if attacker:
                    total_step_asr_success += int(is_asr)

                if not is_correct:
                    all_steps_correct = False
                if attacker and not is_asr:
                    all_steps_asr_success = False

                year_ptr = end_ptr
                end_ptr = min(total_years, end_ptr + int(args.dynamic_step_size))

            # --- 样本结束 ---
            final_response = current_ans if current_ans else "I don't know."
            final_correct = data_tool.eval_response(final_response, data_item)
            final_asr = data_tool.eval_response_asr(final_response, data_item) if attacker else 0

            corr_cnt += int(final_correct)
            if attacker:
                asr_cnt += int(final_asr)

            response_list.append({
                "query": data_item['question'],
                "final_response": final_response,
                "defense": args.defense_method,
                "is_correct": bool(final_correct),
                "steps_taken": step_count
            })

            input_tokens += llm.get_token_count().get("input", 0)
            output_tokens += llm.get_token_count().get("output", 0)
            total_time += time.perf_counter() - start_time

        # ========== 8) rep 级别总结 ==========
        logger.info(f'\n=== Result for rep: {rep_idx} ===')
        num_questions = len(data_list)

        final_acc = corr_cnt / num_questions if num_questions > 0 else 0.0
        final_asr_val = asr_cnt / num_questions if (num_questions > 0 and attacker) else 0.0

        step_acc = total_step_correct / total_steps if total_steps > 0 else 0.0
        step_asr = total_step_asr_success / total_steps if (total_steps > 0 and attacker) else 0.0

        logger.info(f'Total Steps: {total_steps}')
        logger.info(f'Step-wise Avg Accuracy: {step_acc:.4f}')
        if attacker:
            logger.info(f'Step-wise Avg ASR: {step_asr:.4f}')
        logger.info(f'Final (per-question last answer) Accuracy: {final_acc:.4f}')
        if attacker:
            logger.info(f'Final (per-question last answer) ASR: {final_asr_val:.4f}')

        if args.use_cache:
            llm.dump_cache()

        result_current = {
            "rep_idx": rep_idx,
            "step_acc": step_acc,
            "step_asr": step_asr,
            "total_steps": total_steps,
            "final_acc": final_acc,
            "final_asr": final_asr_val,
            "num_questions": num_questions,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_time_sec": round(total_time, 2),
            "defense_method": args.defense_method,
            "attack_method": args.attack_method,
            "dataset_name": "dynamic_serpapi",
            "step_size": args.dynamic_step_size,
            "init_k": args.initial_k
        }
        df = pd.DataFrame([result_current])
        df.to_csv(output_csv_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    main()