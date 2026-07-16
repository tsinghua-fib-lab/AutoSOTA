import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import json

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    return examples


def worker_process(rank, world_size, args, data_name, examples_chunk):
    """每个GPU进程的工作函数"""
    # 设置CUDA设备
    if world_size == 1:
        # 单GPU模式，使用逻辑GPU 0（对应CUDA_VISIBLE_DEVICES中指定的第一个GPU）
        # 不修改CUDA_VISIBLE_DEVICES，让vLLM使用环境变量中指定的GPU
        torch.cuda.set_device(0)
        print(f"进程 {rank}: 使用单GPU模式，加载模型到逻辑GPU 0")
        print(f"进程 {rank}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        print(f"进程 {rank}: 实际物理GPU将通过CUDA_VISIBLE_DEVICES映射")
    else:
        # 多GPU模式
        torch.cuda.set_device(rank)
        # 在多GPU模式下，需要根据CUDA_VISIBLE_DEVICES来设置
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # 如果设置了CUDA_VISIBLE_DEVICES，需要映射到正确的GPU
            visible_gpus = cuda_visible.split(',')
            if rank < len(visible_gpus):
                # 只使用当前rank对应的GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus[rank]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        print(f"进程 {rank}: 加载模型到GPU {rank}")
        print(f"进程 {rank}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
    # 设置随机种子
    torch.manual_seed(args.seed + rank)
    
    # 加载模型
    if args.use_vllm:
        # 确保在vLLM初始化前，CUDA_VISIBLE_DEVICES已经正确设置
        # vLLM会读取这个环境变量来决定使用哪些GPU
        current_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
        print(f"进程 {rank}: 初始化vLLM前，CUDA_VISIBLE_DEVICES={current_cuda_visible}")
        print(f"进程 {rank}: torch.cuda.device_count()={torch.cuda.device_count()}")
        print(f"进程 {rank}: torch.cuda.current_device()={torch.cuda.current_device()}")
        
        # 确保环境变量在vLLM初始化时仍然有效
        # vLLM会自动使用CUDA_VISIBLE_DEVICES环境变量
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,  # 禁用编译后端，使用eager模式以避免torch dynamo编译错误
            # max_model_len=8192,
        )
        
        print(f"进程 {rank}: vLLM初始化完成")
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # 处理数据
    result = main_worker(llm, tokenizer, data_name, args, examples_chunk, rank)
    
    # 保存当前进程的结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{rank}.json")
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"进程 {rank}: 完成，处理了 {len(examples_chunk)} 个样本")


def merge_results(args, data_name, world_size):
    """合并所有进程的结果"""
    print(f"开始合并 {data_name} 的所有进程结果...")
    
    all_samples = []
    
    # 读取所有进程的结果
    for rank in range(world_size):
        result_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{rank}.json")
        
        with open(result_file, "r", encoding="utf8") as f:
            result = json.load(f)
            all_samples.extend(result["samples"])
        
        # 删除临时文件
        os.remove(result_file)
    with open(os.path.join(args.output_dir, f"{data_name}_final_results.json"), "w", encoding="utf8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    # 重新评估合并后的结果
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    with open(os.path.join(args.output_dir, f"{data_name}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    
    print(f"{data_name} 合并完成！总共处理了 {len(all_samples)} 个样本")
    return result_json


def setup(args):
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 检测可用GPU数量
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个GPU")
    
    if world_size == 0:
        raise RuntimeError("没有检测到可用的GPU")
    
    # 如果设置了CUDA_VISIBLE_DEVICES且只指定了一个GPU，强制使用单GPU模式
    if cuda_visible_devices and len(cuda_visible_devices.split(',')) == 1:
        print(f"检测到CUDA_VISIBLE_DEVICES指定了单个GPU ({cuda_visible_devices})，使用单GPU模式")
        world_size = 1
    elif world_size == 1:
        print("检测到单GPU，使用单GPU模式")
        world_size = 1

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    
    for data_name in data_list:
        examples = prepare_data(data_name, args)
        print("=" * 50)
        print("data:", data_name, " ,remain samples:", len(examples))
        if len(examples) > 0:
            print(f"example: {examples[0]}")
        
        # 将数据分割给各个进程
        chunk_size = len(examples) // world_size
        processes = []
        
        for rank in range(world_size):
            start_idx = rank * chunk_size
            if rank == world_size - 1:  # 最后一个进程处理剩余的所有数据
                end_idx = len(examples)
            else:
                end_idx = (rank + 1) * chunk_size
            
            examples_chunk = examples[start_idx:end_idx]
            
            print(f"进程 {rank}: 处理索引 {start_idx} 到 {end_idx-1} ({len(examples_chunk)} 个样本)")
            
            p = mp.Process(
                target=worker_process,
                args=(rank, world_size, args, data_name, examples_chunk)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 合并结果
        result_json = merge_results(args, data_name, world_size)
        
        results.append(result_json)

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main_worker(llm, tokenizer, data_name, args, examples, rank):
    """工作进程的主函数"""
    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples), desc=f"GPU {rank}"):
        idx = example["idx"]


        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<｜end▁of▁sentence｜>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
        
    if "qwen2" in args.model_name_or_path.lower():
        stop_token_ids=[151645, 151643]
    elif "deepseek" in args.model_name_or_path.lower():
        stop_token_ids=[100001]
    else:
        stop_token_ids=None
        
        

    # start inference
    # measure time use
    start_time = time.time()

    # get all outputs in one inference
    if args.use_vllm:
        outputs = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
            ),
        )

        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
    else:
        outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=input_prompts,
            max_new_tokens=args.max_tokens_per_call,
            batch_size=16,
            stop_id_sequences=stop_words,
        )

    assert len(outputs) == len(input_prompts)

    # remove input_prompt from output and clean stop words
    codes = []
    for i in range(len(input_prompts)):
        output = outputs[i].rstrip()
        code = output
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    return {
        "samples": all_samples,
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    }

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to avoid CUDA issues
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    set_seed(args.seed)
    setup(args)
