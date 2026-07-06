import os
from lvlm.Qwen2VL import Qwen2VL
from lvlm.InternVL import InternVL
from lvlm.LLaVA import LLaVA
from lvlm.LLaVANeXT import LLaVANeXT
from benchmark.LLaVABench import LLaVABench
from benchmark.MMVet import MMVet
from benchmark.MMMU import MMMU
from benchmark.ScienceQA import ScienceQA
from llm.Qwen import Qwen
from util.misc import *
import torch
import argparse
from tqdm import tqdm
import json
import random
from datetime import datetime
import warnings
from sentence_transformers import SentenceTransformer
from PIL import Image

warnings.filterwarnings("ignore")

# Use local model path if available
_nli_model_path = "/models/nli-roberta-large"
if not os.path.exists(_nli_model_path):
    _nli_model_path = "nli-roberta-large"
SenSimModel = SentenceTransformer(_nli_model_path)

LVLM_MAP = {
    'Qwen2-VL-72B-Instruct': Qwen2VL,
    'Qwen2-VL-7B-Instruct': Qwen2VL,
    'Qwen2-VL-2B-Instruct': Qwen2VL,
    'InternVL2-26B': InternVL,
    'InternVL2-8B': InternVL,
    'InternVL2-1B': InternVL,
    'llava-v1.6-vicuna-13b-hf': LLaVANeXT,
    'llava-v1.6-mistral-7b-hf': LLaVANeXT,
    'llava-1.5-13b-hf': LLaVA,
    'llava-1.5-7b-hf': LLaVA
}

BENCHMARK_MAP = {
    'MMVet': MMVet,
    'LLaVABench': LLaVABench,
    'MMMU': MMMU,
    'ScienceQA': ScienceQA,
}

LLM_MAP = {
    'Qwen2.5-0.5B-Instruct': Qwen,
    'Qwen2.5-1.5B-Instruct': Qwen,
    'Qwen2.5-3B-Instruct': Qwen,
    'Qwen2.5-7B-Instruct': Qwen,
}

BENCHMARK_TYPE = {
    'MMVet': 'FREE_FORM',
    'LLaVABench': 'FREE_FORM',
    'MMMU': 'MULTI_CHOICE',
    'ScienceQA': 'MULTI_CHOICE'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvlm', type=str, default='Qwen2-VL-2B-Instruct')
    parser.add_argument('--benchmark', type=str, default='LLaVABench')
    parser.add_argument('--llm', type=str, default='Qwen2.5-3B-Instruct')
    parser.add_argument('--inference_temp', type=float, default=0.1)
    parser.add_argument('--sampling_temp', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.99)
    parser.add_argument('--sampling_time', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.48)
    args = parser.parse_args()
    return args


def obtain_lvlm(args):
    lvlm_class = LVLM_MAP.get(args.lvlm)
    if not lvlm_class:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")
    return lvlm_class(args.lvlm)


def obtain_benchmark(args):
    benchmark_class = BENCHMARK_MAP.get(args.benchmark)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    return benchmark_class()


def obtain_llm(args):
    llm_class = LLM_MAP.get(args.llm)
    if not llm_class:
        raise ValueError(f"Unsupported LLM: {args.llm}")
    return llm_class(args.llm)


def obatin_single_sample(args, benchmark, idx, log_dict):
    sample = benchmark.retrieve(idx)
    if sample is None or sample['img'] is None or sample['question'] is None or sample['gt_ans'] is None:
        return sample
    log_dict[idx]['question'] = sample['question']
    log_dict[idx]['gt_ans'] = sample['gt_ans']
    return sample

def get_cur_time():
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def infer_single_sample(args, lvlm, sample, is_sampling, llm, log_dict):
    if not is_sampling:
        ans = lvlm.generate(
            sample['img'],
            sample['question'],
            args.inference_temp if not is_sampling else args.sampling_temp
        )
        log_dict[sample['idx']]['ans'] = ans
        flag_ans_correct = True
        if BENCHMARK_TYPE[args.benchmark] == 'MULTI_CHOICE':
            flag_ans_correct = str(sample['gt_ans']) in ans
        else:
            question = f"Ground truth: {sample['gt_ans']}. Model answer: {ans}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
            llm_ans_check = llm.generate(
                question,
                0.1
            )
            log_dict[sample['idx']]['llm_ans_check'] = llm_ans_check
            flag_ans_correct = 'Correct' in llm_ans_check or 'correct' in llm_ans_check or 'C' in llm_ans_check or 'c' in llm_ans_check
        log_dict[sample['idx']]['flag_ans_correct'] = flag_ans_correct
    else:
        log_dict[sample['idx']]['ans_sampling_list_original'] = lvlm.generate_batch(
            sample['img'],
            sample['question'],
            args,
        )

        log_dict[sample['idx']]['ans_sampling_list_prompt'] = []

        prompt = "Identify up to 3 main objects in the image. Note: do not use abbreviations for object names, please provide the full name. Format your output exactly as follows: {Object 1, Object 2, Object 3}."

        objects_list = lvlm.generate(sample['img'], prompt, args.inference_temp)
        log_dict[sample['idx']]['objects_list'] = objects_list

        prompt = sample['question'] + "At first, use the image description to ensure you understand the image correctly. Then only output your answer of the question.\nImage Description:" + objects_list
        log_dict[sample['idx']]['ans_sampling_list_prompt'] = lvlm.generate_batch(
            sample['img'],
            prompt,
            args,
        )

def vmf(args, lvlm, sample, llm, log_dict, type):
    infer_single_sample(args, lvlm, sample, True, llm, log_dict)
    log_dict[sample['idx']]['vmf_score'] = getvmf(log_dict, sample, SenSimModel, type, debias=True, device=None)

def handle_single(args, idx, lvlm, benchmark, llm, log_dict, llm_llama):
    sample = obatin_single_sample(args, benchmark, idx, log_dict)
    if sample is None or sample['img'] is None or sample['question'] is None or sample['gt_ans'] is None:
        log_dict[idx]['flag_sample_valid'] = False
        return
    log_dict[idx]['flag_sample_valid'] = True
    infer_single_sample(args, lvlm, sample, False, llm, log_dict)
    vmf(args, lvlm, sample, llm, log_dict, BENCHMARK_TYPE[args.benchmark])

def handle_batch(args, lvlm, benchmark, llm, llm_llama):
    log_dict = {}
    log_dict['args'] = str(args)
    begin_time_str = get_cur_time()
    log_dict['begin_time_str'] = begin_time_str

    total = 0
    benchmark_size = benchmark.obtain_size()
    Label = []
    vmfscore = []

    for idx in tqdm(range(benchmark_size)):
        log_dict[idx] = {}
        handle_single(args, idx, lvlm, benchmark, llm, log_dict, llm_llama)
        if not log_dict[idx]['flag_sample_valid']:
            continue
        else:
            log_dict[idx]['success'] = []
            Label.append(log_dict[idx]['flag_ans_correct'])
            vmfscore.append(log_dict[idx]['vmf_score'])
        total += 1

    cnt = 0
    for idx,item in enumerate(Label):
        if item and vmfscore[idx]>args.threshold:
            cnt += 1
        if not item and vmfscore[idx]<=args.threshold:
            cnt += 1
    log_dict['Accuracy'] = cnt/total
    log_dict['Total samples'] = total

    end_time_str = get_cur_time()
    log_dict['end_time_str'] = end_time_str
    if not os.path.exists('exp'):
        os.makedirs('exp')
    with open(f'exp/log_vmf_{begin_time_str}_{args.lvlm}_{args.benchmark}.json', "w") as f:
        json.dump(log_dict, f)
    print(f"- Full log is saved at exp/log_vmf_{begin_time_str}_{args.lvlm}_{args.benchmark}.json.")


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    fix_seed(0)
    args = parse_args()
    lvlm = obtain_lvlm(args)
    llm = obtain_llm(args)
    llm_llama = None
    benchmark = obtain_benchmark(args)
    handle_batch(args, lvlm, benchmark, llm, llm_llama)

if __name__ == "__main__":
    main()