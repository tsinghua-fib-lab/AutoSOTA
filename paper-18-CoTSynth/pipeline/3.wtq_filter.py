import json
import os
import time
from tqdm import tqdm
import random
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse

# Base prompt definition
base_prompt = """你是一个擅长评价文本质量的助手。
请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。你需要从下面的几个维度对回答进行评估:
事实正确性, 满足用户需求, 逻辑连贯性, 完备性。我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。
当你开始你的评估时，你需要按照遵守以下的流程：
1. 从不同维度对AI助手的答案进行评价，在每个维度的评价之前，给每一个维度一个1～10的分数，再进行评价，以句号分隔。
2. 综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
3. 最后，将AI助手的答案与参考答案进行比较，结合每个维度的评价结果指出AI助手的答案有哪些不足，并提供可能的改进方法。
4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
   其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。
   当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
   当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
   当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
   当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
   只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
   作为示例，参考答案可以得到8分。
请记住，你需要严格遵守1~4的评价流程。第1步中你在展开每个维度评价之时，先给出对该维度的打分。
最后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：
{{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}}，例如：{{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}}。
用户的提问： {question}
[参考答案开始]
{ref_answer}
[参考答案结束]
[助手的答案开始]
{answer}
[助手的答案结束]
"""

# Function to extract scores from the judgement text
def extract_score(judgement):
    try:
        # Use regex to extract the score
        match = re.search(r'综合得分[^\d]*(\d+)', judgement)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def model_infer(ckpt, data_path, out_path):
    # Load test data
    with open(data_path, 'r') as fp:
        if data_path.endswith('jsonl'):
            lines = [json.loads(i) for i in fp.readlines()]
        else:
            lines = json.load(fp)

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=1,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
    )
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=stop_token_ids)

    # Prepare prompts

    prompts = []
    for data in lines:
        prompt = data['prompt']
        gold_answer = data['answer']
        data['judgement'] = []
        for response in data['assistant_answer']:
            data['instruction'] = base_prompt.format(question=prompt , ref_answer=gold_answer , answer=response)
            prompts.append(data['instruction'])

    inputs = []
    for instruction in prompts:
        input = [{"role": "user", "content": instruction}]
        input = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        inputs.append(input)
    llm_responses = llm.generate(inputs, sampling_params=sampling_params)

    f_out = open(out_path, 'w', encoding='utf-8')
    response_index = 0
    for data in lines:
        num_responses = len(data['assistant_answer'])
        judgements = [response.outputs[0].text.strip(' ') for response in llm_responses[response_index:response_index + num_responses]]
        scores = [extract_score(judgement) for judgement in judgements]
        data['judgement'] = [item for item in judgements]
        data['scores'] = [item for item in scores]
        data['model_answer'] = [response for score, response in zip(scores, data['assistant_answer']) if score >= 8]
        response_index += num_responses
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # Use argparse for command-line argument parsing
    parser = argparse.ArgumentParser(description="model inference judge script")
    
    # Add command-line arguments
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL data file.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save the output JSONL file.")

    # Parse arguments
    args = parser.parse_args()

    # Call the inference function
    model_infer(
        ckpt=args.ckpt,
        data_path=args.data_path,
        out_path=args.out_path
    )