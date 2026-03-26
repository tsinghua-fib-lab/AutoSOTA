import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import Counter
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_prompt(question, responses):
    return f'''[Instruction]\nPlease act as an excellent summarizer, summarize the following AI responses to the questions. Your summary should fully consider the connection between the question and AI responses, resulting in a correct, high-quality answer. In most cases, the same response that appears most often in the response may be the correct answer. If you find that there is no correct answer, please try to generate a correct answer yourself. Do not copy The candidate's answer, give your summarized answer and reasons, and give the correct answer at the end of the sentence, in the format: The answer is...

[The Start of Original Question]
{question}
[The End of Original Question]

[The Start of AI Responses]
{responses}
[The End of AI Responses]
'''

def summary_infer(model_name, input_file, dataset, output_filename):
    # Load the input data from the specified file
    with open(input_file, 'r', encoding='utf-8') as file:
        datas = [json.loads(line) for line in file]

    prompts = []
    for data in tqdm(datas, desc="Preparing prompts"):
        question = data['prompt']
        responses = "\n".join([f"Response {i + 1}:\n{text.strip()}\n" for i, text in enumerate(data['assistant_answer']) if text is not None])
        data['summary_prompt'] = get_prompt(question=question, responses=responses)
        prompts.append(data['summary_prompt'])

    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    inputs = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    responses = llm.generate(prompts=inputs, sampling_params=sampling_params)

    # Ensure the output directory exists
    os.makedirs(f'../outputs/{dataset}', exist_ok=True)
    
    with open(f'../outputs/{dataset}/{output_filename}.jsonl', 'w', encoding='utf-8') as fp:
        for data, response in zip(datas, responses):
            data['summary_answer'] = [output.text.strip() for output in response.outputs]
            fp.write(json.dumps(data, ensure_ascii=False) + "\n")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize AI responses.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use.')
    parser.add_argument('--input_file', type=str, help='Path to the input JSONL file containing the data.' , default='../data/MATH/math_llama3-8b_ans.jsonl')
    parser.add_argument('--dataset', type=str, help='Name of the dataset being processed.', default = 'MATH')
    parser.add_argument('--output_file', type=str, help='Base name for the output file.' , default = 'math_llama3-8b_summary')

    args = parser.parse_args()

    # Call the function with the parsed arguments
    summary_infer(args.model_name, args.input_file, args.dataset, args.output_file)