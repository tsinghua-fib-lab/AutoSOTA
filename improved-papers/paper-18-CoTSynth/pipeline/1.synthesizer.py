from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def get_prompt(question, responses):
    return f'''[Instruction]\nPlease act as an excellent summarizer, summarize the following AI responses to the questions. Your summary should fully consider the connection between the question and AI responses, resulting in a correct, high-quality answer. In most cases, the same response that appears most often in the response may be the correct answer. If you find that there is no correct answer, please try to generate a correct answer yourself. Do not copy The candidate's answer, give your summarized answer and reasons, and give the correct answer at the end of the sentence, in the format: The answer is...

[The Start of Original Question]
{question}
[The End of Original Question]

[The Start of AI Responses]
{responses}
[The End of AI Responses]
'''

def llamainfer(ckpt , base_dir , data_path , name , type_prompt , n , tokens):

    # load test data
    with open(data_path, 'r' , encoding='utf-8') as fp:
        if data_path.endswith('jsonl'):
            lines = [json.loads(i) for i in fp.readlines()]
        else:
            lines = json.load(fp)
    
    model_name = ckpt

    prompts = []
    for data in tqdm(datas, desc="Preparing prompts"):
        question = data['prompt']
        responses = "\n".join([f"Response {i + 1}:\n{text.strip()}\n" for i, text in enumerate(data['assistant_answer']) if text is not None])
        data['summary_prompt'] = get_prompt(question=question, responses=responses)
        prompts.append(data['summary_prompt'])
    

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
        # gpu_memory_utilization=0.8
    )
    
    sampling_params = SamplingParams(temperature=0.9,top_p=0.9, max_tokens=tokens,n=n , stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]) 
    inputs = []
    for prompt in prompts:
        input = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs.append(input)
    responses = llm.generate(prompts=inputs, sampling_params=sampling_params)
    # save to file
    with open(f'{base_dir}/Infer_{name}.jsonl', 'w', encoding = 'utf-8') as fp:
        for i in range(len(lines)):
            data = lines[i]
            data['model_answer'] = [response.text.strip() for response in responses[i].outputs]
            data['flag'] = type_prompt
            fp.write(json.dumps(data , ensure_ascii=False ) + "\n")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLaMA Inference Script")
    
    # Command-line arguments
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for output files.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--name", type=str, required=True, help="Name for the output file.")
    parser.add_argument("--type_prompt", type=str, required=True, help="Key in the input data to use as a prompt.")
    parser.add_argument("-n", type=int, required=True, help="Number of generations per prompt.")
    parser.add_argument("--tokens", type=int, required=True, help="Maximum number of tokens to generate.")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the function
    llamainfer(
        ckpt=args.ckpt,
        base_dir=args.base_dir,
        data_path=args.data_path,
        name=args.name,
        type_prompt=args.type_prompt,
        n=args.n,
        tokens=args.tokens
    )