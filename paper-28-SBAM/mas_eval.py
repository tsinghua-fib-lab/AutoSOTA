"""
Evaluation script for MAS (Masked Attention by Segment) using MasLlamaForCausalLM.
Evaluates on 8 commonsense reasoning datasets.
"""
import copy
import json
import os
import re
import sys
import argparse
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer
from peft import PeftModel

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from utils_dora import set_pad_token, model_path_to_model_name
import ft_utils
from mas_llama_impl.modeling_mas_llama import MasLlamaForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def evaluate(
    instructions,
    model,
    tokenizer,
    temperature=0.0,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=32,
    dataset=None,
    **kwargs,
):
    prompts = [generate_prompt(instruction, dataset=dataset) for instruction in instructions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
    outputs = [o.split("### Response:")[-1].strip() for o in outputs]
    return outputs


def generate_prompt(instruction, input=None, dataset=None):
    # Use training-format (indented blank lines) for arc_c/obqa; original format for others
    if dataset in ['ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}
                
                ### Input:
                {input}
                
                ### Response:
                """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {instruction}
                
                ### Response:
                """
    else:
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """


def load_data(dataset_name):
    file_path = os.path.join(curr_dir, 'dataset', dataset_name, 'test.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    return json.load(open(file_path, 'r'))


def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def extract_answer(dataset, sentence: str):
    # Case-insensitive search, return last match (model may self-correct)
    sentence_ = sentence.strip().lower()
    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]
    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]
    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]
    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, 
                        choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", 
                                 "ARC-Challenge", "ARC-Easy", "openbookqa"])
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--top_p', default=0.75, type=float)
    parser.add_argument('--top_k', default=40, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--max_new_tokens', default=32, type=int)
    args = parser.parse_args()

    base_model = args.base_model
    preferred_dtype = ft_utils.get_preferred_dtype_per_model(base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    set_pad_token(tokenizer, model_name=base_model)

    model = MasLlamaForCausalLM.from_pretrained(
        base_model,
        MAS_template='commonsense',
        attn_implementation="eager",
        trust_remote_code=True,
        torch_dtype=preferred_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        torch_dtype=preferred_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model = model.eval().requires_grad_(False)

    dataset = load_data(args.dataset)
    batches = create_batch(dataset, args.batch_size)

    total = len(batches)
    correct = 0
    current = 0

    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]
        outputs = evaluate(
            instructions, model, tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
        )
        for data, output in zip(batch, outputs):
            label = data.get('answer')
            predict = extract_answer(args.dataset, output)
            if label == predict:
                correct += 1
        print(f'test:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        pbar.update(1)
    pbar.close()
    
    accuracy = correct / len(dataset) * 100
    print(f'\n{args.dataset}: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    main()
