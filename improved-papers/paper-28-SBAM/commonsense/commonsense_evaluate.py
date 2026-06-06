'''
NOTE: our experiments are based on the following code, which is a modified version of the original code from the DoRA (https://arxiv.org/pdf/2402.09353)repository.
Therfore, we kept the original license
'''

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# REF: https://github.com/NVlabs/DoRA

import copy
import json
import os
import re
import sys
import argparse
import fire
import torch

# sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

curr_dir = os.path.dirname(__file__)
sys.path.append(curr_dir)
sys.path.append(os.path.dirname(curr_dir))

from utils_dora import set_pad_token, model_path_to_model_name
import utils_mas
import ft_utils

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
        # MAS_flag: bool = True,
        # save_model_predictions_into_seperate_file: int = 0,
):
    args = parse_args()
    print(f'args: {args}')

    def evaluate(
            instructions,
            mas_helper_list,  # list of ptr to the attention modules of the model
            mas_helper_func=None,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"].to(device)
        if args.MAS_flag:
            _ = utils_mas.set_mas_ptr(mas_helper_list, input_ids, mod='predefined partitions', position=-1, aux_func=mas_helper_func)

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
        print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs

    if args.model == "AUTO":
        args.model = model_path_to_model_name[args.base_model]
        print(f'Auto detected model name: {args.model}')

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    mas_helper_list = []
    for model_name, model_instance in model.named_modules():
        if 'attn' in model_name or 'attention' in model_name:
            mas_helper_list.append(model_instance)

    # MAS utils
    partiotion_ngram = ft_utils.get_partiotion_ngram(base_model, partiotion_ngram=' ###')
    get_mask_partiations = utils_mas.get_mask_partiations_wrapper(tokenizer, partiotion_ngram=partiotion_ngram)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions, mas_helper_list=mas_helper_list, mas_helper_func=get_mask_partiations)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print('_ _ _')
            print('instruction:', data["instruction"])
            print('model output:', output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        if args.save_model_predictions_into_seperate_file:
            with open(save_file, 'w+') as f:
                json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--model', default='AUTO', required=False)
    parser.add_argument('--adapter', default='LoRA', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'DoRA'])
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--save_model_predictions_into_seperate_file', default=0, type=int)
    parser.add_argument('--MAS_flag', type=int, default=0)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.base_model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    base_model_path = base_model
    preferred_dtype = ft_utils.get_preferred_dtype_per_model(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=load_8bit,
            torch_dtype=preferred_dtype,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=preferred_dtype,
            device_map={"":0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map={"": device},
            torch_dtype=preferred_dtype,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=preferred_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    set_pad_token(tokenizer, model_name=base_model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    print(f'Set pad token id to {tokenizer.pad_token_id}, which is "{tokenizer.pad_token}"')

    if not load_8bit and '32' not in str(model.dtype):
        model.half()

    model = model.eval().requires_grad_(False)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()

