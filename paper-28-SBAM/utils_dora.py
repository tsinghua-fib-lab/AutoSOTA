#%%

'''
NOTE: our experiments are based on the code from the DoRA (https://arxiv.org/pdf/2402.09353) repository.
The following script contains the utility functions taken from there.
'''

# REF: https://github.com/NVlabs/DoRA

import os
import sys

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path_to_model_name = {
    'meta-llama/Llama-3-8b-hf': 'LLaMA3-8B',
    'meta-llama/Meta-Llama-3-8B': 'LLaMA3-8B',
    'meta-llama/Llama-3.1-8B': 'LLaMA3.1-8B',
    'meta-llama/Llama-3.2-1B': 'LLaMA3.2-1B',
    'meta-llama/Llama-3.2-3B': 'LLaMA3.2-3B',
    'Qwen/Qwen2.5-3B': 'Qwen2.5-3B',
    'Qwen/Qwen2.5-7B': 'Qwen2.5-7B',
    'mistralai/Mistral-7B-v0.1': 'Mistral-7B-v0.1',
}


def set_pad_token(tokenizer, model_name):
    if tokenizer.pad_token_id is not None:
        print(f'Pad token is already set with id: {tokenizer.pad_token_id}')
    elif 'llama-7' in model_name.lower() or 'llama-2-' in model_name.lower():
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id == 0:
            tokenizer.pad_token_id = (0)  # could also just write = 0, but due legacy code, we keep it like this
        else:
            raise ValueError(f'Can not find pad token id in this version of Llama: {model_name}')
    elif 'llama-3.2-' in model_name.lower() or 'llama-3.1-' in model_name.lower() or 'llama-3-' in model_name.lower():
        # REF https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/101
        # REF https://github.com/lm-sys/FastChat/issues/3266
        # does not have unk token id. use eot (end of text) token id
        eot = "<|eot_id|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        tokenizer.pad_token_id = eot_id
    elif 'mistral' in model_name.lower() and '-v0.' in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(f'Can not find model name in model name: {model_name}')
    
    tokenizer.padding_side = "left"  # Allow batched inference
    # print(f'Set pad token id to token id {tokenizer.pad_token_id}: {tokenizer.pad_token}')


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


def tokenize(tokenizer, prompt, add_eos_token=True, cutoff_len=256):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        # if "chatglm" not in base_model:
        #     result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    # if "chatglm" in base_model:
    #     return {"input_ids": result["input_ids"], "labels": result["labels"]}
    # else:
    #     return result
    return result


def generate_and_tokenize_prompt(tokenizer, train_on_inputs=True):
    def aux(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(tokenizer, full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt
    return aux


def load_training_ds(tokenizer, data_path: str, val_set_size=120, seed=42):
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt(tokenizer))
        )
        val_data = (
            train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt(tokenizer))
        )
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt(tokenizer))
        val_data = None
    
    return train_data, val_data


