
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

os.environ['HF_TOKEN'] = "Enter your huggingface token here"
os.environ['HF_HOME'] = '../models'

import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tokenizers.processors import TemplateProcessing
from utils import construct_inference_prompt
import re
import json
from datasets import load_dataset, DatasetDict, load_from_disk
from openai import AzureOpenAI
import torch
import transformers
   


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default="./dataset/ifqa")
    parser.add_argument('--data_split', type=str, help='dataset name', default="test") # "test" "validation"
    
    parser.add_argument('--weak_model_name', type=str, help='weak model name', default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--adapter_name', type=str, help='weak model name', default=None) # "lora_weights/llama3-8b-sft-ifqa"
    parser.add_argument('--output_file', type=str, help='output path', default="./outputs/ifqa/test/llama3-8b.json") 
    parser.add_argument('--dpo_adapter_name', type=str, help='dpo adapter name', default=None)
    
    parser.add_argument('--batch_size', type=int, help='batch size of inference', default=32)
    return parser.parse_args()


if __name__ == "__main__":
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args = parse_arguments()
    dataset = load_from_disk(args.dataset_name)
    subset = dataset[args.data_split] #.select(range(2000))

    tokenizer = AutoTokenizer.from_pretrained(args.weak_model_name, padding_side='left', use_fast=True, TOKENIZERS_PARALLELISM=True) 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    weak_model = AutoModelForCausalLM.from_pretrained(
        args.weak_model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pipeline = transformers.pipeline(
    "text-generation",
    model=weak_model,
    tokenizer=tokenizer,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    )

    if args.adapter_name is not None:
        pipeline.model = PeftModel.from_pretrained(
            pipeline.model, 
            args.adapter_name,
            torch_dtype=torch.float16,
        )
    
    if args.dpo_adapter_name is not None:
        pipeline.model = PeftModel.from_pretrained(
            pipeline.model, 
            args.dpo_adapter_name,
            torch_dtype=torch.float16,
        )

    # messages = [construct_inference_prompt(d['prompt']) for d in dataset[args.data_split]]
    messages = [construct_inference_prompt(d['prompt']) for d in subset]
    prompts = [
        pipeline.tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True,
            padding=True, 
            truncation=True,
        )
        for message in messages
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompts,
        max_new_tokens=1028,
        eos_token_id=terminators,
        do_sample=True,
        batch_size=args.batch_size,
        return_full_text=False,
    )

    outputs = [output[0]['generated_text'] for output in outputs]

    with open(args.output_file, "w") as json_file:
        json.dump(outputs, json_file, indent=4)


