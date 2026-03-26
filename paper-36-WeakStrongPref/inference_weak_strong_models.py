
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['HF_HOME'] = '../models'

os.environ["AZURE_OPENAI_API_KEY"] = "Enter your key here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aiops-llm-ch.openai.azure.com/"

import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tokenizers.processors import TemplateProcessing
from utils import construct_inference_prompt, construct_weak_model_prompt, construct_strong_model_prompt
import re
import json
from datasets import load_dataset, DatasetDict, load_from_disk
import openai 
import torch
import transformers
from tqdm import tqdm



client = openai.AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )


def generate_chat_completion(messages, model="gpt-35-turbo", temperature=1, max_tokens=None):
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=messages
        )
        print(response) 
        return response.choices[0].message.content
    
    except openai.OpenAIError as e:
        print("Prompt was filtered by content policy:", e)
        return None 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default="./dataset/ifqa")
    parser.add_argument('--data_split', type=str, help='data split', default="test")
    parser.add_argument('--weak_model_name', type=str, help='weak model name', default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--strong_model_name', type=str, help='strong model name', default="gpt-4") # "gpt-4" "gpt-35-turbo"
    parser.add_argument('--adapter_name', type=str, help='sft adapter name', default="./lora_weights/llama3-8b-sft-ifqa/checkpoint-1000") 
    parser.add_argument('--dpo_adapter_name', type=str, help='dpo adapter name', default="./lora_weights/llama3-8b-dpo-ifqa") 
    parser.add_argument('--output_file', type=str, help='output path', default="./outputs/ifqa/test/llama3-8b-dpo-gpt4.json")

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
        
    # messages = [construct_weak_model_prompt(d['prompt']) for d in dataset[args.data_split]]
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

    weak_outputs = pipeline(
        prompts,
        max_new_tokens=1028,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1.,
        top_p=0.9,
        batch_size=args.batch_size,
        return_full_text=False,
    )

    weak_outputs = [output[0]["generated_text"] for output in weak_outputs]

    strong_outputs = []
    for data, weak_output in tqdm(zip(subset, weak_outputs)):
        print(weak_output)
        query, gt = data['prompt'], data['completion']
        strong_model_prompt = construct_strong_model_prompt(query, weak_output)
        strong_model_output = generate_chat_completion(strong_model_prompt, model=args.strong_model_name)
        if strong_model_output is None:
            strong_model_output = ""
        strong_outputs.append(strong_model_output)
        
        with open(args.output_file, "w") as json_file:
            json.dump(strong_outputs, json_file, indent=4)


