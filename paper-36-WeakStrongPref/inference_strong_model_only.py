
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["AZURE_OPENAI_API_KEY"] = "Enter your key here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aiops-llm-ch.openai.azure.com/"


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
import openai
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
        # print(response) 
        return response.choices[0].message.content
    
    except openai.OpenAIError as e:
        print("Prompt was filtered by content policy:", e)
        return None 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default="./dataset/ifqa")
    parser.add_argument('--data_split', type=str, help='dataset name', default="test") # "test" "validation"
    parser.add_argument('--strong_model_name', type=str, help='model name', default="gpt-4") # "gpt-4" "gpt-35-turbo"
    parser.add_argument('--output_file', type=str, help='output path', default="./outputs/ifqa/test/gpt-4.json")
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()
    dataset = load_from_disk(args.dataset_name)

    # Load existing data if available
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as file:
            outputs = json.load(file)
    else:
        outputs = []

    existing_num = len(outputs)
    print("Number of existing data: ", existing_num )
    for data in tqdm(dataset[args.data_split]):
        if existing_num > 0:
             existing_num -= 1
             continue
        query, gt = data['prompt'], data['completion']
        message = construct_inference_prompt(query)
        output = generate_chat_completion(message, model=args.strong_model_name)
        if output is None:
            output = ""
        outputs.append(output)

        with open(args.output_file, "w") as json_file:
            json.dump(outputs, json_file, indent=4)


