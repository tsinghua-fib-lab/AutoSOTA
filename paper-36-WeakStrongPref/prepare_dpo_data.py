

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = './models'

os.environ["AZURE_OPENAI_API_KEY"] = "e7fa8dca1f454a4193c2f02367d0b836"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aiops-llm-ch.openai.azure.com/"

import re
from datasets import load_dataset, DatasetDict, load_from_disk
import openai
from utils import construct_weak_model_prompt, construct_strong_model_prompt, construct_evalution_prompt
import json
import torch
import transformers
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tokenizers.processors import TemplateProcessing





client = openai.AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )


def generate_chat_completion(prompt, model="gpt-3.5-turbo", temperature=1, max_tokens=None):
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": prompt}
            ]
        )
        print(response) 
        return response.choices[0].message.content
    
    except openai.OpenAIError as e:
        print("Prompt was filtered by content policy:", e)
        return None 




# Dataset loading
dataset_name = 'ifqa'
dataset_path = f'./dataset/{dataset_name}'
dataset = load_from_disk(dataset_path)
subset = dataset['train'].select(range(5000))

weak_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_name = "lora_weights/llama3-8b-sft-medical"
strong_model_name = "gpt-35-turbo"
evaluator_name = "gpt-4"
output_file_name = f"weak_model_output/{dataset_name}_dpo.json"
INFERENCE_TIMES = 5
SCORE_THRESHOLD = 3

# Load existing data if available
if os.path.exists(output_file_name):
    with open(output_file_name, 'r') as file:
        instances = json.load(file)
else:
    instances = []
instance_set = set([i['prompt'] for i in instances])


tokenizer = AutoTokenizer.from_pretrained(weak_model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
bos = tokenizer.bos_token
eos = tokenizer.eos_token
tokenizer._tokenizer.post_processor =TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    )

weak_model = AutoModelForCausalLM.from_pretrained(
    weak_model_name,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)

weak_model = PeftModel.from_pretrained(
    weak_model, 
    adapter_name,
    torch_dtype=torch.float16,
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


for data in tqdm(subset):
    query, gt = data['prompt'], data['completion']
    print("query:", query, "\n\n")
    print("gt:", gt, "\n\n")
    if gt is None or query is None or query in instance_set:
        continue

    # Weak model inference
    weak_model_prompt = construct_weak_model_prompt(query)
    input_ids = tokenizer.apply_chat_template(
        weak_model_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(weak_model.device)

    outputs = weak_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=INFERENCE_TIMES,
    )

    responses = [output[input_ids.shape[-1]:] for output in outputs]
    weak_model_outputs = [tokenizer.decode(response, skip_special_tokens=True) for response in responses]
    # weak_model_outputs = [tokenizer.decode(response, add_special_tokens=True) for response in responses]
    '''
    instance = {
        "prompt": query,
        "groundtruth": gt,
        "weak_model_outputs": weak_model_outputs
    }
    '''

    intermediate_outputs = []
    for weak_model_output in weak_model_outputs:
        print("weak_model_output:", weak_model_output, "\n\n")
        # weak_model_output = remove_repetitive_patterns(weak_model_output)
        
        # Strong model inference with weak model outputs
        strong_model_prompt = construct_strong_model_prompt(query, weak_model_output)
        strong_model_output = generate_chat_completion(strong_model_prompt, model=strong_model_name)
        print("strong_model_output without weak model output:", strong_model_output, "\n\n")
        if strong_model_output is None:
            continue
        
        # Evaluate_model_output
        prompt = construct_evalution_prompt(strong_model_output, gt)
        score = generate_chat_completion(prompt, model=evaluator_name)
        if score is None:
            continue
        print("evaluation_score:", score, "\n\n")

        def is_valid_number(s):
            try:
                number = float(s)
                return number
            except ValueError:
                return 0.0

        intermediate_output = {
            "weak_model_output": weak_model_output,
            "strong_model_output": strong_model_output,
            "score": is_valid_number(score)
        }
        intermediate_outputs.append(intermediate_output)
    
    # Strong model inference with weak model outputs
    vanilla_strong_model_output = generate_chat_completion(query, model=strong_model_name)
    print("vanilla_strong_model_output:", vanilla_strong_model_output, "\n\n")
    if vanilla_strong_model_output is None:
        continue
        
    # Evaluate_model_output
    evaluation_prompt = construct_evalution_prompt(vanilla_strong_model_output, gt)
    vanilla_score = generate_chat_completion(evaluation_prompt, model=evaluator_name)
    if vanilla_score is None:
        continue
    print("vanilla_evaluation_score:", vanilla_score, "\n\n")
    vanilla_score = is_valid_number(vanilla_score)
    vanilla_strong_model_output = {
        "model_output": vanilla_strong_model_output,
        "score": vanilla_score
    }

    # filtered_outputs = [o for o in intermediate_outputs if o["score"] >= SCORE_THRESHOLD]
    filtered_outputs = [o for o in intermediate_outputs]
    '''
    if len(filtered_outputs) > 1:
        chosen_sample = max(filtered_outputs, key=lambda x: x["score"])
        rejected_sample = min(filtered_outputs, key=lambda x: x["score"])
        if chosen_sample["score"] == rejected_sample["score"]:
            chosen_sample = rejected_sample = None
    else:
        chosen_sample = rejected_sample = None
    '''
    chosen_sample = max((d for d in filtered_outputs if d['score'] > vanilla_score), key=lambda x: x['score'], default=None)
    rejected_sample = min((d for d in filtered_outputs if d['score'] <= vanilla_score), key=lambda x: x['score'], default=None)
    
    
    instance = {
        "prompt": query,
        "groundtruth": gt,
        "intermediate_outputs": intermediate_outputs,
        "vanilla_strong_model_output": vanilla_strong_model_output,
        "chosen": chosen_sample,
        "rejected": rejected_sample
    }

    instances.append(instance)

    with open(output_file_name, 'w') as json_file:
        json.dump(instances, json_file, indent=4)

