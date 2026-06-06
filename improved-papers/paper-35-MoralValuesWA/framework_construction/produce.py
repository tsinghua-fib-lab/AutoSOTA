import os
import re
import torch
import transformers
from transformers import pipeline
from huggingface_hub import login
from tqdm import tqdm
import json
from collections import Counter

token = "Your Hugging Face Token here"

os.environ['HF_TOKEN'] = token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

huggingface_token = os.environ.get('HF_TOKEN')
login(token=huggingface_token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device ="cuda", 
)

def create_messages(keyword):
    return [
        {"role": "system", 
        "content": """
        You are a human participant in a psychology experiment.
        ### Background ###
         On average, an adult knows about 40,000 words, but what do these words mean to people like you and me? You can help scientists understand how meaning is organized in our mental dictionary by playing the game of word associations. This game is easy: Just give the first three words that come to mind for a given cue word.
         ### OUTPUT FORMAT ###
            Output your response in the following format:

            response1, response2, response3
            
            Do not provide any additional context, or explanations. Just the words as comma-separated values.
         ### End of instructions ###
         """},
        {"role": "user", "content": f"""
                Cue word: {keyword}
            """}
    ]

def generate(cue_words):
    responses = {}
    useless = {}
    for cue in tqdm(cue_words, desc="Generating responses"):
        responses[cue] = []
        useless[cue] = []
        for i in range(100):
            messages = create_messages(cue)
            outputs = pipeline(
                messages,
                max_new_tokens=20, 
                temperature=2.1,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            response_text = outputs[0]["generated_text"][-1]['content'].lower()
            cleaned_response = re.sub(r'[^\w\s,]', '', response_text)
            words = re.split(r',\s*', cleaned_response)

            if len(words) == 3:
                response1, response2, response3 = words
                responses[cue].extend([response1, response2, response3])
            else:
                useless[cue].extend(cleaned_response)
    return responses,useless


def produce():
    with open('./data/cue.txt', 'r',encoding='utf-8') as f:
        cue_words = [line.strip() for line in f]
    response,useless = generate(cue_words)
    with open(f"llama_2.1.json", 'w') as f:
        json.dump(response, f)
    with open(f"llama_2.1.json", 'w') as f:
        json.dump(useless, f)


produce()

with open("./data/llama_2.1.json", 'r', encoding='utf-8') as file:
    dictionary = json.load(file)
processed = {cue: Counter(dictionary[cue]) for cue in dictionary}
with open("./data/llama_2.1_association.json", 'w', encoding='utf-8') as file:
    json.dump(processed, file, indent=4)
