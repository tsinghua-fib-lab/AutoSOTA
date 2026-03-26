#!/usr/bin/env python3

import time
from pathlib import Path
import textwrap

import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import os
import hashlib
import spacy
import pandas as pd
import random
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import itertools

# Import custom modules
import modules.shared as shared
from modules.model import load_model

# Note: Do NOT import modules.text_generation here
# We'll import it inside main() after initializing shared.nlp

def main():
    # Change working directory
    os.chdir("/home/feline/master-generation")

    # Load the English language model
    nlp = spacy.load("en_core_web_sm")
    shared.nlp = nlp

    # Set model parameters
    shared.model_name = "Meta-Llama-3.1-8B"
    shared.act_order = True

    print(f"Loading {shared.model_name}...")
    t0 = time.time()

    shared.groupsize = 128
    shared.wbits = 4
    shared.use_flash_attention_2 = True

    # Load the model and tokenizer
    shared.model, shared.tokenizer = load_model(shared.model_name, gptq=False, awq=False)

    print(f"Loaded the model in {(time.time() - t0):.2f} seconds.")

    # Load sensorimotor data
    df = pd.read_csv('Lancaster_sensorimotor_norms_for_39707_words.csv', header=0)
    shared.sensorimotor = df.set_index('Word').T.to_dict('dict')
    shared.classes = [
        'Auditory', 'Gustatory', 'Haptic', 'Interoceptive',
        'Olfactory', 'Visual', 'Foot_leg', 'Hand_arm',
        'Head', 'Mouth', 'Torso'
    ]

    # Now that shared.nlp is set, import modules.text_generation
    from modules.text_generation import generate_reply

    # Define hashing functions
    def secure_hash_for_sentence(last_sentence, range_min, range_max):
        doc = shared.nlp(last_sentence)
        core_sentence = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        core_sentence_str = " ".join(core_sentence)
        print(core_sentence_str)

        hashed_sentence_bytes = hashlib.sha256(core_sentence_str.encode()).digest()
        hashed_sentence_integers = int.from_bytes(hashed_sentence_bytes[:4], byteorder='big')

        mapped_sentence_number = (hashed_sentence_integers % (range_max - range_min + 1)) + range_min
        return mapped_sentence_number

    def secure_hash_for_token(last_token, range_min, range_max):
        hashed_token_bytes = hashlib.sha256(str(last_token.item()).encode()).digest()
        hashed_token_integers = int.from_bytes(hashed_token_bytes[:4], byteorder='big')

        mapped_token_number = (hashed_token_integers % (range_max - range_min + 1)) + range_min
        return mapped_token_number

    def secure_hash_to_numbers(last_sentence, last_token, range_list):
        result_numbers = []

        # Compute the secure hash for the last token if last_token is not None
        if last_token is not None:
            token_hash = secure_hash_for_token(last_token, range_list[0][0], range_list[0][1])
        else:
            token_hash = shared.secret_key[0]  # Reuse the shared key if last_token is missing

        result_numbers.append(token_hash)

        # Compute the secure hash for the last sentence if last_sentence is not None
        if last_sentence is not None:
            sentence_hash = secure_hash_for_sentence(last_sentence, range_list[1][0], range_list[1][1])
        else:
            sentence_hash = shared.secret_key[1]  # Reuse the shared key if last_sentence is missing

        result_numbers.append(sentence_hash)

        return result_numbers

    def get_last_sentence(text):
        doc = shared.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        if sentences:
            return sentences[-1]
        else:
            return None  # Return None if there are no sentences

    # Initialize shared variables
    shared.secret_key = [0, 0]
    shared.new_sentence = False

    # Parameters
    num_samples = 400  # Number of random samples to extract (adjust as needed)
    num_tokens_trimmed = 200  # Number of tokens to trim from the end
    tokenizer_name = "gpt2"  # Tokenizer model name

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load the RealNews-like subset of the C4 dataset
    dataset = load_dataset("c4", "realnewslike", split="train")

    # Function to tokenize and process text
    def process_text(text, num_tokens_trimmed):
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        # Check if there are enough tokens to trim
        if len(tokens) <= num_tokens_trimmed:
            return None, None
        # Split tokens into prompt and baseline completion
        prompt_tokens = tokens[:-num_tokens_trimmed]
        baseline_tokens = tokens[-num_tokens_trimmed:]
        # Convert back to string format
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
        baseline_completion = tokenizer.convert_tokens_to_string(baseline_tokens)
        return prompt, baseline_completion

    # Select random samples
    sampled_indices = random.sample(range(len(dataset)), num_samples)

    # Process each sampled text
    c4_random_samples = []
    for idx in sampled_indices:
        text = dataset[idx]["text"]
        prompt, baseline_completion = process_text(text, num_tokens_trimmed)
        if prompt and baseline_completion:
            c4_random_samples.append({"prompt": prompt, "baseline": baseline_completion})

    # Define watermark feature settings
    feature_levels = {
        # Uncomment and adjust the levels as needed
        # 'weak': {'delta_senso': 1.0, 'delta_acro': 10.0, 'delta_redgreen': 1.0},
        'medium': {'delta_senso': 1.0, 'delta_acro': 10.0, 'delta_redgreen': 1.0},
        # 'strong': {'delta_senso': 5.0, 'delta_acro': 40.0, 'delta_redgreen': 10.0}
    }

    # Define configurations
    configurations = {
        'baseline': {
            'description': 'Baseline Completion (No LLM)',
            'type': 'baseline',
        },
        'llm_baseline': {
            'description': 'LLM Baseline (No Watermark)',
            'type': 'llm',
            'delta_senso': 0.0,
            'delta_acro': 0.0,
            'delta_redgreen': 0.0  # Initialize delta_redgreen
        }
    }

    # Add configurations for each feature individually and combinations with different levels
    for level_name, levels in feature_levels.items():
        # Single feature: delta_senso
        configurations[f'llm_senso_{level_name}'] = {
            'description': f'LLM with delta_senso ({level_name} setting)',
            'type': 'llm',
            'delta_senso': levels['delta_senso'],
            'delta_acro': 0.0,
            'delta_redgreen': 0.0
        }
        # Single feature: delta_acro
        configurations[f'llm_acro_{level_name}'] = {
            'description': f'LLM with delta_acro ({level_name} setting)',
            'type': 'llm',
            'delta_senso': 0.0,
            'delta_acro': levels['delta_acro'],
            'delta_redgreen': 0.0
        }
        # Single feature: delta_redgreen
        configurations[f'llm_redgreen_{level_name}'] = {
            'description': f'LLM with delta_redgreen ({level_name} setting)',
            'type': 'llm',
            'delta_senso': 0.0,
            'delta_acro': 0.0,
            'delta_redgreen': levels['delta_redgreen']
        }
        # Combination: delta_senso and delta_acro
        configurations[f'llm_both_{level_name}'] = {
            'description': f'LLM with delta_senso and delta_acro ({level_name} setting)',
            'type': 'llm',
            'delta_senso': levels['delta_senso'],
            'delta_acro': levels['delta_acro'],
            'delta_redgreen': 0.0
        }
        # Combination: delta_senso and delta_redgreen
        configurations[f'llm_senso_redgreen_{level_name}'] = {
            'description': f'LLM with delta_senso and delta_redgreen ({level_name} setting)',
            'type': 'llm',
            'delta_senso': levels['delta_senso'],
            'delta_acro': 0.0,
            'delta_redgreen': levels['delta_redgreen']
        }
        # Combination: delta_acro and delta_redgreen
        configurations[f'llm_acro_redgreen_{level_name}'] = {
            'description': f'LLM with delta_acro and delta_redgreen ({level_name} setting)',
            'type': 'llm',
            'delta_senso': 0.0,
            'delta_acro': levels['delta_acro'],
            'delta_redgreen': levels['delta_redgreen']
        }
        # All three features together
        configurations[f'llm_all_three_{level_name}'] = {
            'description': f'LLM with delta_senso, delta_acro, and delta_redgreen ({level_name} setting)',
            'type': 'llm',
            'delta_senso': levels['delta_senso'],
            'delta_acro': levels['delta_acro'],
            'delta_redgreen': levels['delta_redgreen']
        }

    # Function to generate replies based on configuration
    def generate_replies(sample, config, generate_params):
        if config['type'] == 'baseline':
            return sample['baseline']
        elif config['type'] == 'llm':
            # Set watermark features
            shared.delta_senso = config.get('delta_senso', 0.0)
            shared.delta_acro = config.get('delta_acro', 0.0)
            shared.delta_redgreen = config.get('delta_redgreen', 0.0)  # Set delta_redgreen
            shared.secret_key = [0, 0]  # Adjust if needed per configuration
            shared.new_sentence = False  # Adjust if needed per configuration

            question = f'''<|begin_of_text|> {sample["prompt"]}'''
            reply, done = generate_reply(question, generate_params, eos_token='<|end_of_text|>')
            return reply
        else:
            raise ValueError("Unknown configuration type")

    # Generation parameters (adjust as needed)
    generate_params = {
        'max_new_tokens': 200,
        'add_bos_token': False,
        'truncation_length': 4096,
        'custom_stopping_strings': [
            "### Human:", "Human:", "user:", "USER:",
            "Q:", "<|im_end|>", "<|im_start|>system"
        ],
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'do_sample': False,
        'early_stopping': False,
        'seed': 0,
    }

    # Initialize results list
    results = []

    print("Starting generation...")

    # Iterate through each sample
    for i, sample in enumerate(c4_random_samples, start=1):
        print(f"\nProcessing Sample {i}/{num_samples}")
        # Initialize the sample_result with prompt and baseline
        sample_result = {
            'prompt': sample['prompt'],
            'baseline': sample['baseline']
        }

        # Iterate through each configuration
        for config_name, config in configurations.items():
            description = config['description']
            print(f"  Generating for configuration: {description}")

            # Generate the reply based on the configuration
            try:
                reply = generate_replies(sample, config, generate_params)
            except Exception as e:
                print(f"    Error generating reply for {config_name}: {e}")
                reply = None  # Or handle as needed

            # Define the key for the configuration
            json_key = config_name  # e.g., 'llm_baseline', 'llm_senso_medium', etc.

            # Store the reply in the sample_result
            sample_result[json_key] = reply

        # Append the sample_result to results
        results.append(sample_result)

        print(f"Completed Sample {i}/{num_samples}")

    # Save the results to a JSON file
    output_filename = 'generation_results_soft.json'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nAll results have been saved to '{output_filename}'.")
    except Exception as e:
        print(f"\nFailed to save results to '{output_filename}': {e}")

if __name__ == "__main__":
    main()
