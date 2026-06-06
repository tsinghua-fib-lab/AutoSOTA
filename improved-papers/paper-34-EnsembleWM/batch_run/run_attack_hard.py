#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Python script to process generation results and perform text replacements
using a T5 model. The script reads input from 'generation_results3.json', processes
specified fields by replacing a percentage of words, and saves the output to
'attack_data3.json'.

Dependencies:
- Python 3.6+
- torch
- transformers
- tqdm

Usage:
    python process_generation.py

Optional:
    Adjust parameters and paths as needed within the script.
"""

import json
import re
import random
import math
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
import sys

def main():
    # =========================
    # Configuration Parameters
    # =========================

    # Set the working directory (optional)
    # You can comment out the following line if not needed
    os.chdir("/home/feline/master-generation")

    # Parameters
    replace_percentage = 0.10  # Default 10%
    num_samples = None         # Set to None to process all samples
    max_iterations_multiplier = 5  # To prevent infinite loops
    batch_size = 8  # Batch size for processing masked sentences

    # Debug flag
    debug = False  # Set to True to enable debug outputs

    # Input and Output file paths
    input_file = 'generation_results_hard.json'
    output_file = 'attack_data_hard.json'

    # =========================
    # Setup Device
    # =========================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"Using device: {device}")

    # =========================
    # Load JSON Data
    # =========================

    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' not found. Please ensure the file exists in the current directory.")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{input_file}': {e}")
            sys.exit(1)

    if not isinstance(data, list):
        print(f"Expected the JSON data to be a list of samples. Please check the format of '{input_file}'.")
        sys.exit(1)

    # =========================
    # Initialize T5 Tokenizer and Model
    # =========================

    if debug:
        print("Loading T5 tokenizer and model...")

    try:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    except Exception as e:
        print(f"Error loading T5 model/tokenizer: {e}")
        sys.exit(1)

    model.to(device)  # Move model to GPU if available

    # =========================
    # Define Helper Functions
    # =========================

    def reconstruct_text(words):
        """
        Reconstructs text from a list of words and punctuation, adding spaces where appropriate.
        """
        text = ''
        for i in range(len(words)):
            if i > 0:
                if re.match(r'\w', words[i]) and re.match(r'\w', words[i - 1]):
                    text += ' '
                elif re.match(r'\w', words[i]) and words[i - 1] in (')', ']', '}', '"', "'"):
                    text += ' '
                elif words[i] in ('(', '[', '{', '"', "'") and not words[i - 1].isspace():
                    text += ' '
            text += words[i]
        return text

    # =========================
    # Define Fields to Process
    # =========================

    fields_to_process = [
        "llm_baseline", "llm_senso_medium", "llm_acro_medium", "llm_redgreen_medium",
        "llm_both_medium", "llm_senso_redgreen_medium", "llm_acro_redgreen_medium",
        "llm_all_three_medium"
    ]

    # =========================
    # Process Samples
    # =========================

    processed_data = []
    samples_to_process = data if num_samples is None else data[:num_samples]

    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Processing samples")):
        if debug:
            print(f"\nProcessing sample {sample_idx + 1}/{len(samples_to_process)}")
        new_sample = sample.copy()
        for field in fields_to_process:
            if field in sample:
                if debug:
                    print(f"\nProcessing field: {field}")
                text = sample[field]
                # Split text into words and punctuation
                words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
                T = len(words)
                epsilon_T = max(1, math.ceil(replace_percentage * T))
                successful_replacements = 0
                iterations = 0
                max_iterations = max_iterations_multiplier * epsilon_T
                processed_indices = set()

                if debug:
                    print(f"Total words: {T}")
                    print(f"Replacement budget (epsilon_T): {epsilon_T}")
                    print(f"Max iterations: {max_iterations}")

                # Create a list of available indices and shuffle them for randomness
                available_indices = list(range(T))
                random.shuffle(available_indices)

                while successful_replacements < epsilon_T and iterations < max_iterations and available_indices:
                    # Prepare batch of masked sentences
                    batch_input_texts = []
                    batch_word_indices = []
                    batch_original_words = []
                    batch_size_current = min(batch_size, len(available_indices))
                    indices_to_remove = []
                    for idx in range(batch_size_current):
                        word_idx = available_indices[idx]
                        original_word = words[word_idx]
                        # Skip if the word is a special token
                        if original_word in tokenizer.all_special_tokens:
                            continue
                        words_masked = words.copy()
                        words_masked[word_idx] = '<extra_id_0>'
                        masked_text = reconstruct_text(words_masked)
                        batch_input_texts.append(masked_text)
                        batch_word_indices.append(word_idx)
                        batch_original_words.append(original_word)
                        indices_to_remove.append(word_idx)
                        iterations += 1

                    # Remove processed indices
                    for idx in indices_to_remove:
                        if idx in available_indices:
                            available_indices.remove(idx)
                            processed_indices.add(idx)

                    if not batch_input_texts:
                        continue

                    # Tokenize batch input texts
                    try:
                        input_ids = tokenizer(
                            batch_input_texts,
                            return_tensors='pt',
                            padding=True,
                            truncation=True
                        ).input_ids.to(device)
                    except Exception as e:
                        print(f"Error during tokenization: {e}")
                        continue

                    # Generate outputs
                    try:
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_length=50,
                            num_beams=10,             # Reduced from 50 to 10 for speed
                            num_return_sequences=3,   # Reduced from 20 to 3
                            early_stopping=True
                        )
                    except Exception as e:
                        print(f"Error during generation: {e}")
                        continue

                    # Process outputs
                    num_return_sequences = 3
                    try:
                        outputs = outputs.view(len(batch_input_texts), num_return_sequences, -1)
                    except Exception as e:
                        print(f"Error reshaping outputs: {e}")
                        continue

                    for i in range(len(batch_input_texts)):
                        word_idx = batch_word_indices[i]
                        original_word = batch_original_words[i]
                        output_sequences = outputs[i]
                        replacement_found = False
                        for output in output_sequences:
                            output_text = tokenizer.decode(output, skip_special_tokens=False)
                            # Extract the generated text after '<extra_id_0>'
                            split_text = output_text.split('<extra_id_0>')
                            if len(split_text) < 2:
                                continue
                            gen_text = split_text[1]
                            # Remove any additional '<extra_id_X>' tokens
                            gen_text = re.split(r'<extra_id_\d+>', gen_text)[0]
                            gen_text = gen_text.strip()
                            if gen_text and gen_text.lower() != original_word.lower():
                                # Replace the word
                                words[word_idx] = gen_text
                                successful_replacements += 1
                                replacement_found = True
                                if debug:
                                    print(f"Replaced '{original_word}' with '{gen_text}' at index {word_idx}")
                                break  # Move to next word
                        if successful_replacements >= epsilon_T:
                            break

                # Reconstruct the final text
                final_text = reconstruct_text(words)
                new_sample[field] = final_text
                if debug:
                    print(f"\nFinal text for field '{field}':\n{final_text}")

        processed_data.append(new_sample)

    # =========================
    # Save Processed Data
    # =========================

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        if debug:
            print(f"\nProcessing completed. Results saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving output to '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
