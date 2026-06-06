# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import openai
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=openai.api_key)

try:
    from prompt.lean4_prompt import prompt
except ImportError:
    logging.error("Failed to import prompt from prompt.lean4_prompt.")
    raise

def load_data(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logging.error(f"File {json_file_path} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {json_file_path}.")
        raise

def generate_informalproof(problem, tags, formal_proof):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in Lean 4 and formal mathematics. Your task is to explain how to construct formal proofs using Lean 4 syntax and tactics. Focus on the Lean 4 approach to theorem proving, rather than traditional mathematical reasoning."},
                {"role": "user", "content": prompt.format(problem=problem, tags=tags, informal_proof=formal_proof)}
            ],
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1
        )
        lean_proof = response.choices[0].message.content
        return lean_proof
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise

def process_entry(entry):
    problem = entry.get('natural_language_statement')
    formal_proof = entry.get('formal_statement')
    tags = entry.get('tags')
    if tags:
        tags = ", ".join(tags)
    else:
        tags = ""
            
    if problem and formal_proof:
        informal_solution = generate_informalproof(problem, tags, formal_proof)
        return {
            'problem': problem,
            'formal_proof': formal_proof,
            'informal_solution': informal_solution
        }
    return None

def process_file(json_file_path, output_file_path):
    data = load_data(json_file_path)
    results = []
    for entry in data:
        result = process_entry(entry)
        if result:
            results.append(result)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"LEAN solutions saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file_path}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Lean workbook JSON file and generate informal proof.')
    parser.add_argument('--input_file', type=str, help='Path to the Lean workbook JSON file')
    parser.add_argument('--output_file', type=str, help='Path to save the output JSON file')
    args = parser.parse_args()

    try:
        process_file(args.input_file, args.output_file)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
