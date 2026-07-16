#!/usr/bin/env python3
"""
Generate TinyBenchmarks item IDs with STRICT matching
Matches questions more carefully to avoid false positives
"""

from datasets import load_dataset
import pandas as pd
import re

# Load the TruthfulQA dataset
DATASET_NAME = 'TruthfulQA'
print(f"Loading TinyBenchmarks tiny{DATASET_NAME}...")
tiny_data = load_dataset('tinyBenchmarks/tiny'+DATASET_NAME, 'multiple_choice')['validation']
print(f"  Loaded {len(tiny_data)} items\n")

# Path to the CSV file with item IDs and prompts
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument('--prompts_dir', default='data/metabench_data/benchmark-data')
_known, _ = _parser.parse_known_args()
ID_PATH = _known.prompts_dir + '/' + DATASET_NAME.lower() + '_prompts.csv'

# Read the CSV file with item IDs
prompts_df = pd.read_csv(ID_PATH)
print(f"Loaded {len(prompts_df)} prompts from metabench\n")

# Get questions from TinyBenchmarks
if 'question' in tiny_data.features:
    tiny_data_questions = tiny_data['question']
    print("Using 'question' field from TinyBenchmarks")
else:
    tiny_data_questions = tiny_data['input_formatted']
    print("Using 'input_formatted' field from TinyBenchmarks")

print(f"Number of questions: {len(tiny_data_questions)}\n")

# Create a list to store the matched item IDs
item_ids = []
questions = []
match_info = []

def extract_first_question(prompt):
    """Extract the first Q: ... part from a prompt"""
    # Match "Q: ... ?" or "Q: ... \n"
    match = re.search(r'Q:\s*([^\n]+(?:\?|$))', prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()

# Prepare metabench data with extracted first questions
metabench_questions = {}
for _, row in prompts_df.iterrows():
    item_id = row['item']
    prompt = row['prompt']
    first_q = extract_first_question(prompt)
    metabench_questions[item_id] = {
        'full_prompt': prompt,
        'first_question': first_q
    }

print("Matching questions...\n")
print("=" * 80)

# Match each question in tiny_data to an item ID
for idx, question in enumerate(tiny_data_questions):
    questions.append(question)
    found = False
    match_type = "not_found"
    
    # Strip common prefixes
    clean_question = question.strip()
    if clean_question.startswith("Q:"):
        clean_question = clean_question[2:].strip()
    
    # Try exact match on first question
    for item_id, data in metabench_questions.items():
        first_q = data['first_question']
        
        # Clean the metabench question too
        clean_metabench = first_q
        if clean_metabench.startswith("Q:"):
            clean_metabench = clean_metabench[2:].strip()
        
        # Try exact match
        if clean_question == clean_metabench:
            item_ids.append(item_id)
            match_type = "exact_match"
            found = True
            break
    
    # If no exact match, try substring match but only on first question
    if not found:
        for item_id, data in metabench_questions.items():
            first_q = data['first_question']
            
            # Only match if the tiny question is the main question (not part of context)
            if clean_question in first_q or first_q in clean_question:
                item_ids.append(item_id)
                match_type = "substring_match"
                found = True
                break
    
    # Last resort: check if question appears at the START of the prompt
    if not found:
        for item_id, data in metabench_questions.items():
            full_prompt = data['full_prompt']
            # Check if question appears near the start (within first 200 chars)
            if clean_question in full_prompt[:200]:
                item_ids.append(item_id)
                match_type = "prefix_match"
                found = True
                break
    
    if not found:
        item_ids.append('not_found')
        print(f"NOT FOUND: {clean_question[:80]}...")
    
    match_info.append(match_type)

print("=" * 80)
print()

# Create a DataFrame with questions and item IDs
results_df = pd.DataFrame({
    'item_id': item_ids,
    'question': questions,
    'match_type': match_info
})

# Check for duplicates
duplicates = results_df[results_df.duplicated('item_id', keep=False) & (results_df['item_id'] != 'not_found')]
if len(duplicates) > 0:
    print(f"WARNING: Found {len(duplicates)} duplicate item IDs:")
    print(duplicates[['item_id', 'question']])
    print()

# Count matches
not_found = (results_df['item_id'] == 'not_found').sum()
unique_matches = results_df[results_df['item_id'] != 'not_found']['item_id'].nunique()

print(f"Summary:")
print(f"  Total TinyBenchmarks items: {len(results_df)}")
print(f"  Matched to metabench: {len(results_df) - not_found}")
print(f"  Not found: {not_found}")
print(f"  Unique metabench items: {unique_matches}")
print(f"  Match types:")
for match_type in match_info:
    count = match_info.count(match_type)
    if count > 0:
        print(f"    {match_type}: {count}")
print()

# Save the results to a CSV file
output_path = f'tiny{DATASET_NAME.lower()}_item_ids.csv'

# Remove duplicates - keep first occurrence
results_clean = results_df.drop_duplicates('item_id', keep='first')

# Only save item_id and question columns
results_clean[['item_id', 'question']].to_csv(output_path, index=False)

print(f"Saved {len(results_clean)} questions (after removing duplicates) to {output_path}")
print(f"First few items:")
print(results_clean[['item_id', 'question']].head(10))
