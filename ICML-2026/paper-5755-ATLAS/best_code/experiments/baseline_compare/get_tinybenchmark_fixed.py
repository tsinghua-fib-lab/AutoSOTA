#!/usr/bin/env python3
"""
Generate TinyBenchmarks item IDs with improved matching
Only matches the LAST/PRIMARY question in multi-turn prompts
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

def extract_last_question(prompt):
    """Extract the last Q: from a prompt (the main question, not context)"""
    # Find all Q: patterns
    questions = re.findall(r'Q:\s*([^\n]+)', prompt)
    if questions:
        # Return the last question (the actual question being asked)
        return questions[-1].strip()
    return prompt.strip()

# Prepare metabench data with extracted last questions
metabench_data = {}
for _, row in prompts_df.iterrows():
    item_id = row['item']
    prompt = row['prompt']
    last_q = extract_last_question(prompt)
    metabench_data[item_id] = {
        'full_prompt': prompt,
        'last_question': last_q
    }

print("Matching questions (using LAST question in prompts only)...\n")

# Create a list to store the matched item IDs
item_ids = []
questions = []
match_details = []

# Match each question in tiny_data to an item ID
for idx, question in enumerate(tiny_data_questions):
    questions.append(question)
    found = False
    matched_item = None
    
    # Clean the tinybenchmarks question
    clean_question = question.strip()
    if clean_question.startswith("Q:"):
        clean_question = clean_question[2:].strip()
    
    # Try to match against the LAST question only (not earlier context)
    for item_id, data in metabench_data.items():
        last_q = data['last_question']
        
        # Check if the question appears in the last question
        if clean_question in last_q or last_q in clean_question:
            matched_item = item_id
            found = True
            break
    
    if found:
        item_ids.append(matched_item)
        match_details.append(f"matched to item {matched_item}")
    else:
        item_ids.append('not_found')
        match_details.append("NOT FOUND")
        print(f"NOT FOUND: {clean_question[:80]}...")

# Create a DataFrame with questions and item IDs
results_df = pd.DataFrame({
    'item_id': item_ids,
    'question': questions
})

# Check for duplicates
print("\n" + "="*80)
duplicates = results_df[results_df.duplicated('item_id', keep=False) & (results_df['item_id'] != 'not_found')]
if len(duplicates) > 0:
    print(f"WARNING: Found {len(duplicates)} duplicate item IDs:\n")
    for item_id in duplicates['item_id'].unique():
        dup_rows = results_df[results_df['item_id'] == item_id]
        print(f"Item {item_id} matched by {len(dup_rows)} questions:")
        for _, row in dup_rows.iterrows():
            print(f"  - {row['question'][:80]}...")
        print()
else:
    print("No duplicates found!")

# Count matches
not_found = (results_df['item_id'] == 'not_found').sum()
unique_matches = results_df[results_df['item_id'] != 'not_found']['item_id'].nunique()

print("="*80)
print(f"\nSummary:")
print(f"  Total TinyBenchmarks items: {len(results_df)}")
print(f"  Matched to metabench: {len(results_df) - not_found}")
print(f"  Not found: {not_found}")
print(f"  Unique metabench items: {unique_matches}")
print()

# Save the results
output_path = f'tiny{DATASET_NAME.lower()}_item_ids.csv'
results_df[['item_id', 'question']].to_csv(output_path, index=False)

print(f"Saved {len(results_df)} questions to {output_path}")
print(f"\nFirst few items:")
print(results_df[['item_id', 'question']].head(10))
