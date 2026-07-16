#!/usr/bin/env python3
"""Debug TinyBenchmarks matching to find duplicate item 747"""

from datasets import load_dataset
import pandas as pd

# Load the TruthfulQA dataset
print("Loading TinyTruthfulQA dataset...")
tiny_data = load_dataset('tinyBenchmarks/tinyTruthfulQA', 'multiple_choice')['validation']

print(f"Total items in TinyTruthfulQA: {len(tiny_data)}")
print(f"Features: {list(tiny_data.features.keys())}")
print()

# Load prompts
prompts_path = 'data/metabench_data/benchmark-data/truthfulqa_prompts.csv'
prompts_df = pd.read_csv(prompts_path)

print(f"Total prompts: {len(prompts_df)}")
print()

# Get questions
if 'question' in tiny_data.features:
    questions = tiny_data['question']
    print("Using 'question' field")
else:
    questions = tiny_data['input_formatted']
    print("Using 'input_formatted' field")

print()

# Find matches and track which questions map to item 747
item_747_matches = []
all_matches = []

for idx, question in enumerate(questions):
    matched_item = None
    for _, row in prompts_df.iterrows():
        prompt = str(row['prompt'])
        if question in prompt:
            matched_item = row['item']
            break
    
    if matched_item == 747:
        item_747_matches.append((idx, question))
    
    all_matches.append((idx, matched_item, question[:80]))

print(f"Questions that match item 747:")
print("-" * 80)
for idx, question in item_747_matches:
    print(f"Index {idx}: {question}")
    print()

# Show all matches
print("\nAll matches (first 10):")
print("-" * 80)
for idx, item, question in all_matches[:10]:
    print(f"Tiny idx {idx} -> Item {item}: {question}...")

# Check for duplicates
print(f"\n\nTotal matches to item 747: {len(item_747_matches)}")

# Get the prompt for item 747
item_747_prompt = prompts_df[prompts_df['item'] == 747]['prompt'].values[0] if len(prompts_df[prompts_df['item'] == 747]) > 0 else "Not found"
print(f"\nItem 747 prompt from metabench:\n{item_747_prompt}")
