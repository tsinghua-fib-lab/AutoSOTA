#!/usr/bin/env python3
from datasets import load_dataset

# Load TinyTruthfulQA
print("Loading TinyTruthfulQA dataset...")
tiny_data = load_dataset('tinyBenchmarks/tinyTruthfulQA', 'multiple_choice')['validation']

print(f"Total items in TinyTruthfulQA: {len(tiny_data)}")
print(f"Features: {list(tiny_data.features.keys())}")

if 'question' in tiny_data.features:
    questions = tiny_data['question']
    print(f"\nUsing 'question' field")
    print(f"Number of questions: {len(questions)}")
    print(f"Unique questions: {len(set(questions))}")
    
    # Check for duplicates
    from collections import Counter
    counts = Counter(questions)
    duplicates = [(q, c) for q, c in counts.items() if c > 1]
    
    if duplicates:
        print(f"\nDuplicate questions found:")
        for q, count in duplicates:
            print(f"  {count}x: {q[:80]}...")
    else:
        print("\nNo duplicate questions in TinyBenchmarks dataset")
