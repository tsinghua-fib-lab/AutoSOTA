#!/usr/bin/env python3
"""Check for duplicate questions in TinyBenchmarks datasets"""

from datasets import load_dataset
from collections import Counter

datasets_to_check = [
    ('TruthfulQA', 'tinyBenchmarks/tinyTruthfulQA', 'multiple_choice', 'validation', 'question'),
    ('Hellaswag', 'tinyBenchmarks/tinyHellaswag', None, 'validation', 'ind'),
    ('Winogrande', 'tinyBenchmarks/tinyWinogrande', 'winogrande_xl', 'validation', None),
    ('GSM8K', 'tinyBenchmarks/tinyGSM8K', 'main', 'test', None),
    ('ARC', 'tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge', 'test', 'id'),
]

for name, dataset_path, config, split, question_field in datasets_to_check:
    print(f"\n{'='*80}")
    print(f"Checking {name}")
    print('='*80)
    
    try:
        # Load dataset
        if config:
            ds = load_dataset(dataset_path, config)[split]
        else:
            ds = load_dataset(dataset_path)[split]
        
        print(f"Total items: {len(ds)}")
        print(f"Features: {list(ds.features.keys())}")
        
        # Determine question field
        if question_field and question_field in ds.features:
            questions = ds[question_field]
            field_name = question_field
        elif 'question' in ds.features:
            questions = ds['question']
            field_name = 'question'
        elif 'input_formatted' in ds.features:
            questions = ds['input_formatted']
            field_name = 'input_formatted'
        elif 'ctx' in ds.features:
            # For Hellaswag, combine ctx and endings
            questions = [ds[i]['ctx'] for i in range(len(ds))]
            field_name = 'ctx'
        else:
            print(f"  Cannot determine question field")
            continue
        
        print(f"Using field: '{field_name}'")
        
        # Check for duplicates
        counts = Counter(questions)
        duplicates = [(q, c) for q, c in counts.items() if c > 1]
        
        if duplicates:
            print(f"\n⚠️  Found {len(duplicates)} duplicate questions:")
            for q, count in duplicates[:5]:  # Show first 5
                print(f"\n  {count}x duplicates:")
                print(f"    {q[:100]}...")
                # Show indices
                indices = [i for i, x in enumerate(questions) if x == q]
                print(f"    Indices: {indices}")
        else:
            print(f"\n✓ No duplicate questions found")
        
        # Check unique count
        unique_count = len(set(questions))
        print(f"\nUnique items: {unique_count} / {len(ds)}")
        
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n{'='*80}")
