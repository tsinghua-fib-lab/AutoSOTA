# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import Levenshtein
from tqdm import tqdm

def load_jsonlines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonlines(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def normalize_text(text):
    return text.lower().strip()

def calculate_levenshtein_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    similarity = 1 - (distance / max_len)
    return similarity

def filter_generated_dataset(target_data, generated_data, threshold, key):
    target_questions = [normalize_text(item[key]) for item in target_data if key in item]
    filtered_data = []
    
    print(f"Original generated dataset size: {len(generated_data)}")
    
    removed_items = []
    
    for gen_item in tqdm(generated_data, desc="Filtering dataset"):
        if key not in gen_item:
            filtered_data.append(gen_item)
            continue
            
        gen_question = normalize_text(gen_item[key])
        max_similarity = 0.0
        most_similar_target = ""
        
        for target_question in target_questions:
            similarity = calculate_levenshtein_similarity(gen_question, target_question)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_target = target_question
        
        if max_similarity < threshold:
            filtered_data.append(gen_item)
        else:
            removed_items.append({
                "generated": gen_question,
                "target": most_similar_target,
                "similarity": max_similarity
            })
    
    print(f"Filtered dataset size: {len(filtered_data)}")
    print(f"Removed {len(removed_items)} items ({(len(removed_items)/len(generated_data))*100:.2f}%) due to similarity >= {threshold}")
    
    if removed_items:
        with open("removed_items.json", "w", encoding="utf-8") as f:
            json.dump(removed_items, f, ensure_ascii=False, indent=4)
        print("Saved detailed information about removed items to 'removed_items.json'")
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description="Filter generated dataset to remove potential data leakage")
    parser.add_argument("--target", required=True, help="Path to target dataset")
    parser.add_argument("--generated", required=True, help="Path to generated dataset")
    parser.add_argument("--output", required=True, help="Path to output filtered dataset")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Similarity threshold (0.0-1.0), above which items are considered too similar")
    parser.add_argument("--key", default="question", help="The key to compare in both datasets")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading target dataset from {args.target}")
    target_data = load_jsonlines(args.target)
    print(f"Target dataset size: {len(target_data)}")
    
    print(f"Loading generated dataset from {args.generated}")
    generated_data = load_jsonlines(args.generated)
    print(f"Generated dataset size: {len(generated_data)}")
    
    # Filter data
    filtered_data = filter_generated_dataset(
        target_data=target_data,
        generated_data=generated_data,
        threshold=args.threshold,
        key=args.key
    )
    
    # Save filtered data
    save_jsonlines(filtered_data, args.output)
    print(f"Saved filtered dataset to {args.output}")

if __name__ == "__main__":
    main()
