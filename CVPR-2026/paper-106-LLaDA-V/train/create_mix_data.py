import json
import random
import math
import argparse

def find_closest_simple_ratio(target_ratio, max_val=10):
    """Finds the closest simple integer ratio (up to max_val:1 or 1:max_val) to the target float ratio."""
    if target_ratio <= 0:
        return (0, 1) if target_ratio == 0 else (1, 0) # Handle zero cases

    candidates = []
    # Generate ratios like 1:1, 2:1, ..., max_val:1
    for i in range(1, max_val + 1):
        candidates.append((i, 1))
    # Generate ratios like 1:2, 1:3, ..., 1:max_val
    for i in range(2, max_val + 1):
        candidates.append((1, i))

    best_ratio = (1, 1)
    min_diff = float('inf')

    for n_ratio, i_ratio in candidates:
        candidate_float = n_ratio / i_ratio
        diff = abs(target_ratio - candidate_float)

        # Preference for simpler ratios if difference is very close
        current_best_float = best_ratio[0] / best_ratio[1]
        current_diff = abs(target_ratio - current_best_float)

        if diff < current_diff - 1e-9: # Significantly closer
             min_diff = diff
             best_ratio = (n_ratio, i_ratio)
        elif abs(diff - current_diff) < 1e-9: # Differences are almost identical, prefer simpler ratio
            # Simpler ratio defined as smaller sum of components
            if (n_ratio + i_ratio) < (best_ratio[0] + best_ratio[1]):
                 best_ratio = (n_ratio, i_ratio)

    return best_ratio

def process_and_merge_datasets(normal_file, inference_file, output_file, max_ratio_val=10):
    """
    Processes and merges two datasets (normal and inference) into a single file.
    - Appends '/no_think' to every human turn in normal data.
    - Randomly appends '/think' (50% chance) to every human turn in inference data.
    - Mix ratio is determined by finding the closest simple integer ratio (up to max_ratio_val) 
      to the actual count ratio.

    Args:
        normal_file (str): Path to the normal data JSON file.
        inference_file (str): Path to the inference data JSON file.
        output_file (str): Path for the merged output JSON file.
        max_ratio_val (int): The maximum component value for the simple ratio (e.g., 10 for 10:1 or 1:10).
    """
    try:
        with open(normal_file, 'r', encoding='utf-8') as f:
            normal_data = json.load(f)
        print(f"Read {len(normal_data)} items from normal file: {normal_file}")
    except FileNotFoundError:
        print(f"Error: Normal data file not found at {normal_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from normal data file: {normal_file}")
        return

    try:
        with open(inference_file, 'r', encoding='utf-8') as f:
            inference_data = json.load(f)
        print(f"Read {len(inference_data)} items from inference file: {inference_file}")
    except FileNotFoundError:
        print(f"Error: Inference data file not found at {inference_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from inference data file: {inference_file}")
        return

    len_normal_orig = len(normal_data)
    len_inference_orig = len(inference_data)

    if len_normal_orig == 0 and len_inference_orig == 0:
        print("Error: Both input files are empty.")
        return

    processed_normal = []
    for item in normal_data:
        if 'conversations' in item and isinstance(item['conversations'], list):
            for turn in item['conversations']:
                if turn.get('from') == 'human' and 'value' in turn:
                    turn['value'] += ' /no_think'
        processed_normal.append(item)

    processed_inference = []
    for item in inference_data:
        if 'conversations' in item and isinstance(item['conversations'], list):
            for turn in item['conversations']:
                if turn.get('from') == 'human' and 'value' in turn:
                    if random.random() < 0.5:
                        turn['value'] += ' /think'
        processed_inference.append(item)

    # Calculate the closest simple mix ratio
    if len_inference_orig == 0:
        normal_ratio, inference_ratio = (1, 0) # All normal
    elif len_normal_orig == 0:
        normal_ratio, inference_ratio = (0, 1) # All inference
    else:
        actual_decimal_ratio = len_normal_orig / len_inference_orig
        normal_ratio, inference_ratio = find_closest_simple_ratio(actual_decimal_ratio, max_val=max_ratio_val)
        print(f"Actual count ratio: {len_normal_orig}:{len_inference_orig} (~{actual_decimal_ratio:.2f}:1)")
        print(f"Using closest simple ratio (Normal:Inference) = {normal_ratio}:{inference_ratio}")


    merged_data = []
    normal_idx, inference_idx = 0, 0
    len_normal, len_inference = len(processed_normal), len(processed_inference)

    # Interleave based on the calculated simple ratio
    while normal_idx < len_normal or inference_idx < len_inference:
        # Add normal items for this cycle
        limit_normal = normal_idx + normal_ratio
        # Ensure we handle ratio=0 case correctly
        if normal_ratio > 0:
           while normal_idx < min(limit_normal, len_normal):
                merged_data.append(processed_normal[normal_idx])
                normal_idx += 1
        else: # If normal_ratio is 0, skip adding normal items in this cycle
             pass

        # Add inference items for this cycle
        limit_inference = inference_idx + inference_ratio
        if inference_ratio > 0:
            while inference_idx < min(limit_inference, len_inference):
                merged_data.append(processed_inference[inference_idx])
                inference_idx += 1
        else: # If inference_ratio is 0, skip adding inference items
            pass
         # Safety break to prevent infinite loops if both ratios are 0 (shouldn't happen with logic above)
        if normal_ratio == 0 and inference_ratio == 0:
            print("Warning: Both ratios became 0, breaking merge loop.")
            break
           
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged {len(merged_data)} items into {output_file}")
    except IOError as e:
        print(f"Error writing output file {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Merge normal and inference datasets')
    parser.add_argument('--normal_data', 
                       type=str, 
                       required=True,
                       help='Path to normal dataset JSON file')
    parser.add_argument('--inference_data', 
                       type=str, 
                       required=True,
                       help='Path to inference dataset JSON file') 
    parser.add_argument('--output_path', 
                       type=str, 
                       required=True,
                       help='Output path for merged dataset JSON file')
    
    args = parser.parse_args()
    MAX_SIMPLE_RATIO_VALUE = 5  # Maximum value for ratio components (e.g., 10 means ratios up to 10:1 or 1:10)

    print("Starting dataset processing and merging...")
    process_and_merge_datasets(
        args.normal_data,
        args.inference_data,
        args.output_path,
        MAX_SIMPLE_RATIO_VALUE
    )
    print("Processing finished.")

if __name__ == "__main__":
    main()