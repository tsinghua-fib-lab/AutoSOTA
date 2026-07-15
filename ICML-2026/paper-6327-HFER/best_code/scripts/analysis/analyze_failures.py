import json
import numpy as np
import os
import argparse

def get_best_threshold(valid_vals, invalid_vals):
    # Find threshold that maximizes accuracy
    all_vals = np.concatenate([valid_vals, invalid_vals])
    thresholds = np.unique(np.percentile(all_vals, np.linspace(0, 100, 100)))
    
    best_acc = 0
    best_t = 0
    
    # Expect Valid < Threshold (HFER logic)
    for t in thresholds:
        tp = sum(1 for v in valid_vals if v < t)
        tn = sum(1 for i in invalid_vals if i >= t)
        acc = (tp + tn) / len(all_vals)
        
        if acc > best_acc:
            best_acc = acc
            best_t = t
            
    return best_t, best_acc

def analyze_failures(results_file, output_prefix):
    print(f"Loading {results_file}...")
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    # Using L11 HFER as the standard comparator
    METRIC = "hfer"
    LAYER = 11
    
    valid_vals = []
    invalid_vals = []
    
    valid_items = []
    invalid_items = []

    for item in data["valid"]:
        val = item["trajectory"][LAYER][METRIC]
        if val is not None:
            valid_vals.append(val)
            valid_items.append((item['file'], val))
            
    for item in data["invalid"]:
        val = item["trajectory"][LAYER][METRIC]
        if val is not None:
            invalid_vals.append(val)
            invalid_items.append((item['file'], val))

    # Auto-calc threshold
    threshold, acc = get_best_threshold(valid_vals, invalid_vals)
    print(f"Optimal Threshold (L{LAYER}_{METRIC}): {threshold:.4f} (Acc: {acc*100:.2f}%)")

    # List A: Confusing Valid (False Negatives: Valid but High HFER)
    # Valid should be < T. If >= T, it's a failure.
    list_a = [item for item in valid_items if item[1] >= threshold]
    list_a.sort(key=lambda x: x[1], reverse=True) # Most confused first (highest HFER)

    # List B: Confident Invalid (False Positives: Invalid but Low HFER)
    # Invalid should be >= T. If < T, it's a failure.
    list_b = [item for item in invalid_items if item[1] < threshold]
    list_b.sort(key=lambda x: x[1]) # Most confident first (lowest HFER)

    print(f"List A (Confusing Valid): {len(list_a)} proofs")
    print(f"List B (Confident Invalid): {len(list_b)} proofs")

    # Save to files
    os.makedirs("analysis", exist_ok=True)
    
    with open(f"analysis/{output_prefix}_list_a_confusing_valid.json", "w") as f:
        json.dump([{"file": x[0], "hfer": x[1]} for x in list_a], f, indent=2)
        
    with open(f"analysis/{output_prefix}_list_b_confident_invalid.json", "w") as f:
        json.dump([{"file": x[0], "hfer": x[1]} for x in list_b], f, indent=2)

    print(f"Saved lists to analysis/{output_prefix}_*.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to results json")
    parser.add_argument("--name", type=str, required=True, help="Output prefix name (e.g. 1B)")
    args = parser.parse_args()
    
    analyze_failures(args.file, args.name)
