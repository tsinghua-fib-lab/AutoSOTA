import json
import os
import shutil
import glob
import random

def setup_experiment_1():
    # Config
    reclaimed_json = "data/reclaimed/1B_list_a_confusing_valid.json"
    source_dirs = ["data/experiment_ready/invalid", "data/experiment_ready/valid", "data/proofs_minif2f/invalid", "data/proofs_minif2f/unverified"]
    output_dir = "data/exp1_model_vs_model"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(f"{output_dir}/valid")
    os.makedirs(f"{output_dir}/invalid")
    
    # Load Reclaimed (Valid) List
    with open(reclaimed_json, 'r') as f:
        reclaimed_list = json.load(f)
        
    # Set of reclaimed filenames
    reclaimed_files = set(item['file'] for item in reclaimed_list)
    
    # Find these files in source
    # Problem: JSON has 'aime_1983_p9.lean', source has 'attempt_aime_1983_p9_0.lean' or similar?
    # Or maybe the JSON filenames ARE the exact filenames if they came from an earlier run.
    # Let's check if they exist directly.
    
    valid_count = 0
    found_paths = set()
    
    valid_count = 0
    found_paths = set()
    
    # Try to match
    # Reclaimed JSON names are like 'prob_name.lean'. 
    # Actual files are like 'prob_name_0.lean', 'prob_name_1.lean' etc.
    # Verification notes reference '_0.lean'.
    
    for target in reclaimed_files:
        prob_name = target.replace(".lean", "")
        
        candidates = []
        for s_dir in source_dirs:
            # Try specific glob to catch _0 etc, or exact match
            # Catch all starting with name
            start_search = glob.glob(os.path.join(s_dir, f"{prob_name}*.lean"))
            candidates.extend(start_search)
            
        if candidates:
            # Pick the first one
            src = candidates[0]
            dst = os.path.join(output_dir, "valid", os.path.basename(src))
            if not os.path.exists(dst): # Avoid overwrite if duplicates
                shutil.copy(src, dst)
                found_paths.add(src)
                valid_count += 1
        else:
            print(f"Warning: Could not find source for reclaimed file {target}")
            
    print(f"Found {valid_count} / {len(reclaimed_files)} reclaimed proofs.")
    
    # Sample Invalid
    # Pool all invalid files from all sources
    all_invalid_files = []
    for s_dir in source_dirs:
        files = glob.glob(os.path.join(s_dir, "*.lean"))
        all_invalid_files.extend(files)
        
    remaining_invalid = [f for f in all_invalid_files if f not in found_paths]
    
    # Sample count = valid_count
    if valid_count > 0:
        sample_invalid = random.sample(remaining_invalid, valid_count)
        for src in sample_invalid:
            dst = os.path.join(output_dir, "invalid", os.path.basename(src))
            shutil.copy(src, dst)
        print(f"Sampled {valid_count} invalid proofs.")
        
if __name__ == "__main__":
    setup_experiment_1()
