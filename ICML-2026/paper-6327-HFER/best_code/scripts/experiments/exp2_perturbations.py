import os
import random
import re

def perturb_delete_step(code_lines):
    # Find lines with 'have' or 'show' and comment one out
    candidates = [i for i, line in enumerate(code_lines) if 'have' in line or 'show' in line]
    if not candidates:
        return None, "no_valid_lines"
    
    msg_idx = random.choice(candidates)
    new_lines = code_lines[:]
    new_lines[msg_idx] = "-- " + new_lines[msg_idx] # Comment out
    return "\n".join(new_lines), "delete_step"

def perturb_wrong_lemma(code_lines):
    # Replace common lemmas with wrong ones
    # add_comm -> mul_comm
    # mul_assoc -> add_assoc
    # le_refl -> lt_irrefl
    replacements = [
        ('add_comm', 'mul_comm'),
        ('mul_comm', 'add_comm'),
        ('add_assoc', 'mul_assoc'),
        ('mul_assoc', 'add_assoc'),
        ('le_refl', 'lt_irrefl')
    ]
    
    content = "\n".join(code_lines)
    for src, tgt in replacements:
        if src in content:
            new_content = content.replace(src, tgt, 1) # Replace 1 occurrence
            return new_content, f"replace_{src}_{tgt}"
            
    return None, "no_lemmas"

def perturb_type_error(code_lines):
    # Introduce a subtle type error?
    # e.g. change 0 to "0" (string) if possible? No, Lean is strictly typed.
    # Change a Nat to Int? 
    # Hard to do with regex reliability.
    # Let's try breaking a naming chain.
    content = "\n".join(code_lines)
    if "h1" in content and "h2" in content:
        # Swap h1 for h2
        new_content = content.replace("h1", "h2")
        return new_content, "swap_hypothesis"
    return None, "no_hypotheses"

import shutil
import argparse
import glob

# ... (Previous strategies kept) ...

def generate_perturbations(proof_file, output_dir):
    if not os.path.exists(proof_file):
        print(f"File not found: {proof_file}")
        return

    with open(proof_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    lines = code.split('\n')
    base_name = os.path.basename(proof_file).replace(".lean", "")
    
    # Try strategies
    strategies = [perturb_delete_step, perturb_wrong_lemma, perturb_type_error]
    
    count = 0
    for strategy in strategies:
        new_code, p_type = strategy(lines)
        if new_code:
            out_name = f"{base_name}_{p_type}.lean"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(new_code)
            count += 1
    return count

def batch_process(input_dir, output_root):
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    
    valid_dir = os.path.join(output_root, "valid")
    invalid_dir = os.path.join(output_root, "invalid")
    os.makedirs(valid_dir)
    os.makedirs(invalid_dir)
    
    files = glob.glob(os.path.join(input_dir, "*.lean"))
    print(f"Found {len(files)} input files in {input_dir}")
    
    total_perturbed = 0
    for f_path in files:
        # Copy valid
        shutil.copy(f_path, os.path.join(valid_dir, os.path.basename(f_path)))
        
        # Generate invalid
        n = generate_perturbations(f_path, invalid_dir)
        total_perturbed += n
        
    print(f"Processed {len(files)} valid proofs.")
    print(f"Generated {total_perturbed} perturbed (invalid) proofs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing valid lean files")
    parser.add_argument("--output-dir", required=True, help="Output root for experiment")
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir)
