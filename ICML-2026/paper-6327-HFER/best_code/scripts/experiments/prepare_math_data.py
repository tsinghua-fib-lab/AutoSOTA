import os
import json
import re
import torch
import random
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def extract_boxed_answer(text):
    # Find \boxed{...} matches
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches:
        return matches[-1] # Return the last one
    return None

def normalize_answer(ans):
    if ans is None: return ""
    return ans.replace(" ", "").strip()

def main():
    parser = argparse.ArgumentParser(description="Generate MATH solutions with a given model.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--valid-dir", type=str, default="data/math/valid", help="Output directory for valid proofs")
    parser.add_argument("--invalid-dir", type=str, default="data/math/invalid", help="Output directory for invalid proofs")
    parser.add_argument("--target", type=int, default=50, help="Target number of samples per class (valid/invalid)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max total math problems to attempt")
    
    args = parser.parse_args()

    ensure_dir(args.valid_dir)
    ensure_dir(args.invalid_dir)

    print(f"Loading dataset: lighteval/MATH...")
    try:
        dataset = load_dataset("lighteval/MATH", "all", split="train")
    except Exception as e:
        print(f"Failed load lighteval: {e}. Trying qwedsacf/competition_math...")
        try:
             dataset = load_dataset("qwedsacf/competition_math", split="train")
        except Exception as e2:
             print(f"Failed backup: {e2}. Exiting.")
             return

    # Filter for Level 3-4, Algebra/Number Theory
    print("Filtering for Level 3-4, Algebra/Number Theory...")
    targets_types = ['Algebra', 'Number Theory']
    target_levels = ['Level 3', 'Level 4']
    
    subset = []
    ds_list = list(dataset)
    random.shuffle(ds_list)
    
    for ex in ds_list:
        if len(subset) >= args.max_samples: break
        l = ex.get('level', '')
        t = ex.get('type', '')
        if l in target_levels and t in targets_types:
            subset.append(ex)
            
    print(f"Selected {len(subset)} candidate problems.")
    
    if not subset:
        print("No samples found matching criteria.")
        return

    # Load Model
    print(f"Loading model: {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True 
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    valid_count = len([f for f in os.listdir(args.valid_dir) if f.endswith(".txt")])
    invalid_count = len([f for f in os.listdir(args.invalid_dir) if f.endswith(".txt")])
    
    print(f"Initial counts -> Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"Target: {args.target} each.")

    print("Generating solutions...")
    for i, example in enumerate(tqdm(subset)):
        # Stopping Condition
        if valid_count >= args.target and invalid_count >= args.target:
            print("Target counts reached. Stopping.")
            break

        problem = example['problem']
        gt_solution = example['solution']
        gt_answer = extract_boxed_answer(gt_solution)
        
        # Prepare Prompt (Generic Chat)
        messages = [
            {"role": "user", "content": f"Solve this step-by-step. Put your final answer in \\boxed{{}}.\n\n{problem}"}
        ]
        
        # Apply template if available, else fallback
        if tokenizer.chat_template:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        else:
            # Simple fallback for models without chat template (rare for instruct)
            text = f"User: {messages[0]['content']}\nAssistant:"
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Validation
        model_answer = extract_boxed_answer(generated_text)
        
        is_correct = False
        if model_answer and gt_answer:
            if normalize_answer(model_answer) == normalize_answer(gt_answer):
                is_correct = True
        
        # Determine strict save logic to satisfy user request "Stop when we have 50 v 50"
        filename = f"math_gen_{i}.txt"
        content = f"Problem:\n{problem}\n\nSolution:\n{generated_text}"
        
        if is_correct:
            if valid_count < args.target:
                with open(os.path.join(args.valid_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)
                valid_count += 1
        else:
            if invalid_count < args.target:
                with open(os.path.join(args.invalid_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)
                invalid_count += 1
            
    print(f"Generation Complete.")
    print(f"Final Counts -> Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"Valid Data: {args.valid_dir}")
    print(f"Invalid Data: {args.invalid_dir}")

if __name__ == "__main__":
    main()
