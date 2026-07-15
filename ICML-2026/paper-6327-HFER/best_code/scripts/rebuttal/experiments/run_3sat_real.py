import os
import json
import random
import torch
import numpy as np
import gc
from pysat.solvers import Glucose3
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- MEMORY EFFICIENT HFER EXTRACTOR ---
def compute_hfer_single_layer(attention_matrix):
    A = attention_matrix.mean(dim=0).float().cpu().numpy()
    N = A.shape[0]
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    try:
        vals = np.linalg.eigvalsh(L)
        vals = np.sort(vals)
        fiedler = vals[1] if len(vals) > 1 else 0
        return float(fiedler / (vals[-1] + 1e-6))
    except: return 0.0

def get_hfer_pair(model, tokenizer, text):
    """Returns (hfer_l24, hfer_l30) for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # IMPORTANT: Model MUST be loaded with output_attentions=True
        outputs = model(**inputs, output_attentions=True)
        # Check if attentions exist
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("Model is not returning attentions. Ensure output_attentions=True.")
        
        h24 = compute_hfer_single_layer(outputs.attentions[24][0])
        del outputs
        torch.cuda.empty_cache()
    return h24

# --- 3-SAT LOGIC ---
def generate_3sat_instance(n_vars=30, m_clauses=128):
    clauses = []
    while len(clauses) < m_clauses:
        variables = random.sample(range(1, n_vars + 1), 3)
        lits = [v if random.random() > 0.5 else -v for v in variables]
        if sorted(lits) not in clauses:
            clauses.append(sorted(lits))
    return clauses

def solve_sat(clauses):
    solver = Glucose3()
    for c in clauses: solver.add_clause(c)
    is_sat = solver.solve()
    assignment = solver.get_model() if is_sat else None
    return is_sat, assignment

def format_prompt(clauses, n_vars):
    clause_strs = []
    for c in clauses:
        terms = [f"{'' if lit > 0 else 'NOT '}x{abs(lit)}" for lit in c]
        clause_strs.append("(" + " OR ".join(terms) + ")")
    formula = " AND ".join(clause_strs)
    return f"Find a satisfying assignment for this 3-SAT formula ({n_vars} vars, {len(clauses)} clauses). Format: x1=T, x2=F...\n\nFormula: {formula}\n\nAssignment:"

def check_assignment(clauses, text, n_vars):
    try:
        assignment = {}
        cleaned = text.replace(" ", "").replace("\n", "").split("Assignment:")[-1]
        for part in cleaned.split(","):
            if "=" in part:
                k, v = part.split("=")
                assignment[int(k[1:])] = ("T" in v.upper())
        if len(assignment) < n_vars: return False
        for c in clauses:
            if not any((assignment.get(abs(lit)) == (lit > 0)) for lit in c): return False
        return True
    except: return False

def run_3sat_experiment(n_instances=50):
    n_vars, m_clauses = 30, 128
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"Loading {model_id} (Config: output_attentions=True)...")
    # CRITICAL: bnb_config and config must allow attentions
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # MUST specify output_attentions=True here
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        output_attentions=True
    )

    valid_instances = []
    while len(valid_instances) < n_instances:
        clauses = generate_3sat_instance(n_vars, m_clauses)
        is_sat, _ = solve_sat(clauses)
        if is_sat: valid_instances.append(clauses)

    results = []
    for clauses in tqdm(valid_instances, desc="3-SAT Eval"):
        prompt = format_prompt(clauses, n_vars)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        response = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        hfer = get_hfer_pair(model, tokenizer, prompt + response) # Fix: renamed helper or use correct one
        is_correct = check_assignment(clauses, response, n_vars)
        
        results.append({"is_correct": is_correct, "hfer": hfer})
        torch.cuda.empty_cache()

    # Save
    os.makedirs("data/results/rebuttal", exist_ok=True)
    with open("data/results/rebuttal/sat_real.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    cor_h = [r['hfer'] for r in results if r['is_correct']]
    inc_h = [r['hfer'] for r in results if not r['is_correct']]
    print(f"\nCompleted {len(results)} instances.")
    if cor_h and inc_h:
        d = (np.mean(inc_h) - np.mean(cor_h)) / np.sqrt((np.var(cor_h) + np.var(inc_h))/2)
        print(f"Verified 3-SAT d = {d:.4f}")

if __name__ == "__main__":
    run_3sat_experiment()
