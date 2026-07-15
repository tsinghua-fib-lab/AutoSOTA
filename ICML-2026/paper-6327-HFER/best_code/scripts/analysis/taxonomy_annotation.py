import os
import json

def categorize_proof(content):
    # Simple keyword heuristics
    content = content.lower()
    
    # 1. Incomplete/Cutoff (Often Logic/Planning fail, but user wants Logic vs Calc)
    if "end" not in content and "qed" not in content:
        return "Logic_Incomplete"
        
    # 2. Calculation Indicators
    calc_keywords = ["calc", "ring", "linarith", "norm_num", "simp", "field_simp"]
    for k in calc_keywords:
        if k in content:
            return "Calc"
            
    # 3. Logic Indicators (Default)
    # rw, apply, intro, etc are logic. 
    return "Logic"

def main():
    invalid_dir = "data/experiment_ready/invalid"
    taxonomy_file = "data/experiment_ready/taxonomy.json"
    
    classification = {}
    
    if not os.path.exists(invalid_dir):
        print(f"Directory not found: {invalid_dir}")
        return

    files = [f for f in os.listdir(invalid_dir) if f.endswith(".lean") or f.endswith(".txt")]
    
    counts = {"Logic": 0, "Logic_Incomplete": 0, "Calc": 0}
    
    for f in files:
        path = os.path.join(invalid_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            
        cat = categorize_proof(content)
        classification[f] = cat
        counts[cat] += 1
        
    print(f"Classification complete: {json.dumps(counts, indent=2)}")
    
    with open(taxonomy_file, "w") as f:
        json.dump(classification, f, indent=2)
    print(f"Saved to {taxonomy_file}")

if __name__ == "__main__":
    main()
