"""Run baseline (no MONICA, no cue) evaluation."""
import json, torch, sys, os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
model_path = "/models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

with open("/repo/data/expData/aime_2024_multichoice.json") as f:
    all_questions = json.load(f)

# Select only the first 15 questions (same as MONICA run)
questions = all_questions[:15]
print("Evaluating", len(questions), "questions (no cue, no MONICA)")

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

settings = {
    "do_sample": True,
    "temperature": 0.5,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 4096,
    "eos_token_id": tokenizer.eos_token_id,
}

correct = 0
results = []

for i, q in enumerate(questions):
    qid = q["question_id"]
    correct_ans = q["correct_answer"]
    
    inst = "You are a reasoning assistant for multiple choice questions. Both in thinking stage and final response stage, please put your conclusive answer in the format of \\boxed{your answer}."
    
    # Qwen3 format
    input_text = "<|im_start|>system\n" + inst + "<|im_end|>\n"
    input_text += "<|im_start|>user\n" + q["original_question"] + "<|im_end|>\n"
    input_text += "<|im_start|>assistant\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **settings)[0]
    
    response = tokenizer.decode(output_ids)
    
    # Extract answer from \boxed{...}
    import re
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    model_ans = match.group(1).strip() if match else ""
    
    is_correct = (model_ans == correct_ans)
    if is_correct:
        correct += 1
    
    results.append({
        "question_id": qid,
        "correct_answer": correct_ans,
        "model_answer": model_ans,
        "is_correct": is_correct,
    })
    
    print("[{}/{}] {}: correct={}, model={} {}".format(
        i+1, len(questions), qid[-25:], correct_ans, model_ans, 
        "CORRECT" if is_correct else "WRONG"))

# Save results
output_dir = Path("/repo/outputs/baseline_no_cue")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

acc = correct / len(questions) if questions else 0
print("\nBaseline accuracy (no cue, no MONICA): {:.4f} ({}/{})".format(acc, correct, len(questions)))
