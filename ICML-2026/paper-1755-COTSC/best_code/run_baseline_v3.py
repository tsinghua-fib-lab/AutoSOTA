"""Run baseline with fixed instruction format."""
import json, torch, sys, os, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
model_path = "/models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

with open("/repo/data/expData/aime_2024_multichoice.json") as f:
    all_questions = json.load(f)

questions = all_questions[:5]
print("Evaluating", len(questions), "questions (no cue)")

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto",
)
model.eval()

# Fixed instruction - same as what the dataset questions already include
inst = "You are a reasoning assistant for multiple choice questions. Put your final answer in \\boxed{X} where X is the letter of the correct choice."

settings = {
    "do_sample": True, "temperature": 0.5, "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 4096,
    "eos_token_id": tokenizer.eos_token_id,
}

correct = 0
for i, q in enumerate(questions):
    input_text = "<|im_start|>system\n" + inst + "<|im_end|>\n"
    input_text += "<|im_start|>user\n" + q["original_question"] + "<|im_end|>\n"
    input_text += "<|im_start|>assistant\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **settings)[0]
    
    response = tokenizer.decode(output_ids)
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    model_ans = match.group(1).strip() if match else ""
    
    is_correct = (model_ans == q["correct_answer"])
    if is_correct:
        correct += 1
    
    print("[{}/{}] {}: correct={}, model={} {}".format(
        i+1, len(questions), q["question_id"][-25:], q["correct_answer"], 
        model_ans, "CORRECT" if is_correct else "WRONG"))

acc = correct / len(questions) if questions else 0
print("\nBaseline accuracy: {:.4f} ({}/{})".format(acc, correct, len(questions)))
