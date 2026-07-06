import json, os, sys, re, random
from pathlib import Path
REPO_ROOT = Path("/repo")
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer

def extract_boxed(text):
    matches = list(re.finditer(r"\\\\boxed\\{", text))
    if not matches: return None
    start = matches[-1].end(); depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start:i-1] if depth == 0 else None

def answers_match(pred, gold):
    pn = pred.strip().replace("$","").strip(".,;:!?")
    gn = gold.strip().replace("$","").strip(".,;:!?")
    if pn == gn: return True
    try: return abs(float(pn.replace(",","")) - float(gn.replace(",",""))) < 1e-6
    except: return False

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/models/Qwen3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16,
    attn_implementation="eager", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-8B-Instruct", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f"Model on {model.device}")

with open("/repo/data/math_problems.json") as f:
    problems = json.load(f)
random.seed(42)
random.shuffle(problems)

SP = "You are a math assistant. Solve step by step concisely. Put final answer in \\\\boxed{}."
kept = []
pbar = tqdm(total=100, desc="Correct")
for prob in problems:
    if len(kept) >= 100: break
    msgs = [{"role":"system","content":SP},{"role":"user","content":prob["problem"]}]
    f = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(f, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(o[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pred = extract_boxed(gen)
    if not pred: continue
    if not answers_match(pred, prob["answer"]): continue
    ex = CachedExample(prompt=prob["problem"], target=gen, indices_to_explain=None,
        attr_mask_indices=None, sink_span=None, thinking_span=None,
        metadata={"reference_answer":prob["answer"],"boxed_answer":pred})
    ex = attach_spans_from_answer(ex, tokenizer, pred)
    if not (isinstance(ex.sink_span, list) and len(ex.sink_span) == 2): continue
    kept.append({"prompt":prob["problem"],"target":gen,"indices_to_explain":list(ex.sink_span),
        "attr_mask_indices":None,"sink_span":list(ex.sink_span),"thinking_span":ex.thinking_span,
        "metadata":{"dataset":"math","reference_answer":prob["answer"],"boxed_answer":pred}})
    pbar.update(1)

pbar.close()
os.makedirs("/repo/exp/exp2/data", exist_ok=True)
with open("/repo/exp/exp2/data/math.jsonl","w") as f:
    for e in kept: f.write(json.dumps(e, ensure_ascii=False)+"\n")
print(f"Done: {len(kept)} samples")
