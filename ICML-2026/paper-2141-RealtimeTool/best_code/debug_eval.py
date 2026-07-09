import os, json
os.environ["CUDA_HOME"] = "/opt/conda/lib/python3.10/site-packages/nvidia/cu13"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
from eval_bfcl import (
    build_prompt_v2, HEADS, STOPS, clean_output, parse_multi_head_output,
    load_bfcl_data, load_possible_answers, check_function_accuracy, check_overall_accuracy
)

llm = LLM(
    model="/models/RT-Qwen2.5-0.5B",
    trust_remote_code=True, dtype="auto",
    gpu_memory_utilization=0.80, max_model_len=4096,
    enable_prefix_caching=True, max_num_seqs=8,
)
sp = SamplingParams(temperature=0.0, max_tokens=128, stop=STOPS, include_stop_str_in_output=True)

data_dir = "/repo/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
subsets = ["BFCL_v4_multiple", "BFCL_v4_simple_python"]
samples = load_bfcl_data(data_dir, subsets)
answers = load_possible_answers(data_dir, subsets)

# Debug first 10 samples
for s_idx in range(min(10, len(samples))):
    sample = samples[s_idx]
    base = build_prompt_v2(sample)

    prompts = [base + op for _, op, _ in HEADS]
    outputs = llm.generate(prompts, sp)

    raw = {}
    full = {}
    for j, (name, _, _) in enumerate(HEADS):
        if j < len(outputs) and outputs[j].outputs:
            o = outputs[j].outputs[0]
            raw[name] = clean_output(o.text)
            full[name] = o.text
        else:
            raw[name] = None
            full[name] = ""

    func_name, args = parse_multi_head_output(raw, full)
    gt = answers.get(sample["id"], [])

    # Get expected
    expected_funcs = []
    for g in gt:
        expected_funcs.extend(g.keys())

    print(f"=== Sample {s_idx}: {sample['id']} ({sample.get('_subset', '?')}) ===")
    if sample.get("question") and sample["question"][0]:
        last_msg = sample["question"][0][-1]
        content = last_msg.get("content", "?")
        print(f"Query: {content[:150]}")
    print(f"Expected funcs: {expected_funcs}")
    if gt:
        print(f"Expected GT: {json.dumps(gt[0])[:300]}")
    print(f"Function head raw: '{raw.get('function', 'N/A')}'")
    print(f"Function head full: '{full.get('function', 'N/A')}'")
    for i in range(1, 7):
        v = raw.get(f"arg{i}")
        fv = full.get(f"arg{i}", "")
        if v is not None and v != "<|null|>" and v != "":
            print(f"  arg{i}: raw='{v}'  full='{fv}'")
        elif v == "<|null|>" or v == "":
            print(f"  arg{i}: NULL")
    print(f"Parsed func_name: '{func_name}'")
    print(f"Parsed args: {args}")
    print(f"Func accuracy: {check_function_accuracy(func_name, gt)}")
    print(f"Overall accuracy: {check_overall_accuracy(func_name, args, gt)}")
    print()
