import os, json
os.environ["CUDA_HOME"] = "/opt/conda/lib/python3.10/site-packages/nvidia/cu13"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
from eval_bfcl import build_prompt, build_tool_text, HEADS, STOPS, clean_output, load_possible_answers

llm = LLM(
    model="/models/RT-Qwen2.5-0.5B",
    trust_remote_code=True, dtype="auto",
    gpu_memory_utilization=0.80, max_model_len=4096,
    enable_prefix_caching=True, max_num_seqs=8,
)
sp = SamplingParams(temperature=0.0, max_tokens=128, stop=STOPS, include_stop_str_in_output=True)

data_dir = "/repo/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"

# Load first 3 samples
samples = []
with open(f"{data_dir}/BFCL_v4_multiple.json") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        samples.append(json.loads(line.strip()))

answers = load_possible_answers(data_dir, ["BFCL_v4_multiple"])

for s_idx, sample in enumerate(samples):
    base = build_prompt(sample)

    # Print the prompt
    print(f"=== Sample {s_idx}: {sample['id']} ===")
    print(f"Prompt (first 500 chars):")
    print(base[:500])
    print("...")
    print()

    # Generate
    prompts = [base + op for _, op, _ in HEADS]
    outputs = llm.generate(prompts, sp)

    raw = {}
    for j, (name, _, _) in enumerate(HEADS):
        if j < len(outputs) and outputs[j].outputs:
            raw[name] = clean_output(outputs[j].outputs[0].text)
        else:
            raw[name] = None

    gt = answers.get(sample["id"], [])
    expected_funcs = []
    for g in gt:
        expected_funcs.extend(g.keys())

    print(f"Expected funcs: {expected_funcs}")
    print(f"Generated function: '{raw.get('function', 'N/A')}'")
    for i in range(1, 7):
        v = raw.get(f"arg{i}")
        if v is not None and v != "" and v != "<|null|>":
            print(f"  arg{i}: '{v}'")
    print()
