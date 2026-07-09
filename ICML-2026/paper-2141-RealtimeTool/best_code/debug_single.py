import os, json
os.environ["CUDA_HOME"] = "/opt/conda/lib/python3.10/site-packages/nvidia/cu13"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

HEADS = [("function", "<function>", "</function>")] + [
    (f"arg{i}", f"<arg{i}>", f"</arg{i}>") for i in range(1, 7)
]
STOPS = ["</function>"] + [f"</arg{i}>" for i in range(1, 7)] + ["</content>", "<|null|>", "<|im_end|>"]

llm = LLM(
    model="/models/RT-Qwen2.5-0.5B",
    trust_remote_code=True, dtype="auto",
    gpu_memory_utilization=0.80, max_model_len=4096,
    enable_prefix_caching=True, max_num_seqs=8,
)
sp = SamplingParams(temperature=0.0, max_tokens=128, stop=STOPS, include_stop_str_in_output=True)

# Use OpenAI function-calling format (matching training data)
tools = json.dumps({
    "type": "function",
    "function": {
        "name": "country_info.capital",
        "description": "Fetch the capital city of a specified country.",
        "parameters": {
            "type": "object",
            "properties": {"country": {"type": "string", "description": "Name of the country."}},
            "required": ["country"]
        }
    }
})

base = (
    "<|im_start|>system\nYou are an AI assistant that helps users find and call the right functions.\n\n"
    "## Available Tools:\n\n"
    f"{tools}\n"
    "<|im_end|>\n"
    "<|im_start|>user\nWhat is the capital of Brazil?\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("=== Base prompt ===")
print(base[-300:])
print()

# Test each head individually
for name, open_tag, close_tag in HEADS:
    prompt = base + open_tag
    outputs = llm.generate([prompt], sp)
    if outputs and outputs[0].outputs:
        o = outputs[0].outputs[0]
        tok = llm.get_tokenizer()
        tokens_decoded = [tok.decode([tid]) for tid in o.token_ids]
        print(f"{name}: {repr(o.text.strip())}  tokens={tokens_decoded}")
    print()
