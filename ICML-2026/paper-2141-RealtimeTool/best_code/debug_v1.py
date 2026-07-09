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

# v1 system prompt (from prompts/v1_system.txt)
V1_SYSTEM = """You are a multi-head parallel function calling model.
## Output Heads
**Head 0 - <content>**: Natural language response
- Format: <content>response text</content>
- Answer what you want to say while you are calling a function
**Head 1 - <function>**: Function names to call
- Format: <function>name</function>
- Name: must match tool defined name
**Head 2-7 - <arg1>、<arg2>、<arg3>、<arg4>、<arg5>、<arg6>**: Function arguments by position
- Format: <argN>value</argN>
- Strictly fill in according to the parameter order of the tool you intend to call
- Note the special restrictions of parameter definitions for corresponding positions
- If the corresponding tool definition has required parameters, these must be filled in
- Infer the user's actual needs.
- If Unnecessary: <argN><|null|></argN>
**Environment - The information you have.
**History - The tools you have called."""

# Use OpenAI function-calling format
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

# v1 format
base = (
    f"<|im_start|>system\n{V1_SYSTEM}\n## Available Tools:\n\n{tools}<|im_end|>\n"
    "<|im_start|>user\nenvironment: []\nhistory: []\n\nWhat is the capital of Brazil?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("=== V1 Base prompt (end) ===")
print(base[-300:])
print()

for name, open_tag, close_tag in HEADS[:4]:  # Test first 4 heads
    prompt = base + open_tag
    outputs = llm.generate([prompt], sp)
    if outputs and outputs[0].outputs:
        o = outputs[0].outputs[0]
        tok = llm.get_tokenizer()
        tokens_decoded = [tok.decode([tid]) for tid in o.token_ids]
        print(f"{name}: {repr(o.text.strip())}")
        print(f"  tokens: {tokens_decoded}")
    print()

# Also test v2 format for comparison
base_v2 = (
    "<|im_start|>system\nYou are an AI assistant that helps users find and call the right functions.\n\n"
    "## Available Tools:\n\n"
    f"{tools}\n"
    "<|im_end|>\n"
    "<|im_start|>user\nWhat is the capital of Brazil?\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("=== V2 Base prompt (end) ===")
print(base_v2[-300:])
print()

for name, open_tag, close_tag in HEADS[:4]:
    prompt = base_v2 + open_tag
    outputs = llm.generate([prompt], sp)
    if outputs and outputs[0].outputs:
        o = outputs[0].outputs[0]
        tok = llm.get_tokenizer()
        tokens_decoded = [tok.decode([tid]) for tid in o.token_ids]
        print(f"{name}: {repr(o.text.strip())}")
        print(f"  tokens: {tokens_decoded}")
    print()
