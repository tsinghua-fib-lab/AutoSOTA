import torch
from llamacu.llama import LLM
from llamacu.speculative import LLM_with_medusa, LLM_with_eagle
from transformers import AutoTokenizer, AutoConfig

path = "/yourpath/Llama-3.2-1B-instruct"
eagle_path = "/yourpath/LLaMA3.2-Instruct-1B-FR-Spec"
dtype = torch.float16
cuda_graph = True
num_generate = 100
model_type = ["base", "eagle"][1]
Bench = True
V = 16384 # or -1 to use normal EAGLE without FR-Spec

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
config = AutoConfig.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()


position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if model_type == "eagle":
    llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.9)
else:
    llm = LLM(path, dtype=dtype, memory_limit=0.9)

our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
if model_type == "eagle":
    if V != -1:
        with open(f'{eagle_path}/freq_{V}.pt', 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
    else:
        token_id_remap = torch.arange(config.vocab_size, dtype=torch.int32, device="cpu")
    llm._load("token_id_remap", token_id_remap, cls="eagle")
llm.load_from_hf()

if model_type == "eagle":
    res = our_generate()
    print("average accept length:", sum(res[1]) / len(res[1]))
    print(tokenizer.decode(res[0]))
else:
    print(tokenizer.decode(our_generate()))