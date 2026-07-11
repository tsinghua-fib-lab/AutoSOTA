import os
from dyllm.dllm import dLLM
from dyllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

import time


def main():
    path = os.path.expanduser("/data/model/Dream-v0-Instruct-7B")
    # path = os.path.expanduser("/data/model/LLaDA-8B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dllm = dLLM(path, threshold=0.995, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(
        temperature=None,
        max_new_tokens=256,
        steps=256,
        num_full_steps=4,
        block_size=32,
        ignore_eos=True,
        algorithm="confidence",
    )

    prompts = [
        "Describe the water cycle in detail.",
    ]
    
    prompts=prompts * 4

    templated = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    for _ in range(2):
        outputs = dllm.generate(templated, sampling_params)

    for p, out in zip(prompts, outputs):
        print("\nPrompt:", repr(p))
        print("Completion:", repr(out["text"]))


if __name__ == "__main__":
    main()

