import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import transformers
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import shared_cache
from hogwild.attention import HogwildCache, model_surgery, merge_caches
from hogwild.formatting import FormattingBase, MathFormatting, get_default_options_for_model


device = "cuda"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


def fix_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def reference(problem: str,
              model: transformers.PreTrainedModel,
              tokenizer: transformers.PreTrainedTokenizer,
              fmt: FormattingBase,):
    tokenizer_kwargs = dict(return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)

    caches = tuple(shared_cache.CacheBlock(config=model.config) for _ in range(5))
    cache_common, cache_current_step_header, cache_own_header, cache_w1, cache_w2 = caches
    cm = shared_cache.SharedCacheManager(cache_structure=[
        [cache_common, cache_current_step_header, cache_w2, cache_own_header, cache_w1],
        [cache_common, cache_current_step_header, cache_w1, cache_own_header, cache_w2],
    ])

    with torch.inference_mode():
        # pre-fill common cache parts
        model(**tokenizer(fmt.apply_chat_template(problem), **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_common)  # <-- write to common prompt

        model(**tokenizer(fmt.current_step_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_current_step_header)  # <-- write to the separator after history

        model(**tokenizer(fmt.current_worker_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_own_header)  # <-- write to separator between incomplete steps

        # first step - worker headers
        next_inputs = tokenizer(list(fmt.worker_prompts), **tokenizer_kwargs).to(device)
        model(**cm.get_input_kwargs(**next_inputs))

        model(**cm.get_input_kwargs(**tokenizer(["Alice", "Bob"], **tokenizer_kwargs).to(device)))
        cache_common.append_from(cache_w1)
        cache_w1.clear()

        model(**cm.get_input_kwargs(**tokenizer(["Here", "Let"], **tokenizer_kwargs).to(device)))

        model(**cm.get_input_kwargs(**tokenizer([": Here's", ":"], **tokenizer_kwargs).to(device)))

    return caches


def new_code(problem: str,
             model: transformers.PreTrainedModel,
             tokenizer: transformers.PreTrainedTokenizer,
             fmt: FormattingBase,):
    model_surgery(model)
    #model = torch.compile(model)
    tokenizer_kwargs = dict(return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)

    tokens_since_last_wait = 0
    caches = tuple(transformers.DynamicCache() for _ in range(5))
    cache_common, cache_current_step_header, cache_own_header, cache_w1, cache_w2 = caches
    cm = HogwildCache(cache_structure=[
        [cache_common, cache_current_step_header, cache_w2, cache_own_header, cache_w1],
        [cache_common, cache_current_step_header, cache_w1, cache_own_header, cache_w2],
    ], write_to=[cache_w1, cache_w2], model=model)

    # pre-fill common cache parts
    with torch.inference_mode():
        model(**tokenizer(fmt.apply_chat_template(problem), **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_common]], model=model))  # <-- write to common prompt

        model(**tokenizer(fmt.current_step_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_current_step_header]], model=model))  # <-- write to the separator after history

        model(**tokenizer(fmt.current_worker_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_own_header]], model=model))  # <-- write to separator between incomplete steps

        next_inputs = tokenizer(list(fmt.worker_prompts), **tokenizer_kwargs).to(device)
        model(**cm.get_input_kwargs(**next_inputs))

        model(**cm.get_input_kwargs(**tokenizer(["Alice", "Bob"], **tokenizer_kwargs).to(device)))

        merge_caches(cache_common, cache_w1, model.model)
        cache_w1.crop(0)
        model(**cm.get_input_kwargs(**tokenizer(["Here", "Let"], **tokenizer_kwargs).to(device)))

        model(**cm.get_input_kwargs(**tokenizer([": Here's", ":"], **tokenizer_kwargs).to(device)))
    return caches


def test_match_fp32():
    model_name = "Qwen/QwQ-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', #device_map=device,
                                                 low_cpu_mem_usage=True, trust_remote_code=True)
    model.model.config._attn_implementation = "eager"
    # select only a small subset of the model to prevent OOM (we're in fp32...)
    model.model.layers = model.model.layers[0:5]
    model.to(torch.float32)
    model.to("cuda")
    model.train(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    opts = get_default_options_for_model(model)
    fmt = MathFormatting(tokenizer, extract_result=lambda box: int("".join(x for x in box if x.isdigit())), **opts)
    dataset = load_dataset("GAIR/LIMO", split="train")

    fix_seed(42)
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    problem = system_prompt + str(dataset[1]['question'])

    caches_ref = reference(problem, model, tokenizer, fmt)

    fix_seed(42)
    caches_new = new_code(problem, model, tokenizer, fmt)

    for idx, (cn, cr) in enumerate(zip(caches_new, caches_ref)):
        print(f"Cache {idx} - {cr.get_seq_length()}")
        for layer in range(len(cn.value_cache)):
            assert cn.value_cache[layer].shape == cr.value_cache[layer].shape, (cn.value_cache[layer].shape, cr.value_cache[layer].shape)
            df = []
            for t in range(cn.value_cache[layer].shape[2]):
                df.append(torch.max(torch.abs(cn.value_cache[layer][:, :, t, :] - cr.value_cache[layer][:, :, t, :])).item())
            if cn.value_cache[layer].shape[2] < 10:
                print(f"L: {layer}, max: {df}")
            else:
                print(f"L: {layer}, max: {max(df)}")


if __name__ == "__main__":
    test_match_fp32()

