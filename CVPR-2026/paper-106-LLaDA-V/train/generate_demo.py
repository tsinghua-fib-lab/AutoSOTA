from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

from PIL import Image
import requests
import copy
import torch
import time

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
pretrained = "GSAI-ML/LLaDA-V"

model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()
image = Image.open("test.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 
question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(model)
    print("Testing with Fast dLLM hook enabled")
elif use_dllm_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

start_time = time.time()
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    threshold=1,
)
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)
