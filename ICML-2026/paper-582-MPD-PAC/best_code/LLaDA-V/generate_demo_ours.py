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
import random
import numpy as np

import sys
import warnings

# Seed 고정
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
pretrained = "GSAI-ML/LLaDA-V"

model_name = "llava_llada_ours"
# device_map = "auto"
device_map= "cuda:0"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="eager", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()
image = Image.open("dog_and_cat.jpg")
image_tensor = process_images([image], image_processor, model.config)
device = model.parameters().__next__().device if next(model.parameters(), None) is not None else "cpu"
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 
question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

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

steps=16
gen_length=64
block_length=64
img_start = torch.nonzero(input_ids[0] == -200, as_tuple=False).squeeze(-1).tolist()
prompt_length = input_ids.shape[1] - img_start[0]

# LLaDA-V
lladav = model.generate(
    input_ids,
    images= image_tensor,
    image_sizes=image_sizes,
    steps=steps, gen_length=gen_length, block_length=block_length, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    img_start=img_start,
    prompt_length=prompt_length,
    temperature=0.0,
    prior = 0.0,
    rope = 0.0,
    mode="sigmoid",
    slope=0.0,
    center=0.0
)

lladav_outputs = tokenizer.batch_decode(lladav, skip_special_tokens=False)
print('lladav outputs:', lladav_outputs)

# LLaDA-V + Ours
ours = model.generate(
    input_ids,
    images= image_tensor,
    image_sizes=image_sizes,
    steps=steps, gen_length=gen_length, block_length=block_length, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    img_start=img_start,
    prompt_length=prompt_length,
    temperature=0.0,
    prior = 0.1,
    rope = 0.1,
    mode="sigmoid",
    k=3,
    slope=8.0,
    center=0.6
)

ours_outputs = tokenizer.batch_decode(ours, skip_special_tokens=False)
print('ours:', ours_outputs)

