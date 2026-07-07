import os
import sys
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llada_ours.generate import generate as llada_generate
from llava.model.language_model.llada_ours.log_likelyhood import get_logits as llada_get_logits
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import json
import time

pretrained="jacklishufan/lavida-llada-v1.0-instruct"
model_name = "llada_ours"
device = "cuda"
device_map = "cuda:0"

conv_template = "llada" 
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print(prompt_question)
vision_kwargs = None
vision_kwargs = dict(
    mm_vision_tower="google/siglip-so400m-patch14-384",
    mm_resampler_type=None,
    mm_projector_type='mlp2x_gelu',
    mm_hidden_size=1152,
    use_mm_proj=True
)
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,vision_kwargs=vision_kwargs,torch_dtype='bfloat16') # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()
model.to(torch.bfloat16)


image = Image.open('dog_and_cat.jpg').convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

# LaViDa
cont,hist = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0.1,
    max_new_tokens=64,
    block_length=64,
    step_per_block=16,
    tokenizer=tokenizer,
    prefix_lm=False,
    verbose=True,
    schedule='shift',
    prior=0.0,
    rope=0.0,
    mode="sigmoid",
    slope=0.0,
    center=0.0
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
print('lavida outputs', text_outputs)

# LaViDa + Ours
cont,hist = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0.1,
    max_new_tokens=64,
    block_length=64,
    step_per_block=16,
    tokenizer=tokenizer,
    prefix_lm=False,
    verbose=True,
    schedule='shift',
    prior=0.1,
    rope=0.1,
    mode="sigmoid",
    slope=12.0,
    center=0.6
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
print('ours', text_outputs)