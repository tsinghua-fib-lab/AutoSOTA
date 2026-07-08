# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import torch
from transformers import AutoModel, AutoTokenizer
import time
from model.modeling_dream import DreamModel

# Load model and tokenizer
device = "cuda"
model_path = "/path/to/your/Dream-v0-Instruct-7B"
model = DreamModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()

# Initialize conversation history
messages = []

# Get user input
# user_input = "Brandon's iPhone is four times as old as Ben's iPhone. Ben's iPhone is two times older than Suzy's iPhone. If Suzy’s iPhone is 1 year old, how old is Brandon’s iPhone?"
user_input = "Ellie has found an old bicycle in a field and thinks it just needs some oil to work well again. She needs 10ml of oil to fix each wheel and will need another 5ml of oil to fix the rest of the bike. How much oil does she need in total to fix the bike?"

# Add user message to conversation history
messages.append({"role": "user", "content": user_input})

# Format input with chat template
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device=device)
attention_mask = inputs.attention_mask.to(device=device)

# Generate response
start = time.time()
output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    steps=256,
    temperature=0.,
    top_p=None,
    alg="entropy",
    alg_temp=0.1,
    top_k=None,
    block_length=32,
    method="dc_leap", 
    commit_thres=0.70,
    draft_thres=0.98,
    max_window_size=128,
)
print(f"Time spent: {time.time() - start}")

# Process response
generation = tokenizer.decode(output.reshape(-1)[len(input_ids[0]):].tolist())
generation = generation.split(tokenizer.eos_token)[0].strip()

# Print response
print("Model:", generation)