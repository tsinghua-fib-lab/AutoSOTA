"""Simple inference demo with visual and text patches."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

from models.qwen import VisualPatch, TextPatch

# Config
CACHE_DIR = "${HF_HOME:-$HOME/.cache/huggingface}/"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir=CACHE_DIR,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Prepare input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Use both patches with context manager
with VisualPatch(model.model.visual) as visual_patch, \
     TextPatch(model.model.language_model, capture_hidden_states=False) as text_patch:

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Print captured outputs
    print("=" * 50)
    print("Visual Patch Outputs:")
    print(f"  - rotary_emb shape: {visual_patch.rotary_emb.shape}")
    print(f"  - pos_embeds shape: {visual_patch.pos_embeds.shape}")
    print(f"  - hidden_states layers: {len(visual_patch.hidden_states_per_layer)}")

    print("\nText Patch Outputs:")
    print(f"  - position_ids shape: {text_patch.outputs.position_ids.shape}")
    print(f"  - position_embeddings cos shape: {text_patch.outputs.position_embeddings[0].shape}")
    print(f"  - position_embeddings sin shape: {text_patch.outputs.position_embeddings[1].shape}")
    print(f"  - input_embeds shape: {text_patch.outputs.input_embeds.shape}")

    print("\nGenerated text:")
    print(output_text[0])
