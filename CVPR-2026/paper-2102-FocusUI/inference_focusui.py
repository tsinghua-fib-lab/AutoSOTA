from PIL import Image
import torch
from transformers import AutoProcessor

from focusui.modeling_focusui_qwen25vl import FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer
from focusui.modeling_focusui_qwen3vl import FocusUI_Qwen3VLForConditionalGenerationWithPointer
from focusui.inference import inference_focusui_token_select
from focusui.constants import grounding_system_message_guiactor_qwen25vl

from evaluation.shared_grounding_eval import save_patch_saliency_heatmap_overlay, draw_point


# Load model and processor
model_path = "./checkpoints/FocusUI-3B"   # if not downloaded, use url: "yyyang/FocusUI-Qwen3VL-3B"
model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",  # "flash_attention_2" if available
).eval()

# model_path = "./checkpoints/FocusUI-Qwen-3VL-2B"  # if not downloaded, use url: "yyyang/FocusUI-Qwen-3VL-2B"
# model = FocusUI_Qwen3VLForConditionalGenerationWithPointer.from_pretrained(
#     model_path,  
#     dtype=torch.bfloat16,
#     device_map="cuda",
#     attn_implementation="sdpa",  # "flash_attention_2" if available
# ).eval()
processor = AutoProcessor.from_pretrained(model_path)

# Prepare conversation
image_path = "assets/example_screenshot.png"
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": grounding_system_message_guiactor_qwen25vl}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Go to 'Watch Live'."}
        ]
    }
]

# Configure visual token selection
model.apply_visual_token_select = True
model.visual_reduct_ratio = 0.5  # Keep 50% of visual tokens

# Run inference
result = inference_focusui_token_select(
    conversation=conversation,
    model=model,
    tokenizer=processor.tokenizer,
    data_processor=processor,
    topk=3,
)

# Get predicted coordinates
topk_points = result['topk_points']
top1_point = topk_points[0]
print(f"Top-1 point: {top1_point}")

# Save patch score prediction overlay
heatmap_saved = save_patch_saliency_heatmap_overlay(
    screenshot=image_path,
    pred=result,
    save_name=f"patch_saliency_score_prediction.png",
    image_patch_size=16,
)
print(f"Heatmap saved to: {heatmap_saved}")

# Draw point on the image
image = draw_point(Image.open(image_path), top1_point)
image.save(f"./point_on_image.png")
print(f"Grounding result saved to: ./point_on_image.png")