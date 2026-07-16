"""
VLM Task: Scoring using image neighbor information
- Based on neighbors found via image embeddings
- Provides two images: reference image + image to evaluate
- Uses VLM model to view both images and corresponding captions for evaluation
"""
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
from PIL import Image

# Import project configuration and models
from config import MODEL_CONFIG
from models.vlm_models import Qwen3VLCaptioner, BLIP2Captioner, Llama32VisionCaptioner, OpenAICaptioner, InternVLCaptioner, SAILVLCaptioner, Ministral3VLCaptioner


# Model class mapping (VLM models)
VLM_MODEL_CLASS_MAP = {
    "qwen3_vl": Qwen3VLCaptioner,
    "blip": BLIP2Captioner,
    "blip2": BLIP2Captioner,
    "llama-vision": Llama32VisionCaptioner,
    "openai": OpenAICaptioner,
    "internvl": InternVLCaptioner,
    "sailvl": SAILVLCaptioner,
    "ministral3_vl": Ministral3VLCaptioner,
}


# VLM Caption Scoring Prompt (two-image version)
# Image 1 = Reference image (neighbor's)
# Image 2 = Image to evaluate (current sample's)
# CAPTION_EVALUATION_PROMPT = '''You are given two images and their captions. Please evaluate whether the second caption accurately describes the second image.

# [Image 1 - Reference Image]
# Caption: {ref_caption}

# [Image 2 - Image to Evaluate]
# Predicted Caption: {predicted_caption}

# Scoring Rules:
# - Score 5: Perfect - the predicted caption accurately and completely describes Image 2
# - Score 4: Excellent - captures all key elements with minor differences
# - Score 3: Good (passing) - captures main elements but misses some details
# - Score 2: Fair - captures some elements but has significant gaps
# - Score 1: Poor - barely related to Image 2
# - Score 0: Completely wrong - does not describe Image 2 at all

# Important:
# - Image 1 and its caption are provided as CONTEXT (they show a visually similar scene)
# - Focus ONLY on whether the predicted caption accurately describes Image 2
# - A score of 3 or above means the caption is acceptable

# Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

# Score:'''
# CAPTION_EVALUATION_PROMPT = '''You are given two images and their captions. Please evaluate whether the second caption accurately describes the second image.

# [Image 1 - Reference Image]
# Caption (Score 3 - passing but imperfect): {ref_caption}

# [Image 2 - Image to Evaluate]
# Predicted Caption: {predicted_caption}

# Scoring Rules:
# - Score 5: Perfect - the predicted caption accurately and completely describes Image 2
# - Score 4: Excellent - captures all key elements with minor differences
# - Score 3: Good (passing) - captures main elements but misses some details
# - Score 2: Fair - captures some elements but has significant gaps
# - Score 1: Poor - barely related to Image 2
# - Score 0: Completely wrong - does not describe Image 2 at all

# Important:
# - Image 1 and its caption are provided as CONTEXT only (they show a visually similar scene)
# - The caption for Image 1 is a "Score 3" level description - acceptable but may have errors or miss details
# - Do NOT use Image 1's caption as ground truth
# - Evaluate Image 2's caption based on what YOU see in Image 2, not based on Image 1's caption
# - A score of 3 or above means the caption is acceptable

# Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

# Score:'''
CAPTION_EVALUATION_PROMPT = '''You are given two images and their captions. Evaluate whether the second caption reaches the quality standard shown by the first caption.

[Image 1 - Reference Image (Excellent Example)]
High-Quality Caption (Score 5 standard): {ref_caption}

[Image 2 - Image to Evaluate]
Predicted Caption: {predicted_caption}

Scoring Rules (compare against the Score 5 reference):
- Score 5: Matches the reference quality - detailed, accurate, and comprehensive
- Score 4: Nearly as good - captures most elements with similar detail level
- Score 3: Acceptable - captures main elements but noticeably less detailed than reference
- Score 2: Below standard - missing significant details or less accurate than reference
- Score 1: Poor - much less detailed or accurate than reference
- Score 0: Completely inadequate - does not properly describe Image 2

Evaluation Approach:
1. Look at Image 1 and its caption - this represents excellent captioning quality
2. Look at Image 2 and its predicted caption
3. Compare: Does the predicted caption describe Image 2 with similar quality/detail as the reference describes Image 1?
4. If Image 2's caption has similar comprehensiveness and accuracy → high score
5. If Image 2's caption is less detailed or less accurate → lower score

Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

Score:'''

# Single image scoring Prompt (no-reference mode)
SINGLE_IMAGE_EVALUATION_PROMPT = '''You are evaluating the quality of an image caption. Look at the image and assess how well the caption describes it.

[Image to Evaluate]
Predicted Caption: {predicted_caption}

Scoring Rules:
- Score 5: Excellent - detailed, accurate, and comprehensive description of the image
- Score 4: Very Good - captures most important elements with good detail
- Score 3: Acceptable - captures main subject but lacks some details or has minor inaccuracies
- Score 2: Below Average - missing significant details or contains noticeable errors
- Score 1: Poor - only partially describes the image or has major inaccuracies
- Score 0: Completely Inadequate - does not properly describe the image at all

Evaluation Criteria:
1. Accuracy: Does the caption correctly describe what's in the image?
2. Completeness: Does it mention the main subject and key elements?
3. Clarity: Is the description clear and understandable?
4. Relevance: Does it focus on the important aspects of the image?

Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

Score:'''

# CAPTION_EVALUATION_PROMPT = '''You are evaluating caption quality by comparing against a reference example.

# [Image 1 - Reference Example]
# Caption: {ref_caption}
# (This caption represents high-quality image description)

# [Image 2 - To Evaluate]  
# Predicted Caption: {predicted_caption}

# Your Task:
# Compare the quality and detail level of both image-caption pairs. Does the predicted caption describe Image 2 as well as the reference caption describes Image 1?

# Scoring Guidelines:
# - Score 5: Equal quality - predicted caption describes Image 2 with the same level of detail, accuracy, and comprehensiveness as the reference describes Image 1
# - Score 4: Nearly equal - slightly less detailed but still captures most key elements
# - Score 3: Acceptable - captures main elements but noticeably less comprehensive than the reference standard
# - Score 2: Below standard - missing significant details compared to the reference quality level
# - Score 1: Poor - much less detailed or accurate than the reference standard
# - Score 0: Inadequate - fails to properly describe Image 2

# Key Points:
# - The reference shows what good captioning looks like
# - Judge whether Image 2's caption reaches that same quality bar
# - Consider: detail level, accuracy, comprehensiveness, specificity
# - Both captions should be evaluated relative to their respective images

# Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

# Score:'''

def parse_args():
    parser = argparse.ArgumentParser(description='VLM Task: Scoring using two images + neighbor information')
    
    # Scoring mode
    parser.add_argument('--mode', type=str, default='neighbor', choices=['neighbor', 'single'],
                       help='Scoring mode: neighbor=neighbor reference mode (default), single=no-reference single image mode')
    
    parser.add_argument('--neighbors', type=str, default=None,
                       help='Image neighbor JSONL file path (required for neighbor mode)')
    parser.add_argument('--vlm-data', type=str, required=True,
                       help='VLM results JSON file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output results path (JSON file)')
    parser.add_argument('--model', type=str, default='qwen3-vl-8b',
                       help='VLM scoring model name (e.g., qwen3-vl-8b)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['flickr30k', 'coco'],
                       help='Dataset name (flickr30k/coco)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split')
    parser.add_argument('--ref-type', type=str, default='predicted',
                       choices=['predicted', 'true', 'both'],
                       help='Reference caption type (neighbor mode only)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                       help='Batch size (VLM typically uses 1)')
    parser.add_argument('--save-interval', '-s', type=int, default=50,
                       help='Auto-save interval')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process the first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'neighbor' and not args.neighbors:
        parser.error('--neighbors is required when mode is "neighbor"')
    
    return args


def load_hf_dataset(dataset_name: str, split: str = "test"):
    """Load dataset from HuggingFace"""
    from datasets import load_dataset
    
    if dataset_name.lower() == "flickr30k":
        print(f"Loading Flickr30k from HuggingFace (split={split})...")
        dataset = load_dataset("nlphuji/flickr30k", split=split, trust_remote_code=True)
    elif dataset_name.lower() == "coco":
        print(f"Loading COCO-Caption from HuggingFace (split={split})...")
        # COCO validation split
        if split == "test":
            split = "validation"
        dataset = load_dataset("lmms-lab/COCO-Caption", split=split, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset


def get_image_from_hf(hf_dataset, idx: int) -> Optional[Image.Image]:
    """Get image from HuggingFace dataset"""
    try:
        item = hf_dataset[idx]
        image = item.get('image')
        return image
    except Exception as e:
        print(f"⚠️  Failed to get image at index {idx}: {e}")
        return None


def load_vlm_model(model_name: str):
    """Load VLM model, returns (model, model_type)"""
    if model_name not in MODEL_CONFIG["vlm_tagging"]:
        raise ValueError(f"Model '{model_name}' not found in vlm_tagging configuration")

    config = MODEL_CONFIG["vlm_tagging"][model_name]
    model_type = config["model_type"]

    if model_type not in VLM_MODEL_CLASS_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = VLM_MODEL_CLASS_MAP[model_type]
    
    # Build configuration
    model_config = {
        "model_name": config["model_name"],
        "model_type": model_type,
        "device": config.get("device", "cuda"),
        "max_new_tokens": 5,  # Only need 1 digit
    }
    
    model = model_class(model_config)
    model.load_model()
    
    print(f"✅ Loaded VLM model: {model_name}")
    print(f"   - Type: {model_type}")
    print(f"   - Path: {config['model_name']}")
    
    return model, model_type


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    """Load image neighbor JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_vlm_data(file_path: str) -> List[Dict]:
    """Load VLM results JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_global_id_map(vlm_data: List[Dict]) -> Dict[int, Dict]:
    """Build global_id to VLM data mapping"""
    return {i: item for i, item in enumerate(vlm_data)}


def get_ref_caption(vlm_item: Dict, ref_type: str) -> str:
    """Get reference caption (single entry)"""
    if ref_type == 'predicted':
        return vlm_item.get('predicted_caption', '')
    elif ref_type == 'true':
        true_caps = vlm_item.get('true_captions', [])
        return true_caps[0] if true_caps else ''
    elif ref_type == 'both':
        pred = vlm_item.get('predicted_caption', '')
        true_caps = vlm_item.get('true_captions', [])
        true_cap = true_caps[0] if true_caps else ''
        return f"Predicted: {pred}\nTrue: {true_cap}"
    return ''


def create_evaluation_tasks(neighbors_data: List[Dict], vlm_map: Dict[int, Dict], 
                           ref_type: str = 'predicted', include_self: bool = True) -> List[Dict]:
    """
    Create all scoring tasks
    Each task contains:
    - Reference image index (neighbor_index)
    - Reference caption (ref_caption)
    - Current image index (sample_index)
    - Current predicted_caption (predicted_caption)
    """
    tasks = []
    
    for sample in neighbors_data:
        sample_id = sample['global_id']
        sample_idx = sample.get('index', sample_id)
        sample_vlm = vlm_map[sample_id]
        
        # 1. Use 9 neighbors as reference for scoring
        for rank, neighbor in enumerate(sample['neighbors']):
            neighbor_id = neighbor['global_id']
            neighbor_idx = neighbor.get('index', neighbor_id)
            neighbor_vlm = vlm_map[neighbor_id]
            
            tasks.append({
                'sample_global_id': sample_id,
                'sample_index': sample_idx,
                'neighbor_rank': rank,
                'neighbor_global_id': neighbor_id,
                'neighbor_index': neighbor_idx,  # Neighbor image index
                'neighbor_cosine': neighbor['cosine'],
                'predicted_caption': sample_vlm.get('predicted_caption', ''),
                'true_captions': sample_vlm.get('true_captions', []),
                'ref_caption': get_ref_caption(neighbor_vlm, ref_type),
                'is_self_ref': False,
            })
        
        # 2. Use own predicted_caption as reference for scoring (the 10th)
        if include_self:
            tasks.append({
                'sample_global_id': sample_id,
                'sample_index': sample_idx,
                'neighbor_rank': 9,
                'neighbor_global_id': sample_id,
                'neighbor_index': sample_idx,  # Own image
                'neighbor_cosine': 1.0,
                'predicted_caption': sample_vlm.get('predicted_caption', ''),
                'true_captions': sample_vlm.get('true_captions', []),
                'ref_caption': get_ref_caption(sample_vlm, ref_type),
                'is_self_ref': True,
            })
    
    return tasks


def create_prompt(task: Dict) -> str:
    """Create scoring prompt (text portion)"""
    return CAPTION_EVALUATION_PROMPT.format(
        ref_caption=task['ref_caption'],
        predicted_caption=task['predicted_caption']
    )


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """Parse the returned score"""
    response = response.strip()

    # Method 1: Directly parse the first line
    first_line = response.split('\n')[0].strip()
    try:
        score = int(first_line)
        if 0 <= score <= 5:
            return score, True, str(score)
    except ValueError:
        pass
    
    # Method 2: Regex extraction
    match = re.search(r'\b([0-5])\b', response)
    if match:
        score = int(match.group(1))
        return score, True, str(score)
    
    # Method 3: Find any digit
    match = re.search(r'(\d)', response)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 5:
            return score, True, str(score)
    
    return None, False, "INVALID"


def generate_score_with_two_images(model, ref_image: Image.Image, eval_image: Image.Image, 
                                   prompt: str, model_type: str = None) -> str:
    """
    Generate score using VLM model (two images)
    - ref_image: Reference image (neighbor's)
    - eval_image: Image to evaluate (current sample's)
    - model_type: Model type (used to select the correct generation function)
    """
    try:
        if hasattr(model, 'generate_with_two_images'):
            return model.generate_with_two_images(ref_image, eval_image, prompt)
        
        # Select generation function based on model type
        if model_type == 'sailvl':
            return _generate_sailvl_two_images(model, ref_image, eval_image, prompt)
        elif model_type == 'internvl':
            return _generate_internvl_two_images(model, ref_image, eval_image, prompt)
        elif model_type == 'qwen3_vl':
            return _generate_qwen_two_images(model, ref_image, eval_image, prompt)
        
        # Fallback: Try generic processor + model approach
        if hasattr(model, 'processor') and hasattr(model, 'model'):
            return _generate_qwen_two_images(model, ref_image, eval_image, prompt)
        
        # Fallback: Only use the image to evaluate
        print("⚠️  Model does not support dual-image input, using only the evaluation image")
        if hasattr(model, 'generate_with_image'):
            return model.generate_with_image(eval_image, prompt, max_tokens=5)
        
        return ""
        
    except Exception as e:
        print(f"⚠️  Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def _generate_qwen_two_images(model, ref_image: Image.Image, eval_image: Image.Image, 
                               prompt: str) -> str:
    """Qwen3-VL dual-image generation"""
    from qwen_vl_utils import process_vision_info
    
    # Build message containing two images
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": ref_image},   # Image 1 - Reference image
            {"type": "image", "image": eval_image},  # Image 2 - Image to evaluate
            {"type": "text", "text": prompt}
        ]
    }]

    text = model.processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    return output_text


def _generate_sailvl_two_images(model, ref_image: Image.Image, eval_image: Image.Image, 
                                 prompt: str) -> str:
    """SAIL-VL2 dual-image generation"""
    # Preprocess images
    if ref_image.mode != 'RGB':
        ref_image = ref_image.convert('RGB')
    if eval_image.mode != 'RGB':
        eval_image = eval_image.convert('RGB')
    
    # Build message containing two images
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": ref_image},   # Image 1 - Reference image
            {"type": "image", "image": eval_image},  # Image 2 - Image to evaluate
            {"type": "text", "text": prompt}
        ]
    }]

    # Apply chat template
    text = model.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Process inputs (passing in two images)
    inputs = model.processor(
        images=[ref_image, eval_image],
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.model.device).to(torch.bfloat16)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    
    # Decode output
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Extract generated portion (remove prompt)
    response = response.split('<|im_end|>')[0].strip()
    
    # If the response contains an assistant marker, extract content after it
    if 'assistant' in response.lower():
        parts = response.split('assistant')
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


def _generate_internvl_two_images(model, ref_image: Image.Image, eval_image: Image.Image, 
                                   prompt: str) -> str:
    """InternVL dual-image generation"""
    # Preprocess images
    if ref_image.mode != 'RGB':
        ref_image = ref_image.convert('RGB')
    if eval_image.mode != 'RGB':
        eval_image = eval_image.convert('RGB')
    
    # InternVL uses <image> placeholders, need to mark image positions in the prompt
    # Format: <image>\n<image>\n{prompt}
    full_prompt = f"<image>\n<image>\n{prompt}"
    
    # Use InternVL's chat method (if multi-image is supported)
    if hasattr(model.model, 'chat'):
        # InternVL3.5 supports multi-image input
        pixel_values_list = []
        
        # Use InternVL's image preprocessing
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        input_size = 448
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        for img in [ref_image, eval_image]:
            pixel_values = transform(img).unsqueeze(0).to(
                dtype=torch.bfloat16, device=model.model.device
            )
            pixel_values_list.append(pixel_values)
        
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        generation_config = {
            'max_new_tokens': 5,
            'do_sample': False,
        }
        
        response = model.model.chat(
            model.tokenizer,
            pixel_values,
            full_prompt,
            generation_config
        )
        
        return response.strip()
    
    # Fallback: Use single-image mode (only the evaluation image)
    print("⚠️  InternVL does not support multi-image chat, using single-image mode")
    return model.generate_with_image(eval_image, prompt, max_tokens=5)


# ==================== Single image mode (no reference) ====================

def create_single_prompt(predicted_caption: str) -> str:
    """Create single-image scoring prompt"""
    return SINGLE_IMAGE_EVALUATION_PROMPT.format(
        predicted_caption=predicted_caption
    )


def generate_score_single_image(model, eval_image: Image.Image, prompt: str, 
                                 model_type: str = None) -> str:
    """
    Generate score using VLM model (single image no-reference mode)
    - eval_image: Image to evaluate
    """
    try:
        # All supported models should have a generate_with_image method
        if hasattr(model, 'generate_with_image'):
            return model.generate_with_image(eval_image, prompt, max_tokens=5)
        
        # Fallback: Try using processor + model approach
        if model_type == 'sailvl':
            return _generate_sailvl_single_image(model, eval_image, prompt)
        elif model_type == 'qwen3_vl':
            return _generate_qwen_single_image(model, eval_image, prompt)
        elif model_type == 'internvl':
            return _generate_internvl_single_image(model, eval_image, prompt)
        
        return ""
        
    except Exception as e:
        print(f"⚠️  Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def _generate_qwen_single_image(model, eval_image: Image.Image, prompt: str) -> str:
    """Qwen3-VL single-image generation"""
    from qwen_vl_utils import process_vision_info
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    return output_text


def _generate_sailvl_single_image(model, eval_image: Image.Image, prompt: str) -> str:
    """SAIL-VL2 single-image generation"""
    if eval_image.mode != 'RGB':
        eval_image = eval_image.convert('RGB')
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = model.processor(
        images=eval_image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.model.device).to(torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()
    
    if 'assistant' in response.lower():
        parts = response.split('assistant')
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


def _generate_internvl_single_image(model, eval_image: Image.Image, prompt: str) -> str:
    """InternVL single-image generation"""
    if eval_image.mode != 'RGB':
        eval_image = eval_image.convert('RGB')
    
    full_prompt = f"<image>\n{prompt}"
    
    if hasattr(model.model, 'chat'):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        input_size = 448
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        pixel_values = transform(eval_image).unsqueeze(0).to(
            dtype=torch.bfloat16, device=model.model.device
        )
        
        generation_config = {
            'max_new_tokens': 5,
            'do_sample': False,
        }
        
        response = model.model.chat(
            model.tokenizer,
            pixel_values,
            full_prompt,
            generation_config
        )
        
        return response.strip()
    
    return model.generate_with_image(eval_image, prompt, max_tokens=5)


def create_single_tasks(vlm_data: List[Dict]) -> List[Dict]:
    """Create all scoring tasks (single image no-reference mode)"""
    tasks = []
    
    for i, item in enumerate(vlm_data):
        sample_idx = item.get('index', i)
        predicted_caption = item.get('predicted_caption', '')
        true_captions = item.get('true_captions', [])
        
        tasks.append({
            'sample_global_id': i,
            'sample_index': sample_idx,
            'predicted_caption': predicted_caption,
            'true_captions': true_captions,
        })
    
    return tasks


def reorganize_single_results(all_results: List[Dict]) -> List[Dict]:
    """Organize single-image mode results"""
    final_results = []
    
    for result in all_results:
        final_results.append({
            'global_id': result['sample_global_id'],
            'sample_index': result['sample_index'],
            'predicted_caption': result.get('predicted_caption', ''),
            'true_captions': result.get('true_captions', []),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'is_passing': result.get('is_passing', False),
            'raw_response': result.get('raw_response', ''),
        })
    
    return final_results


def reorganize_results_by_sample(all_results: List[Dict]) -> List[Dict]:
    """Reorganize results by sample"""
    from collections import defaultdict
    
    samples_dict = defaultdict(lambda: {
        'neighbor_scores': [None] * 10,
        'metadata': {}
    })
    
    for result in all_results:
        sample_id = result['sample_global_id']
        rank = result['neighbor_rank']
        
        if not samples_dict[sample_id]['metadata']:
            samples_dict[sample_id]['metadata'] = {
                'global_id': result['sample_global_id'],
                'sample_index': result['sample_index'],
                'predicted_caption': result.get('predicted_caption', ''),
                'true_captions': result.get('true_captions', []),
            }
        
        samples_dict[sample_id]['neighbor_scores'][rank] = {
            'neighbor_rank': rank,
            'neighbor_global_id': result['neighbor_global_id'],
            'neighbor_index': result.get('neighbor_index'),
            'neighbor_cosine': result['neighbor_cosine'],
            'ref_caption': result.get('ref_caption', ''),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'is_passing': result.get('is_passing', False),
            'raw_response': result.get('raw_response', ''),
            'success': result.get('success', False),
            'is_self_ref': result.get('is_self_ref', False),
        }
    
    # Calculate statistics
    final_results = []
    for sample_id in sorted(samples_dict.keys()):
        sample_data = samples_dict[sample_id]
        scores = sample_data['neighbor_scores']
        
        valid_scores = [s['score'] for s in scores if s and s['is_valid']]
        
        score_distribution = {str(i): 0 for i in range(6)}
        score_distribution['INVALID'] = 0
        
        for s in scores:
            if s:
                score_distribution[s['score_str']] += 1
        
        final_results.append({
            **sample_data['metadata'],
            'neighbor_scores': scores,
            'stats': {
                'total_scores': sum(1 for s in scores if s),
                'valid_scores_count': len(valid_scores),
                'invalid_count': score_distribution['INVALID'],
                'passing_count': sum(1 for s in scores if s and s['is_passing']),
                'avg_score': sum(valid_scores) / len(valid_scores) if valid_scores else None,
                'min_score': min(valid_scores) if valid_scores else None,
                'max_score': max(valid_scores) if valid_scores else None,
                'score_distribution': score_distribution
            }
        })
    
    return final_results


def save_results(results, output_path, is_final=False):
    """Save results"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f"\n✅ Final results saved to: {output_path}")
    else:
        print(f"\n💾 Intermediate results saved: {len(results)} samples")


def load_existing_results(output_path) -> List[Dict]:
    """Load existing results (resume from checkpoint)"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Flatten into task result list
        tasks_completed = []
        for sample in results:
            sample_id = sample['global_id']
            sample_index = sample['sample_index']
            predicted_caption = sample.get('predicted_caption', '')
            true_captions = sample.get('true_captions', [])
            
            for score_data in sample['neighbor_scores']:
                if score_data:
                    tasks_completed.append({
                        'sample_global_id': sample_id,
                        'sample_index': sample_index,
                        'neighbor_rank': score_data['neighbor_rank'],
                        'neighbor_global_id': score_data['neighbor_global_id'],
                        'neighbor_index': score_data.get('neighbor_index'),
                        'neighbor_cosine': score_data['neighbor_cosine'],
                        'predicted_caption': predicted_caption,
                        'true_captions': true_captions,
                        'ref_caption': score_data.get('ref_caption', ''),
                        'score': score_data['score'],
                        'score_str': score_data['score_str'],
                        'is_valid': score_data['is_valid'],
                        'is_passing': score_data['is_passing'],
                        'raw_response': score_data.get('raw_response', ''),
                        'success': score_data['success'],
                        'is_self_ref': score_data.get('is_self_ref', False),
                    })
        
        print(f"📂 Found existing results, completed {len(tasks_completed)} scoring tasks")
        return tasks_completed
    except FileNotFoundError:
        print("📂 No existing results found, starting from scratch")
        return []


def main():
    args = parse_args()
    
    mode_desc = 'neighbor reference mode' if args.mode == 'neighbor' else 'single image no-reference mode'
    
    print("=" * 80)
    print("🚀 VLM Caption Scoring")
    if args.mode == 'neighbor':
        print("   - Image 1: Reference image (neighbor) + reference caption")
        print("   - Image 2: Image to evaluate + predicted_caption")
    else:
        print("   - Single image mode: directly evaluate image-caption match quality")
    print(f"   - Scoring mode: {mode_desc}")
    print("=" * 80)
    if args.mode == 'neighbor':
        print(f"📥 Neighbor file: {args.neighbors}")
    print(f"📥 VLM data file: {args.vlm_data}")
    print(f"📤 Output file: {args.output}")
    print(f"🤖 VLM scoring model: {args.model}")
    print(f"📦 Dataset: {args.dataset} (split={args.split})")
    if args.mode == 'neighbor':
        print(f"📝 Reference type: {args.ref_type}")
    if args.test:
        print(f"🧪 Test mode: first {args.test} samples")
    print("=" * 80 + "\n")
    
    # Load VLM model
    print("Loading VLM model...")
    model, model_type = load_vlm_model(args.model)
    print()
    
    # Load HuggingFace dataset (images)
    print("Loading image dataset...")
    hf_dataset = load_hf_dataset(args.dataset, args.split)
    print()
    
    # Load VLM data
    print("Loading VLM data...")
    vlm_data = load_vlm_data(args.vlm_data)
    vlm_map = build_global_id_map(vlm_data)
    print(f"✅ VLM data: {len(vlm_data)} entries\n")
    
    # ==================== Neighbor reference mode ====================
    if args.mode == 'neighbor':
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f"✅ Neighbor data: {len(neighbors_data)} samples\n")
        
        if args.test:
            neighbors_data = neighbors_data[:args.test]
            print(f"🧪 Test mode: only processing first {args.test} samples\n")

        # Create tasks
        print("Creating scoring tasks...")
        all_tasks = create_evaluation_tasks(neighbors_data, vlm_map, ref_type=args.ref_type)
        print(f"✅ Total {len(all_tasks)} tasks ({len(neighbors_data)} x 10)\n")
        
        # Resume from checkpoint
        completed_results = []
        if args.resume:
            completed_results = load_existing_results(args.output)

            if len(completed_results) >= len(all_tasks):
                print("✅ All tasks completed!")
                final_results = reorganize_results_by_sample(completed_results)
                save_results(final_results, args.output, is_final=True)
                return

            completed_keys = {
                (r['sample_global_id'], r['neighbor_rank'])
                for r in completed_results
            }
            all_tasks = [
                t for t in all_tasks
                if (t['sample_global_id'], t['neighbor_rank']) not in completed_keys
            ]
            print(f"🔄 Remaining {len(all_tasks)} tasks\n")

        # Scoring
        print("=== Starting scoring (neighbor reference mode) ===\n")
        all_results = completed_results.copy()

        for i, task in enumerate(tqdm(all_tasks, desc="Scoring progress")):
            try:
                sample_idx = task['sample_index']
                neighbor_idx = task['neighbor_index']

                eval_image = get_image_from_hf(hf_dataset, sample_idx)
                ref_image = get_image_from_hf(hf_dataset, neighbor_idx)

                if eval_image is None or ref_image is None:
                    print(f"⚠️  Cannot get image (sample={sample_idx}, neighbor={neighbor_idx})")
                    response = ""
                else:
                    prompt = create_prompt(task)
                    response = generate_score_with_two_images(model, ref_image, eval_image, prompt, model_type=model_type)
                
                score, is_valid, score_str = parse_score(response)
                
                result = {
                    'sample_global_id': task['sample_global_id'],
                    'sample_index': task['sample_index'],
                    'neighbor_rank': task['neighbor_rank'],
                    'neighbor_global_id': task['neighbor_global_id'],
                    'neighbor_index': task['neighbor_index'],
                    'neighbor_cosine': task['neighbor_cosine'],
                    'predicted_caption': task['predicted_caption'],
                    'true_captions': task['true_captions'],
                    'ref_caption': task['ref_caption'],
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'is_passing': score >= 3 if is_valid else False,
                    'raw_response': response,
                    'success': is_valid,
                    'is_self_ref': task.get('is_self_ref', False),
                }
                all_results.append(result)
                
                if (i + 1) % args.save_interval == 0:
                    reorganized = reorganize_results_by_sample(all_results)
                    save_results(reorganized, args.output, is_final=False)
            
            except Exception as e:
                print(f"\n⚠️  Task {i} failed: {str(e)}")
                if all_results:
                    reorganized = reorganize_results_by_sample(all_results)
                    save_results(reorganized, args.output, is_final=False)
                continue

        # Save final results
        print("\nReorganizing results...")
        final_results = reorganize_results_by_sample(all_results)

        # Statistics
        total_tasks = len(all_results)
        success_count = sum(1 for r in all_results if r.get('success', False))
        
        all_valid_scores = []
        invalid_count = 0
        for sample in final_results:
            for s in sample['neighbor_scores']:
                if s and s['is_valid']:
                    all_valid_scores.append(s['score'])
                elif s and s['score_str'] == 'INVALID':
                    invalid_count += 1
    
    # ==================== Single image no-reference mode ====================
    else:
        if args.test:
            vlm_data = vlm_data[:args.test]
            print(f"🧪 Test mode: only processing first {args.test} samples\n")

        # Create tasks
        all_tasks = create_single_tasks(vlm_data)
        print(f"✅ Total {len(all_tasks)} scoring tasks\n")

        # Resume from checkpoint (single image mode)
        completed_ids = set()
        if args.resume:
            try:
                with open(args.output, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                completed_ids = {item['global_id'] for item in existing}
                print(f"📂 Found existing results: {len(completed_ids)} samples completed")
            except FileNotFoundError:
                pass
        
        tasks_to_process = [t for t in all_tasks if t['sample_global_id'] not in completed_ids]
        print(f"🔄 To process: {len(tasks_to_process)} tasks\n")

        if not tasks_to_process:
            print("✅ All tasks completed!")
            return

        # Scoring
        print("=== Starting scoring (single image no-reference mode) ===\n")
        all_results = []
        
        for i, task in enumerate(tqdm(tasks_to_process, desc="Scoring progress")):
            try:
                sample_idx = task['sample_index']
                eval_image = get_image_from_hf(hf_dataset, sample_idx)
                
                if eval_image is None:
                    print(f"⚠️  Cannot get image (sample={sample_idx})")
                    response = ""
                else:
                    prompt = create_single_prompt(task['predicted_caption'])
                    response = generate_score_single_image(model, eval_image, prompt, model_type=model_type)
                
                score, is_valid, score_str = parse_score(response)
                
                result = {
                    'sample_global_id': task['sample_global_id'],
                    'sample_index': task['sample_index'],
                    'predicted_caption': task['predicted_caption'],
                    'true_captions': task['true_captions'],
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'is_passing': score >= 3 if is_valid else False,
                    'raw_response': response,
                }
                all_results.append(result)
                
                if (i + 1) % args.save_interval == 0:
                    final_results = reorganize_single_results(all_results)
                    save_results(final_results, args.output, is_final=False)
            
            except Exception as e:
                print(f"\n⚠️  Task {i} failed: {str(e)}")
                if all_results:
                    final_results = reorganize_single_results(all_results)
                    save_results(final_results, args.output, is_final=False)
                continue

        # Save final results
        final_results = reorganize_single_results(all_results)

        # Statistics
        total_tasks = len(all_results)
        success_count = sum(1 for r in all_results if r.get('is_valid', False))
        all_valid_scores = [r['score'] for r in all_results if r.get('is_valid', False)]
        invalid_count = total_tasks - len(all_valid_scores)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("📊 Final Statistics")
    print("=" * 80)
    print(f"Total samples: {len(final_results)}")
    print(f"Total scoring tasks: {total_tasks}")
    print(f"Succeeded: {success_count} ({success_count/total_tasks*100:.1f}%)" if total_tasks > 0 else "Succeeded: 0")

    if all_valid_scores:
        avg_score = sum(all_valid_scores) / len(all_valid_scores)
        passing_count = sum(1 for s in all_valid_scores if s >= 3)

        print(f"\n📈 Score statistics:")
        print(f"   Valid scores: {len(all_valid_scores)}")
        print(f"   Invalid scores: {invalid_count}")
        print(f"   Average score: {avg_score:.2f}/5.0")
        print(f"   Pass rate: {passing_count/len(all_valid_scores)*100:.1f}%")

        print(f"\n📊 Score distribution:")
        total = len(all_valid_scores) + invalid_count
        for score in range(5, -1, -1):
            count = all_valid_scores.count(score)
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            mark = '✅' if score >= 3 else '❌'
            print(f"   {score}: {count:4d} ({pct:5.1f}%) {bar} {mark}")
    
    print("=" * 80)
    save_results(final_results, args.output, is_final=True)


if __name__ == "__main__":
    main()