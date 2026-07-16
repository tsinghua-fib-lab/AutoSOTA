"""
Image Classification task: scoring using image neighbor information (non-vLLM version)
- Supports SAIL-VL, InternVL, Qwen and other models
- Supports neighbor (neighbor reference) and single (no reference) modes
- Supports CIFAR-10 and ImageNet-1k
"""
import sys
import os
from pathlib import Path

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
from models.vlm_models import Qwen3VLCaptioner, InternVLCaptioner, SAILVLCaptioner, Ministral3VLCaptioner, Step3VLCaptioner


# Model class mapping
VLM_MODEL_CLASS_MAP = {
    "qwen3_vl": Qwen3VLCaptioner,
    "internvl": InternVLCaptioner,
    "sailvl": SAILVLCaptioner,
    "ministral3_vl": Ministral3VLCaptioner,
    "step3vl": Step3VLCaptioner,
}


# Neighbor reference mode Prompt (consistent with vLLM version)
IC_NEIGHBOR_EVALUATION_PROMPT = '''You are evaluating an image classification prediction.

[Image 1 - Reference (Correct Example)]
Label: {ref_label}
This is a HIGH-QUALITY correct prediction.

[Image 2 - To Evaluate]
Label: {eval_label}

Task: Compare Image 2's prediction quality against the reference standard.

Step 1: Are Image 1 and Image 2 visually similar?

If SIMILAR (same type of content):
Labels should be similar or related. Score based on label consistency:
- Score 5: Labels match or are equivalent
- Score 4: Labels are closely related (e.g., sub-category, synonym)
- Score 3: Labels are in the same general domain
- Score 2: Labels differ but Image 2's label still fits its content
- Score 1: Labels differ significantly
- Score 0: Image 2's label is wrong for its content

If NOT SIMILAR (different content):
Judge Image 2's prediction by the same high standard as the reference:
- Score 5: Prediction is as accurate as the reference
- Score 4: Prediction is good, nearly reference quality
- Score 3: Prediction is acceptable
- Score 2: Prediction is below reference standard
- Score 1: Prediction is poor
- Score 0: Prediction is clearly incorrect

Return ONLY a number (0-5).

Score:'''

# Single image no-reference mode Prompt
IC_SINGLE_EVALUATION_PROMPT = '''You are evaluating an image classification prediction. Look at the image and assess whether the predicted label is correct.

[Image to Evaluate]
Predicted Label: {eval_label}

Scoring Rules:
- Score 5: Prediction is excellent - clearly correct and matches the visual content perfectly
- Score 4: Prediction is good - mostly correct, captures the main subject
- Score 3: Prediction is acceptable - somewhat correct but could be more accurate
- Score 2: Prediction is questionable - may not accurately describe the image
- Score 1: Prediction is poor - does not match the visual content well
- Score 0: Prediction is wrong - completely incorrect for the image

Evaluation Criteria:
1. Does the predicted label correctly identify what's in the image?
2. Is it the most appropriate label for this image?
3. Would a human agree with this classification?

Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

Score:'''


def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification scoring (supports SAIL-VL)')

    # Scoring mode
    parser.add_argument('--mode', type=str, default='neighbor', choices=['neighbor', 'single'],
                       help='Scoring mode: neighbor=neighbor reference mode (default), single=no-reference single image mode')

    parser.add_argument('--neighbors', type=str, default=None,
                       help='Neighbor file path (required for neighbor mode)')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Prediction results file path (JSON)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--model', type=str, default='sailvl-8b',
                       help='Model name (e.g., sailvl-8b, internvl3.5-8b, qwen3-vl-8b)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar-10', 'cifar10', 'imagenet-1k', 'imagenet'],
                       help='Dataset name')
    parser.add_argument('--save-interval', '-s', type=int, default=50,
                       help='Auto-save interval')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process the first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    # Data sharding parameters (for multi-GPU parallelism)
    parser.add_argument('--shard', type=int, default=0,
                       help='Current shard ID (0-based)')
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of shards (e.g., 4 means split into 4 parts)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'neighbor' and not args.neighbors:
        parser.error('--neighbors is required when mode is "neighbor"')
    
    return args


def load_hf_dataset(dataset_name: str):
    """Load dataset from HuggingFace"""
    from datasets import load_dataset
    
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['cifar-10', 'cifar10']:
        print(f'Loading CIFAR-10 from HuggingFace...')
        dataset = load_dataset('cifar10', split='test', trust_remote_code=True)
        image_field = 'img'
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name in ['imagenet-1k', 'imagenet']:
        print(f'Loading ImageNet-1K from HuggingFace (streaming mode, first 10000)...')
        # Use streaming mode to avoid downloading the entire dataset
        stream_dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True, trust_remote_code=True)
        # Take the first 10000 samples and convert to a regular Dataset
        samples = []
        for i, sample in enumerate(stream_dataset):
            if i >= 10000:
                break
            samples.append(sample)
            if (i + 1) % 1000 == 0:
                print(f'  Loaded {i + 1}/10000 samples...')
        
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        image_field = 'image'
        # ImageNet-1k has 1000 classes, get label names from the streaming dataset
        label_names = stream_dataset.features['label'].names if hasattr(stream_dataset.features['label'], 'names') else None
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    print(f'✅ Loaded {len(dataset)} samples')
    return dataset, image_field, label_names


def get_image_from_hf(hf_dataset, idx: int, image_field: str) -> Optional[Image.Image]:
    """Get image from HuggingFace dataset"""
    try:
        item = hf_dataset[idx]
        image = item.get(image_field)
        if image and image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"⚠️  Failed to get image at index {idx}: {e}")
        return None


def load_vlm_model(model_name: str):
    """Load VLM model"""
    if model_name not in MODEL_CONFIG["vlm_tagging"]:
        raise ValueError(f"Model '{model_name}' not found in vlm_tagging configuration")

    config = MODEL_CONFIG["vlm_tagging"][model_name]
    model_type = config["model_type"]

    if model_type not in VLM_MODEL_CLASS_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = VLM_MODEL_CLASS_MAP[model_type]
    
    model_config = {
        "model_name": config["model_name"],
        "model_type": model_type,
        "device": config.get("device", "cuda"),
        "max_new_tokens": 5,
    }
    
    model = model_class(model_config)
    model.load_model()
    
    print(f"✅ Loaded VLM model: {model_name}")
    print(f"   - Type: {model_type}")
    print(f"   - Path: {config['model_name']}")
    
    return model, model_type


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    """Load neighbor data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_predictions_json(file_path: str) -> List[Dict]:
    """Load prediction results"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_label_name(label_id: int, label_names: List[str] = None) -> str:
    """Get label name"""
    if label_names and label_id < len(label_names):
        return label_names[label_id]
    return f"class_{label_id}"


def get_correct(pred: Dict) -> bool:
    """Get the correct field, compute if not present"""
    if 'correct' in pred:
        return pred['correct']
    return pred['true_label'] == pred['predicted_label']


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """Parse score"""
    response = response.strip()

    # Clean thinking content
    if '<think>' in response:
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        else:
            response = response.replace('<think>', '').strip()
    response = response.replace('</think>', '').strip()
    
    # 1. If the response is very short (<10 chars) and contains only one number, return directly
    if len(response) < 10:
        try:
            score = int(response.strip())
            if 0 <= score <= 5:
                return score, True, str(score)
        except ValueError:
            pass
        # Search for number in short response
        match = re.search(r'([0-5])', response)
        if match:
            return int(match.group(1)), True, match.group(1)
    
    # 2. Detect if it is a thinking process (contains thinking indicator words)
    thinking_indicators = [
        "let's", "Let's", "first", "First", "check", "Check",
        "wait", "Wait", "think", "Think", "step by step",
        "user", "firstly", "let me", "examine", "look at"
    ]
    is_thinking = any(ind in response for ind in thinking_indicators)
    
    # 3. If it is a thinking process, only accept very explicit final answer formats
    if is_thinking:
        # Only accept these explicit final answer formats
        final_patterns = [
            r'[Ff]inal\s+[Ss]core[:\s]+([0-5])\s*$',      # Final Score: 5 (at the end)
            r'[Ff]inal\s+[Aa]nswer[:\s]+([0-5])\s*$',     # Final Answer: 5 (at the end)
            r'[Ss]core[:\s]+([0-5])\s*$',                  # Score: 5 (at the end)
            r'\n([0-5])\s*$',                              # A standalone number at the end
            r'最终[评打]分[：:\s]*([0-5])',                # Final score: 5 (Chinese)
            r'所以[，,]?\s*([0-5])\s*分',                  # So, 5 points (Chinese)
            r'答案[是为：:\s]*([0-5])',                    # Answer is 5 (Chinese)
        ]
        for pattern in final_patterns:
            match = re.search(pattern, response)
            if match:
                return int(match.group(1)), True, match.group(1)
        # Thinking process has no explicit final answer, mark as invalid
        return None, False, 'INVALID'
    
    # 4. Non-thinking process, search for score patterns
    score_patterns = [
        r'[Ss]core[:\s]+([0-5])',           # Score: 5
        r'[Ff]inal\s+[Ss]core[:\s]+([0-5])', # Final Score: 5
        r'[Ff]inal\s+[Aa]nswer[:\s]+([0-5])', # Final Answer: 5
        r'评分[：:\s]*([0-5])',              # Score: 5 (Chinese)
        r'分数[是为：:\s]*([0-5])',          # Score is 5 (Chinese)
        r'得分[：:\s]*([0-5])',              # Points: 5 (Chinese)
        r'([0-5])\s*分',                     # 5 points (Chinese)
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            return int(match.group(1)), True, match.group(1)
    
    # 5. Find the last standalone 0-5 digit
    cleaned = re.sub(r'[Ii]mage\s*[12]', '', response)
    matches = re.findall(r'(?<![0-9])([0-5])(?![0-9])', cleaned)
    if matches:
        return int(matches[-1]), True, matches[-1]
    
    return None, False, 'INVALID'


# ==================== Neighbor reference mode generation functions ====================

def generate_neighbor_score_qwen(model, ref_image: Image.Image, eval_image: Image.Image, 
                                  ref_label: str, eval_label: str) -> str:
    """Qwen3-VL neighbor reference scoring"""
    from qwen_vl_utils import process_vision_info
    
    prompt = IC_NEIGHBOR_EVALUATION_PROMPT.format(ref_label=ref_label, eval_label=eval_label)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": ref_image},
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = model.processor(text=[text], images=image_inputs, videos=video_inputs, 
                             padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = model.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    return output_text


def generate_neighbor_score_sailvl(model, ref_image: Image.Image, eval_image: Image.Image, 
                                    ref_label: str, eval_label: str) -> str:
    """SAIL-VL neighbor reference scoring"""
    prompt = IC_NEIGHBOR_EVALUATION_PROMPT.format(ref_label=ref_label, eval_label=eval_label)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": ref_image},
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = model.processor(
        images=[ref_image, eval_image],
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.model.device).to(torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()
    
    if 'assistant' in response.lower():
        parts = response.split('assistant')
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


def generate_neighbor_score_internvl(model, ref_image: Image.Image, eval_image: Image.Image, 
                                      ref_label: str, eval_label: str) -> str:
    """InternVL neighbor reference scoring"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    prompt = IC_NEIGHBOR_EVALUATION_PROMPT.format(ref_label=ref_label, eval_label=eval_label)
    full_prompt = f"<image>\n<image>\n{prompt}"
    
    input_size = 448
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    pixel_values_list = []
    for img in [ref_image, eval_image]:
        pixel_values = transform(img).unsqueeze(0).to(dtype=torch.bfloat16, device=model.model.device)
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list, dim=0)
    
    generation_config = {'max_new_tokens': 5, 'do_sample': False}
    response = model.model.chat(model.tokenizer, pixel_values, full_prompt, generation_config)
    
    return response.strip()


def generate_neighbor_score_step3vl(model, ref_image: Image.Image, eval_image: Image.Image, 
                                     ref_label: str, eval_label: str) -> str:
    """Step3-VL neighbor reference scoring"""
    import io
    import base64
    
    prompt = IC_NEIGHBOR_EVALUATION_PROMPT.format(ref_label=ref_label, eval_label=eval_label)
    
    # Convert image to base64 URL
    def img_to_base64_url(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    
    ref_url = img_to_base64_url(ref_image)
    eval_url = img_to_base64_url(eval_image)
    
    # Step3-VL cannot disable thinking mode, use regular prompt directly (consistent with other models)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": ref_url},
                {"type": "image", "url": eval_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    inputs = model.processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Step3-VL needs more tokens to complete thinking before outputting the score
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=500, do_sample=False)
    
    input_len = inputs['input_ids'].shape[1]
    output_ids = generated_ids[0][input_len:]
    response = model.processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Handle possible thinking output
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    return response


def generate_neighbor_score(model, ref_image: Image.Image, eval_image: Image.Image, 
                            ref_label: str, eval_label: str, model_type: str) -> str:
    """Select generation function based on model type"""
    try:
        if model_type == 'sailvl':
            return generate_neighbor_score_sailvl(model, ref_image, eval_image, ref_label, eval_label)
        elif model_type == 'internvl':
            return generate_neighbor_score_internvl(model, ref_image, eval_image, ref_label, eval_label)
        elif model_type == 'qwen3_vl':
            return generate_neighbor_score_qwen(model, ref_image, eval_image, ref_label, eval_label)
        elif model_type == 'step3vl':
            return generate_neighbor_score_step3vl(model, ref_image, eval_image, ref_label, eval_label)
        else:
            print(f"⚠️  Unknown model type: {model_type}")
            return ""
    except Exception as e:
        print(f"⚠️  Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


# ==================== Single image no-reference mode generation functions ====================

def generate_single_score_qwen(model, eval_image: Image.Image, eval_label: str) -> str:
    """Qwen3-VL single image scoring"""
    from qwen_vl_utils import process_vision_info
    
    prompt = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = model.processor(text=[text], images=image_inputs, videos=video_inputs, 
                             padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = model.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    return output_text


def generate_single_score_sailvl(model, eval_image: Image.Image, eval_label: str) -> str:
    """SAIL-VL single image scoring"""
    prompt = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": eval_image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = model.processor(
        images=eval_image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.model.device).to(torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()
    
    if 'assistant' in response.lower():
        parts = response.split('assistant')
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


def generate_single_score_internvl(model, eval_image: Image.Image, eval_label: str) -> str:
    """InternVL single image scoring"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    prompt = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    full_prompt = f"<image>\n{prompt}"
    
    input_size = 448
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    pixel_values = transform(eval_image).unsqueeze(0).to(dtype=torch.bfloat16, device=model.model.device)
    
    generation_config = {'max_new_tokens': 5, 'do_sample': False}
    response = model.model.chat(model.tokenizer, pixel_values, full_prompt, generation_config)
    
    return response.strip()


def generate_single_score_step3vl(model, eval_image: Image.Image, eval_label: str) -> str:
    """Step3-VL single image scoring"""
    import io
    import base64
    
    prompt = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    
    # Convert image to base64 URL
    buffered = io.BytesIO()
    eval_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    eval_url = f"data:image/png;base64,{img_base64}"
    
    # Step3-VL cannot disable thinking mode, use regular prompt directly (consistent with other models)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": eval_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    inputs = model.processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Step3-VL needs more tokens to complete thinking before outputting the score
    with torch.no_grad():
        generated_ids = model.model.generate(**inputs, max_new_tokens=500, do_sample=False)
    
    input_len = inputs['input_ids'].shape[1]
    output_ids = generated_ids[0][input_len:]
    response = model.processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Handle possible thinking output
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    return response


def generate_single_score(model, eval_image: Image.Image, eval_label: str, model_type: str) -> str:
    """Select generation function based on model type"""
    try:
        if model_type == 'sailvl':
            return generate_single_score_sailvl(model, eval_image, eval_label)
        elif model_type == 'internvl':
            return generate_single_score_internvl(model, eval_image, eval_label)
        elif model_type == 'qwen3_vl':
            return generate_single_score_qwen(model, eval_image, eval_label)
        elif model_type == 'step3vl':
            return generate_single_score_step3vl(model, eval_image, eval_label)
        else:
            print(f"⚠️  Unknown model type: {model_type}")
            return ""
    except Exception as e:
        print(f"⚠️  Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


# ==================== Task creation and result organization ====================

def create_neighbor_tasks(neighbors_data: List[Dict], predictions: List[Dict],
                          label_names: List[str]) -> List[Dict]:
    """Create tasks for neighbor reference mode"""
    pred_map = {p['index']: p for p in predictions}
    
    tasks = []
    for sample in neighbors_data:
        sample_idx = sample.get('index', sample.get('global_id'))
        
        if sample_idx not in pred_map:
            continue
            
        sample_pred = pred_map[sample_idx]
        eval_label = sample_pred.get('predicted_label_name') or get_label_name(sample_pred['predicted_label'], label_names)
        sample_correct = get_correct(sample_pred)
        
        # 9 neighbors
        for rank, neighbor in enumerate(sample['neighbors']):
            neighbor_idx = neighbor.get('global_id', neighbor.get('index'))
            
            if neighbor_idx not in pred_map:
                continue
                
            neighbor_pred = pred_map[neighbor_idx]
            ref_label = neighbor_pred.get('predicted_label_name') or get_label_name(neighbor_pred['predicted_label'], label_names)
            
            tasks.append({
                'sample_index': sample_idx,
                'sample_true_label': sample_pred['true_label'],
                'sample_predicted_label': sample_pred['predicted_label'],
                'sample_correct': sample_correct,
                'eval_label': eval_label,
                'neighbor_rank': rank,
                'neighbor_index': neighbor_idx,
                'neighbor_cosine': neighbor['cosine'],
                'ref_label': ref_label,
                'neighbor_correct': get_correct(neighbor_pred),
                'is_self_ref': False,
            })
        
        # Self as reference (the 10th)
        tasks.append({
            'sample_index': sample_idx,
            'sample_true_label': sample_pred['true_label'],
            'sample_predicted_label': sample_pred['predicted_label'],
            'sample_correct': sample_correct,
            'eval_label': eval_label,
            'neighbor_rank': 9,
            'neighbor_index': sample_idx,
            'neighbor_cosine': 1.0,
            'ref_label': eval_label,
            'neighbor_correct': sample_correct,
            'is_self_ref': True,
        })
    
    return tasks


def create_single_tasks(predictions: List[Dict], label_names: List[str]) -> List[Dict]:
    """Create tasks for single image no-reference mode"""
    tasks = []
    for pred in predictions:
        sample_idx = pred['index']
        eval_label = pred.get('predicted_label_name') or get_label_name(pred['predicted_label'], label_names)
        
        tasks.append({
            'sample_index': sample_idx,
            'sample_true_label': pred['true_label'],
            'sample_predicted_label': pred['predicted_label'],
            'sample_correct': get_correct(pred),
            'eval_label': eval_label,
        })
    
    return tasks


def reorganize_neighbor_results(all_results: List[Dict]) -> List[Dict]:
    """Reorganize neighbor mode results by sample"""
    from collections import defaultdict
    
    samples_dict = defaultdict(lambda: {'neighbor_scores': [None] * 10, 'metadata': {}})
    
    for result in all_results:
        sample_idx = result['sample_index']
        rank = result['neighbor_rank']
        
        if not samples_dict[sample_idx]['metadata']:
            samples_dict[sample_idx]['metadata'] = {
                'index': result['sample_index'],
                'true_label': result['sample_true_label'],
                'predicted_label': result['sample_predicted_label'],
                'correct': result['sample_correct'],
            }
        
        samples_dict[sample_idx]['neighbor_scores'][rank] = {
            'neighbor_rank': rank,
            'neighbor_index': result['neighbor_index'],
            'neighbor_cosine': result['neighbor_cosine'],
            'ref_label': result.get('ref_label', ''),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'raw_response': result.get('raw_response', ''),
            'is_self_ref': result.get('is_self_ref', False),
        }
    
    final_results = []
    for sample_idx in sorted(samples_dict.keys()):
        sample_data = samples_dict[sample_idx]
        scores = sample_data['neighbor_scores']
        valid_scores = [s['score'] for s in scores if s and s['is_valid']]
        
        final_results.append({
            **sample_data['metadata'],
            'neighbor_scores': scores,
            'stats': {
                'total_scores': sum(1 for s in scores if s),
                'valid_scores_count': len(valid_scores),
                'avg_score': sum(valid_scores) / len(valid_scores) if valid_scores else None,
                'passing_count': sum(1 for s in valid_scores if s >= 3),
            }
        })
    
    return final_results


def reorganize_single_results(all_results: List[Dict]) -> List[Dict]:
    """Organize single image mode results"""
    return [{
        'index': r['sample_index'],
        'true_label': r['sample_true_label'],
        'predicted_label': r['sample_predicted_label'],
        'correct': r['sample_correct'],
        'score': r['score'],
        'score_str': r.get('score_str', 'INVALID'),
        'is_valid': r.get('is_valid', False),
        'raw_response': r.get('raw_response', ''),
    } for r in all_results]


def save_results(results, output_path, is_final=False):
    """Save results"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if is_final:
        print(f'\n✅ Results saved: {output_path}')


def load_existing_results(output_path, mode: str) -> Tuple[List[Dict], set]:
    """Load existing results"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if mode == 'neighbor':
            completed_keys = set()
            for sample in results:
                sample_idx = sample['index']
                for score_data in sample['neighbor_scores']:
                    if score_data:
                        completed_keys.add((sample_idx, score_data['neighbor_rank']))
            print(f'📂 Found existing results: {len(completed_keys)} tasks completed')
            return results, completed_keys
        else:
            completed_ids = {item['index'] for item in results}
            print(f'📂 Found existing results: {len(completed_ids)} samples completed')
            return results, completed_ids
    except FileNotFoundError:
        return [], set()


def main():
    args = parse_args()
    
    mode_desc = 'neighbor reference mode' if args.mode == 'neighbor' else 'single image no-reference mode'

    # Handle output filename (add suffix if sharded)
    output_path = args.output
    if args.num_shards > 1:
        base, ext = os.path.splitext(args.output)
        output_path = f"{base}_shard{args.shard}{ext}"
    
    print('=' * 70)
    print('🚀 Image Classification Scoring (supports SAIL-VL)')
    print(f'   - Scoring mode: {mode_desc}')
    if args.num_shards > 1:
        print(f'   - Data shard: {args.shard + 1}/{args.num_shards}')
    print('=' * 70)
    if args.mode == 'neighbor':
        print(f'📥 Neighbor file: {args.neighbors}')
    print(f'📥 Prediction file: {args.predictions}')
    print(f'📤 Output: {output_path}')
    print(f'🤖 Model: {args.model}')
    print(f'📦 Dataset: {args.dataset}')
    print('=' * 70 + '\n')
    
    # Load model
    model, model_type = load_vlm_model(args.model)
    print()

    # Load dataset
    hf_dataset, image_field, label_names = load_hf_dataset(args.dataset)

    # Load prediction data
    predictions = load_predictions_json(args.predictions)
    print(f'✅ Prediction data: {len(predictions)} entries\n')
    
    # ==================== Neighbor reference mode ====================
    if args.mode == 'neighbor':
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f'✅ Neighbor data: {len(neighbors_data)} samples')

        if args.test:
            neighbors_data = neighbors_data[:args.test]
            print(f'🧪 Test mode: first {args.test} samples')

        # Data sharding (by sample, ensuring all neighbors of the same sample are in the same shard)
        if args.num_shards > 1:
            total_samples = len(neighbors_data)
            shard_size = (total_samples + args.num_shards - 1) // args.num_shards  # Round up
            start_idx = args.shard * shard_size
            end_idx = min(start_idx + shard_size, total_samples)
            neighbors_data = neighbors_data[start_idx:end_idx]
            print(f'📊 Shard {args.shard + 1}/{args.num_shards}: samples [{start_idx}, {end_idx}), total {len(neighbors_data)}')
        
        print()
        
        all_tasks = create_neighbor_tasks(neighbors_data, predictions, label_names)
        print(f'✅ Total {len(all_tasks)} scoring tasks\n')
        
        # Resume from checkpoint
        completed_keys = set()
        if args.resume:
            _, completed_keys = load_existing_results(output_path, 'neighbor')
        
        tasks_to_process = [t for t in all_tasks if (t['sample_index'], t['neighbor_rank']) not in completed_keys]
        print(f'🔄 To process: {len(tasks_to_process)} tasks\n')
        
        if not tasks_to_process:
            print('✅ All tasks completed!')
            return
        
        print('=== Starting scoring (neighbor reference mode) ===\n')
        all_results = []
        
        for i, task in enumerate(tqdm(tasks_to_process, desc='Scoring progress')):
            try:
                eval_image = get_image_from_hf(hf_dataset, task['sample_index'], image_field)
                ref_image = get_image_from_hf(hf_dataset, task['neighbor_index'], image_field)
                
                if eval_image is None or ref_image is None:
                    response = ""
                else:
                    response = generate_neighbor_score(
                        model, ref_image, eval_image,
                        task['ref_label'], task['eval_label'],
                        model_type
                    )
                
                score, is_valid, score_str = parse_score(response)
                
                all_results.append({
                    **task,
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'raw_response': response,
                })
                
                if (i + 1) % args.save_interval == 0:
                    save_results(reorganize_neighbor_results(all_results), output_path)
                    
            except Exception as e:
                print(f'\n⚠️  Task {i} failed: {str(e)}')
                continue
        
        final_results = reorganize_neighbor_results(all_results)
        save_results(final_results, output_path, is_final=True)
    
    # ==================== Single image no-reference mode ====================
    else:
        if args.test:
            predictions = predictions[:args.test]
            print(f'🧪 Test mode: first {args.test} samples\n')
        
        all_tasks = create_single_tasks(predictions, label_names)
        print(f'✅ Total {len(all_tasks)} scoring tasks\n')
        
        # Resume from checkpoint
        completed_ids = set()
        if args.resume:
            _, completed_ids = load_existing_results(output_path, 'single')
        
        tasks_to_process = [t for t in all_tasks if t['sample_index'] not in completed_ids]
        print(f'🔄 To process: {len(tasks_to_process)} tasks\n')
        
        if not tasks_to_process:
            print('✅ All tasks completed!')
            return
        
        print('=== Starting scoring (single image no-reference mode) ===\n')
        all_results = []
        
        for i, task in enumerate(tqdm(tasks_to_process, desc='Scoring progress')):
            try:
                eval_image = get_image_from_hf(hf_dataset, task['sample_index'], image_field)
                
                if eval_image is None:
                    response = ""
                else:
                    response = generate_single_score(
                        model, eval_image, task['eval_label'], model_type
                    )
                
                score, is_valid, score_str = parse_score(response)
                
                all_results.append({
                    **task,
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'raw_response': response,
                })
                
                if (i + 1) % args.save_interval == 0:
                    save_results(reorganize_single_results(all_results), output_path)
                    
            except Exception as e:
                print(f'\n⚠️  Task {i} failed: {str(e)}')
                continue
        
        final_results = reorganize_single_results(all_results)
        save_results(final_results, output_path, is_final=True)
    
    # Statistics
    all_scores = [r['score'] for r in all_results if r.get('is_valid')]
    if all_scores:
        print(f'\n📊 Statistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate (>=3): {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')


if __name__ == '__main__':
    main()

