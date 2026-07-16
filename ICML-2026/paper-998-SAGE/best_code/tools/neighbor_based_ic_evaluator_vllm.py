"""
Image Classification Task: Scoring using image neighbor information (vLLM accelerated version)
- Evaluates whether current sample's predicted label is reasonable based on neighbor images and predicted labels
- Supports neighbor (neighbor reference) and single (no-reference) modes
- Uses vLLM for efficient inference
- Supports CIFAR-10 and ImageNet-1k
- Supports InternVL and Qwen models
"""
import os
# Must set offline mode before importing datasets
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
import re
import glob
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image


# Neighbor reference mode prompt
# IC_NEIGHBOR_EVALUATION_PROMPT = '''You are evaluating image classification predictions by comparing with a reference.

# [Image 1 - Reference Image (Excellent Example)]
# Predicted Label: {ref_label}

# [Image 2 - Image to Evaluate]  
# Predicted Label: {eval_label}

# Task: Evaluate whether the prediction for Image 2 is reasonable, using Image 1's prediction quality as reference.

# Scoring Rules (compare against the reference):
# - Score 5: Prediction is excellent - matches reference quality, clearly correct
# - Score 4: Prediction is good - nearly as accurate as reference
# - Score 3: Prediction is acceptable - reasonable but not as accurate as reference
# - Score 2: Prediction is questionable - noticeably less accurate than reference
# - Score 1: Prediction is poor - much less accurate than reference
# - Score 0: Prediction is wrong - completely incorrect

# Evaluation Approach:
# 1. Look at Image 1 and its label - this represents the reference quality
# 2. Look at Image 2 and its predicted label
# 3. Compare: Is the prediction for Image 2 as accurate as the reference?

# Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

# Score:'''
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
# Single image no-reference mode prompt
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
    parser = argparse.ArgumentParser(description='Image Classification Scoring (vLLM accelerated version)')

    # Scoring mode
    parser.add_argument('--mode', type=str, default='neighbor', choices=['neighbor', 'single'],
                       help='Scoring mode: neighbor=neighbor reference mode (default), single=no-reference single image mode')
    
    parser.add_argument('--neighbors', type=str, default=None,
                       help='Neighbor file path (required for neighbor mode)')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Prediction results file path (JSON)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar-10', 'cifar10', 'imagenet-1k', 'imagenet'],
                       help='Dataset name')
    parser.add_argument('--save-interval', '-s', type=int, default=100)
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    # vLLM parameters
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization')
    parser.add_argument('--max-model-len', type=int, default=16384,
                       help='Maximum model length')
    parser.add_argument('--max-image-size', type=int, default=384,
                       help='Maximum image edge length (resize images to reduce tokens)')
    # Data sharding parameters
    parser.add_argument('--shard', type=int, default=0,
                       help='Current shard ID (0-based)')
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of shards')
    
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
        print(f'Loading ImageNet-1K from HuggingFace (offline mode)...')
        # Use correct dataset path, prefer loading from local cache
        dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', trust_remote_code=True)
        image_field = 'image'
        label_names = dataset.features['label'].names if hasattr(dataset.features['label'], 'names') else None
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    print(f'✅ Loaded {len(dataset)} samples')
    return dataset, image_field, label_names


def load_vllm_model(model_name: str, tensor_parallel_size: int = 1, 
                    gpu_memory_utilization: float = 0.45, max_model_len: int = 16384,
                    mode: str = 'neighbor'):
    """Load vLLM model"""
    import os
    from vllm import LLM

    # Force spawn method for subprocesses to ensure environment variables are passed correctly
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    
    print(f'Loading vLLM model: {model_name}')
    print(f'  - Tensor Parallel Size: {tensor_parallel_size}')
    print(f'  - GPU Memory Utilization: {gpu_memory_utilization}')
    print(f'  - Mode: {mode}')
    
    max_images = 2 if mode == 'neighbor' else 1
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        limit_mm_per_prompt={'image': max_images},
        enforce_eager=True,
        max_num_seqs=2,
    )
    
    print('✅ vLLM model loaded')
    return llm


def resize_image(image: Image.Image, max_size: int = 384) -> Image.Image:
    """Resize image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


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


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """Parse score"""
    response = response.strip()
    
    # Clean InternVL thinking content
    if '<think>' in response:
        # Try to extract content after </think>
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        else:
            # Only <think> without </think>, remove <think> tag
            response = response.replace('<think>', '').strip()
    
    # Remove other common special tokens
    response = response.replace('</think>', '').strip()
    
    first_line = response.split('\n')[0].strip()
    try:
        score = int(first_line)
        if 0 <= score <= 5:
            return score, True, str(score)
    except ValueError:
        pass
    
    # Search for digit 0-5 in the entire response
    match = re.search(r'\b([0-5])\b', response)
    if match:
        return int(match.group(1)), True, match.group(1)
    return None, False, 'INVALID'


def get_processor(model_name: str):
    """Get the processor for the model (with caching)"""
    if not hasattr(get_processor, '_cache'):
        get_processor._cache = {}
    if model_name not in get_processor._cache:
        from transformers import AutoProcessor
        print(f'Loading processor for {model_name}...')
        get_processor._cache[model_name] = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
    return get_processor._cache[model_name]


# ==================== Neighbor reference mode prompt ====================

def create_neighbor_prompt_qwen(ref_image: Image.Image, eval_image: Image.Image, 
                                 ref_label: str, eval_label: str,
                                 model_name: str = '') -> dict:
    """Create neighbor reference prompt for Qwen VL"""
    prompt_text = IC_NEIGHBOR_EVALUATION_PROMPT.format(
        ref_label=ref_label,
        eval_label=eval_label
    )
    
    processor = get_processor(model_name)
    
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': ref_image},
            {'type': 'image', 'image': eval_image},
            {'type': 'text', 'text': prompt_text}
        ]
    }]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [ref_image, eval_image]
        }
    }


def create_neighbor_prompt_internvl(ref_image: Image.Image, eval_image: Image.Image, 
                                     ref_label: str, eval_label: str,
                                     model_name: str = '') -> dict:
    """Create neighbor reference prompt for InternVL"""
    prompt_text = IC_NEIGHBOR_EVALUATION_PROMPT.format(
        ref_label=ref_label,
        eval_label=eval_label
    )
    
    # Simplified system prompt, directly guide digit output at the end
    system_prompt = "You are a direct assistant. Output only single digits without explanation."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\nThe score is: "
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [ref_image, eval_image]
        }
    }


def create_neighbor_prompt(ref_image: Image.Image, eval_image: Image.Image,
                           ref_label: str, eval_label: str,
                           model_name: str = '') -> dict:
    """Select prompt creation function based on model (neighbor reference mode)"""
    model_lower = model_name.lower()
    if 'internvl' in model_lower:
        return create_neighbor_prompt_internvl(
            ref_image, eval_image, ref_label, eval_label, model_name
        )
    else:
        return create_neighbor_prompt_qwen(
            ref_image, eval_image, ref_label, eval_label, model_name
        )


# ==================== Single image no-reference mode prompt ====================

def create_single_prompt_qwen(eval_image: Image.Image, eval_label: str,
                               model_name: str = '') -> dict:
    """Create single-image scoring prompt for Qwen VL"""
    prompt_text = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    
    processor = get_processor(model_name)
    
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': eval_image},
            {'type': 'text', 'text': prompt_text}
        ]
    }]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [eval_image]
        }
    }


def create_single_prompt_internvl(eval_image: Image.Image, eval_label: str,
                                   model_name: str = '') -> dict:
    """Create single-image scoring prompt for InternVL"""
    prompt_text = IC_SINGLE_EVALUATION_PROMPT.format(eval_label=eval_label)
    
    # Simplified system prompt, directly guide digit output at the end
    system_prompt = "You are a direct assistant. Output only single digits without explanation."
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\nThe score is: "
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [eval_image]
        }
    }


def create_single_prompt(eval_image: Image.Image, eval_label: str,
                         model_name: str = '') -> dict:
    """Select prompt creation function based on model (single image no-reference mode)"""
    model_lower = model_name.lower()
    if 'internvl' in model_lower:
        return create_single_prompt_internvl(eval_image, eval_label, model_name)
    else:
        return create_single_prompt_qwen(eval_image, eval_label, model_name)


def batch_generate(llm, prompts: List[dict], batch_size: int = 8, model_name: str = '') -> List[str]:
    """Batch generation"""
    from vllm import SamplingParams
    
    stop_tokens = ['<|im_end|>', '\n']
    if 'internvl' in model_name.lower():
        stop_tokens = ['<|im_end|>', '<|endoftext|>', '\n']
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5,
        stop=stop_tokens
    )
    
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            all_outputs.append(generated_text)
    
    return all_outputs


def get_correct(pred: Dict) -> bool:
    """Get the correct field, compute if not present"""
    if 'correct' in pred:
        return pred['correct']
    return pred['true_label'] == pred['predicted_label']


def create_neighbor_tasks(neighbors_data: List[Dict], predictions: List[Dict],
                          label_names: List[str]) -> List[Dict]:
    """Create neighbor reference mode tasks"""
    # Build index -> prediction mapping
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
        
        # Self as reference (10th entry)
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
    """Create single image no-reference mode tasks"""
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
    
    samples_dict = defaultdict(lambda: {
        'neighbor_scores': [None] * 10,
        'metadata': {}
    })
    
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
        print(f'\nResults saved: {output_path}')


def main():
    args = parse_args()
    
    mode_desc = 'Neighbor reference mode' if args.mode == 'neighbor' else 'Single image no-reference mode'

    print('=' * 70)
    print('Image Classification Scoring (vLLM accelerated version)')
    print(f'   - Scoring mode: {mode_desc}')
    print(f'   - Tensor Parallel: {args.tensor_parallel_size} GPUs')
    print(f'   - Batch Size: {args.batch_size}')
    if args.num_shards > 1:
        print(f'   - Data shard: {args.shard + 1}/{args.num_shards}')
    print('=' * 70)
    if args.mode == 'neighbor':
        print(f'Neighbor file: {args.neighbors}')
    print(f'Prediction file: {args.predictions}')
    print(f'Output: {args.output}')
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}')
    print('=' * 70 + '\n')
    
    # Load vLLM model
    llm = load_vllm_model(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        mode=args.mode
    )
    
    max_image_size = args.max_image_size
    print(f'Images will be resized to max {max_image_size}x{max_image_size}\n')

    # Load dataset
    hf_dataset, image_field, label_names = load_hf_dataset(args.dataset)
    
    # Load prediction data
    predictions = load_predictions_json(args.predictions)
    print(f'Prediction data: {len(predictions)} entries')

    # ==================== Neighbor reference mode ====================
    if args.mode == 'neighbor':
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f'Neighbor data: {len(neighbors_data)} samples')
        
        if args.test:
            neighbors_data = neighbors_data[:args.test]
            print(f'Test mode: first {args.test} samples\n')

        # Data sharding
        if args.num_shards > 1:
            total = len(neighbors_data)
            shard_size = total // args.num_shards
            start = args.shard * shard_size
            end = total if args.shard == args.num_shards - 1 else start + shard_size
            neighbors_data = neighbors_data[start:end]
            print(f'Shard {args.shard + 1}/{args.num_shards}: [{start}, {end})\n')

        all_tasks = create_neighbor_tasks(neighbors_data, predictions, label_names)
        print(f'Total {len(all_tasks)} scoring tasks\n')

        print('=== Starting batch scoring (neighbor reference mode) ===\n')
        all_results = []
        
        for batch_start in tqdm(range(0, len(all_tasks), args.batch_size), desc='Batch progress'):
            batch_tasks = all_tasks[batch_start:batch_start + args.batch_size]
            
            batch_prompts = []
            for task in batch_tasks:
                try:
                    eval_image = resize_image(hf_dataset[task['sample_index']][image_field], max_image_size)
                    ref_image = resize_image(hf_dataset[task['neighbor_index']][image_field], max_image_size)
                    
                    prompt_data = create_neighbor_prompt(
                        ref_image, eval_image,
                        task['ref_label'], task['eval_label'],
                        model_name=args.model
                    )
                    batch_prompts.append(prompt_data)
                except Exception as e:
                    print(f'Warning: Failed to load image: {e}')
                    batch_prompts.append(None)
            
            valid_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
            valid_prompts = [batch_prompts[i] for i in valid_indices]
            
            if valid_prompts:
                try:
                    responses = batch_generate(llm, valid_prompts, len(valid_prompts), args.model)
                except Exception as e:
                    print(f'Warning: Batch generation failed: {e}')
                    responses = [''] * len(valid_prompts)
            else:
                responses = []
            
            response_idx = 0
            for i, task in enumerate(batch_tasks):
                response = responses[response_idx] if i in valid_indices else ''
                if i in valid_indices:
                    response_idx += 1
                
                score, is_valid, score_str = parse_score(response)
                
                all_results.append({
                    **task,
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'raw_response': response,
                })
            
            if (batch_start // args.batch_size + 1) % max(1, args.save_interval // args.batch_size) == 0:
                save_results(reorganize_neighbor_results(all_results), args.output)
        
        final_results = reorganize_neighbor_results(all_results)
        save_results(final_results, args.output, is_final=True)
    
    # ==================== Single image no-reference mode ====================
    else:
        if args.test:
            predictions = predictions[:args.test]
            print(f'Test mode: first {args.test} samples\n')

        # Data sharding
        if args.num_shards > 1:
            total = len(predictions)
            shard_size = total // args.num_shards
            start = args.shard * shard_size
            end = total if args.shard == args.num_shards - 1 else start + shard_size
            predictions = predictions[start:end]
            print(f'Shard {args.shard + 1}/{args.num_shards}: [{start}, {end})\n')

        all_tasks = create_single_tasks(predictions, label_names)
        print(f'Total {len(all_tasks)} scoring tasks\n')

        print('=== Starting batch scoring (single image no-reference mode) ===\n')
        all_results = []
        
        for batch_start in tqdm(range(0, len(all_tasks), args.batch_size), desc='Batch progress'):
            batch_tasks = all_tasks[batch_start:batch_start + args.batch_size]
            
            batch_prompts = []
            for task in batch_tasks:
                try:
                    eval_image = resize_image(hf_dataset[task['sample_index']][image_field], max_image_size)
                    
                    prompt_data = create_single_prompt(
                        eval_image, task['eval_label'],
                        model_name=args.model
                    )
                    batch_prompts.append(prompt_data)
                except Exception as e:
                    print(f'Warning: Failed to load image: {e}')
                    batch_prompts.append(None)
            
            valid_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
            valid_prompts = [batch_prompts[i] for i in valid_indices]
            
            if valid_prompts:
                try:
                    responses = batch_generate(llm, valid_prompts, len(valid_prompts), args.model)
                except Exception as e:
                    print(f'Warning: Batch generation failed: {e}')
                    responses = [''] * len(valid_prompts)
            else:
                responses = []
            
            response_idx = 0
            for i, task in enumerate(batch_tasks):
                response = responses[response_idx] if i in valid_indices else ''
                if i in valid_indices:
                    response_idx += 1
                
                score, is_valid, score_str = parse_score(response)
                
                all_results.append({
                    **task,
                    'score': score,
                    'score_str': score_str,
                    'is_valid': is_valid,
                    'raw_response': response,
                })
            
            if (batch_start // args.batch_size + 1) % max(1, args.save_interval // args.batch_size) == 0:
                save_results(reorganize_single_results(all_results), args.output)
        
        final_results = reorganize_single_results(all_results)
        save_results(final_results, args.output, is_final=True)
    
    # Statistics
    all_scores = [r['score'] for r in all_results if r.get('is_valid')]
    if all_scores:
        print(f'\nStatistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate (>=3): {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')


if __name__ == '__main__':
    main()
