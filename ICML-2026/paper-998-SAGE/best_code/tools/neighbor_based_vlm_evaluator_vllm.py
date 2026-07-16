"""
VLM Task: Scoring using image neighbor information (vLLM accelerated version)
- Uses vLLM for efficient inference
- Supports multi-GPU tensor parallelism
- Supports batch inference
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
from PIL import Image
import base64
from io import BytesIO


# vLLM two-image scoring prompt (neighbor reference mode)
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

# Single-image scoring prompt (no-reference mode)
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


def parse_args():
    parser = argparse.ArgumentParser(description='VLM Scoring (vLLM accelerated version)')
    
    # Scoring mode
    parser.add_argument('--mode', type=str, default='neighbor', choices=['neighbor', 'single'],
                       help='Scoring mode: neighbor=neighbor reference mode (default), single=no-reference single image mode')
    
    parser.add_argument('--neighbors', type=str, default=None,
                       help='Neighbor file path (required for neighbor mode)')
    parser.add_argument('--vlm-data', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, required=True, choices=['flickr30k', 'coco'])
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--ref-type', type=str, default='predicted', choices=['predicted', 'true', 'both'])
    parser.add_argument('--save-interval', '-s', type=int, default=100)
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    # vLLM parameters
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization')
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


def load_hf_dataset(dataset_name: str, split: str = 'val'):
    """Load dataset from HuggingFace (consistent with original version)"""
    from datasets import load_dataset
    
    if dataset_name.lower() == 'flickr30k':
        # Must use nlphuji/flickr30k, consistent with knn_image.py!
        print(f'Loading Flickr30k from HuggingFace (split={split})...')
        dataset = load_dataset('nlphuji/flickr30k', split=split, trust_remote_code=True)
    elif dataset_name.lower() == 'coco':
        print(f'Loading COCO-Caption from HuggingFace (split={split})...')
        # COCO: test -> validation (consistent with original version)
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('lmms-lab/COCO-Caption', split=split, trust_remote_code=True)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    print(f'✅ Loaded {len(dataset)} samples from {dataset_name}')
    return dataset


def load_vllm_model(model_name: str, tensor_parallel_size: int = 1, 
                    gpu_memory_utilization: float = 0.45, mode: str = 'neighbor'):
    """Load vLLM model"""
    from vllm import LLM
    
    print(f'Loading vLLM model: {model_name}')
    print(f'  - Tensor Parallel Size: {tensor_parallel_size}')
    print(f'  - GPU Memory Utilization: {gpu_memory_utilization}')
    print(f'  - Mode: {mode}')
    
    # Set image count limit based on mode
    max_images = 2 if mode == 'neighbor' else 1
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192*2,  # Single image needs ~6400 tokens
        # Limit image token count
        limit_mm_per_prompt={'image': max_images},
        # Reduce memory usage
        enforce_eager=True,  # Disable CUDA graph to reduce memory
        max_num_seqs=2,  # Reduce concurrent sequences to avoid OOM during profiling
    )
    
    print('✅ vLLM model loaded')
    return llm


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_vlm_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ref_caption(vlm_item: Dict, ref_type: str) -> str:
    if ref_type == 'predicted':
        return vlm_item.get('predicted_caption', '')
    elif ref_type == 'true':
        true_caps = vlm_item.get('true_captions', [])
        return true_caps[0] if true_caps else ''
    elif ref_type == 'both':
        pred = vlm_item.get('predicted_caption', '')
        true_caps = vlm_item.get('true_captions', [])
        return f'{pred} | {true_caps[0] if true_caps else ""}'
    return ''


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    response = response.strip()
    
    first_line = response.split('\n')[0].strip()
    try:
        score = int(first_line)
        if 0 <= score <= 5:
            return score, True, str(score)
    except ValueError:
        pass
    
    match = re.search(r'\b([0-5])\b', response)
    if match:
        return int(match.group(1)), True, match.group(1)
    
    return None, False, 'INVALID'


def image_to_base64(image: Image.Image) -> str:
    """Convert image to base64"""
    buffered = BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize to reduce tokens
    max_size = 384
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(buffered, format='JPEG', quality=80)
    return base64.b64encode(buffered.getvalue()).decode()


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


def create_vllm_prompt_qwen(ref_image: Image.Image, eval_image: Image.Image, 
                            ref_caption: str, predicted_caption: str,
                            model_name: str = '') -> dict:
    """
    Create vLLM prompt for Qwen VL
    Uses processor.apply_chat_template to ensure format consistency with original version
    """
    prompt_text = CAPTION_EVALUATION_PROMPT.format(
        ref_caption=ref_caption,
        predicted_caption=predicted_caption
    )
    
    # Use the same approach as original: processor.apply_chat_template
    # This automatically generates the correct token format
    processor = get_processor(model_name)
    
    # Build message format (consistent with original _generate_qwen_two_images)
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': ref_image},   # Image 1 - Reference image
            {'type': 'image', 'image': eval_image},  # Image 2 - Image to evaluate
            {'type': 'text', 'text': prompt_text}
        ]
    }]
    
    # Use apply_chat_template to generate correctly formatted prompt
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


def create_vllm_prompt_internvl(ref_image: Image.Image, eval_image: Image.Image, 
                                ref_caption: str, predicted_caption: str,
                                model_name: str = '') -> dict:
    """
    Create vLLM prompt for InternVL
    InternVL uses a different chat template format
    """
    prompt_text = CAPTION_EVALUATION_PROMPT.format(
        ref_caption=ref_caption,
        predicted_caption=predicted_caption
    )
    
    # InternVL prompt format: uses <image> placeholders
    # Format reference: https://huggingface.co/OpenGVLab/InternVL3_5-8B
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [ref_image, eval_image]
        }
    }


def create_vllm_prompt(ref_image: Image.Image, eval_image: Image.Image,
                       ref_caption: str, predicted_caption: str,
                       model_name: str = '') -> dict:
    """
    Select the correct prompt creation function based on model type (neighbor reference mode)
    """
    model_lower = model_name.lower()
    
    if 'internvl' in model_lower:
        return create_vllm_prompt_internvl(
            ref_image, eval_image, ref_caption, predicted_caption, model_name
        )
    else:
        # Qwen and other models use the original approach
        return create_vllm_prompt_qwen(
            ref_image, eval_image, ref_caption, predicted_caption, model_name
        )


# ==================== Single image mode (no reference) ====================

def create_single_prompt_qwen(eval_image: Image.Image, predicted_caption: str,
                               model_name: str = '') -> dict:
    """
    Create single-image scoring prompt for Qwen VL (no-reference mode)
    """
    prompt_text = SINGLE_IMAGE_EVALUATION_PROMPT.format(
        predicted_caption=predicted_caption
    )
    
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


def create_single_prompt_internvl(eval_image: Image.Image, predicted_caption: str,
                                   model_name: str = '') -> dict:
    """
    Create single-image scoring prompt for InternVL (no-reference mode)
    """
    prompt_text = SINGLE_IMAGE_EVALUATION_PROMPT.format(
        predicted_caption=predicted_caption
    )
    
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    return {
        'prompt': prompt,
        'multi_modal_data': {
            'image': [eval_image]
        }
    }


def create_single_prompt(eval_image: Image.Image, predicted_caption: str,
                         model_name: str = '') -> dict:
    """
    Select the correct single-image prompt creation function based on model type (no-reference mode)
    """
    model_lower = model_name.lower()
    
    if 'internvl' in model_lower:
        return create_single_prompt_internvl(eval_image, predicted_caption, model_name)
    else:
        return create_single_prompt_qwen(eval_image, predicted_caption, model_name)


def batch_generate(llm, prompts: List[dict], batch_size: int = 8, model_name: str = '') -> List[str]:
    """Batch generation"""
    from vllm import SamplingParams
    
    # Set stop tokens based on model type
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


def create_all_tasks(neighbors_data: List[Dict], vlm_map: Dict, 
                     ref_type: str) -> List[Dict]:
    """Create all scoring tasks (neighbor reference mode)"""
    tasks = []
    
    for sample in neighbors_data:
        sample_id = sample['global_id']
        sample_idx = sample.get('index', sample_id)
        sample_vlm = vlm_map[sample_id]
        predicted_caption = sample_vlm.get('predicted_caption', '')
        true_captions = sample_vlm.get('true_captions', [])
        
        # 9 neighbors
        for rank, neighbor in enumerate(sample['neighbors']):
            neighbor_id = neighbor['global_id']
            neighbor_idx = neighbor.get('index', neighbor_id)
            neighbor_vlm = vlm_map[neighbor_id]
            ref_caption = get_ref_caption(neighbor_vlm, ref_type)
            
            tasks.append({
                'sample_global_id': sample_id,
                'sample_index': sample_idx,
                'neighbor_rank': rank,
                'neighbor_global_id': neighbor_id,
                'neighbor_index': neighbor_idx,
                'neighbor_cosine': neighbor['cosine'],
                'predicted_caption': predicted_caption,
                'true_captions': true_captions,
                'ref_caption': ref_caption,
                'is_self_ref': False,
            })
        
        # Self as reference (10th entry)
        self_ref_caption = get_ref_caption(sample_vlm, ref_type)
        tasks.append({
            'sample_global_id': sample_id,
            'sample_index': sample_idx,
            'neighbor_rank': 9,
            'neighbor_global_id': sample_id,
            'neighbor_index': sample_idx,
            'neighbor_cosine': 1.0,
            'predicted_caption': predicted_caption,
            'true_captions': true_captions,
            'ref_caption': self_ref_caption,
            'is_self_ref': True,
        })
    
    return tasks


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
    """Organize single image mode results"""
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
    
    # Compute statistics
    final_results = []
    for sample_id in sorted(samples_dict.keys()):
        sample_data = samples_dict[sample_id]
        scores = sample_data['neighbor_scores']
        
        valid_scores = [s['score'] for s in scores if s and s['is_valid']]
        
        final_results.append({
            **sample_data['metadata'],
            'neighbor_scores': scores,
            'stats': {
                'total_scores': sum(1 for s in scores if s),
                'valid_scores_count': len(valid_scores),
                'passing_count': sum(1 for s in valid_scores if s >= 3),
                'avg_score': sum(valid_scores) / len(valid_scores) if valid_scores else None,
            }
        })
    
    return final_results


def save_results(results, output_path, is_final=False):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f'\nResults saved: {output_path}')


def load_existing_results(output_path) -> Tuple[List[Dict], set]:
    """Load existing results"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Extract completed tasks from sample results
        completed_keys = set()
        for sample in results:
            sample_id = sample['global_id']
            for score_data in sample['neighbor_scores']:
                if score_data:
                    completed_keys.add((sample_id, score_data['neighbor_rank']))
        
        print(f'Found existing results: {len(completed_keys)} tasks completed')
        return results, completed_keys
    except FileNotFoundError:
        return [], set()


def main():
    args = parse_args()
    
    mode_desc = 'Neighbor reference mode' if args.mode == 'neighbor' else 'Single image no-reference mode'

    print('=' * 70)
    print('VLM Caption Scoring (vLLM accelerated version)')
    print('   - Using vLLM for efficient inference')
    print(f'   - Scoring mode: {mode_desc}')
    print(f'   - Tensor Parallel: {args.tensor_parallel_size} GPUs')
    print(f'   - Batch Size: {args.batch_size}')
    if args.num_shards > 1:
        print(f'   - Data shard: {args.shard + 1}/{args.num_shards}')
    print('=' * 70)
    if args.mode == 'neighbor':
        print(f'Neighbor file: {args.neighbors}')
    print(f'VLM data: {args.vlm_data}')
    print(f'Output: {args.output}')
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset} (split={args.split})')
    print('=' * 70 + '\n')
    
    # Load vLLM model
    llm = load_vllm_model(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mode=args.mode
    )
    
    # Load dataset
    hf_dataset = load_hf_dataset(args.dataset, args.split)

    # Load data
    vlm_data = load_vlm_data(args.vlm_data)
    vlm_map = {i: item for i, item in enumerate(vlm_data)}
    
    print(f'VLM data: {len(vlm_data)} entries\n')

    # ==================== Neighbor reference mode ====================
    if args.mode == 'neighbor':
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f'Neighbor data: {len(neighbors_data)} samples')
    
    if args.test:
        neighbors_data = neighbors_data[:args.test]
        print(f'Test mode: first {args.test} samples\n')

    # Data sharding
    if args.num_shards > 1:
        total_samples = len(neighbors_data)
        shard_size = total_samples // args.num_shards
        start_idx = args.shard * shard_size
        if args.shard == args.num_shards - 1:
                end_idx = total_samples
        else:
            end_idx = start_idx + shard_size
        neighbors_data = neighbors_data[start_idx:end_idx]
        print(f'Shard {args.shard + 1}/{args.num_shards}: samples [{start_idx}, {end_idx}) total {len(neighbors_data)}\n')

    # Create all tasks
    all_tasks = create_all_tasks(neighbors_data, vlm_map, args.ref_type)
    print(f'Total {len(all_tasks)} scoring tasks\n')

    # Resume from checkpoint
    completed_keys = set()
    if args.resume:
            _, completed_keys = load_existing_results(args.output)
    
    # Filter completed tasks
    tasks_to_process = [
        t for t in all_tasks 
        if (t['sample_global_id'], t['neighbor_rank']) not in completed_keys
    ]
    print(f'Pending: {len(tasks_to_process)} tasks\n')

    if not tasks_to_process:
        print('All tasks completed!')
        return

    # Batch processing
        print('=== Starting batch scoring (neighbor reference mode) ===\n')
    
    all_results = []
    
    for batch_start in tqdm(range(0, len(tasks_to_process), args.batch_size), 
                            desc='Batch progress'):
        batch_tasks = tasks_to_process[batch_start:batch_start + args.batch_size]
        
        batch_prompts = []
        for task in batch_tasks:
            try:
                eval_image = hf_dataset[task['sample_index']]['image']
                ref_image = hf_dataset[task['neighbor_index']]['image']
                
                prompt_data = create_vllm_prompt(
                    ref_image, eval_image,
                    task['ref_caption'], task['predicted_caption'],
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
                responses = batch_generate(llm, valid_prompts, batch_size=len(valid_prompts), model_name=args.model)
            except Exception as e:
                print(f'Warning: Batch generation failed: {e}')
                responses = [''] * len(valid_prompts)
        else:
            responses = []
        
        response_idx = 0
        for i, task in enumerate(batch_tasks):
            if i in valid_indices:
                response = responses[response_idx]
                response_idx += 1
            else:
                response = ''
            
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
        
        if (batch_start // args.batch_size + 1) % (args.save_interval // args.batch_size + 1) == 0:
            reorganized = reorganize_results_by_sample(all_results)
            save_results(reorganized, args.output)
            print(f'\nSaved {len(all_results)} tasks')
    
    final_results = reorganize_results_by_sample(all_results)
    save_results(final_results, args.output, is_final=True)
    
    all_scores = [r['score'] for r in all_results if r['is_valid']]
    if all_scores:
        print(f'\nStatistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate: {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')

    # ==================== Single image no-reference mode ====================
    else:
        if args.test:
            vlm_data = vlm_data[:args.test]
            print(f'Test mode: first {args.test} samples\n')

        # Data sharding
        if args.num_shards > 1:
            total_samples = len(vlm_data)
            shard_size = total_samples // args.num_shards
            start_idx = args.shard * shard_size
            if args.shard == args.num_shards - 1:
                end_idx = total_samples
            else:
                end_idx = start_idx + shard_size
            vlm_data = vlm_data[start_idx:end_idx]
            print(f'Shard {args.shard + 1}/{args.num_shards}: samples [{start_idx}, {end_idx}) total {len(vlm_data)}\n')

        # Create tasks
        all_tasks = create_single_tasks(vlm_data)
        print(f'Total {len(all_tasks)} scoring tasks\n')

        # Resume from checkpoint (single image mode)
        completed_ids = set()
        if args.resume:
            try:
                with open(args.output, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                completed_ids = {item['global_id'] for item in existing}
                print(f'Found existing results: {len(completed_ids)} samples completed')
            except FileNotFoundError:
                pass
        
        tasks_to_process = [t for t in all_tasks if t['sample_global_id'] not in completed_ids]
        print(f'Pending: {len(tasks_to_process)} tasks\n')

        if not tasks_to_process:
            print('All tasks completed!')
            return

        # Batch processing
        print('=== Starting batch scoring (single image no-reference mode) ===\n')
        
        all_results = []
        
        for batch_start in tqdm(range(0, len(tasks_to_process), args.batch_size), 
                                desc='Batch progress'):
            batch_tasks = tasks_to_process[batch_start:batch_start + args.batch_size]
            
            batch_prompts = []
            for task in batch_tasks:
                try:
                    eval_image = hf_dataset[task['sample_index']]['image']
                    
                    prompt_data = create_single_prompt(
                        eval_image, task['predicted_caption'],
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
                    responses = batch_generate(llm, valid_prompts, batch_size=len(valid_prompts), model_name=args.model)
                except Exception as e:
                    print(f'Warning: Batch generation failed: {e}')
                    responses = [''] * len(valid_prompts)
            else:
                responses = []
            
            response_idx = 0
            for i, task in enumerate(batch_tasks):
                if i in valid_indices:
                    response = responses[response_idx]
                    response_idx += 1
                else:
                    response = ''
                
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
            
            if (batch_start // args.batch_size + 1) % (args.save_interval // args.batch_size + 1) == 0:
                final_results = reorganize_single_results(all_results)
                save_results(final_results, args.output)
                print(f'\nSaved {len(all_results)} tasks')
        
        final_results = reorganize_single_results(all_results)
        save_results(final_results, args.output, is_final=True)
        
    all_scores = [r['score'] for r in all_results if r['is_valid']]
    if all_scores:
        print(f'\nStatistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate: {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')


if __name__ == '__main__':
    main()

