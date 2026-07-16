"""
Text Classification Task: Scoring using neighbor information (vLLM accelerated version)
- Evaluates whether current sample's predicted category is reasonable based on neighbor text and predicted categories
- Uses vLLM for efficient batch inference
- Supports AG_News and MMLU datasets
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
from collections import defaultdict


# AG_News categories
AGNEWS_LABELS = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Sci/Tech"
}

# ==================== Neighbor reference mode prompts ====================

# AG_News neighbor scoring prompt
AGNEWS_NEIGHBOR_PROMPT = '''Evaluate the classification against the reference. Be STRICT.

Reference (verified correct):
Text: {ref_text}
Category: {ref_category}

To Evaluate:
Text: {eval_text}
Category: {eval_category}

Scoring Rules:
For SIMILAR texts (same topic/domain): Categories should match.
- Score 5: Categories match exactly
- Score 4: Categories match, minor text ambiguity
- Score 3: Categories closely related (e.g., Business/Sci-Tech for tech companies)
- Score 2: Categories differ - suspicious
- Score 1: Categories differ significantly - likely wrong
- Score 0: Categories clearly contradict

For DIFFERENT texts: Learn from reference quality, apply same standard.
- Score 5: Classification as clear and correct as reference
- Score 4: Correct, nearly as clear as reference
- Score 3: Reasonable but less clear than reference
- Score 2: Questionable compared to reference standard
- Score 1: Likely wrong
- Score 0: Clearly incorrect

Return ONLY one digit (0-5):'''

# MMLU neighbor scoring prompt
# MMLU_NEIGHBOR_PROMPT = '''Evaluate the answer against the reference. Be STRICT.

# Reference (verified correct):
# Question: {ref_text}
# Choices: {ref_choices}
# Answer: {ref_category}

# To Evaluate:
# Question: {eval_text}
# Choices: {eval_choices}
# Answer: {eval_category}

# Scoring Rules:
# For SIMILAR domain questions: Answers should follow similar reasoning.
# - Score 5: Answer aligns perfectly with domain knowledge
# - Score 4: Answer correct based on domain knowledge
# - Score 3: Reasonable but some uncertainty
# - Score 2: Questionable given the domain
# - Score 1: Contradicts domain knowledge patterns
# - Score 0: Clearly wrong

# For DIFFERENT domain questions: Learn from reference quality, apply same standard.
# - Score 5: Answer as clearly correct as reference
# - Score 4: Correct, nearly as clear as reference
# - Score 3: Reasonable but less certain than reference
# - Score 2: Questionable compared to reference standard
# - Score 1: Likely wrong
# - Score 0: Clearly incorrect

# Return ONLY one digit (0-5):'''
MMLU_NEIGHBOR_PROMPT = '''Evaluate the answer against the reference. Be STRICT.

Reference (verified correct, usefulness unknown):
Question: {ref_text}
Choices: {ref_choices}
Answer: {ref_category}

To Evaluate:
Question: {eval_text}
Choices: {eval_choices}
Answer: {eval_category}

IMPORTANT:
- The reference question MAY be related or completely unrelated.
- First assess whether the reference is semantically relevant to the evaluated question.
- If relevant, it may be used to calibrate strictness and domain expectations.
- If irrelevant, IGNORE its content and use it ONLY as a score-scale anchor.

General Rules:
- This is a multiple-choice task. The answer must map to ONE option (A/B/C/D) or exactly one choice text.
- Do NOT infer correctness from style, confidence, or similarity alone.

Scoring Rules (0–5):

If the reference is RELEVANT:
- 5: Answer is clearly correct and consistent with domain knowledge.
- 4: Correct with minor ambiguity.
- 3: Reasonable but some uncertainty.
- 2: Questionable given domain knowledge.
- 1: Likely incorrect.
- 0: Clearly wrong.

If the reference is IRRELEVANT:
- Be CONSERVATIVE. Use the reference ONLY to calibrate score strictness.
- 5: Clearly and unambiguously correct; no reasonable doubt.
- 4: Very likely correct, minor ambiguity.
- 3: Plausible but uncertain (default when unsure).
- 2: Likely incorrect.
- 1: Very likely incorrect.
- 0: Clearly wrong or invalid mapping.

Return ONLY a number (0–5).
Score:
'''

# ==================== Single sample no-reference mode prompts ====================

# AG_News single sample scoring prompt
AGNEWS_SINGLE_PROMPT = '''Evaluate the news classification prediction.

Categories: World, Sports, Business, Sci/Tech

Text: {eval_text}
Predicted Category: {eval_category}

Scoring Rules:
- Score 5: Prediction is clearly correct - text unambiguously belongs to this category
- Score 4: Prediction is correct - text fits this category well
- Score 3: Prediction is reasonable - text mostly fits, but has some ambiguity
- Score 2: Prediction is questionable - another category might be more appropriate
- Score 1: Prediction is likely wrong - text doesn't fit this category well
- Score 0: Prediction is wrong - text clearly belongs to a different category

Return ONLY one digit (0-5):'''

# MMLU single sample scoring prompt
MMLU_SINGLE_PROMPT = '''Evaluate the multiple-choice answer.

Question: {eval_text}
Choices: {eval_choices}
Predicted Answer: {eval_category}

Scoring Rules:
- Score 5: Answer is clearly correct - definitely the right choice
- Score 4: Answer is correct - the right choice with high confidence
- Score 3: Answer is reasonable - likely correct but some uncertainty
- Score 2: Answer is questionable - may not be the correct choice
- Score 1: Answer is likely wrong - another choice seems better
- Score 0: Answer is wrong - clearly incorrect

Return ONLY one digit (0-5):'''




def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification Scoring (vLLM accelerated version)')

    # Scoring mode
    parser.add_argument('--mode', type=str, default='neighbor', choices=['neighbor', 'single'],
                       help='Scoring mode: neighbor=neighbor reference mode (default), single=no-reference single sample mode')
    
    parser.add_argument('--neighbors', type=str, default=None,
                       help='Neighbor file path (JSONL), required for neighbor mode')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Prediction results file path (JSON, containing input_text)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ag_news', 'mmlu'],
                       help='Dataset type')
    parser.add_argument('--save-interval', '-s', type=int, default=500,
                       help='Save interval (number of tasks)')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    # vLLM parameters
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization')
    parser.add_argument('--max-model-len', type=int, default=8192,
                       help='Maximum model length')
    # Data sharding parameters
    parser.add_argument('--shard', type=int, default=0,
                       help='Current shard ID (0-based)')
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of shards')
    
    return parser.parse_args()


def load_vllm_model(model_name: str, tensor_parallel_size: int = 1,
                    gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
    """Load vLLM model"""
    from vllm import LLM
    from transformers import AutoTokenizer
    
    print(f'Loading vLLM model: {model_name}')
    print(f'  - Tensor Parallel: {tensor_parallel_size}')
    print(f'  - GPU Memory: {gpu_memory_utilization * 100:.0f}%')
    print(f'  - Max Model Len: {max_model_len}')
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    
    # Load tokenizer (for formatting prompts)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f'vLLM model loaded\n')
    return llm, tokenizer


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    """Load neighbor data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_predictions_json(file_path: str) -> Dict[int, Dict]:
    """Load prediction results (JSON format, containing input_text)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pred_map = {}
    for item in data:
        idx = item.get('index', item.get('idx'))
        pred_map[idx] = item
    
    return pred_map


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """Parse score (consistent with LLM evaluator)"""
    response = response.strip()
    
    # Method 1: Directly parse first character (more lenient, handles "4\n" "4." "4 " etc.)
    if response and response[0].isdigit():
        score = int(response[0])
        if 0 <= score <= 5:
            return score, True, str(score)
    
    # Method 2: Regex extraction
    match = re.search(r'\b([0-5])\b', response)
    if match:
        score = int(match.group(1))
        return score, True, str(score)
    
    return None, False, 'INVALID'


def create_neighbor_prompt(task: Dict) -> str:
    """Create neighbor reference mode prompt"""
    dataset = task.get('dataset', 'ag_news')
    
    if dataset == 'ag_news':
        return AGNEWS_NEIGHBOR_PROMPT.format(
            ref_text=task['ref_text'],
            ref_category=task['ref_category'],
            eval_text=task['eval_text'],
            eval_category=task['eval_category']
        )
    else:  # mmlu
        return MMLU_NEIGHBOR_PROMPT.format(
            ref_text=task['ref_text'],
            ref_choices=', '.join(str(c) for c in task.get('ref_choices', [])),
            ref_category=task['ref_category'],
            eval_text=task['eval_text'],
            eval_choices=', '.join(str(c) for c in task.get('eval_choices', [])),
            eval_category=task['eval_category']
        )


def create_single_prompt(task: Dict) -> str:
    """Create single sample no-reference mode prompt"""
    dataset = task.get('dataset', 'ag_news')
    
    if dataset == 'ag_news':
        return AGNEWS_SINGLE_PROMPT.format(
            eval_text=task['eval_text'],
            eval_category=task['eval_category']
        )
    else:  # mmlu
        return MMLU_SINGLE_PROMPT.format(
            eval_text=task['eval_text'],
            eval_choices=', '.join(str(c) for c in task.get('eval_choices', [])),
            eval_category=task['eval_category']
        )


def format_prompt_for_model(prompt: str, model_name: str, tokenizer=None) -> str:
    """Format prompt based on model (consistent with LLM evaluator)"""
    if 'llama' in model_name.lower():
        # Consistent with LLM evaluator: pass raw prompt directly, no chat template
        # Llama in completion mode tends to output answers directly
        return prompt
    elif 'qwen' in model_name.lower():
        # Qwen3 needs thinking mode disabled
        if tokenizer is not None:
            try:
                messages = [{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking mode!
                )
            except Exception:
                pass
        # Fallback format
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif 'ministral' in model_name.lower():
        return f"[INST] {prompt} [/INST]"
    else:
        return prompt


def batch_generate(llm, prompts: List[str], batch_size: int = 32, 
                   model_name: str = '', tokenizer=None) -> List[str]:
    """Batch generation"""
    from vllm import SamplingParams

    # Format prompts
    formatted_prompts = [format_prompt_for_model(p, model_name, tokenizer) for p in prompts]
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5,  # Increased to 5, consistent with LLM evaluator
        stop=['\n', '<|im_end|>', '<|eot_id|>', '</s>']
    )
    
    all_outputs = []
    
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            all_outputs.append(generated_text)
    
    return all_outputs


def create_neighbor_tasks(neighbors_data: List[Dict], pred_map: Dict[int, Dict],
                          dataset: str) -> List[Dict]:
    """Create all scoring tasks for neighbor reference mode"""
    tasks = []
    
    for sample in neighbors_data:
        sample_idx = sample['index']
        
        if sample_idx not in pred_map:
            continue
        
        sample_pred = pred_map[sample_idx]
        eval_text = sample_pred.get('input_text', '')
        
        if dataset == 'ag_news':
            eval_category = sample_pred.get('predicted_label_name', 
                                            AGNEWS_LABELS.get(sample_pred.get('predicted_label', -1), 'Unknown'))
            eval_choices = None
        else:  # mmlu
            eval_category = sample_pred.get('predicted_answer', sample_pred.get('predicted_label_name', 'Unknown'))
            eval_choices = sample_pred.get('choices', [])
        
        # 9 neighbors
        for rank, neighbor in enumerate(sample['neighbors']):
            neighbor_idx = neighbor['index']
            
            if neighbor_idx not in pred_map:
                continue
            
            neighbor_pred = pred_map[neighbor_idx]
            ref_text = neighbor_pred.get('input_text', '')
            
            if dataset == 'ag_news':
                ref_category = neighbor_pred.get('predicted_label_name',
                                                AGNEWS_LABELS.get(neighbor_pred.get('predicted_label', -1), 'Unknown'))
                ref_choices = None
            else:  # mmlu
                ref_category = neighbor_pred.get('predicted_answer', neighbor_pred.get('predicted_label_name', 'Unknown'))
                ref_choices = neighbor_pred.get('choices', [])
            
            tasks.append({
                'sample_index': sample_idx,
                'sample_true_label': sample_pred.get('true_label'),
                'sample_predicted_label': sample_pred.get('predicted_label'),
                'sample_correct': sample_pred.get('correct', sample_pred.get('predicted_label') == sample_pred.get('true_label')),
                'eval_text': eval_text,
                'eval_category': eval_category,
                'eval_choices': eval_choices,
                'neighbor_rank': rank,
                'neighbor_index': neighbor_idx,
                'neighbor_cosine': neighbor['cosine'],
                'ref_text': ref_text,
                'ref_category': ref_category,
                'ref_choices': ref_choices,
                'is_self_ref': False,
                'dataset': dataset,
            })
        
        # Self as reference (10th entry)
        tasks.append({
            'sample_index': sample_idx,
            'sample_true_label': sample_pred.get('true_label'),
            'sample_predicted_label': sample_pred.get('predicted_label'),
            'sample_correct': sample_pred.get('correct', sample_pred.get('predicted_label') == sample_pred.get('true_label')),
            'eval_text': eval_text,
            'eval_category': eval_category,
            'eval_choices': eval_choices,
            'neighbor_rank': 9,
            'neighbor_index': sample_idx,
            'neighbor_cosine': 1.0,
            'ref_text': eval_text,
            'ref_category': eval_category,
            'ref_choices': eval_choices,
            'is_self_ref': True,
            'dataset': dataset,
        })
    
    return tasks


def create_single_tasks(pred_map: Dict[int, Dict], dataset: str, 
                        test_limit: Optional[int] = None) -> List[Dict]:
    """Create scoring tasks for single sample no-reference mode"""
    tasks = []
    
    indices = sorted(pred_map.keys())
    if test_limit:
        indices = indices[:test_limit]
    
    for sample_idx in indices:
        sample_pred = pred_map[sample_idx]
        eval_text = sample_pred.get('input_text', '')
        
        if dataset == 'ag_news':
            eval_category = sample_pred.get('predicted_label_name', 
                                            AGNEWS_LABELS.get(sample_pred.get('predicted_label', -1), 'Unknown'))
            eval_choices = None
        else:  # mmlu
            eval_category = sample_pred.get('predicted_answer', sample_pred.get('predicted_label_name', 'Unknown'))
            eval_choices = sample_pred.get('choices', [])
        
        tasks.append({
            'sample_index': sample_idx,
            'sample_true_label': sample_pred.get('true_label'),
            'sample_predicted_label': sample_pred.get('predicted_label'),
            'sample_correct': sample_pred.get('correct', sample_pred.get('predicted_label') == sample_pred.get('true_label')),
            'eval_text': eval_text,
            'eval_category': eval_category,
            'eval_choices': eval_choices,
            'dataset': dataset,
        })
    
    return tasks


def reorganize_neighbor_results(all_results: List[Dict]) -> List[Dict]:
    """Reorganize neighbor mode results by sample"""
    samples_dict = defaultdict(lambda: {
        'neighbor_scores': [None] * 10,
        'metadata': {}
    })
    
    for result in all_results:
        sample_idx = result['sample_index']
        rank = result['neighbor_rank']
        
        if not samples_dict[sample_idx]['metadata']:
            samples_dict[sample_idx]['metadata'] = {
                'index': sample_idx,
                'true_label': result['sample_true_label'],
                'predicted_label': result['sample_predicted_label'],
                'correct': result['sample_correct'],
            }
        
        samples_dict[sample_idx]['neighbor_scores'][rank] = {
            'neighbor_rank': rank,
            'neighbor_index': result['neighbor_index'],
            'neighbor_cosine': result.get('neighbor_cosine'),
            'ref_category': result.get('ref_category'),
            'score': result['score'],
            'score_str': result['score_str'],
            'is_valid': result['is_valid'],
            'raw_response': result.get('raw_response', ''),
            'is_self_ref': result.get('is_self_ref', False),
        }
    
    # Compute statistics
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
    """Organize single sample mode results"""
    final_results = []
    
    for result in all_results:
        final_results.append({
            'index': result['sample_index'],
            'true_label': result['sample_true_label'],
            'predicted_label': result['sample_predicted_label'],
            'correct': result['sample_correct'],
            'score': result['score'],
            'score_str': result['score_str'],
            'is_valid': result['is_valid'],
            'raw_response': result.get('raw_response', ''),
        })
    
    return final_results


def save_results(results, output_path, is_final=False):
    """Save results"""
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
        
        completed_keys = set()
        for sample in results:
            sample_idx = sample['index']
            for score_data in sample['neighbor_scores']:
                if score_data:
                    completed_keys.add((sample_idx, score_data['neighbor_rank']))
        
        print(f'Found existing results: {len(completed_keys)} tasks completed')
        return results, completed_keys
    except FileNotFoundError:
        return [], set()


def main():
    args = parse_args()
    
    # Validate arguments
    if args.mode == 'neighbor' and not args.neighbors:
        print('Error: neighbor mode requires --neighbors argument')
        return

    mode_str = 'Neighbor reference mode' if args.mode == 'neighbor' else 'Single sample no-reference mode'

    print('=' * 70)
    print(f'Text Classification Scoring (vLLM accelerated version) - {mode_str}')
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
    print(f'Mode: {args.mode}')
    print('=' * 70 + '\n')
    
    # Load vLLM model
    llm, tokenizer = load_vllm_model(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Load data
    pred_map = load_predictions_json(args.predictions)
    print(f'Prediction data: {len(pred_map)} entries')
    
    if args.mode == 'neighbor':
        # Neighbor mode
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f'Neighbor data: {len(neighbors_data)} samples\n')
        
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

        # Create neighbor tasks
        all_tasks = create_neighbor_tasks(neighbors_data, pred_map, args.dataset)
        print(f'Total {len(all_tasks)} scoring tasks\n')

        # Resume from checkpoint
        completed_keys = set()
        if args.resume:
            _, completed_keys = load_existing_results(args.output)
        
        # Filter completed tasks
        tasks_to_process = [
            t for t in all_tasks 
            if (t['sample_index'], t['neighbor_rank']) not in completed_keys
        ]
        
        create_prompt_fn = create_neighbor_prompt
        reorganize_fn = reorganize_neighbor_results
        
    else:
        # Single sample mode
        print()
        if args.test:
            print(f'Test mode: first {args.test} samples\n')
        
        all_tasks = create_single_tasks(pred_map, args.dataset, args.test)
        print(f'Total {len(all_tasks)} scoring tasks\n')

        # Resume from checkpoint (simplified for single sample mode)
        completed_keys = set()
        if args.resume:
            try:
                with open(args.output, 'r') as f:
                    existing = json.load(f)
                completed_keys = {r['index'] for r in existing}
                print(f'Found existing results: {len(completed_keys)} tasks completed')
            except FileNotFoundError:
                pass
        
        tasks_to_process = [t for t in all_tasks if t['sample_index'] not in completed_keys]
        
        create_prompt_fn = create_single_prompt
        reorganize_fn = reorganize_single_results
    
    print(f'Pending: {len(tasks_to_process)} tasks\n')

    if not tasks_to_process:
        print('All tasks completed!')
        return

    # Batch processing
    print('=== Starting batch scoring ===\n')
    
    all_results = []
    
    for batch_start in tqdm(range(0, len(tasks_to_process), args.batch_size), 
                            desc='Batch progress'):
        batch_tasks = tasks_to_process[batch_start:batch_start + args.batch_size]
        
        # Prepare batch prompts
        batch_prompts = [create_prompt_fn(task) for task in batch_tasks]
        
        # Batch generation
        try:
            responses = batch_generate(llm, batch_prompts, batch_size=len(batch_prompts),
                                       model_name=args.model, tokenizer=tokenizer)
        except Exception as e:
            print(f'Warning: Batch generation failed: {e}')
            responses = [''] * len(batch_prompts)
        
        # Process results
        for task, response in zip(batch_tasks, responses):
            score, is_valid, score_str = parse_score(response)
            
            result = {
                'sample_index': task['sample_index'],
                'sample_true_label': task['sample_true_label'],
                'sample_predicted_label': task['sample_predicted_label'],
                'sample_correct': task['sample_correct'],
                'eval_category': task['eval_category'],
                'score': score,
                'score_str': score_str,
                'is_valid': is_valid,
                'raw_response': response,
            }
            
            # Neighbor mode specific fields
            if args.mode == 'neighbor':
                result.update({
                    'neighbor_rank': task['neighbor_rank'],
                    'neighbor_index': task['neighbor_index'],
                    'neighbor_cosine': task['neighbor_cosine'],
                    'ref_category': task['ref_category'],
                    'is_self_ref': task.get('is_self_ref', False),
                })
            
            all_results.append(result)
        
        # Periodic save
        processed_count = batch_start + len(batch_tasks)
        if processed_count % args.save_interval < args.batch_size:
            reorganized = reorganize_fn(all_results)
            save_results(reorganized, args.output)
            tqdm.write(f'Saved {len(all_results)} tasks')

    # Final save
    final_results = reorganize_fn(all_results)
    save_results(final_results, args.output, is_final=True)
    
    # Statistics
    all_scores = [r['score'] for r in all_results if r['is_valid']]
    if all_scores:
        print(f'\nStatistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate (>=3): {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')


if __name__ == '__main__':
    main()
