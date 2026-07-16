"""
Text Classification task: scoring using neighbor information
- Evaluate whether the current sample's predicted category is reasonable based on the neighbor's text and predicted category
- Supports AG_News dataset
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

from config import MODEL_CONFIG
from models import (
    LlamaGenerator,
    Qwen3Generator,
    MinistralGenerator
)

# Model class mapping
MODEL_CLASS_MAP = {
    "llama": LlamaGenerator,
    "qwen": Qwen3Generator,
    "qwen3": Qwen3Generator,
    "ministral": MinistralGenerator,
}

# AG_News categories
AGNEWS_LABELS = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Sci/Tech"
}

# AG_News scoring Prompt
AGNEWS_EVALUATION_PROMPT = '''You are evaluating a news article classification. The categories are: World, Sports, Business, Sci/Tech.

[Reference Example]
Text: {ref_text}
Category: {ref_category}

[Text to Evaluate]
Text: {eval_text}
Predicted Category: {eval_category}

Task: Evaluate whether "{eval_category}" is a correct category for the text to evaluate.

Important: The reference text is semantically similar but may belong to a DIFFERENT category. Judge the evaluation text on its own content.

Scoring Rules:
- Score 5: Excellent - The predicted category perfectly matches the text content
- Score 4: Good - The category is appropriate, captures the main topic
- Score 3: Acceptable - The category is reasonable but another might fit better
- Score 2: Poor - The category seems mismatched with the text content
- Score 1: Very Poor - The category is clearly inappropriate
- Score 0: Wrong - The category is completely incorrect

Return ONLY a single digit (0, 1, 2, 3, 4, or 5) with NO explanation.

Score:'''

# MMLU scoring Prompt
MMLU_EVALUATION_PROMPT = '''You are evaluating a multiple-choice question answer.

[Reference Example]
Question: {ref_text}
Choices: {ref_choices}
Answer: {ref_category}

[Question to Evaluate]
Question: {eval_text}
Choices: {eval_choices}
Predicted Answer: {eval_category}

Task: Evaluate whether the predicted answer "{eval_category}" is correct for the question.

Important: The reference question is semantically similar but may have a DIFFERENT correct answer. Judge the evaluation question independently based on the question content and choices.

Scoring Rules:
- Score 5: Excellent - The answer is definitely correct
- Score 4: Good - The answer is most likely correct
- Score 3: Acceptable - The answer could be correct
- Score 2: Poor - The answer seems incorrect
- Score 1: Very Poor - The answer is likely wrong
- Score 0: Wrong - The answer is definitely incorrect

Return ONLY a single digit (0, 1, 2, 3, 4, or 5) with NO explanation.

Score:'''


def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification neighbor scoring')

    parser.add_argument('--neighbors', type=str, required=True,
                       help='Neighbor file path (JSONL)')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Prediction results file path (JSON, containing input_text)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Scoring model name (e.g., llama3.1-8b)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ag_news', 'mmlu'],
                       help='Dataset type')
    parser.add_argument('--save-interval', '-s', type=int, default=100,
                       help='Save interval')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process the first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--max-text-len', type=int, default=10000,
                       help='Maximum text length (truncation), default 10000 means no truncation')
    
    return parser.parse_args()


def load_model(model_name: str):
    """Load scoring model"""
    # Find model configuration
    model_config = None
    
    for task_type, configs in MODEL_CONFIG.items():
        for name, cfg in configs.items():
            if name == model_name or cfg.get('short_name') == model_name:
                model_config = cfg
                break
        if model_config:
            break
    
    if not model_config:
        raise ValueError(f"Model configuration not found: {model_name}")

    # Get model type from configuration
    model_type = model_config.get('model_type')
    print(f"Loading model: {model_name} (type: {model_type})")
    
    # Get model class
    model_class = MODEL_CLASS_MAP.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create model instance, scoring task only needs to output 1 digit
    model = model_class({
        **model_config,
        'model_name': model_config['model_name'],
        'model_type': model_type,
        'max_new_tokens': 2,  # Scoring only needs a single digit 0-5
    })
    
    return model


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    """Load neighbor data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_predictions_json(file_path: str) -> Dict[int, Dict]:
    """Load prediction results (containing input_text)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pred_map = {}
    for item in data:
        pred_map[item['index']] = item
    return pred_map


def truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """Parse score"""
    response = response.strip()

    # Try to directly parse the first line
    first_line = response.split('\n')[0].strip()
    try:
        score = int(first_line)
        if 0 <= score <= 5:
            return score, True, str(score)
    except ValueError:
        pass
    
    # Try regex matching
    match = re.search(r'\b([0-5])\b', response)
    if match:
        return int(match.group(1)), True, match.group(1)
    
    return None, False, 'INVALID'


def create_evaluation_tasks(neighbors_data: List[Dict], pred_map: Dict[int, Dict],
                           max_text_len: int, dataset: str) -> List[Dict]:
    """Create scoring tasks"""
    tasks = []
    
    for sample in neighbors_data:
        sample_idx = sample['index']
        
        if sample_idx not in pred_map:
            continue
        
        sample_pred = pred_map[sample_idx]
        eval_text = truncate_text(sample_pred.get('input_text', ''), max_text_len)
        
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
            ref_text = truncate_text(neighbor_pred.get('input_text', ''), max_text_len)
            
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
        
        # Self as reference (the 10th)
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


def generate_score(model, task: Dict) -> Dict:
    """Generate a single score"""
    dataset = task.get('dataset', 'ag_news')
    
    if dataset == 'ag_news':
        prompt = AGNEWS_EVALUATION_PROMPT.format(
            ref_text=task['ref_text'],
            ref_category=task['ref_category'],
            eval_text=task['eval_text'],
            eval_category=task['eval_category']
        )
    else:  # mmlu
        prompt = MMLU_EVALUATION_PROMPT.format(
            ref_text=task['ref_text'],
            ref_choices=', '.join(str(c) for c in task.get('ref_choices', [])),
            ref_category=task['ref_category'],
            eval_text=task['eval_text'],
            eval_choices=', '.join(str(c) for c in task.get('eval_choices', [])),
            eval_category=task['eval_category']
        )
    
    try:
        response = model.generate(prompt)
        score, is_valid, score_str = parse_score(response)
    except Exception as e:
        response = f"ERROR: {e}"
        score, is_valid, score_str = None, False, 'ERROR'
    
    return {
        **task,
        'score': score,
        'score_str': score_str,
        'is_valid': is_valid,
        'raw_response': response,
    }


def reorganize_results_by_sample(all_results: List[Dict]) -> List[Dict]:
    """Reorganize results by sample"""
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
            'ref_category': result.get('ref_category', ''),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'raw_response': result.get('raw_response', ''),
            'is_self_ref': result.get('is_self_ref', False),
        }
    
    # Calculate statistics
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


def save_results(results, output_path, is_final=False):
    """Save results"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f'\n✅ Results saved: {output_path}')


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
        
        print(f'📂 Found existing results: {len(completed_keys)} tasks completed')
        return results, completed_keys
    except FileNotFoundError:
        return [], set()


def main():
    args = parse_args()
    
    print('=' * 70)
    print('🔤 Text Classification Neighbor Scoring')
    print('=' * 70)
    print(f'📥 Neighbor file: {args.neighbors}')
    print(f'📥 Prediction file: {args.predictions}')
    print(f'📤 Output: {args.output}')
    print(f'🤖 Scoring model: {args.model}')
    print('=' * 70 + '\n')
    
    # Load model
    model = load_model(args.model)

    # Load data
    neighbors_data = load_neighbors_jsonl(args.neighbors)
    pred_map = load_predictions_json(args.predictions)
    
    print(f'✅ Neighbor data: {len(neighbors_data)} samples')
    print(f'✅ Prediction data: {len(pred_map)} entries\n')
    
    if args.test:
        neighbors_data = neighbors_data[:args.test]
        print(f'🧪 Test mode: first {args.test} samples\n')

    # Create tasks
    all_tasks = create_evaluation_tasks(neighbors_data, pred_map, args.max_text_len, args.dataset)
    print(f'✅ Total {len(all_tasks)} scoring tasks\n')

    # Resume from checkpoint
    completed_keys = set()
    if args.resume:
        _, completed_keys = load_existing_results(args.output)
    
    # Filter completed
    tasks_to_process = [
        t for t in all_tasks
        if (t['sample_index'], t['neighbor_rank']) not in completed_keys
    ]
    print(f'🔄 To process: {len(tasks_to_process)} tasks\n')

    if not tasks_to_process:
        print('✅ All tasks completed!')
        return
    
    # Process tasks
    print('=== Starting scoring ===\n')
    all_results = []
    
    for i, task in enumerate(tqdm(tasks_to_process, desc='Scoring progress')):
        result = generate_score(model, task)
        all_results.append(result)
        
        # Periodic save
        if (i + 1) % args.save_interval == 0:
            reorganized = reorganize_results_by_sample(all_results)
            save_results(reorganized, args.output)
            print(f'\n💾 Saved {len(all_results)} tasks')
    
    # Final save
    final_results = reorganize_results_by_sample(all_results)
    save_results(final_results, args.output, is_final=True)
    
    # Statistics
    all_scores = [r['score'] for r in all_results if r['is_valid']]
    if all_scores:
        print(f'\n📊 Statistics:')
        print(f'   Samples: {len(final_results)}')
        print(f'   Valid scores: {len(all_scores)}')
        print(f'   Average score: {sum(all_scores)/len(all_scores):.2f}')
        print(f'   Pass rate (>=3): {sum(1 for s in all_scores if s >= 3)/len(all_scores)*100:.1f}%')


if __name__ == '__main__':
    main()
