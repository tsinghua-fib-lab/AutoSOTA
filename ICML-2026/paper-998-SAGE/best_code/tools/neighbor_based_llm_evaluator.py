"""
Use the answer-generating model itself for scoring
- Uses project configuration file
- Supports short model names (e.g., llama3.1-8b)
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

# Import project configuration and models
from config import MODEL_CONFIG
from models import (
    LlamaGenerator,
    OpenAIGenerator,
    Qwen3Generator,
    MinistralGenerator
)

# Model class mapping
MODEL_CLASS_MAP = {
    "llama": LlamaGenerator,
    "openai": OpenAIGenerator,
    "qwen": Qwen3Generator,
    "qwen3": Qwen3Generator,  # Compatible with qwen3 in config file
    "ministral": MinistralGenerator,
}


# QA_EVALUATION_PROMPT = '''Please evaluate the quality of the predicted answer by comparing it with a reference Q&A pair from a similar question.

# Reference Q&A Pair (from a semantically similar question):
# Question: {ref_question}
# Answer: {ref_answer}

# Question to Evaluate:
# {question}

# Predicted Answer (to be evaluated):
# {predicted_answer}

# Scoring Rules:
# - Score 5: Excellent - accurate and comprehensive
# - Score 4: Good - mostly accurate with minor issues
# - Score 3: Acceptable - partially correct but has gaps
# - Score 2: Poor - significant errors or incomplete
# - Score 1: Very Poor - mostly incorrect or irrelevant
# - Score 0: Completely Wrong - entirely incorrect

# Important Notes:
# - The reference Q&A may not be perfectly related, use your judgment
# - Focus on factual accuracy and completeness
# - A score of 3 or above means the answer is acceptable

# CRITICAL: Return ONLY one digit (0, 1, 2, 3, 4, or 5) with NO explanation.

# Score:'''
QA_EVALUATION_PROMPT = '''Please evaluate the quality of the predicted answer using the context provided below.

Context:
Question: {ref_question}
Answer: {ref_answer}

Question to Evaluate:
{question}

Predicted Answer (to be evaluated):
{predicted_answer}

Scoring Rules:
- Score 5: Excellent - accurate and comprehensive
- Score 4: Good - mostly accurate with minor issues
- Score 3: Acceptable - partially correct but has gaps
- Score 2: Poor - significant errors or incomplete
- Score 1: Very Poor - mostly incorrect or irrelevant
- Score 0: Completely Wrong - entirely incorrect

Important Notes:
- The context is for reference only, use your own judgment
- Focus on whether the predicted answer is factually correct and addresses the question
- The predicted answer doesn't need to match the context exactly
- A score of 3 or above means the answer is acceptable

CRITICAL: Return ONLY one digit (0, 1, 2, 3, 4, or 5) with NO explanation.

Score:'''

# No-reference evaluation prompt (evaluate self only)
QA_EVALUATION_PROMPT_SINGLE = '''Please evaluate the quality of the predicted answer for the given question.

Question:
{question}

Predicted Answer:
{predicted_answer}

Scoring Rules:
- Score 5: Excellent - accurate and comprehensive, addresses all aspects of the question
- Score 4: Good - mostly accurate with minor issues or omissions
- Score 3: Acceptable - partially correct but has notable gaps or minor errors
- Score 2: Poor - significant errors or incomplete, misses key points
- Score 1: Very Poor - mostly incorrect or irrelevant to the question
- Score 0: Completely Wrong - entirely incorrect or does not address the question at all

CRITICAL: Return ONLY one digit (0, 1, 2, 3, 4, or 5) with NO explanation.

Score:'''

def parse_args():
    parser = argparse.ArgumentParser(description='Self-scoring using the original model')

    parser.add_argument('--mode', type=str, choices=['neighbor', 'single'], default='neighbor',
                       help='Evaluation mode: neighbor (use neighbors as reference) or single (evaluate self only)')
    parser.add_argument('--neighbors', type=str, default=None,
                       help='Neighbors JSONL file path (required for neighbor mode)')
    parser.add_argument('--qa-data', type=str, required=True,
                       help='Original QA JSON file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output results path (JSON file)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., llama3.1-8b)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--save-interval', '-s', type=int, default=100,
                       help='Auto-save interval')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: only process the first N samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from interruption')
    
    args = parser.parse_args()
    
    # Neighbor mode requires neighbors file
    if args.mode == 'neighbor' and args.neighbors is None:
        parser.error("--neighbors is required when --mode is 'neighbor'")
    
    return args


def load_model(model_name: str):
    """
    Load model by model name
    Read configuration from config.py
    """
    if model_name not in MODEL_CONFIG["llm_generation"]:
        raise ValueError(f"Model '{model_name}' not found in configuration")

    config = MODEL_CONFIG["llm_generation"][model_name]
    model_type = config["model_type"]

    if model_type not in MODEL_CLASS_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = MODEL_CLASS_MAP[model_type]
    
    # Instantiate model - pass config dictionary
    model = model_class({
        "model_name": config["model_name"],
        "model_type": model_type,
        "device": config.get("device", "cuda"),
        "max_new_tokens": 3,  # Only need 1 digit
        "temperature": 0.0    # Deterministic output
    })
    
    print(f"✅ Loaded model: {model_name}")
    print(f"   - Type: {model_type}")
    print(f"   - Path: {config['model_name']}")
    print(f"   - Device: {config.get('device', 'cuda')}")
    
    return model


def load_neighbors_jsonl(file_path: str) -> List[Dict]:
    """Load neighbors JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_qa_data(file_path: str) -> List[Dict]:
    """Load original QA JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_global_id_map(qa_data: List[Dict]) -> Dict[int, Dict]:
    """Build mapping from global_id to QA data"""
    return {i: qa for i, qa in enumerate(qa_data)}


def get_question_field(qa: Dict) -> str:
    """Get question field, compatible with TruthfulQA and HaluEval formats"""
    if 'question' in qa:
        return qa['question']
    elif 'knowledge' in qa and 'context' in qa:
        return f"Knowledge: {qa['knowledge']}\n\nConversation: {qa['context']}"
    else:
        return qa.get('context', '')


def get_best_answer_field(qa: Dict) -> str:
    """Get best_answer field, compatible with TruthfulQA and HaluEval formats"""
    return qa.get('best_answer') or qa.get('right_response', '')


def create_evaluation_tasks_neighbor(neighbors_data: List[Dict], qa_map: Dict[int, Dict], include_self: bool = True) -> List[Dict]:
    """
    Create all scoring tasks (neighbor mode: use neighbors as reference)

    Args:
        neighbors_data: Neighbor data
        qa_map: QA data mapping
        include_self: Whether to include scoring using own best_answer as reference (the 10th)
    """
    tasks = []
    
    for sample in neighbors_data:
        sample_id = sample['global_id']
        sample_qa = qa_map[sample_id]
        sample_question = get_question_field(sample_qa)
        sample_best_answer = get_best_answer_field(sample_qa)
        
        # 1. Use 9 neighbors as reference for scoring
        for rank, neighbor in enumerate(sample['neighbors']):
            neighbor_id = neighbor['global_id']
            neighbor_qa = qa_map[neighbor_id]
            neighbor_question = get_question_field(neighbor_qa)
            
            tasks.append({
                'sample_global_id': sample_id,
                'sample_index': sample.get('index', sample_id),
                'neighbor_rank': rank,  # 0-8
                'neighbor_global_id': neighbor_id,
                'neighbor_cosine': neighbor['cosine'],
                'question': sample_question,
                'predicted_answer': sample_qa['generated_answer'],
                'best_answer': sample_best_answer,
                'ref_question': neighbor_question,
                'ref_answer': neighbor_qa['generated_answer'],
                'is_self_ref': False,
            })
        
        # 2. Use own generated_answer as reference for scoring (the 10th, self-consistency check)
        if include_self:
            tasks.append({
                'sample_global_id': sample_id,
                'sample_index': sample.get('index', sample_id),
                'neighbor_rank': 9,  # The 10th (index 9), indicates self-reference
                'neighbor_global_id': sample_id,  # Points to self
                'neighbor_cosine': 1.0,  # Similarity with self is 1
                'question': sample_question,
                'predicted_answer': sample_qa['generated_answer'],
                'best_answer': sample_best_answer,
                'ref_question': sample_question,  # Same question
                'ref_answer': sample_qa['generated_answer'],  # Use own generated_answer as reference
                'is_self_ref': True,
            })
    
    return tasks


def create_evaluation_tasks_single(qa_data: List[Dict]) -> List[Dict]:
    """
    Create all scoring tasks (single mode: evaluate self only, no reference)

    Args:
        qa_data: List of QA data
    """
    tasks = []
    
    for i, qa in enumerate(qa_data):
        tasks.append({
            'sample_global_id': i,
            'sample_index': qa.get('index', i),
            'question': get_question_field(qa),
            'predicted_answer': qa['generated_answer'],
            'best_answer': get_best_answer_field(qa),
        })
    
    return tasks


def create_prompt_neighbor(task: Dict) -> str:
    """Create scoring prompt (neighbor mode: use neighbors as reference)"""
    return QA_EVALUATION_PROMPT.format(
        ref_question=task['ref_question'],
        ref_answer=task['ref_answer'],
        question=task['question'],
        predicted_answer=task['predicted_answer']
    )


def create_prompt_single(task: Dict) -> str:
    """Create scoring prompt (single mode: evaluate self only)"""
    return QA_EVALUATION_PROMPT_SINGLE.format(
        question=task['question'],
        predicted_answer=task['predicted_answer']
    )


def parse_score(response: str) -> Tuple[Optional[int], bool, str]:
    """
    Parse the returned score
    Returns: (score, is_valid, score_str)
    """
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


def batch_generate_scores_neighbor(
    model,
    tasks: List[Dict],
) -> List[Dict]:
    """Batch generate scores (neighbor mode)"""

    # Create prompts
    prompts = [create_prompt_neighbor(task) for task in tasks]

    # Batch generate (using model's generate method)
    responses = []
    for prompt in prompts:
        try:
            response = model.generate(prompt)
            responses.append(response)
        except Exception as e:
            print(f"⚠️  Generation failed: {str(e)}")
            responses.append("")
    
    # Parse results
    results = []
    for task, response in zip(tasks, responses):
        score, is_valid, score_str = parse_score(response)
        
        results.append({
            'sample_global_id': task['sample_global_id'],
            'sample_index': task['sample_index'],
            'neighbor_rank': task['neighbor_rank'],
            'neighbor_global_id': task['neighbor_global_id'],
            'neighbor_cosine': task['neighbor_cosine'],
            'question': task['question'],
            'predicted_answer': task['predicted_answer'],
            'best_answer': task['best_answer'],
            'ref_question': task['ref_question'],
            'ref_answer': task['ref_answer'],
            'score': score,
            'score_str': score_str,
            'is_valid': is_valid,
            'is_passing': score >= 3 if is_valid else False,
            'raw_response': response,
            'success': is_valid,
        })
    
    return results


def batch_generate_scores_single(
    model,
    tasks: List[Dict],
) -> List[Dict]:
    """Batch generate scores (single mode: evaluate self only)"""

    # Create prompts
    prompts = [create_prompt_single(task) for task in tasks]

    # Batch generate (using model's generate method)
    responses = []
    for prompt in prompts:
        try:
            response = model.generate(prompt)
            responses.append(response)
        except Exception as e:
            print(f"⚠️  Generation failed: {str(e)}")
            responses.append("")
    
    # Parse results
    results = []
    for task, response in zip(tasks, responses):
        score, is_valid, score_str = parse_score(response)
        
        results.append({
            'sample_global_id': task['sample_global_id'],
            'sample_index': task['sample_index'],
            'question': task['question'],
            'predicted_answer': task['predicted_answer'],
            'best_answer': task['best_answer'],
            'score': score,
            'score_str': score_str,
            'is_valid': is_valid,
            'is_passing': score >= 3 if is_valid else False,
            'raw_response': response,
            'success': is_valid,
        })
    
    return results


def reorganize_results_by_sample_neighbor(all_results: List[Dict]) -> List[Dict]:
    """Reorganize results by sample (neighbor mode: supports 9 neighbors + 1 self-reference = 10 scores)"""
    from collections import defaultdict
    
    samples_dict = defaultdict(lambda: {
        'neighbor_scores': [None] * 10,  # 0-8: neighbors, 9: self-reference
        'metadata': {}
    })
    
    for result in all_results:
        sample_id = result['sample_global_id']
        rank = result['neighbor_rank']
        
        if not samples_dict[sample_id]['metadata']:
            samples_dict[sample_id]['metadata'] = {
                'global_id': result['sample_global_id'],
                'sample_index': result['sample_index'],
                'question': result.get('question', ''),
                'predicted_answer': result.get('predicted_answer', ''),
                'best_answer': result.get('best_answer', ''),
            }
        
        samples_dict[sample_id]['neighbor_scores'][rank] = {
            'neighbor_rank': rank,
            'neighbor_global_id': result['neighbor_global_id'],
            'neighbor_cosine': result['neighbor_cosine'],
            'ref_question': result.get('ref_question', ''),
            'ref_answer': result.get('ref_answer', ''),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'is_passing': result.get('is_passing', False),
            'raw_response': result.get('raw_response', ''),
            'success': result.get('success', False),
            'is_self_ref': result.get('is_self_ref', False),  # Marks whether this is a self-reference score
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
                'total_neighbors': len(scores),
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


def reorganize_results_by_sample_single(all_results: List[Dict]) -> List[Dict]:
    """Reorganize results by sample (single mode: one score per sample)"""
    final_results = []
    
    for result in all_results:
        final_results.append({
            'global_id': result['sample_global_id'],
            'sample_index': result['sample_index'],
            'question': result.get('question', ''),
            'predicted_answer': result.get('predicted_answer', ''),
            'best_answer': result.get('best_answer', ''),
            'score': result['score'],
            'score_str': result.get('score_str', 'INVALID'),
            'is_valid': result.get('is_valid', False),
            'is_passing': result.get('is_passing', False),
            'raw_response': result.get('raw_response', ''),
            'success': result.get('success', False),
        })
    
    # Sort by global_id
    final_results.sort(key=lambda x: x['global_id'])
    
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


def load_existing_results(output_path):
    """Load existing results (resume from checkpoint)"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        tasks_completed = []
        for sample in results:
            for score_data in sample['neighbor_scores']:
                if score_data:
                    tasks_completed.append(score_data)
        
        print(f"📂 Found existing results, {len(tasks_completed)} scoring tasks completed")
        return tasks_completed
    except FileNotFoundError:
        print("📂 No existing results found, starting from scratch")
        return []


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🚀 Model Self-Scoring Task Started")
    print("=" * 80)
    print(f"📋 Evaluation mode: {args.mode}")
    if args.mode == 'neighbor':
        print(f"📥 Neighbors file: {args.neighbors}")
    print(f"📥 QA data file: {args.qa_data}")
    print(f"📤 Output file: {args.output}")
    print(f"🤖 Model: {args.model}")
    print(f"📦 Batch size: {args.batch_size}")
    if args.test:
        print(f"🧪 Test mode: first {args.test} samples")
    print("=" * 80 + "\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    print()

    # Load data
    print("Loading data...")
    qa_data = load_qa_data(args.qa_data)
    qa_map = build_global_id_map(qa_data)
    
    if args.mode == 'neighbor':
        neighbors_data = load_neighbors_jsonl(args.neighbors)
        print(f"✅ Data loaded")
        print(f"   - Neighbors: {len(neighbors_data)} samples")
        print(f"   - QA data: {len(qa_data)} entries\n")

        if args.test:
            neighbors_data = neighbors_data[:args.test]
            print(f"🧪 Test mode: only processing first {args.test} samples\n")

        # Create tasks
        print("Creating scoring tasks...")
        all_tasks = create_evaluation_tasks_neighbor(neighbors_data, qa_map)
        print(f"✅ Total {len(all_tasks)} tasks ({len(neighbors_data)} x 10)\n")
    else:
        # single mode
        print(f"✅ Data loaded")
        print(f"   - QA data: {len(qa_data)} entries\n")

        if args.test:
            qa_data = qa_data[:args.test]
            print(f"🧪 Test mode: only processing first {args.test} samples\n")

        # Create tasks
        print("Creating scoring tasks...")
        all_tasks = create_evaluation_tasks_single(qa_data)
        print(f"✅ Total {len(all_tasks)} tasks (1 score per sample)\n")
    
    # Resume from checkpoint
    completed_results = []
    if args.resume:
        completed_results = load_existing_results(args.output)
        completed_count = len(completed_results)
        
        if completed_count >= len(all_tasks):
            print("✅ All tasks completed!")
            if args.mode == 'neighbor':
                final_results = reorganize_results_by_sample_neighbor(completed_results)
            else:
                final_results = reorganize_results_by_sample_single(completed_results)
            save_results(final_results, args.output, is_final=True)
            return
        
        # Filter completed tasks
        if args.mode == 'neighbor':
            completed_keys = {
                (r['sample_global_id'], r['neighbor_rank']) 
                for r in completed_results
            }
            all_tasks = [
                t for t in all_tasks 
                if (t['sample_global_id'], t['neighbor_rank']) not in completed_keys
            ]
        else:
            completed_keys = {r['sample_global_id'] for r in completed_results}
            all_tasks = [
                t for t in all_tasks 
                if t['sample_global_id'] not in completed_keys
            ]
        print(f"🔄 Remaining {len(all_tasks)} tasks\n")

    # Batch processing
    print("=== Starting scoring ===\n")
    all_results = completed_results.copy()
    
    # Select correct batch generation and reorganization functions
    if args.mode == 'neighbor':
        batch_generate_func = batch_generate_scores_neighbor
        reorganize_func = reorganize_results_by_sample_neighbor
    else:
        batch_generate_func = batch_generate_scores_single
        reorganize_func = reorganize_results_by_sample_single
    
    for i in tqdm(range(0, len(all_tasks), args.batch_size), desc="Scoring progress"):
        batch_tasks = all_tasks[i:i + args.batch_size]
        
        try:
            batch_results = batch_generate_func(model, batch_tasks)
            all_results.extend(batch_results)
            
            # Periodic save
            if len(all_results) % args.save_interval < args.batch_size:
                reorganized = reorganize_func(all_results)
                save_results(reorganized, args.output, is_final=False)
        
        except Exception as e:
            print(f"\n⚠️  Batch {i//args.batch_size} failed: {str(e)}")
            # Save current progress
            if all_results:
                reorganized = reorganize_func(all_results)
                save_results(reorganized, args.output, is_final=False)
            raise
    
    # Reorganize and save final results
    print("\nReorganizing results...")
    final_results = reorganize_func(all_results)
    
    # Statistics
    total_tasks = len(all_results)
    success_count = sum(1 for r in all_results if r.get('success', False))
    fail_count = total_tasks - success_count
    
    # Collect all valid scores
    all_valid_scores = []
    invalid_count = 0
    
    if args.mode == 'neighbor':
        for sample in final_results:
            for s in sample['neighbor_scores']:
                if s and s['is_valid']:
                    all_valid_scores.append(s['score'])
                elif s and s['score_str'] == 'INVALID':
                    invalid_count += 1
    else:
        for sample in final_results:
            if sample['is_valid']:
                all_valid_scores.append(sample['score'])
            elif sample['score_str'] == 'INVALID':
                invalid_count += 1
    
    print("\n" + "=" * 80)
    print("📊 Final Statistics")
    print("=" * 80)
    print(f"Total samples: {len(final_results)}")
    print(f"Total scoring tasks: {total_tasks}")
    print(f"Succeeded: {success_count} ({success_count/total_tasks*100:.1f}%)")
    print(f"Failed: {fail_count} ({fail_count/total_tasks*100:.1f}%)")
    
    if all_valid_scores:
        avg_score = sum(all_valid_scores) / len(all_valid_scores)
        passing_count = sum(1 for s in all_valid_scores if s >= 3)
        
        print(f"\n📈 Score statistics:")
        print(f"   Valid scores: {len(all_valid_scores)}")
        print(f"   Invalid scores: {invalid_count}")
        print(f"   Average score: {avg_score:.2f}/5.0")
        print(f"   Passing count (>=3): {passing_count}")
        print(f"   Pass rate: {passing_count/len(all_valid_scores)*100:.1f}%")

        print(f"\n📊 Score distribution:")
        score_dist = {str(i): all_valid_scores.count(i) for i in range(6)}
        score_dist['INVALID'] = invalid_count
        
        total = len(all_valid_scores) + invalid_count
        for score in range(5, -1, -1):
            count = score_dist[str(score)]
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            mark = '✅' if score >= 3 else '❌'
            print(f"   {score}: {count:4d} ({pct:5.1f}%) {bar} {mark}")
        
        pct = invalid_count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"   INVALID: {invalid_count:4d} ({pct:5.1f}%) {bar} ⚠️")
    
    print("=" * 80)
    
    save_results(final_results, args.output, is_final=True)


if __name__ == "__main__":
    main()