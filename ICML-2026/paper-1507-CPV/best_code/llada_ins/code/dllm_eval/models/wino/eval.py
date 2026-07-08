import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import tqdm
from datasets import load_dataset
import torch
import yaml
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_llada import LLaDAModelLM
from decoding import decoding_default, decoding_wino
import dataset_utils
from human_eval.evaluation import evaluate_functional_correctness
import tempfile
from dataset_utils.eval_correctness_mbpp.evaluation import evaluate_functional_correctness as evaluate_functional_correctness_mbpp


def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset config YAML file (e.g., configs/gsm8k.yaml)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"==> Loaded config for: {config['dataset_name']}")

    model_path = config['model_path']
    print(f"==> Loading model: {model_path}")
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    run_single_task_evaluation(config, model, tokenizer)


def get_generation_function(method_name):
    if method_name == 'default': return decoding_default
    elif method_name == 'wino': return decoding_wino
    else: raise ValueError(f"Unknown method: {method_name}")

def run_single_task_evaluation(config, model, tokenizer):
    dataset_name = config['dataset_name']
    print(f"==> Running Single-Task Evaluation for {dataset_name}")
    dataset_cfg = config['dataset_config']
    doc_to_text_fn = getattr(dataset_utils, dataset_cfg['doc_to_text_fn'])
    eval_fn = getattr(dataset_utils, dataset_cfg.get('eval_fn')) if dataset_cfg.get('eval_fn') else None
    extract_answer_fn = getattr(dataset_utils, dataset_cfg.get('extract_answer_fn')) if dataset_cfg.get('extract_answer_fn') else None
    is_correct_fn = getattr(dataset_utils, dataset_cfg.get('is_correct_fn')) if dataset_cfg.get('is_correct_fn') else None
    
    gen_cfg = config['generation_args']
    method_name = config['method']
    method_params = config.get('method_args', {}).get(method_name, {})
    generation_fn = get_generation_function(method_name)
    print(f"==> Loading dataset...")
    loader_type = dataset_cfg.get('data_loader', 'huggingface')
    dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
    
    if loader_type == 'huggingface':
        dataset_name_hf = dataset_cfg['load_dataset_args'].get('name')
        dataset = load_dataset(dataset_path, dataset_name_hf, trust_remote_code=True)[dataset_cfg['split']]
    elif dataset_name == 'mbpp':
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = list(loader_fn(dataset_path))
    # elif loader_type == 'custom_loader':
    else:
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = loader_fn(dataset_path)

    num_samples = config.get('num_samples')
    if num_samples is not None:
        print(f"==> Evaluating on the first {num_samples} samples.")
        if hasattr(dataset, 'select'): 
            dataset = dataset.select(range(num_samples))
        else: 
            dataset = dataset[:num_samples]

    # ---Warm-up---
    print("==> Performing warm-up run with one sample...")
    warmup_doc = dataset[0]
    trailing_prompt = "" 
    if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
        context, _, trailing_prompt = doc_to_text_fn(warmup_doc)
    else:
        context, _ = doc_to_text_fn(warmup_doc) 
    context = ""   
    prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    _ = generation_fn(model, input_ids, **gen_cfg, **method_params)   
    print("==> Warm-up complete.")
    # --- Warm-up complete ---

    total_len = len(dataset)
    raw_outputs, correct_count, total_count, total_steps = [], 0, 0, 0
    trailing_prompt = "" 
    for i in tqdm.tqdm(range(total_len), desc=f"Evaluating {dataset_name} with method '{method_name}'"):
        doc = dataset[i]
        
        gt_doc, gt_for_eval, trailing_prompt = None, None, ""
        if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
            context, gt_doc, trailing_prompt = doc_to_text_fn(doc)
        else:
            context, gt_for_eval = doc_to_text_fn(doc)
        
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        gen_output, steps = generation_fn(model, input_ids, **gen_cfg, **method_params)
        gen_str = tokenizer.batch_decode(gen_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        total_steps += steps
        
        pred, is_correct = None, "N/A"
        result_item = {'index': i, 'full_response': gen_str, 'steps': steps}

        if dataset_cfg['eval_mode'] == 'binary':
            pred = extract_answer_fn(gen_str)
            if dataset_name == 'countdown':
                target = int(doc['output'])
                num_str = doc['input']
                numbers = [int(i) for i in num_str.split(',')]
                is_correct = is_correct_fn(pred, target, numbers)
            else:
                is_correct = is_correct_fn(pred, gt_for_eval)
            if is_correct: correct_count += 1
            result_item.update({'prediction': pred, 'is_correct': is_correct})
        elif dataset_cfg['eval_mode'] == 'partial_credit':
            target = doc['Solution']
            correct, total, pred = eval_fn(gen_str, target, doc['Puzzle'])
            correct_count += correct
            total_count += total
            result_item.update({'prediction': pred, 'correct_cells': correct, 'total_cells': total})
        elif dataset_cfg['eval_mode'] == 'save_only':
            if dataset_name == 'mbpp':
                gen_str = f"```python\n" + gen_str
                gen_code = extract_answer_fn(gen_str, doc['entry_point'])
                result_item['completion'] = gen_code
            else:
                pred = extract_answer_fn(gen_str, doc)
                result_item['completion'] = pred
            if 'task_id' in gt_doc: result_item['task_id'] = gt_doc['task_id']
            
        raw_outputs.append(result_item)
    
    final_metrics = {}
    output_formatter = config['dataset_config'].get('output_formatter', 'default')
    if output_formatter == 'humaneval':
        if evaluate_functional_correctness is None:
            print("Warning: 'human-eval' library not found. Skipping functional correctness evaluation.")
        else:
            print("\n==> Generations complete. Calling official evaluation script...")
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
                for item in raw_outputs:
                    temp_f.write(json.dumps(item) + "\n")
                    temp_file_path = temp_f.name
            if dataset_name == 'humaneval':
                final_metrics = evaluate_functional_correctness(temp_file_path)
            elif dataset_name == 'mbpp':
                problem_file = config['dataset_config'].get('problem_file', 'default')
                final_metrics = evaluate_functional_correctness_mbpp(temp_file_path,problem_file=problem_file,is_mbpp=True)
    
        os.unlink(temp_file_path)
    save_results(config, method_name, raw_outputs, correct_count, total_count, total_steps, total_len, external_metrics=final_metrics)


def save_results(config, method_name, raw_outputs, correct_count, total_count, total_steps, total_len, external_metrics):
    summary = {'config': config, 'metrics': {}}
    eval_mode = config['dataset_config']['eval_mode']
    if external_metrics:
        summary['metrics'].update(external_metrics)
        accuracy = external_metrics['pass@1']
    else:
        accuracy = "N/A"
        if eval_mode == 'binary':
            accuracy = correct_count / total_len if total_len > 0 else 0
        elif eval_mode == 'partial_credit':
            accuracy = correct_count / total_count if total_count > 0 else 0
        summary['metrics']['accuracy'] = accuracy
    
    avg_steps = total_steps / total_len if total_len > 0 else 0
    summary['metrics']['average_steps'] = avg_steps
    summary = {'raw_output': raw_outputs}
    
    output_path = f"./results/{config['dataset_name']}_{method_name}.json"
    if config.get('num_samples') is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_samples{config['num_samples']}{ext}"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with jsonlines.Writer(open(output_path.replace('.json', '.jsonl'), "w", encoding="utf-8")) as writer:
        writer.write_all(raw_outputs)
    print(f"Results saved in .jsonl format to {output_path.replace('.json', '.jsonl')}")
    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Method: {method_name}")
    print(f"Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}")
    print(f"Average Steps: {avg_steps:.2f}")

if __name__ == "__main__":
    main()
