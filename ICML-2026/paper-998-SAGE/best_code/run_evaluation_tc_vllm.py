"""
vLLM accelerated Text Classification evaluation script
Supports batch inference for Llama, Qwen, Ministral
Datasets: AG_News, MMLU
"""

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Import datasets
from data import AGNewsDataset, MMLUDataset


def run_vllm_tc_evaluation(
    dataset_name: str,
    model_name: str,
    output_dir: str = "./outputs/text_classification",
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    num_samples: int = None
):
    """Run text classification batch inference using vLLM"""
    
    from vllm import LLM, SamplingParams
    
    # Load dataset
    print(f"\n{'='*50}")
    print(f"Task: text_classification")
    print(f"Loading dataset: {dataset_name}")
    if num_samples:
        print(f"⚠️ TEST MODE: Only processing {num_samples} samples")
    print(f"{'='*50}")
    
    # Dataset configuration
    if dataset_name == "AG_News":
        dataset = AGNewsDataset(split='train', num_samples=num_samples or 10000)
    elif dataset_name == "MMLU":
        dataset = MMLUDataset(split='test', num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset.load_data()
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.class_names}")
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    
    # Model configuration
    MODEL_MAP = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
    }
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}")
    
    model_path = MODEL_MAP[model_name]
    
    # Initialize vLLM
    print(f"\n{'='*50}")
    print(f"Loading vLLM model: {model_path}")
    print(f"{'='*50}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=10,  # Classification only needs a short answer
        temperature=0,
    )
    
    # Get tokenizer
    tokenizer = llm.get_tokenizer()
    
    # Prepare all inputs
    print(f"\nPreparing inputs...")
    all_prompts = []
    all_metadata = []
    
    for idx in tqdm(range(len(dataset)), desc="Preparing"):
        item = dataset[idx]
        text = item['input']
        true_label = item['label']
        metadata = item.get('metadata', {})
        choices = metadata.get('choices', None)  # MMLU has choices
        
        # Build prompt
        if dataset_name == "MMLU" and choices:
            # MMLU: Multiple-choice format
            options = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            class_list = ", ".join(class_names)
            prompt_text = (
                f"Question: {text}\n\n"
                f"Options:\n{options}\n\n"
                f"Answer with ONLY the letter ({class_list}):"
            )
            system_prompt = "You are a helpful assistant. Answer directly with just the letter."
        else:
            # AG_News: Text classification format
            class_list = ", ".join(class_names)
            prompt_text = (
                f"Classify the following text into one of these categories: {class_list}.\n"
                f"Answer only with the category name.\n\n"
                f"Text: {text}\n\n"
                f"Category:"
            )
            system_prompt = "You are a concise text classifier."
        
        # Format prompt based on model
        if "llama" in model_name.lower():
            formatted_prompt = format_llama_prompt(prompt_text, system_prompt, tokenizer)
        elif "qwen" in model_name.lower():
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        elif "ministral" in model_name.lower():
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt_text} [/INST]"
        else:
            formatted_prompt = f"{system_prompt}\n\n{prompt_text}\n\nAnswer:"
        
        all_prompts.append(formatted_prompt)
        all_metadata.append({
            'index': idx,
            'text': text[:200],  # Save first 200 characters
            'true_label': true_label,
            'true_label_name': class_names[true_label] if isinstance(true_label, int) else str(true_label),
            'choices': choices,
        })
    
    # Batch inference
    import time
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print(f"Running batch inference with vLLM...")
    print(f"Total samples: {len(all_prompts)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    pbar = tqdm(
        range(0, len(all_prompts), batch_size), 
        desc="Generating",
        total=total_batches,
        unit="batch",
    )
    
    for i in pbar:
        batch_prompts = all_prompts[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        
        # vLLM batch generation
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for output, meta in zip(outputs, batch_metadata):
            generated_text = output.outputs[0].text.strip()
            
            # Parse prediction result
            predicted_class, confidence = parse_prediction(
                generated_text, 
                class_names, 
                meta.get('choices')
            )
            
            result = {
                'index': meta['index'],
                'text': meta['text'],
                'true_label': meta['true_label'],
                'true_label_name': meta['true_label_name'],
                'prediction': predicted_class,
                'prediction_name': class_names[predicted_class] if predicted_class < len(class_names) else "Unknown",
                'confidence': confidence,
                'raw_output': generated_text,
            }
            
            results.append(result)
        
        # Update progress bar
        samples_done = len(results)
        correct = sum(1 for r in results if r['prediction'] == r['true_label'])
        accuracy = correct / samples_done * 100 if samples_done > 0 else 0
        pbar.set_postfix({
            'done': f'{samples_done}/{len(all_prompts)}',
            'acc': f'{accuracy:.1f}%'
        })
    
    # Calculate final accuracy
    total_time = time.time() - start_time
    correct = sum(1 for r in results if r['prediction'] == r['true_label'])
    accuracy = correct / len(results) * 100
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{model_name}_{timestamp}"
    
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {json_path}")
    print(f"Total samples: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Total time: {total_time:.1f}s ({len(results)/total_time:.1f} samples/s)")
    print(f"{'='*50}")
    
    # Show examples
    print(f"\nSample predictions (first 5):")
    for result in results[:5]:
        status = "✓" if result['prediction'] == result['true_label'] else "✗"
        print(f"\n[{status}] Index {result['index']}")
        print(f"  Text: {result['text'][:80]}...")
        print(f"  Predicted: {result['prediction_name']}")
        print(f"  True: {result['true_label_name']}")
        print(f"  Raw output: {result['raw_output']}")
    
    return results


def format_llama_prompt(prompt: str, system_prompt: str, tokenizer) -> str:
    """Format Llama chat prompt"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        # Fallback
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_prediction(generated_text: str, class_names: list, choices: list = None) -> tuple:
    """Parse generated text, return predicted class and confidence"""
    generated_lower = generated_text.lower().strip()
    num_classes = len(class_names)
    
    # If multiple-choice (MMLU), try matching letters first
    if choices:
        for i, letter in enumerate(['a', 'b', 'c', 'd'][:num_classes]):
            if generated_lower.startswith(letter) or f"({letter})" in generated_lower or f"{letter})" in generated_lower:
                return i, 1.0
    
    # Try direct class name matching
    scores = np.zeros(num_classes)
    for i, class_name in enumerate(class_names):
        class_lower = class_name.lower().strip()
        
        if class_lower == generated_lower:
            scores[i] = 10.0
        elif class_lower in generated_lower:
            scores[i] = 5.0 + len(class_lower) * 0.1
        elif generated_lower in class_lower:
            scores[i] = 3.0
        else:
            # Word-level matching
            class_words = set(class_lower.split())
            output_words = set(generated_lower.split())
            common = class_words.intersection(output_words)
            if common:
                scores[i] = sum(len(w) * 0.1 for w in common if len(w) > 2)
    
    # Normalize
    if scores.sum() > 0:
        scores = scores / scores.sum()
    else:
        scores = np.ones(num_classes) / num_classes
    
    predicted_class = int(np.argmax(scores))
    confidence = float(scores[predicted_class])
    
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="vLLM accelerated Text Classification evaluation")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['AG_News', 'MMLU'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       choices=['llama3.1-8b', 'qwen3-8b', 'ministral-8b'],
                       help='Model name')
    parser.add_argument('--output_dir', type=str, default='./outputs/text_classification',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--tp', type=int, default=1,
                       help='Tensor parallel size (number of GPUs)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (for testing, default: all)')
    
    args = parser.parse_args()
    
    run_vllm_tc_evaluation(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tp,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()

