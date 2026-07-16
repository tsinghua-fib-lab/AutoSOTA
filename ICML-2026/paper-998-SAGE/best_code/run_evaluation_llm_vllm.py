"""
vLLM accelerated LLM evaluation script
Supports batch inference for Llama, Qwen, Ministral
Datasets: TruthfulQA, HaluEval
"""

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Import datasets
from data import TruthfulQADataset, HaluEvalDataset


def run_vllm_evaluation(
    dataset_name: str,
    model_name: str,
    output_dir: str = "./outputs/llm_generation",
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    max_tokens: int = 64,
    num_samples: int = None
):
    """Run batch inference using vLLM"""

    from vllm import LLM, SamplingParams

    # Load dataset
    print(f"\n{'='*50}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*50}")
    
    if dataset_name == "TruthfulQA":
        dataset = TruthfulQADataset(split='validation', num_samples=num_samples or 817)
    elif dataset_name == "HaluEval":
        dataset = HaluEvalDataset(subset='dialogue', split='data', num_samples=num_samples or 10000)
    elif dataset_name == "HaluEval-qa":
        dataset = HaluEvalDataset(subset='qa', split='data', num_samples=num_samples or 10000)
    elif dataset_name == "HaluEval-summarization":
        dataset = HaluEvalDataset(subset='summarization', split='data', num_samples=num_samples or 10000)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset.load_data()
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Model configuration
    MODEL_MAP = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen3-8b": "/models/Qwen3-8B",
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
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        stop=["[Human]", "[Human]:", "\n[", "Okay, let me", "Let me break", "Wait,", "You are a friendly"],
    )
    
    # Get tokenizer for building chat format
    tokenizer = llm.get_tokenizer()
    
    # System prompt - optimized for dialogue continuation tasks
    SYSTEM_PROMPT = (
        "You are a friendly assistant in a casual conversation. "
        "Respond naturally and briefly (under 25 words). "
    )
    
    # Prepare all inputs
    print(f"\nPreparing inputs...")
    all_prompts = []
    all_metadata = []
    
    for idx in tqdm(range(len(dataset)), desc="Preparing"):
        item = dataset[idx]
        prompt = item['input']
        reference = item['label']
        metadata = item['metadata']
        
        # Unified prompt preprocessing (clean special characters and formatting issues)
        prompt = preprocess_prompt(prompt)
        
        # Build chat format prompt (with system message)
        if "llama" in model_name.lower():
            formatted_prompt = format_llama_prompt(prompt, SYSTEM_PROMPT)
        elif "qwen" in model_name.lower():
            # Qwen uses raw prompt directly in vLLM (chat template has compatibility issues)
            formatted_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nResponse:"
        elif "ministral" in model_name.lower():
            # Ministral uses raw prompt directly in vLLM (does not use chat template)
            formatted_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nResponse:"
        else:
            formatted_prompt = prompt
        
        all_prompts.append(formatted_prompt)
        
        # Save metadata
        if dataset_name.startswith("HaluEval"):
            all_metadata.append({
                'index': idx,
                'original_prompt': prompt,
                'right_response': reference,
                'hallucinated_response': metadata.get('hallucinated_response', 
                                                      metadata.get('hallucinated_answer', 
                                                      metadata.get('hallucinated_summary', ''))),
                'knowledge': metadata.get('knowledge', metadata.get('document', '')[:500]),
                'context': metadata.get('dialogue_history', metadata.get('question', '')),
            })
        else:  # TruthfulQA
            all_metadata.append({
                'index': idx,
                'question': prompt,
                'best_answer': reference,
                'correct_answers': metadata.get('correct_answers', []),
                'incorrect_answers': metadata.get('incorrect_answers', []),
                'category': metadata.get('category', ''),
            })
    
    # Batch inference
    print(f"\n{'='*50}")
    print(f"Running batch inference with vLLM...")
    print(f"Total samples: {len(all_prompts)}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*50}\n")
    
    results = []
    
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        
        # vLLM batch generation
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for j, (output, meta) in enumerate(zip(outputs, batch_metadata)):
            raw_text = output.outputs[0].text
            generated_answer = raw_text.strip()
            
            # If generated output is empty, retry individually
            if not generated_answer:
                print(f"\n[RETRY] Empty output for index {meta['index']}, retrying...")
                retry_prompt = create_retry_prompt(meta, model_name)
                retry_outputs = llm.generate([retry_prompt], sampling_params)
                raw_text = retry_outputs[0].outputs[0].text
                generated_answer = raw_text.strip()
                
                if not generated_answer:
                    print(f"  [RETRY FAILED] Still empty after retry")
                    generated_answer = "[No response generated]"
                else:
                    print(f"  [RETRY SUCCESS] Got: {generated_answer[:50]}...")
            
            # Post-processing
            generated_answer = clean_answer(generated_answer)
            
            if dataset_name.startswith("HaluEval"):
                result = {
                    'index': meta['index'],
                    'prompt': meta['original_prompt'],
                    'generated_answer': generated_answer,
                    'right_response': meta['right_response'],
                    'hallucinated_response': meta['hallucinated_response'],
                    'knowledge': meta['knowledge'],
                    'context': meta['context'],
                }
            else:  # TruthfulQA
                result = {
                    'index': meta['index'],
                    'question': meta['question'],
                    'generated_answer': generated_answer,
                    'best_answer': meta['best_answer'],
                    'correct_answers': meta['correct_answers'],
                    'incorrect_answers': meta['incorrect_answers'],
                    'category': meta['category'],
                }
            
            results.append(result)
    
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
    print(f"{'='*50}")
    
    # Show examples
    print(f"\nSample predictions (first 3):")
    for result in results[:3]:
        print(f"\n[Index {result['index']}]")
        if 'question' in result:
            print(f"  Question: {result['question'][:80]}...")
            print(f"  Generated: {result['generated_answer'][:80]}...")
            print(f"  Best Answer: {result['best_answer'][:80]}...")
        else:
            print(f"  Generated: {result['generated_answer'][:80]}...")
            print(f"  Right Response: {result['right_response'][:80]}...")
    
    return results


def format_llama_prompt(prompt: str, system_prompt: str = None) -> str:
    """Format Llama prompt"""
    if system_prompt:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_qwen_prompt(prompt: str, tokenizer=None, system_prompt: str = None) -> str:
    """Format Qwen prompt"""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    if tokenizer:
        try:
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            pass
    # Fallback to manual format
    if system_prompt:
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    return f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""


def format_mistral_prompt(prompt: str, tokenizer=None, system_prompt: str = None) -> str:
    """Format Mistral/Ministral prompt - consistent with models/llm_models.py"""
    # Build messages (Ministral supports system role)
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    # Prefer using tokenizer (consistent with original version)
    if tokenizer:
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            print(f"Warning: tokenizer.apply_chat_template failed: {e}")
            # Fallback - consistent with original version
            if system_prompt:
                return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    # Fallback to manual format
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def preprocess_prompt(prompt: str) -> str:
    """Preprocess prompt, clean content that may cause issues (common for all models)"""
    import re
    
    # Remove special characters
    prompt = prompt.replace('`', '').replace('~', '')

    # Fix repeated concatenated words (e.g., "ZodiacZodiac" -> "Zodiac. Zodiac")
    prompt = re.sub(r'([A-Z][a-z]+)(\1)', r'\1. \2', prompt)
    
    # Fix formats like "XXX : Award Won:" etc.
    prompt = re.sub(r':\s*Award Won\s*:', ' won ', prompt)
    prompt = re.sub(r':\s*Lyricist\s*:', ' wrote the lyrics for ', prompt)
    prompt = re.sub(r':\s*Subject\s*:', ' is about ', prompt)
    
    # Add spaces between sentences (if missing)
    prompt = re.sub(r'\.([A-Z])', r'. \1', prompt)
    
    # Remove extra whitespace
    prompt = ' '.join(prompt.split())
    
    return prompt


def create_retry_prompt(meta: dict, model_name: str) -> str:
    """Create simplified prompt for retry"""
    import re
    
    # Clean special characters and formatting in knowledge
    knowledge = meta.get('knowledge', meta.get('question', ''))
    # Remove special characters
    knowledge = knowledge.replace('`', '').replace('~', '')
    # Remove formats like "XXX : Award Won:", convert to more natural sentences
    knowledge = re.sub(r'(\w+)\s*:\s*Award Won\s*:', r'\1 won', knowledge)
    knowledge = re.sub(r'(\w+)\s*:\s*Lyricist\s*:', r'\1 wrote the lyrics for', knowledge)
    knowledge = re.sub(r'(\w+)\s*:\s*Subject\s*:', r'\1 is about', knowledge)
    # Remove extra whitespace and duplicates
    knowledge = ' '.join(knowledge.split())
    
    context = meta.get('context', meta.get('dialogue_history', ''))
    
    # Use a simpler and more direct prompt
    simple_prompt = f"""Information: {knowledge}

Conversation so far:
{context}

Continue with a brief, natural response:"""
    
    return simple_prompt


def clean_answer(answer: str) -> str:
    """
    Clean the generated answer

    Goal: Mimic the style of right_response
    - Colloquial, conversational
    - Typically 5-25 words
    - Can be multiple sentences
    """
    import re
    
    # Remove thinking markers
    if "<think>" in answer:
        if "</think>" in answer:
            answer = answer.split("</think>")[-1]
        else:
            answer = answer.split("<think>")[0]
    
    # Remove emoji (Qwen tends to add emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    answer = emoji_pattern.sub('', answer)
    
    # Remove Qwen thinking mode leakage
    thinking_patterns = [
        "Okay, let me break down",
        "Let me break down",
        "Wait, the user",
        "Let me correct that",
        "You are a friendly assistant",
    ]
    for pattern in thinking_patterns:
        if pattern in answer:
            answer = answer.split(pattern)[0]
    
    # Remove continued dialogue parts
    dialog_markers = ["[Human]", "[Assistant]", "[Human]:", "[Assistant]:"]
    for marker in dialog_markers:
        if marker in answer:
            answer = answer.split(marker)[0]
    
    # Remove extra whitespace and newlines
    answer = ' '.join(answer.split())
    
    # Remove leading markers
    prefixes_to_remove = ["[Assistant]:", "[Me]:", "Assistant:", "Response:", "[Assistant]"]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove duplicate sentences (e.g., "You're welcome! ... You're welcome!")
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > 1:
        seen = set()
        unique_sentences = []
        for s in sentences:
            s_normalized = s.lower().strip()
            if s_normalized not in seen:
                seen.add(s_normalized)
                unique_sentences.append(s)
        answer = ' '.join(unique_sentences)
    
    # If too long (over 50 words), truncate to nearest sentence end
    words = answer.split()
    if len(words) > 50:
        # Find the last sentence-ending punctuation within 50 words
        truncated = ' '.join(words[:50])
        for sep in ['. ', '! ', '? ']:
            last_idx = truncated.rfind(sep)
            if last_idx > 0:
                answer = truncated[:last_idx + 1].strip()
                break
        else:
            # No sentence-ending punctuation found, truncate directly
            answer = truncated + "..."
    
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="vLLM accelerated LLM evaluation")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['TruthfulQA', 'HaluEval', 'HaluEval-qa', 'HaluEval-summarization'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       choices=['llama3.1-8b', 'qwen3-8b', 'ministral-8b'],
                       help='Model name')
    parser.add_argument('--output_dir', type=str, default='./outputs/llm_generation',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--tp', type=int, default=1,
                       help='Tensor parallel size (number of GPUs)')
    parser.add_argument('--max_tokens', type=int, default=128,
                       help='Maximum tokens to generate (will be post-processed to first sentence)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    run_vllm_evaluation(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tp,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()

