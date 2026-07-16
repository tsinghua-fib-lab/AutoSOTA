#!/usr/bin/env python3
"""
Regenerate empty answers for Qwen HaluEval
"""

import json
import argparse
import re
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def clean_answer(answer: str) -> str:
    """Clean generated answer"""
    # Handle Qwen3 thinking mode
    if "<think>" in answer:
        if "</think>" in answer:
            # Extract the actual answer after </think>
            answer = answer.split("</think>")[-1]
        else:
            # Thinking not finished, try to find from the last sentence
            # Or just return empty (thinking too long, not completed)
            return ""
    
    # Remove extra whitespace and newlines
    answer = ' '.join(answer.split())
    
    # Remove leading markers
    prefixes_to_remove = ["[Assistant]:", "[Me]:", "Assistant:", "Response:",
                          "Okay, let me break down how I approached this.",
                          "Let me break down how I arrived at that response.",
                          "Wait, the user asked for",
                          "You are a friendly assistant"]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    seen_sentences = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen_sentences:
            unique_sentences.append(s)
            seen_sentences.add(s)
    answer = ' '.join(unique_sentences)

    # Remove emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    answer = emoji_pattern.sub(r'', answer)
    
    # If too long, truncate
    words = answer.split()
    if len(words) > 50:
        truncated = ' '.join(words[:50])
        for sep in ['. ', '! ', '? ']:
            last_idx = truncated.rfind(sep)
            if last_idx > 0:
                answer = truncated[:last_idx + 1].strip()
                break
        else:
            answer = truncated + "..."
    
    return answer.strip()


def aggressive_clean_knowledge(knowledge: str) -> str:
    """
    More aggressively clean the knowledge field
    """
    # Remove all backticks and tildes
    knowledge = knowledge.replace('`', '').replace('~', '')
    
    # Fix repeated concatenated word patterns (e.g. SingerSinger -> Singer. Singer)
    # Match: words starting with uppercase letter, immediately followed by the same pattern
    knowledge = re.sub(r'([A-Z][a-z]+)([A-Z])', r'\1. \2', knowledge)
    
    # Fix patterns like A.A. MilneA.A. Milne
    knowledge = re.sub(r'(\b[A-Z]\.[A-Z]\.\s*[A-Z][a-z]+)([A-Z])', r'\1. \2', knowledge)
    
    # Fix formatting markers
    knowledge = re.sub(r':\s*Award Won\s*:', ' won ', knowledge)
    knowledge = re.sub(r':\s*Lyricist\s*:', ' wrote the lyrics for ', knowledge)
    knowledge = re.sub(r':\s*Subject\s*:', ' is about ', knowledge)
    knowledge = re.sub(r':\s*Position\(s\)\s*:', ' plays as ', knowledge)
    knowledge = re.sub(r':\s*Sport played\s*:', ' plays ', knowledge)
    knowledge = re.sub(r':\s*Sport\s*:', ' is a sport: ', knowledge)
    knowledge = re.sub(r':\s*Composer\s*:', ' composed ', knowledge)
    knowledge = re.sub(r':\s*Matches Lost\s*:', ' lost to ', knowledge)
    
    # Add space after period
    knowledge = re.sub(r'\.([A-Z])', r'. \1', knowledge)
    
    # Remove extra whitespace
    knowledge = ' '.join(knowledge.split())
    
    return knowledge


def build_simple_prompt(item: dict) -> str:
    """
    Build a more concise prompt to avoid special character issues
    """
    knowledge = item.get('knowledge', '')
    context = item.get('context', '')
    
    # Aggressive cleaning
    knowledge = aggressive_clean_knowledge(knowledge)
    
    # Build concise prompt
    prompt = f"""You are having a casual conversation. Continue naturally in 1-2 sentences.

Background info: {knowledge}

Conversation so far:
{context}

Your brief response:"""
    
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input Qwen generation file')
    parser.add_argument('--output', type=str, help='Output file (default: overwrite original file)')
    parser.add_argument('--tp', type=int, default=1, help='tensor parallel')
    args = parser.parse_args()
    
    output_file = args.output or args.input
    
    # Load data
    print(f"Loading data: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Find empty answers
    empty_indices = []
    empty_items = []
    for i, item in enumerate(data):
        if not item.get('generated_answer', '').strip():
            empty_indices.append(i)
            empty_items.append(item)
    
    print(f"Found {len(empty_indices)} empty answers")
    
    if not empty_items:
        print("No samples need to be retried")
        return
    
    # Load Qwen model and tokenizer
    print("Loading Qwen model...")
    model = LLM(
        model="Qwen/Qwen3-8B",
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    
    # Load tokenizer for disabling thinking mode (using local cache)
    print("Loading tokenizer...")
    import os
    local_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
    
    # Build prompts - use enable_thinking=False to disable thinking mode
    prompts = []
    for item in empty_items:
        prompt = build_simple_prompt(item)
        messages = [{"role": "user", "content": prompt}]
        # Key: enable_thinking=False disables thinking mode
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode
        )
        prompts.append(formatted)
    
    # Sampling parameters - don't need too many tokens since there's no thinking content
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,  # Direct answer output, don't need too many
        stop=["<|im_end|>", "<|endoftext|>", "[Human]:", "[Assistant]:"],
    )
    
    # Batch generation
    print(f"Starting generation of {len(prompts)} samples...")
    outputs = model.generate(prompts, sampling_params)
    
    # Update data
    success_count = 0
    still_empty = []
    for idx, (array_idx, output) in enumerate(zip(empty_indices, outputs)):
        raw_text = output.outputs[0].text
        cleaned = clean_answer(raw_text)
        
        data[array_idx]['generated_answer'] = cleaned
        data[array_idx]['retry_raw'] = raw_text
        data[array_idx]['retried'] = True
        
        if cleaned.strip():
            success_count += 1
        else:
            still_empty.append(data[array_idx]['index'])
    
    print(f"Successfully generated: {success_count}/{len(empty_indices)}")
    if still_empty:
        print(f"Still empty indices: {still_empty}")
    
    # Save
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    main()

