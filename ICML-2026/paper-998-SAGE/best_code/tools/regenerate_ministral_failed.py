"""
Regenerate Ministral failed samples
- Remove stop tokens that may cause issues
- Add min_tokens to ensure content generation
- Modify prompt format to more explicitly request a response
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Regenerate Ministral failed samples')
    parser.add_argument('--input', type=str,
                       default='outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json',
                       help='Original generation result file')
    parser.add_argument('--output', type=str,
                       default='outputs/llm_generation/HaluEval_ministral-8b_regenerated.json',
                       help='Output file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=150)
    parser.add_argument('--test', type=int, default=None, help='Only process the first N failed samples')
    args = parser.parse_args()
    
    # Load original data
    print(f"Loading original data: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Find failed samples
    failed_samples = [d for d in all_data if d.get('generated_answer', '') == '[No response generated]']
    success_samples = [d for d in all_data if d.get('generated_answer', '') != '[No response generated]']
    
    print(f"Total samples: {len(all_data)}")
    print(f"Failed samples: {len(failed_samples)}")
    print(f"Successful samples: {len(success_samples)}")
    
    if args.test:
        failed_samples = failed_samples[:args.test]
        print(f"Test mode: only processing the first {args.test} failed samples")
    
    if not failed_samples:
        print("No failed samples need to be regenerated!")
        return
    
    # Load vLLM
    print("\nLoading vLLM...")
    from vllm import LLM, SamplingParams
    
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    # Improved sampling params
    # 1. Remove stop tokens that may cause issues
    # 2. Add min_tokens to ensure content generation
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=5,  # Generate at least 5 tokens
        temperature=0.1,  # Slightly higher temperature to avoid deterministic empty output
        stop=["[Human]", "[Human]:"],  # Only keep necessary stop tokens
    )
    
    # Get tokenizer
    tokenizer = llm.get_tokenizer()
    
    # Improved system prompt - more explicit response requirement
    SYSTEM_PROMPT = (
        "You are a helpful assistant in a conversation. "
        "You MUST provide a response to continue the dialogue. "
        "Keep your response brief (under 30 words) but always say something relevant."
    )
    
    def build_prompt(sample):
        """Build improved prompt"""
        context = sample.get('context', '')
        knowledge = sample.get('knowledge', '')
        
        # Extract conversation history
        user_content = f"Knowledge: {knowledge}\n\nConversation:\n{context}\n\nPlease provide the next assistant response:"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    # Build all prompts
    print("\nBuilding prompts...")
    prompts = []
    for sample in failed_samples:
        prompt = build_prompt(sample)
        prompts.append(prompt)
    
    # Batch generation
    print(f"\nStarting batch generation ({len(prompts)} samples)...")
    
    regenerated = []
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+args.batch_size]
        batch_samples = failed_samples[i:i+args.batch_size]
        
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for output, sample in zip(outputs, batch_samples):
            raw_text = output.outputs[0].text
            generated_answer = raw_text.strip()
            
            # Clean Assistant: prefix from response
            if generated_answer.startswith("Assistant:"):
                generated_answer = generated_answer[len("Assistant:"):].strip()
            if generated_answer.startswith("[Assistant]:"):
                generated_answer = generated_answer[len("[Assistant]:"):].strip()
            
            # If still empty, try regenerating with higher temperature
            if not generated_answer:
                retry_params = SamplingParams(
                    max_tokens=args.max_tokens,
                    min_tokens=10,
                    temperature=0.5,  # Higher temperature
                    stop=["[Human]", "[Human]:"],
                )
                retry_output = llm.generate([build_prompt(sample)], retry_params)
                generated_answer = retry_output[0].outputs[0].text.strip()
                
                # Clean prefix again
                if generated_answer.startswith("Assistant:"):
                    generated_answer = generated_answer[len("Assistant:"):].strip()
                if generated_answer.startswith("[Assistant]:"):
                    generated_answer = generated_answer[len("[Assistant]:"):].strip()
                
                if not generated_answer:
                    generated_answer = "[Still no response]"
                    print(f"  Index {sample['index']} still failed")
            
            regenerated.append({
                'index': sample['index'],
                'original_generated_answer': sample['generated_answer'],
                'regenerated_answer': generated_answer,
                'prompt': sample.get('prompt', ''),
                'right_response': sample.get('right_response', ''),
                'hallucinated_response': sample.get('hallucinated_response', ''),
                'knowledge': sample.get('knowledge', ''),
                'context': sample.get('context', ''),
            })
    
    # Statistics
    still_failed = sum(1 for r in regenerated if r['regenerated_answer'] in ['[Still no response]', ''])
    success_count = len(regenerated) - still_failed
    
    print(f"\n{'='*50}")
    print(f"Regeneration results:")
    print(f"  Total processed: {len(regenerated)}")
    print(f"  Succeeded: {success_count} ({success_count/len(regenerated)*100:.1f}%)")
    print(f"  Still failed: {still_failed} ({still_failed/len(regenerated)*100:.1f}%)")
    print(f"{'='*50}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(regenerated, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Show some successful examples
    print("\nSuccessfully regenerated examples:")
    print("-"*50)
    success_examples = [r for r in regenerated if r['regenerated_answer'] not in ['[Still no response]', '']][:5]
    for r in success_examples:
        print(f"Index {r['index']}:")
        print(f"  Expected: {r['right_response'][:80]}...")
        print(f"  Regenerated: {r['regenerated_answer'][:80]}...")
        print()


if __name__ == "__main__":
    main()

