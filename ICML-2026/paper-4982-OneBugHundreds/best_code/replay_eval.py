#!/usr/bin/env python3
"""
Replay cached evaluation prompts with modified LLM parameters.
Does NOT require the Linux kernel git repo - uses pre-computed prompts.
"""
import json, os, sys, time
from openai import OpenAI

sys.path.insert(0, '/repo')
sys.path.insert(0, '/autosota_cache/pip_pkgs')

def replay_evaluation(cached_results_path, output_path, model_name, api_key, base_url,
                      temperature=1.0, top_p=1.0, max_tokens=4096,
                      system_prompt=None, frequency_penalty=0.0, presence_penalty=0.0,
                      max_retries=5, max_samples=None):
    """
    Load cached prompts, re-send to API with new parameters, save results.
    """
    with open(cached_results_path) as f:
        cached = json.load(f)
    
    if max_samples:
        cached = cached[:max_samples]
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    messages_template = [
        {"role": "system", "content": system_prompt or "You are a helpful assistant with expertise in coding and security."},
    ]
    
    results = []
    for i, entry in enumerate(cached):
        data_point = {
            'sec_rule': entry.get('sec_rule', ''),
            'seed_cmt': entry.get('seed_cmt', ''),
            'target_cmt': entry.get('target_cmt', ''),
            'unpatched_prompt': entry['unpatched_prompt'],
            'patched_prompt': entry['patched_prompt'],
        }
        
        for prompt_type in ['unpatched', 'patched']:
            prompt_key = f'{prompt_type}_prompt'
            user_prompt = entry[prompt_key]
            
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant with expertise in coding and security."},
                {"role": "user", "content": user_prompt}
            ]
            
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )
                    data_point[f'{prompt_type}_output'] = completion.choices[0].message.content
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = min(60, 2 ** attempt)
                        print(f"  [Retry {attempt+1}/{max_retries}] {e}. Waiting {wait}s...", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  [FAILED] {e}", flush=True)
                        data_point[f'{prompt_type}_output'] = f"Error: {e}"
        
        results.append(data_point)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(cached)}", flush=True)
            # Save intermediate results
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} results to {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cached', required=True, help='Path to cached results JSON')
    ap.add_argument('--output', required=True, help='Output path for new results')
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key', required=True)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=1.0, dest='top_p')
    ap.add_argument('--max-tokens', type=int, default=4096, dest='max_tokens')
    ap.add_argument('--system-prompt', default=None, dest='system_prompt')
    ap.add_argument('--frequency-penalty', type=float, default=0.0, dest='frequency_penalty')
    ap.add_argument('--presence-penalty', type=float, default=0.0, dest='presence_penalty')
    ap.add_argument('--max-samples', type=int, default=None, dest='max_samples')
    args = ap.parse_args()
    
    replay_evaluation(
        cached_results_path=args.cached,
        output_path=args.output,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        max_samples=args.max_samples,
    )
    
    # Run analysis
    from accuracy_checking import OverAllResultsAnalysis
    OverAllResultsAnalysis(args.output)
