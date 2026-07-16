"""
VLM Caption single-image reference-free evaluation - using external API endpoint
- Uses text-only LLM API to evaluate caption quality
- Does not require reference images or true captions
- Scoring criteria consistent with neighbor_based_vlm_evaluator_vllm.py single mode
- Batch concurrent requests

Usage:
    python tools/vlm_single_eval_api.py \
        --input outputs/vlm_tagging/COCO_internvl3.5-8b_v1.json \
        --output outputs/vlm_tagging/COCO_internvl_single_api_scores.json \
        --batch-size 32 --concurrency 4
"""
import asyncio
import aiohttp
import json
import time
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm
from typing import List, Tuple

# API address
URL = "http://your-api-server:port/v1/completions"

# Batch configuration
BATCH_SIZE = 32          # Number of prompts aggregated per request
BATCH_CONCURRENCY = 4    # Number of concurrent batches
MAX_TOKENS = 2           # Only need to output a single number
TEMPERATURE = 0

# Single-image reference-free scoring prompt (consistent with neighbor_based_vlm_evaluator_vllm.py single mode)
SINGLE_IMAGE_EVALUATION_PROMPT = '''You are evaluating the quality of an image caption. Assess how well the caption describes a typical image based on the caption text alone.

Caption to Evaluate: {predicted_caption}

Scoring Rules:
- Score 5: Excellent - detailed, specific, and sounds like a comprehensive description
- Score 4: Very Good - captures important elements with good detail
- Score 3: Acceptable - describes a main subject but lacks some details or specificity
- Score 2: Below Average - vague, generic, or missing significant details
- Score 1: Poor - very brief, unclear, or seems incomplete
- Score 0: Completely Inadequate - not a valid caption or makes no sense

Evaluation Criteria:
1. Specificity: Does it describe specific objects, actions, or scenes?
2. Completeness: Does it seem like a full description?
3. Clarity: Is the description clear and understandable?
4. Quality: Is it grammatically correct and well-formed?

Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

Score:'''


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VLM Caption single-image reference-free evaluation (external API version)')

    parser.add_argument('--input', '-i', type=str, required=True, help='Input dataset path (JSON file)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output result path (JSON file)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Number of prompts per request, default 32')
    parser.add_argument('--concurrency', '-c', type=int, default=4, help='Number of concurrent batches, default 4')
    parser.add_argument('--save-interval', '-s', type=int, default=200, help='Auto-save interval for intermediate results, default every 200 entries')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum retry count per failed batch, default 5')
    parser.add_argument('--retry-delay', type=float, default=2.0, help='Delay between retries (seconds), default 2s')
    parser.add_argument('--test', type=int, default=None, help='Test mode: only process the first N entries')
    parser.add_argument('--resume', action='store_true', help='Resume from interruption (read existing output file)')
    parser.add_argument('--url', type=str, default=URL, help=f'API address, default: {URL}')
    
    return parser.parse_args()


def load_dataset(file_path):
    """Load dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_prompt(item):
    """Create prompt based on data item"""
    return SINGLE_IMAGE_EVALUATION_PROMPT.format(
        predicted_caption=item.get('predicted_caption', '')
    )


def chunk_with_index(items: List, batch_size: int) -> List[List[Tuple[int, dict]]]:
    """Split items into batches, preserving original index for result backfill"""
    chunks = []
    cur = []
    for i, item in enumerate(items):
        cur.append((i, item))
        if len(cur) == batch_size:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def parse_score(response: str) -> Tuple[int, bool]:
    """Parse the returned score"""
    try:
        score = int(response.strip())
        if 0 <= score <= 5:
            return score, True
    except ValueError:
        pass
    return -1, False


async def send_one_batch(session: aiohttp.ClientSession, batch: List[Tuple[int, dict]], 
                         batch_id: int, max_retries: int, retry_delay: float, url: str):
    """Send one batch (containing multiple prompts) with retry support"""
    indices = [i for i, _ in batch]
    items = [item for _, item in batch]
    prompts = [create_prompt(item) for item in items]
    
    for attempt in range(max_retries):
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompts,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": False,
            }
            
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if attempt < max_retries - 1:
                        print(f"\n[Batch {batch_id}] Attempt {attempt+1}/{max_retries}: HTTP {resp.status}")
                        await asyncio.sleep(retry_delay)
                        continue
                    raise RuntimeError(f"Batch {batch_id} failed: HTTP {resp.status}, body={text[:300]}")
                
                data = await resp.json()
            
            elapsed_time = time.time() - start_time
            choices = data.get("choices", [])
            
            if len(choices) != len(prompts):
                if attempt < max_retries - 1:
                    print(f"\n[Batch {batch_id}] Attempt {attempt+1}/{max_retries}: choices count mismatch")
                    await asyncio.sleep(retry_delay)
                    continue
                raise RuntimeError(f"Batch {batch_id} choices mismatch: {len(choices)} vs {len(prompts)}")
            
            # Parse results
            results = []
            for idx, item, choice in zip(indices, items, choices):
                raw_response = choice.get("text", "").strip()
                score, is_valid = parse_score(raw_response)
                is_passing = score >= 3 if is_valid else False  # Consistent with vLLM version, >= 3 is passing
                
                results.append({
                    'global_id': idx,
                    'sample_index': item.get('index', idx),
                    'predicted_caption': item.get('predicted_caption', ''),
                    'true_captions': item.get('true_captions', []),
                    'score': score if is_valid else None,
                    'score_str': str(score) if is_valid else 'INVALID',
                    'is_valid': is_valid,
                    'is_passing': is_passing,
                    'raw_response': raw_response,
                    'time': elapsed_time / len(prompts),
                    'retry_count': attempt
                })
            
            print(f"[batch {batch_id}] size={len(prompts)} time={elapsed_time:.2f}s "
                  f"avg={elapsed_time/len(prompts):.3f}s/prompt")
            
            return results
            
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(f"\n[Batch {batch_id}] Attempt {attempt+1}/{max_retries}: request timeout")
                await asyncio.sleep(retry_delay)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n[Batch {batch_id}] Attempt {attempt+1}/{max_retries}: {str(e)}")
                await asyncio.sleep(retry_delay)
    
    # All retries failed, return failure results
    results = []
    for idx, item in zip(indices, items):
        results.append({
            'global_id': idx,
            'sample_index': item.get('index', idx),
            'predicted_caption': item.get('predicted_caption', ''),
            'true_captions': item.get('true_captions', []),
            'score': None,
            'score_str': 'ERROR',
            'is_valid': False,
            'is_passing': False,
            'raw_response': f"Still failed after {max_retries} retries",
            'retry_count': max_retries
        })
    return results


def save_results(results, output_path, is_final=False):
    """Save results to file"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f"\nFinal results saved to: {output_path}")
    else:
        print(f"\nIntermediate results saved: {len(results)} entries")


def load_existing_results(output_path):
    """Load existing results (for resumption)"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Found existing results file, completed {len(results)} entries")
        return results
    except FileNotFoundError:
        print("No existing results file found, starting from scratch")
        return []


async def main():
    """Main function: batch aggregation concurrent execution"""
    args = parse_args()
    
    global BATCH_SIZE, BATCH_CONCURRENCY
    BATCH_SIZE = args.batch_size
    BATCH_CONCURRENCY = args.concurrency
    
    print("=" * 80)
    print("VLM Caption Single-Image Reference-Free Evaluation (External API Version)")
    print("   - Scoring criteria consistent with vLLM version single mode")
    print("   - Score >= 3 is passing")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"API address: {args.url}")
    print(f"Prompts per request: {args.batch_size}")
    print(f"Concurrent batches: {args.concurrency}")
    print(f"Auto-save interval: {args.save_interval} entries")
    print(f"Max retries: {args.max_retries}")
    if args.test:
        print(f"Test mode: only process the first {args.test} entries")
    if args.resume:
        print(f"Resume mode: enabled")
    print("=" * 80 + "\n")
    
    print(f"Loading dataset...")
    dataset = load_dataset(args.input)
    print(f"Dataset loaded, total {len(dataset)} entries\n")
    
    if args.test:
        dataset = dataset[:args.test]
        print(f"Test mode enabled, only processing the first {args.test} entries\n")
    
    start_idx = 0
    all_results = []
    if args.resume:
        all_results = load_existing_results(args.output)
        start_idx = len(all_results)
        if start_idx >= len(dataset):
            print("All data has been processed!")
            return all_results
        dataset = dataset[start_idx:]
        print(f"Resuming from entry {start_idx}, {len(dataset)} remaining\n")
    
    print(f"=== Starting batch aggregation concurrent requests ===\n")
    start_total = time.time()
    
    # Split into batches
    batches = chunk_with_index(dataset, BATCH_SIZE)
    print(f"Total {len(dataset)} entries, split into {len(batches)} batches\n")
    
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=1800)
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        
        async def runner(batch_id: int, batch):
            async with sem:
                return await send_one_batch(session, batch, batch_id, 
                                           args.max_retries, args.retry_delay, args.url)
        
        # Create all tasks
        tasks = [asyncio.create_task(runner(bid, b)) for bid, b in enumerate(batches)]
        
        # Use tqdm to display progress
        with tqdm(total=len(dataset), desc="Overall progress", unit="items") as pbar:
            for coro in asyncio.as_completed(tasks):
                batch_results = await coro
                all_results.extend(batch_results)
                pbar.update(len(batch_results))
                
                # Periodically save
                if len(all_results) % args.save_interval < BATCH_SIZE:
                    save_results(all_results, args.output, is_final=False)
    
    # Sort by original order
    all_results.sort(key=lambda x: x['global_id'])
    
    # Calculate overall statistics
    total_time = time.time() - start_total
    valid_count = sum(1 for r in all_results if r.get('is_valid', False))
    passing_count = sum(1 for r in all_results if r.get('is_passing', False))
    
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    print(f"Total samples: {len(all_results)}")
    print(f"Valid scores: {valid_count} ({valid_count/len(all_results)*100:.1f}%)")
    
    if valid_count > 0:
        valid_scores = [r['score'] for r in all_results if r.get('is_valid', False)]
        avg_score = sum(valid_scores) / len(valid_scores)
        passing_rate = (passing_count / valid_count) * 100
        
        score_distribution = {i: valid_scores.count(i) for i in range(6)}
        
        print(f"\nScore statistics:")
        print(f"   Average score: {avg_score:.2f}/5.0")
        print(f"   Passing count (>=3): {passing_count} / {valid_count}")
        print(f"   Passing rate: {passing_rate:.2f}%")
        
        print(f"\nScore distribution:")
        for score in range(5, -1, -1):
            count = score_distribution.get(score, 0)
            percentage = (count / valid_count) * 100
            bar = '█' * int(percentage / 2)
            
            if score >= 3:
                label = f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar} PASS"
            else:
                label = f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar} FAIL"
            
            print(label)
        
        print(f"\nTime statistics:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"   Throughput: {len(all_results)/total_time:.2f} items/s")
    
    print("=" * 80)
    
    save_results(all_results, args.output, is_final=True)
    
    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())

