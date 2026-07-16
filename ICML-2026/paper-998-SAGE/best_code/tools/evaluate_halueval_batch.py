"""
Batch evaluate HaluEval LLM generation quality - using /v1/completions batch aggregation method
- Determine whether the generated response is a hallucination
- Evaluate based on knowledge, right_response, hallucinated_response
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
MAX_TOKENS = 10          # Only need to output true/false
TEMPERATURE = 0

# HaluEval Evaluation Prompt
HALUEVAL_EVALUATION_PROMPT = '''Evaluate if the generated response is factually correct.

You may use both the provided Knowledge and your general world knowledge.
If the generated response is correct in the real world, mark it as "true",
even if the fact is not explicitly stated in Knowledge.

Knowledge (may be incomplete):
{knowledge}

Conversation Context:
{context}

Generated Response (to evaluate):
{generated_answer}

Evaluation Rules:
1. INCORRECTNESS CHECK:
   - Contains factual errors according to real-world knowledge → "false"
   - Contradicts the provided Knowledge → "false"

2. CORRECTNESS CHECK:
   - Factually correct in the real world → "true"
   - Reasonable and truthful continuation of the context → "true"

3. SPECIAL CASES:
   - Evasive or non-committal answers → "false"
   - Opinions without factual claims → "true"
   - Empty or nonsensical responses → "false"

Output ONLY "true" or "false":'''


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch evaluate HaluEval LLM generation quality')

    parser.add_argument('--input', '-i', type=str, required=True, help='Input dataset path (JSON file)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output result path (default: append _evaluated to input filename)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Number of prompts per request, default 32')
    parser.add_argument('--concurrency', '-c', type=int, default=4, help='Number of concurrent batches, default 4')
    parser.add_argument('--save-interval', '-s', type=int, default=200, help='Auto-save interval for intermediate results, default every 200 entries')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum retry count per failed batch, default 5')
    parser.add_argument('--retry-delay', type=float, default=2.0, help='Delay between retries (seconds), default 2s')
    parser.add_argument('--test', type=int, default=None, help='Test mode: only process the first N entries')
    parser.add_argument('--resume', action='store_true', help='Resume from interruption (read existing output file)')
    
    return parser.parse_args()


def load_dataset(file_path):
    """Load dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_prompt(item):
    """Create evaluation prompt based on data item"""
    return HALUEVAL_EVALUATION_PROMPT.format(
        knowledge=item.get('knowledge', ''),
        context=item.get('context', ''),
        right_response=item.get('right_response', ''),
        hallucinated_response=item.get('hallucinated_response', ''),
        generated_answer=item.get('generated_answer', '')
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


def parse_result(response: str) -> Tuple[bool, bool]:
    """Parse returned result, returns (is_correct, is_valid)"""
    response = response.strip().lower()
    if response in ['true', '"true"', "'true'"]:
        return True, True
    elif response in ['false', '"false"', "'false'"]:
        return False, True
    return False, False


async def send_one_batch(session: aiohttp.ClientSession, batch: List[Tuple[int, dict]], 
                         batch_id: int, max_retries: int, retry_delay: float):
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
            
            async with session.post(URL, json=payload) as resp:
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
                is_correct, is_valid = parse_result(raw_response)
                
                # Preserve original data and add evaluation results
                result = item.copy()
                result.update({
                    'evaluation_index': idx,
                    'is_correct': is_correct if is_valid else None,
                    'is_hallucination': (not is_correct) if is_valid else None,
                    'is_valid_eval': is_valid,
                    'eval_raw_response': raw_response,
                    'eval_time': elapsed_time / len(prompts),
                    'eval_success': is_valid,
                    'eval_retry_count': attempt
                })
                results.append(result)
            
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
        result = item.copy()
        result.update({
            'evaluation_index': idx,
            'eval_error': f"Still failed after {max_retries} retries",
            'eval_success': False,
            'eval_retry_count': max_retries
        })
        results.append(result)
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
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_evaluated{input_path.suffix}")
    
    print("=" * 80)
    print("HaluEval LLM Generation Quality Evaluation Task Started")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Prompts per request: {args.batch_size}")
    print(f"Concurrent batches: {args.concurrency}")
    print(f"Auto-save interval: {args.save_interval} entries")
    print(f"Max retries: {args.max_retries}")
    print(f"Retry delay: {args.retry_delay}s")
    print(f"Criteria: true=correct (non-hallucination), false=hallucination/error")
    if args.test:
        print(f"Test mode: only process the first {args.test} entries")
    if args.resume:
        print(f"Resume mode: enabled")
    print("=" * 80 + "\n")
    
    print(f"Loading dataset...")
    dataset = load_dataset(args.input)
    print(f"Dataset loaded, total {len(dataset)} entries\n")
    
    # Filter out entries with empty answers (filter first, then handle resume)
    original_count = len(dataset)
    dataset = [item for item in dataset if item.get('generated_answer', '').strip()]
    filtered_count = original_count - len(dataset)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} entries with empty answers\n")
    
    if args.test:
        dataset = dataset[:args.test]
        print(f"Test mode enabled, only processing the first {args.test} entries\n")
    
    all_results = []
    if args.resume:
        all_results = load_existing_results(args.output)
        # Match processed data by index instead of simple slicing
        processed_indices = set(r.get('index') for r in all_results if 'index' in r)
        remaining_dataset = [item for item in dataset if item.get('index') not in processed_indices]
        
        if not remaining_dataset:
            print("All data has been processed!")
            return all_results
        
        print(f"Processed {len(all_results)} entries, {len(remaining_dataset)} remaining\n")
        dataset = remaining_dataset
    
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
                                           args.max_retries, args.retry_delay)
        
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
    all_results.sort(key=lambda x: x.get('evaluation_index', x.get('index', 0)))
    
    # Calculate overall statistics
    total_time = time.time() - start_total
    success_count = sum(1 for r in all_results if r.get('eval_success', False))
    fail_count = len(all_results) - success_count
    valid_count = sum(1 for r in all_results if r.get('is_valid_eval', False))
    correct_count = sum(1 for r in all_results if r.get('is_correct', False))
    hallucination_count = sum(1 for r in all_results if r.get('is_hallucination', False))
    
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    print(f"Total evaluations: {len(all_results)}")
    print(f"Evaluation succeeded: {success_count} ({success_count/len(all_results)*100:.1f}%)")
    print(f"Evaluation failed: {fail_count} ({fail_count/len(all_results)*100:.1f}%)")
    print(f"Valid evaluations: {valid_count}")
    
    if valid_count > 0:
        correct_rate = (correct_count / valid_count) * 100
        hallucination_rate = (hallucination_count / valid_count) * 100
        
        print(f"\nHallucination detection statistics:")
        print(f"   Correct (non-hallucination): {correct_count} / {valid_count} ({correct_rate:.2f}%)")
        print(f"   Hallucination/error: {hallucination_count} / {valid_count} ({hallucination_rate:.2f}%)")
        
        print(f"\nTime statistics:")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"   Throughput: {len(all_results)/total_time:.2f} items/s")
    
    if fail_count > 0:
        print(f"\n{fail_count} entries failed evaluation:")
        failed = [r for r in all_results if not r.get('eval_success', False)]
        for r in failed[:10]:
            print(f"   - Index {r.get('index', '?')}: {r.get('eval_error', 'Unknown')}")
        if fail_count > 10:
            print(f"   ... and {fail_count - 10} more")
    
    print("=" * 80)
    
    save_results(all_results, args.output, is_final=True)
    
    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())

