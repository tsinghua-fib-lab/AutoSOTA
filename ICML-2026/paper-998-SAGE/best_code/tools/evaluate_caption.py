import asyncio
import aiohttp
import json
import time
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm

# API address
url = "http://your-api-server:port/v1/completions"
# url = "http://your-api-server:port/v1/chat/completions"

CAPTION_EVALUATION_PROMPT = '''Please evaluate the quality of the predicted caption compared to the true captions.

True Captions (reference captions, all describe the same image):
{true_captions}

Predicted Caption (to be evaluated):
{predicted_caption}

Scoring Rules:
- Score 5: Perfect match - the predicted caption has the same semantic meaning as the true captions
- Score 4: Excellent - captures all key elements with minor differences in wording
- Score 3: Good (passing) - captures main elements but misses some details
- Score 2: Fair - captures some elements but has significant gaps
- Score 1: Poor - barely related to the true captions
- Score 0: Completely unrelated to the true captions

Important:
- Focus on SEMANTIC MEANING, not exact wording
- A score of 3 or above means the predicted caption is acceptable
- Consider if the predicted caption describes the same scene/object as the true captions

Return ONLY a single number (0, 1, 2, 3, 4, or 5) with no explanation.

Your score:'''

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch evaluate VLM generated caption quality (with auto-retry)')

    parser.add_argument('--input', '-i', type=str, required=True, help='Input dataset path (JSON file)')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output result path (JSON file)')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size per processing round, default 50')
    parser.add_argument('--save-interval', '-s', type=int, default=200, help='Auto-save interval for intermediate results, default every 200 entries')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum retry count per failed request, default 5')
    parser.add_argument('--retry-delay', type=float, default=2.0, help='Delay between retries (seconds), default 2s')
    parser.add_argument('--test', type=int, default=None, help='Test mode: only process the first N entries')
    parser.add_argument('--resume', action='store_true', help='Resume from interruption (read existing output file)')
    
    return parser.parse_args()

def load_dataset(file_path):
    """Load dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_prompt(item):
    """Create prompt based on data item"""
    true_captions_str = '\n'.join(f"{i+1}. {cap.strip()}" for i, cap in enumerate(item['true_captions']))
    return CAPTION_EVALUATION_PROMPT.format(
        true_captions=true_captions_str,
        predicted_caption=item['predicted_caption']
    )

async def send_request_with_retry(session, item, index, max_retries, retry_delay, pbar=None):
    """Send a single async request with auto-retry support"""
    prompt = create_prompt(item)
    payload = {"messages": [{"role": "user", "content": prompt}], "stream": False}
    
    for attempt in range(max_retries):
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload) as response:  # Use session's timeout configuration
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        assistant_message = data['choices'][0]['message']['content'].strip()
                        elapsed_time = time.time() - start_time
                        
                        try:
                            score = int(assistant_message)
                            if 0 <= score <= 5:
                                is_valid = True
                                # Modified: score >= 4 to be considered excellent
                                is_passing = score >= 4
                                
                                if pbar:
                                    pbar.update(1)
                                
                                return {
                                    'index': index,
                                    'original_index': item.get('index', index),
                                    'image_id': item.get('metadata', {}).get('image_id', ''),
                                    'predicted_caption': item['predicted_caption'],
                                    'true_captions': item['true_captions'],
                                    'score': score,
                                    'is_passing': is_passing,
                                    'is_valid_score': is_valid,
                                    'raw_response': assistant_message,
                                    'time': elapsed_time,
                                    'success': True,
                                    'retry_count': attempt
                                }
                            else:
                                if attempt < max_retries - 1:
                                    print(f"\n[Index {index}] Attempt {attempt+1}/{max_retries}: score out of range ({score})")
                        except ValueError:
                            if attempt < max_retries - 1:
                                print(f"\n[Index {index}] Attempt {attempt+1}/{max_retries}: unable to parse score ({assistant_message[:50]})")
                else:
                    if attempt < max_retries - 1:
                        print(f"\n[Index {index}] Attempt {attempt+1}/{max_retries}: status code {response.status}")
                
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(f"\n[Index {index}] Attempt {attempt+1}/{max_retries}: request timeout")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n[Index {index}] Attempt {attempt+1}/{max_retries}: {str(e)}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
    
    if pbar:
        pbar.update(1)
    
    return {
        'index': index,
        'original_index': item.get('index', index),
        'error': f"Still failed after {max_retries} retries",
        'success': False,
        'retry_count': max_retries
    }

async def process_batch(session, batch, start_idx, max_retries, retry_delay, pbar):
    """Process a batch of data"""
    tasks = [
        send_request_with_retry(session, item, start_idx + i, max_retries, retry_delay, pbar) 
        for i, item in enumerate(batch)
    ]
    return await asyncio.gather(*tasks)

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
    """Main function: batch concurrent execution with auto-retry"""
    args = parse_args()
    
    print("=" * 80)
    print("VLM Caption Quality Evaluation Task Started (score >= 4 is excellent)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size} entries/batch")
    print(f"Auto-save interval: {args.save_interval} entries")
    print(f"Max retries: {args.max_retries}")
    print(f"Retry delay: {args.retry_delay}s")
    print(f"Criteria: >= 4 is excellent, <= 3 is below standard")
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
    
    print(f"=== Starting batch concurrent requests ===\n")
    start_total = time.time()
    
    timeout = aiohttp.ClientTimeout(total=1800)  # 30 minutes timeout
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with tqdm(
            total=len(dataset), 
            desc="Overall progress",
            unit="items",
            initial=0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ) as pbar:
            for i in range(0, len(dataset), args.batch_size):
                batch = dataset[i:i+args.batch_size]
                batch_results = await process_batch(
                    session, batch, start_idx + i, 
                    args.max_retries, args.retry_delay, pbar
                )
                all_results.extend(batch_results)
                
                if (i + args.batch_size) % args.save_interval == 0 or (i + args.batch_size) >= len(dataset):
                    save_results(all_results, args.output, is_final=False)
                    
                    current_success = sum(1 for r in all_results if r['success'])
                    current_failed = len(all_results) - current_success
                    current_valid = sum(1 for r in all_results if r.get('is_valid_score', False))
                    current_excellent = sum(1 for r in all_results if r.get('is_passing', False))  # >= 4 points
                    
                    if current_valid > 0:
                        valid_scores = [r['score'] for r in all_results if r.get('is_valid_score', False)]
                        avg_score = sum(valid_scores) / len(valid_scores)
                        excellent_rate = (current_excellent / current_valid) * 100
                        
                        print(f"   Current stats: succeeded {current_success}, failed {current_failed}, "
                              f"avg score {avg_score:.2f}, excellent rate (>=4) {excellent_rate:.1f}%")
    
    # Calculate overall statistics
    total_time = time.time() - start_total
    success_count = sum(1 for r in all_results if r['success'])
    fail_count = len(all_results) - success_count
    valid_count = sum(1 for r in all_results if r.get('is_valid_score', False))
    excellent_count = sum(1 for r in all_results if r.get('is_passing', False))  # >= 4 points
    
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    print(f"Total requests: {len(all_results)}")
    print(f"Succeeded: {success_count} ({success_count/len(all_results)*100:.1f}%)")
    print(f"Failed: {fail_count} ({fail_count/len(all_results)*100:.1f}%)")
    print(f"Valid scores: {valid_count}")
    
    if valid_count > 0:
        valid_scores = [r['score'] for r in all_results if r.get('is_valid_score', False)]
        avg_score = sum(valid_scores) / len(valid_scores)
        excellent_rate = (excellent_count / valid_count) * 100
        
        # Additional: count below-standard scores
        poor_count = sum(1 for r in all_results if r.get('is_valid_score', False) and r['score'] <= 3)
        poor_rate = (poor_count / valid_count) * 100
        
        score_distribution = {i: valid_scores.count(i) for i in range(6)}
        
        print(f"\nScore statistics:")
        print(f"   Average score: {avg_score:.2f}/5.0")
        print(f"   Excellent count (>=4): {excellent_count} / {valid_count}")
        print(f"   Excellent rate: {excellent_rate:.2f}%")
        print(f"   Below standard count (<=3): {poor_count} / {valid_count}")
        print(f"   Below standard rate: {poor_rate:.2f}%")
        
        print(f"\nScore distribution:")
        for score in range(5, -1, -1):
            count = score_distribution.get(score, 0)
            percentage = (count / valid_count) * 100
            bar = '█' * int(percentage / 2)
            
            if score >= 4:
                label = f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar} Excellent"
            elif score == 3:
                label = f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar} Below standard (borderline)"
            else:
                label = f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar} Below standard"
            
            print(label)
        
        successful_times = [r['time'] for r in all_results if r['success']]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            min_time = min(successful_times)
            max_time = max(successful_times)
            
            print(f"\nTime statistics:")
            print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
            print(f"   Average response: {avg_time:.2f}s")
            print(f"   Fastest response: {min_time:.2f}s")
            print(f"   Slowest response: {max_time:.2f}s")
            print(f"   Throughput: {len(all_results)/total_time:.2f} items/s")
    
    retry_counts = [r.get('retry_count', 0) for r in all_results if r['success']]
    if retry_counts:
        retried = sum(1 for c in retry_counts if c > 0)
        print(f"\nRetry statistics:")
        print(f"   Needed retry: {retried}/{success_count} ({retried/success_count*100:.1f}%)")
        print(f"   Average retry count: {sum(retry_counts)/len(retry_counts):.2f}")
    
    if fail_count > 0:
        print(f"\n{fail_count} entries failed:")
        failed = [r for r in all_results if not r['success']]
        for r in failed[:10]:
            print(f"   - Index {r['original_index']}: {r.get('error', 'Unknown')}")
        if fail_count > 10:
            print(f"   ... and {fail_count - 10} more")
    
    print("=" * 80)
    
    save_results(all_results, args.output, is_final=True)
    
    return all_results

if __name__ == "__main__":
    results = asyncio.run(main())
