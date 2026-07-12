"""
Async benchmark client for vLLM serving throughput.
Measures RPS and TPOT with configurable concurrency.
"""
import json, time, sys, os, asyncio
import aiohttp

BASE_URL = os.environ.get("BASE_URL", "http://localhost:11014")
MODEL = "/models/Qwen3-8B"
DATASET_PATH = "/repo/reproduce/outputs/sharegpt_prompts.jsonl"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "4000"))
MAX_CONCURRENCY = int(os.environ.get("CONCURRENCY", "2048"))
WARMUP = int(os.environ.get("WARMUP", "20"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/repo/reproduce/outputs/bench_v1_sharegpt")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "bench_v1.json")

# Load prompts
prompts = []
output_lens = []
with open(DATASET_PATH) as f:
    for i, line in enumerate(f):
        if i >= NUM_PROMPTS + WARMUP:
            break
        item = json.loads(line)
        prompts.append(item["prompt"])
        output_lens.append(min(item.get("output_len", 100), 500))  # paper cap at 500

# Split warmup and benchmark prompts
warmup_prompts = prompts[:WARMUP]
warmup_lens = output_lens[:WARMUP]
bench_prompts = prompts[WARMUP:WARMUP+NUM_PROMPTS]
bench_lens = output_lens[WARMUP:WARMUP+NUM_PROMPTS]

total_prompts = len(bench_prompts)
print(f"Loaded {total_prompts} benchmark prompts + {len(warmup_prompts)} warmup")

async def send_request(session, prompt, output_len, sem):
    async with sem:
        data = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        req_start = time.time()
        try:
            async with session.post(
                f"{BASE_URL}/v1/completions",
                json=data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                resp_data = await resp.json()
                req_end = time.time()
                latency = req_end - req_start
                tokens = resp_data.get("usage", {}).get("completion_tokens", output_len)
                return {"latency": latency, "output_tokens": tokens, "tpot_ms": latency / max(tokens, 1) * 1000}
        except Exception as e:
            return {"error": str(e)[:100]}

async def main():
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2, limit_per_host=MAX_CONCURRENCY * 2)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        if warmup_prompts:
            print(f"Running {len(warmup_prompts)} warmup requests...")
            warmup_tasks = [send_request(session, warmup_prompts[i], warmup_lens[i], asyncio.Semaphore(MAX_CONCURRENCY)) for i in range(len(warmup_prompts))]
            await asyncio.gather(*warmup_tasks)
            print("Warmup done")

        # Benchmark
        print(f"Starting benchmark: concurrency={MAX_CONCURRENCY}, prompts={total_prompts}")
        start_time = time.time()

        tasks = [send_request(session, bench_prompts[i], bench_lens[i], sem) for i in range(total_prompts)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

    # Calculate metrics
    successful = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not successful:
        print("All requests failed!")
        for e in errors[:5]:
            print(f"  Error: {e}")
        return

    latencies = [r["latency"] for r in successful]
    tpots = [r["tpot_ms"] for r in successful]
    total_output_tokens = sum(r["output_tokens"] for r in successful)

    rps = len(successful) / total_time
    avg_tpot = sum(tpots) / len(tpots)
    avg_latency = sum(latencies) / len(latencies)

    latencies.sort()
    tpots.sort()

    p50_lat = latencies[len(latencies)//2]
    p95_lat = latencies[int(len(latencies)*0.95)]
    p99_lat = latencies[int(len(latencies)*0.99)]

    p50_tpot = tpots[len(tpots)//2]
    p95_tpot = tpots[int(len(tpots)*0.95)]
    p99_tpot = tpots[int(len(tpots)*0.99)]

    tokens_per_sec = total_output_tokens / total_time

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS (v1 baseline)")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {len(successful)}, Failed: {len(errors)}")
    print(f"RPS: {rps:.2f}")
    print(f"Token throughput: {tokens_per_sec:.2f} tokens/s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"")
    print(f"Latency (s):  Mean={avg_latency:.3f}, P50={p50_lat:.3f}, P95={p95_lat:.3f}, P99={p99_lat:.3f}")
    print(f"TPOT (ms):    Mean={avg_tpot:.2f}, P50={p50_tpot:.2f}, P95={p95_tpot:.2f}, P99={p99_tpot:.2f}")

    result_dict = {
        "scheduler": "v1",
        "rps": rps,
        "tpot_ms_mean": avg_tpot,
        "tpot_ms_p50": p50_tpot,
        "tpot_ms_p95": p95_tpot,
        "tpot_ms_p99": p99_tpot,
        "latency_s_mean": avg_latency,
        "latency_s_p50": p50_lat,
        "latency_s_p95": p95_lat,
        "latency_s_p99": p99_lat,
        "total_time_s": total_time,
        "successful_requests": len(successful),
        "failed_requests": len(errors),
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_sec,
        "concurrency": MAX_CONCURRENCY,
        "num_prompts": total_prompts,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
