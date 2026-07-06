#!/bin/bash
# Comprehensive StreamFlow evaluation matching paper Table 1 configuration
# Run inside container: cd /repo && bash run_comprehensive_eval.sh

set -e
cd /repo

echo "============================================"
echo "StreamFlow Comprehensive Evaluation"
echo "Paper: xformer+taesd+VFB, 512x512, 4 steps"
echo "============================================"

# Run the main test_demo_gen.py with paper-matched config
python3 << 'PYEOF'
import torch
import torchvision
import time
import os
import sys
import json
import subprocess
import threading

sys.path.insert(0, '/repo')
from src.scheduler_perflow import PeRFlowScheduler
from diffusers import StableDiffusionPipeline
from src.streamflow.pipeline_batch_pipeline import PipelineBatchStreamFlow

def get_gpu_power():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        return sum(float(l.strip()) for l in lines if l.strip())
    except:
        return 0

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")

# Config matching paper: xformer+taesd+VFB, 512x512, 4-step
MODEL_PATH = "/models/perflow-sd15-dreamshaper"
NUM_IMAGES = 100
WARMUP = 10
WIDTH, HEIGHT = 512, 512
SEED = 1024
GUIDANCE_SCALE = 1.0  # cfg_type="none" in paper's default

print(f"\n[1] Loading model from {MODEL_PATH}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, safety_checker=None,
)
pipe.scheduler = PeRFlowScheduler.from_config(
    pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4
)
pipe.to("cuda", torch.float16)

# Enable xformers
pipe.enable_xformers_memory_efficient_attention()
print("   xformers enabled")

# Try loading TAESD, fallback to original VAE
vae_type = "original"
try:
    from diffusers import AutoencoderTiny
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )
    vae_type = "taesd"
    print("   TinyVAE (taesd) loaded")
except Exception as e:
    print(f"   TAESD unavailable ({e}), using original VAE")

# Test both pipeline batch (VFB) and sequential modes
configs = [
    ("xformer+VFB", True),
    ("xformer+noVFB", False),
]

all_results = []

for config_name, use_vfb in configs:
    print(f"\n[2] Testing: {config_name} (vae={vae_type})")

    stream = PipelineBatchStreamFlow(
        pipe, t_index_list=[0,1,2,3], torch_dtype=torch.float16,
        width=WIDTH, height=HEIGHT, frame_buffer_size=1,
        cfg_type="none", use_pipeline_batch=use_vfb,
        vae_decode_method="normalize", do_add_noise=True,
    )

    prompt = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; A man with brown skin, a beard, and dark eyes"
    neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
    stream.prepare(prompt, neg_prompt, num_inference_steps=4, guidance_scale=GUIDANCE_SCALE)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(WARMUP):
        _ = stream.txt2img()
    torch.cuda.synchronize()

    # Power monitoring
    power_samples = []
    stop_mon = threading.Event()
    def monitor():
        while not stop_mon.is_set():
            p = get_gpu_power()
            if p > 0: power_samples.append(p)
            time.sleep(0.1)
    mon_thread = threading.Thread(target=monitor, daemon=True)
    mon_thread.start()

    # Main eval
    times = []
    eval_start = time.time()
    for i in range(NUM_IMAGES):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = stream.txt2img()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        if i % 25 == 0 or i < 3:
            print(f"   {i+1:3d}/{NUM_IMAGES} | {1.0/times[-1]:.2f} FPS")

    eval_end = time.time()
    stop_mon.set()
    mon_thread.join(timeout=2)

    peak_mem = torch.cuda.max_memory_reserved() / (1024*1024)
    avg_fps = len(times) / sum(times)
    wall_fps = NUM_IMAGES / (eval_end - eval_start)
    avg_power = sum(power_samples)/len(power_samples) if power_samples else 0

    result = {
        "config": config_name,
        "vae_type": vae_type,
        "avg_fps": round(avg_fps, 2),
        "wall_fps": round(wall_fps, 2),
        "avg_power_w": round(avg_power, 2),
        "peak_mem_mb": round(peak_mem, 0),
        "total_time_s": round(sum(times), 2),
    }
    all_results.append(result)

    print(f"   Result: FPS={avg_fps:.2f}, Power={avg_power:.1f}W, PeakMem={peak_mem:.0f}MB")

# Summary
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
for r in all_results:
    print(f"  {r['config']}: FPS={r['avg_fps']}, Power={r['avg_power_w']}W, Mem={r['peak_mem_mb']}MB")

if len(all_results) >= 2:
    speedup = all_results[0]['avg_fps'] / all_results[1]['avg_fps']
    print(f"\n  VFB Speedup: {speedup:.2f}x")

# Save
with open("/repo/eval_final.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to /repo/eval_final.json")
PYEOF
