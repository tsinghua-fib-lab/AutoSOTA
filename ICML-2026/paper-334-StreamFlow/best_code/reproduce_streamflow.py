#!/usr/bin/env python3
"""
StreamFlow Reproduction Script
Paper: StreamFlow - High-Efficiency Rectified Flow Generation
Configuration: xformer+taesd+VFB, 512x512, 4-step, 100 images

Usage (inside container):
    cd /repo && python3 reproduce_streamflow.py

Measures: FPS, Power (W), Peak Memory (MB)
Output: /repo/reproduce_results.json
"""

import torch, time, sys, json, subprocess, threading, os
sys.path.insert(0, '/repo')

from src.scheduler_perflow import PeRFlowScheduler
from diffusers import StableDiffusionPipeline
from src.streamflow.pipeline_batch_pipeline import PipelineBatchStreamFlow


def get_gpu_power():
    """Get total GPU power draw in watts."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=5)
        return sum(float(l.strip()) for l in r.stdout.strip().split('\n') if l.strip())
    except:
        return 0.0


def main():
    MODEL_PATH = "/models/perflow-sd15-dreamshaper"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "hansyan/perflow-sd15-dreamshaper"

    WIDTH, HEIGHT = 512, 512
    NUM_IMAGES = 100
    WARMUP = 10

    print(f"StreamFlow Reproduction | PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("[1/3] Loading PeRFlow model...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
    pipe.to("cuda", torch.float16)
    pipe.enable_xformers_memory_efficient_attention()
    print("   Model loaded, xformers enabled")

    # Replace original VAE with TAESD (paper Table 1 config, ~300MB memory savings)
    print("   Loading TAESD (Tiny AutoEncoder) from local cache...")
    from diffusers import AutoencoderTiny
    taesd_cache = "/autosota_cache/hf/hub/models--madebyollin--taesd/snapshots/main"
    taesd = AutoencoderTiny.from_pretrained(taesd_cache, local_files_only=True, torch_dtype=torch.float16).to("cuda")
    pipe.vae = taesd
    print(f"   TAESD VAE loaded ({sum(p.numel() for p in taesd.parameters()):,} params vs 49.5M original)")

    # GPU optimizations: TF32 matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    print("   TF32 enabled")

    # Create StreamFlow pipeline (VFB = Velocity Field Batching enabled)
    print("[2/3] Building StreamFlow pipeline (VFB)...")
    stream = PipelineBatchStreamFlow(
        pipe, t_index_list=[0, 1, 2, 3], torch_dtype=torch.float16,
        width=WIDTH, height=HEIGHT, frame_buffer_size=2,
        cfg_type="none", use_pipeline_batch=True,
        vae_decode_method="normalize", do_add_noise=True,
    )
    prompt = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; A man with brown skin, a beard, and dark eyes"
    neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
    stream.prepare(prompt, neg_prompt, num_inference_steps=4, guidance_scale=1.0)
    print("   Pipeline ready (4-step, 512x512)")

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Warmup
    print(f"[3/3] Warmup ({WARMUP} images) + Evaluation ({NUM_IMAGES} images)...")
    for _ in range(WARMUP):
        _ = stream.txt2img()
    torch.cuda.synchronize()

    # Power monitoring
    power_readings = []
    evt_stop = threading.Event()
    def power_mon():
        while not evt_stop.is_set():
            p = get_gpu_power()
            if p > 0: power_readings.append(p)
            time.sleep(0.1)
    mon = threading.Thread(target=power_mon, daemon=True)
    mon.start()

    # Main evaluation with batched VAE decode (TAESD decoder is small enough for batching)
    VAE_BATCH_SIZE = 4
    times = []
    latent_accum = []
    for i in range(NUM_IMAGES):
        if len(latent_accum) == 0:
            torch.cuda.synchronize()
            batch_t0 = time.time()
        latent = stream.generate_latent()
        latent_accum.append(latent)

        if len(latent_accum) >= VAE_BATCH_SIZE or i == NUM_IMAGES - 1:
            batch = torch.cat(latent_accum, dim=0)
            images = stream.decode_latents(batch)
            torch.cuda.synchronize()
            elapsed = time.time() - batch_t0
            per_image = elapsed / len(latent_accum)
            for _ in range(len(latent_accum)):
                times.append(per_image)
            if i < 3 or (i+1) % 25 == 0:
                print(f"   Image {i+1:3d}/{NUM_IMAGES} | {1.0/per_image:.2f} FPS (batch={len(latent_accum)})")
            latent_accum = []

    evt_stop.set()
    mon.join(timeout=2)

    # Compute metrics
    avg_fps = len(times) / sum(times)
    peak_mem = torch.cuda.max_memory_reserved() / (1024 * 1024)
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0.0

    results = {
        "config": "xformer+taesd+VFB",
        "paper_config": "xformer+taesd+VFB",
        "image_size": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": 4,
        "num_images": NUM_IMAGES,
        "avg_fps": round(avg_fps, 2),
        "avg_power_w": round(avg_power, 2),
        "peak_mem_mb": round(peak_mem, 0),
        "gpu": torch.cuda.get_device_name(0),
        "vae_type": "TAESD (madebyollin/taesd)",
    }

    print(f"\n{'='*50}")
    print(f"REPRODUCTION RESULTS")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open("/repo/reproduce_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /repo/reproduce_results.json")
    return results


if __name__ == "__main__":
    main()
