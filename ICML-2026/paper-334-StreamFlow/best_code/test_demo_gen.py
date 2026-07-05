#!/usr/bin/env python3
"""
Demo: Concise high-performance generator

- 🚀 TensorRT fix (automatic timestep compatibility)
- ⚡ Pipeline batch denoising
- 🔧 Simple config, direct generation
- 📊 Clear performance report
- 📄 YAML config support
"""

import torch, torchvision
from src.scheduler_perflow import PeRFlowScheduler
import time
import os
import sys
import yaml
from pathlib import Path

from diffusers import AutoencoderTiny, StableDiffusionPipeline
from src.streamflow.pipeline_batch_pipeline import PipelineBatchStreamFlow


def load_config_from_yaml(yaml_path):
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def add_tensorrt_timestep_compatibility(stream):
    """
    🚀 Key fix: add TensorRT timestep compatibility

    Problem: TensorRT cannot handle mixed timesteps [1000, 750, 500, 250]
    Solution: when mixed timesteps are detected, split into individual calls
    """
    if not hasattr(stream.unet, 'forward'):
        print("⚠️  UNet has no forward method, skip compatibility patch")
        return None
    
    original_forward = stream.unet.forward
    
    def tensorrt_compatible_forward(sample, timestep, encoder_hidden_states, **kwargs):
        """
        TensorRT-compatible forward
        Automatically handles mixed timesteps
        """
        # Check for mixed timesteps
        if isinstance(timestep, torch.Tensor) and timestep.dim() > 0 and len(timestep) > 1:
            unique_timesteps = torch.unique(timestep)
            if len(unique_timesteps) > 1:
                # 🎯 Detected mixed timesteps, process one by one
                batch_size = sample.shape[0]
                results = []
                for i in range(batch_size):
                    single_sample = sample[i:i+1]
                    single_timestep = timestep[i:i+1]
                    single_encoder_states = encoder_hidden_states[i:i+1]
                    
                    single_result = original_forward(
                        single_sample, single_timestep, single_encoder_states, **kwargs
                    )
                    results.append(single_result)
                
                # Reassemble results
                if isinstance(results[0], tuple):
                    assembled = []
                    for i in range(len(results[0])):
                        assembled.append(torch.cat([r[i] for r in results], dim=0))
                    return tuple(assembled)
                else:
                    return torch.cat(results, dim=0)
        
        # Same timestep or single timestep: call TensorRT directly
        return original_forward(sample, timestep, encoder_hidden_states, **kwargs)
    
    # Apply compatibility fix
    stream.unet.forward = tensorrt_compatible_forward
    print("✅ TensorRT timestep compatibility added")
    return original_forward


# ================================
# 🔧 Configuration section
# ================================

# 🔧 Check if a YAML config is provided via CLI
if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
    CONFIG_PATH = sys.argv[1]
    print(f"📄 Loading config from YAML: {CONFIG_PATH}")
    config = load_config_from_yaml(CONFIG_PATH)
    CONFIG_NAME = config.get('name', 'unnamed')

    # Load config fields
    USE_TINY_VAE = config['model']['use_tiny_vae']
    USE_INT8_VAE = config['model']['use_int8_vae']
    ACCELERATION = config['acceleration']['type']
    USE_CUDA_GRAPH = config['acceleration'].get('use_cuda_graph', False)
    ITERATIONS = config['test']['iterations']

    USE_PIPELINE_BATCH = config['pipeline']['use_pipeline_batch']
    FRAME_BUFFER_SIZE = config['pipeline'].get('frame_buffer_size', 1)
    VAE_DECODE_METHOD = config['pipeline'].get('vae_decode_method', 'normalize')
    DO_ADD_NOISE = config['pipeline'].get('do_add_noise', True)
    CFG_TYPE = config['pipeline']['cfg_type']
    GUIDANCE_SCALE = config['pipeline']['guidance_scale']

    USE_DYNAMIC_STEPS = config['denoising']['use_dynamic_steps']
    NUM_INFERENCE_STEPS = config['denoising']['num_inference_steps']

    USE_TENSORRT_COMPATIBILITY = config['tensorrt']['use_compatibility']
    TENSORRT_OPTIMIZATION = config['tensorrt'].get('optimization', {})

    VAE_BATCH_SIZE = config['vae']['batch_size']

    PROMPT_BASE = config['prompts']['base']
    PROMPT_SUBJECT = config['prompts']['subject']
    NEGATIVE_PROMPT = config['prompts']['negative']

    OUTPUT_DIR = os.path.join(config['test']['output_dir'], CONFIG_NAME)
    SEED = config['test']['seed']
    WIDTH = config['test'].get('width', 512)
    HEIGHT = config['test'].get('height', 512)

else:
    # Default config (keep original hardcoded setup)
    CONFIG_NAME = "default"

    # Base config
    USE_TINY_VAE = True              # True: faster, slight quality drop
    USE_INT8_VAE = False             # Experimental INT8 VAE
    ACCELERATION = "xformers"        # "xformers", "tensorrt", "none"
    USE_CUDA_GRAPH = False           # CUDA Graph optimization
    ITERATIONS = 100                 # Number of images

    # Pipeline config
    USE_PIPELINE_BATCH = True        # True=pipeline batch denoising, False=original
    FRAME_BUFFER_SIZE = 1            # Frame buffer size
    VAE_DECODE_METHOD = "normalize"  # VAE decode method
    DO_ADD_NOISE = True              # Add noise
    CFG_TYPE = "none"               # "none", "full", "self", "initialize"
    GUIDANCE_SCALE = 7.5            # CFG strength

    # Denoising steps config
    USE_DYNAMIC_STEPS = False       # False=fixed 4 steps [0,1,2,3]; True=NUM_INFERENCE_STEPS
    NUM_INFERENCE_STEPS = 4         # Used only when USE_DYNAMIC_STEPS=True

    # TensorRT config
    USE_TENSORRT_COMPATIBILITY = False  # Timestep compatibility layer
    TENSORRT_OPTIMIZATION = {}       # TensorRT build options

    # VAE config
    VAE_BATCH_SIZE = 1  # 1=per image; 4+=batched decode faster but higher latency

    # Prompts
    PROMPT_BASE = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece"
    PROMPT_SUBJECT = "A man with brown skin, a beard, and dark eyes"
    NEGATIVE_PROMPT = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"

    # Output config
    OUTPUT_DIR = "test59_simple_output"
    SEED = 1024
    WIDTH = 512
    HEIGHT = 512

# Optional environment overrides for quick resolution testing
WIDTH = int(os.getenv("STREAMFLOW_WIDTH", WIDTH))
HEIGHT = int(os.getenv("STREAMFLOW_HEIGHT", HEIGHT))

print(f"🚀 PeRFlow high-performance generator - config: {CONFIG_NAME}")
print("=" * 50)
print(f"🔧 Config details:")
vae_desc = "TinyVAE" if USE_TINY_VAE else "Original VAE"
if USE_INT8_VAE:
    vae_desc += " + INT8 quantization"
print(f"   VAE: {vae_desc}")
print(f"   Acceleration: {ACCELERATION}")
print(f"   Pipeline batch: {'✅' if USE_PIPELINE_BATCH else '❌'}")
print(f"   Num images: {ITERATIONS}")
print(f"   Resolution:     {WIDTH}x{HEIGHT}")

# ================================
# 📦 Model loading
# ================================
print(f"\\n📦 Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "hansyan/perflow-sd15-dreamshaper", 
    torch_dtype=torch.float16
)

pipe.scheduler = PeRFlowScheduler.from_config(
    pipe.scheduler.config, 
    prediction_type="diff_eps", 
    num_time_windows=4
)
pipe.to("cuda", torch.float16)
# Reset peak memory stats for monitoring
torch.cuda.reset_peak_memory_stats()

if USE_TINY_VAE:
    if USE_INT8_VAE:
        # Load pre-quantized INT8 TinyVAE
        from utils.quantization import load_quantized_tinyvae
        pipe.vae = load_quantized_tinyvae(device=pipe.device, dtype=pipe.dtype)
    else:
        # Load standard TinyVAE
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=pipe.device, dtype=pipe.dtype
        )

# ================================
# 🚀 Create pipeline
# ================================
print("🚀 Building pipeline...")

# 🔧 Choose timesteps per config
if USE_DYNAMIC_STEPS:
    # Dynamic mode: follow NUM_INFERENCE_STEPS
    t_index_list = list(range(NUM_INFERENCE_STEPS))
    prepare_steps = NUM_INFERENCE_STEPS
    print(f"   Denoise mode: dynamic steps")
    print(f"   Step count: {NUM_INFERENCE_STEPS}")
    print(f"   Timestep indices: {t_index_list}")
else:
    # Fixed mode: preset 4 steps (best quality)
    t_index_list = [0, 1, 2, 3]
    prepare_steps = 4
    print(f"   Denoise mode: fixed 4 steps (quality first)")
    print(f"   Timestep indices: {t_index_list}")

stream = PipelineBatchStreamFlow(
    pipe,
    t_index_list=t_index_list,  # Dynamically generated, follows NUM_INFERENCE_STEPS
    torch_dtype=torch.float16,
    width=WIDTH,
    height=HEIGHT,
    frame_buffer_size=FRAME_BUFFER_SIZE,  # Frame buffer size: 1=no buffer, 2-8=multi-frame buffer
    cfg_type=CFG_TYPE,  # none, full, self, initialize
    use_pipeline_batch=USE_PIPELINE_BATCH,  # Enable pipeline batch denoising
    vae_decode_method=VAE_DECODE_METHOD,  # "normalize", "dynamic", "clamp"
    do_add_noise=DO_ADD_NOISE,  # True=standard, False=fast
)

# ================================
# ⚡ Acceleration
# ================================
if ACCELERATION == "xformers":
    pipe.enable_xformers_memory_efficient_attention()
    print("⚡ xformers enabled")
elif ACCELERATION == "tensorrt":
    print("🚀 Enable TensorRT acceleration...")
    try:
        from src.streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
        from src.streamdiffusion.pipeline import StreamDiffusion

        # 🔧 Choose compile config based on pipeline batch
        if USE_PIPELINE_BATCH:
            # Pipeline mode: handle 4 stages (batch_size=4)
            compile_use_denoising_batch = True
            compile_max_batch_size = 4
            engine_dir = os.path.join("tensorrt_engines_pipeline_batch", CONFIG_NAME)
        else:
            # Sequential mode
            compile_use_denoising_batch = False
            compile_max_batch_size = 2 if GUIDANCE_SCALE > 1.0 and CFG_TYPE != "none" else 1
            engine_dir = os.path.join("tensorrt_engines_sequential", CONFIG_NAME)

        os.makedirs(engine_dir, exist_ok=True)
        print(f"   Engine dir: {engine_dir}")
        print(f"   Compile batch_size: {compile_max_batch_size}")

        temp_stream = StreamDiffusion(
            pipe, t_index_list=t_index_list, torch_dtype=torch.float16,
            frame_buffer_size=1, cfg_type=CFG_TYPE,
            use_denoising_batch=compile_use_denoising_batch,  # 🔧 Based on mode
            width=WIDTH, height=HEIGHT,
        )

        temp_stream.prepare(PROMPT_BASE, NEGATIVE_PROMPT, num_inference_steps=prepare_steps, guidance_scale=GUIDANCE_SCALE)

        print("   Compiling TensorRT engines...")
        accelerated_stream = accelerate_with_tensorrt(
            temp_stream, engine_dir=engine_dir,
            max_batch_size=compile_max_batch_size,  # 🔧 Match runtime batch size
            min_batch_size=1,
            use_cuda_graph=False,
        )
        
        stream.unet = accelerated_stream.unet
        stream.vae = accelerated_stream.vae  # 🚀 TensorRT-accelerated VAE

        # 🚀 Optional timestep compatibility patch
        if USE_PIPELINE_BATCH and USE_TENSORRT_COMPATIBILITY:
            add_tensorrt_timestep_compatibility(stream)
            print("   ⚠️  Timestep compatibility patch applied (slower)")
        elif USE_PIPELINE_BATCH and not USE_TENSORRT_COMPATIBILITY:
            print("   🚀 Compatibility layer disabled, using native TensorRT batch")
        
        del temp_stream, accelerated_stream
        torch.cuda.empty_cache()
        print("✅ TensorRT acceleration enabled")
        
    except Exception as e:
        print(f"❌ TensorRT failed, fallback to no acceleration: {e}")
        ACCELERATION = "none"

# ================================
# 🔥 Warmup
# ================================
# Warmup phase
print(f"\\n🔥 Warming up...")

generator = torch.Generator("cuda").manual_seed(SEED)

prompt_text = f"{PROMPT_BASE}; {PROMPT_SUBJECT}"

# Prepare StreamFlow
stream.prepare(prompt_text, NEGATIVE_PROMPT, num_inference_steps=prepare_steps, guidance_scale=GUIDANCE_SCALE)

# Warmup generation
for i in range(10):
    _ = stream.txt2img()
    if i == 0:
        print("✅ Warmup done")

# ================================
# 🎨 Image generation
# ================================
print(f"\\n🎨 Start generating {ITERATIONS} images...")
print(f"📝 Prompt: {prompt_text}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

if VAE_BATCH_SIZE > 1:
    print(f"🚀 VAE batch decode mode: batch_size={VAE_BATCH_SIZE}")

    # Batched VAE decode
    latent_buffer = []
    buffer_start_idx = []
    buffer_times = []

    for i in range(ITERATIONS):
        torch.cuda.synchronize()
        start_time = time.time()

        # Generate latents only
        latent = stream.generate_latent()

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Accumulate in buffer
        latent_buffer.append(latent)
        buffer_start_idx.append(i)
        buffer_times.append(elapsed)

        # When buffer is full or at the end, decode batch
        if len(latent_buffer) == VAE_BATCH_SIZE or i == ITERATIONS - 1:
            torch.cuda.synchronize()
            decode_start = time.time()

            # Batch decode
            latents_batch = torch.cat(latent_buffer, dim=0)
            images_batch = stream.decode_latents(latents_batch)

            torch.cuda.synchronize()
            decode_time = time.time() - decode_start

            # Spread decode time across images
            decode_time_per_image = decode_time / len(latent_buffer)

            # Save images and record time
            for j, (img_idx, gen_time) in enumerate(zip(buffer_start_idx, buffer_times)):
                total_time = gen_time + decode_time_per_image
                results.append(total_time)

                # Save image
                torchvision.utils.save_image(
                    images_batch[j:j+1],
                    os.path.join(OUTPUT_DIR, f"image_{WIDTH}_{img_idx:06d}.png")
                )

                # Progress
                if img_idx % 10 == 0 or img_idx < 10:
                    img_per_sec = 1 / total_time
                    avg_fps = len(results) / sum(results)
                    print(f"Image {img_idx+1:3d}/{ITERATIONS} | FPS: {img_per_sec:6.2f} | Avg FPS: {avg_fps:6.2f} | UNet: {gen_time:.3f}s | Decode: {decode_time_per_image:.3f}s")

            # Clear buffer
            latent_buffer = []
            buffer_start_idx = []
            buffer_times = []
else:
    print(f"📝 Per-image decode mode (VAE_BATCH_SIZE=1)")

    # Original per-image decode (separate UNet/VAE timing)
    for i in range(ITERATIONS):
        torch.cuda.synchronize()
        unet_start = time.time()

        # Generate latent only
        latent = stream.generate_latent()

        torch.cuda.synchronize()
        unet_time = time.time() - unet_start

        torch.cuda.synchronize()
        vae_start = time.time()

        # Decode
        sample = stream.decode_latents(latent)

        torch.cuda.synchronize()
        vae_time = time.time() - vae_start

        total_time = unet_time + vae_time
        results.append(total_time)

        # Performance
        img_per_sec = 1 / total_time
        avg_fps = len(results) / sum(results)

        # Save image
        torchvision.utils.save_image(
            sample,
            os.path.join(OUTPUT_DIR, f"image_{WIDTH}_{i:06d}.png")
        )

        # Progress
        if i % 10 == 0 or i < 10:
            print(f"Image {i+1:3d}/{ITERATIONS} | FPS: {img_per_sec:6.2f} | Avg FPS: {avg_fps:6.2f} | UNet: {unet_time:.3f}s | Decode: {vae_time:.3f}s")

# ================================
# 📊 Final stats
# ================================
if results:
    avg_time = sum(results) / len(results)
    total_fps = len(results) / sum(results)
    min_time = min(results)
    max_time = max(results)
    
    print(f"\\n" + "=" * 50)
    print(f"📊 Performance")
    print(f"=" * 50)
    print(f"Total images:      {len(results)}")
    print(f"Avg gen time:      {avg_time:.3f}s")
    print(f"Avg FPS:           {total_fps:.2f}")
    print(f"Fastest FPS:       {1/min_time:.2f}")
    print(f"Slowest FPS:       {1/max_time:.2f}")
    print(f"Total time:        {sum(results):.2f}s")
    print(f"Acceleration:      {ACCELERATION}")
    print(f"Pipeline batch:    {'✅' if USE_PIPELINE_BATCH else '❌'}")
    print(f"VAE INT8:          {'✅' if USE_INT8_VAE else '❌'}")
    print(f"VAE batch decode:  {VAE_BATCH_SIZE}")

    # 🎯 Performance verdict
    print(f"\\n💡 Performance tips:")
    if total_fps >= 12:
        print(f"   🎉 Great! Hit 12 FPS target ({total_fps:.1f} FPS)")
    elif total_fps >= 8:
        print(f"   ✅ Good! Close to target ({total_fps:.1f} FPS)")
    elif total_fps >= 6:
        print(f"   ⚡ Not bad! Room to improve ({total_fps:.1f} FPS)")
    else:
        print(f"   🔧 Needs optimization ({total_fps:.1f} FPS)")
    
    # 🎨 Quality note
    if ACCELERATION == "tensorrt" and USE_PIPELINE_BATCH:
        print(f"   🎨 TensorRT fix: noise should be removed")
    
    print(f"\\n📁 Images saved to: {OUTPUT_DIR}")
    
    # 🚀 Suggestions
    print(f"\\n💡 Suggestions:")
    if total_fps < 12:
        if ACCELERATION != "tensorrt":
            print(f"   - Try ACCELERATION = 'tensorrt' for higher performance")
        if not USE_TINY_VAE:
            print(f"   - Try USE_TINY_VAE = True for speed")
        if not USE_PIPELINE_BATCH:
            print(f"   - Try USE_PIPELINE_BATCH = True to enable pipeline speedup")
    else:
        print(f"   🎉 Configuration looks optimized!")

print(f"\\n🎉 Test finished!")
print(f"🔍 Please check image quality/continuity")

if ACCELERATION == "tensorrt" and USE_PIPELINE_BATCH:
    print(f"\\n💡 TensorRT fix usage:")
    print(f"   1) Set ACCELERATION = 'tensorrt'")
    print(f"   2) First run auto-compiles engines")
    print(f"   3) Later runs load engines directly")
    print(f"   4) Handles timestep compatibility automatically")
    print(f"   5) Enjoy noise-free TensorRT acceleration!")

print(f"\\n🎉 Generation done!")

# Print peak memory (backup monitoring)
try:
    peak_mem_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    print(f"PEAK_MEM_MB: {peak_mem_mb:.2f}")
except Exception:
    pass
