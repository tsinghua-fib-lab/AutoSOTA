"""EntQuant quickstart: quantize, compress, evaluate, and benchmark an LLM.

Covers the core workflow (quantize + ANS-compress) and optionally supports
multi-GPU, super weight detection, perplexity evaluation, and inference
benchmarking. See the README for a step-by-step walkthrough.
"""

from run import setup_env

setup_env()

import logging

import torch
from transformers import AutoConfig, AutoTokenizer

from entquant import EntQuantModel
from entquant.quantization.optimizer import SymmetricEntropyOptimizer, WrappedAbsmaxOptimizer
from entquant.super_weights import detect_fallback_layers, SuperWeightsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------

MODEL = "meta-llama/Llama-2-7b-hf"
DTYPE = torch.bfloat16
WEIGHT_QTYPE = "qfloat8"  # or "qint8"
NUM_GPUS = 1  # >1 builds a device map splitting layers across GPUs
SAVE_DIR = None  # Set to a path to persist the checkpoint

# Entropy regularization strength (LAMBDA) and learning rate (LR).
# Together they control the effective bit rate:
#   (LR, LAMBDA) ~ target bit rate
LAMBDA, LR = 3.9, 1.0  # ~ 4-bit (default)
# LAMBDA, LR = 14.5, 1.0   # ~ 3-bit
# LAMBDA, LR = 58.0, 0.25  # ~ 2-bit

# Super weight detection (off by default; useful for models with outliers)
DO_SUPER_WEIGHTS = False
SW_INCLUDE = "*mlp.down_proj*"
SW_SPIKE_THRESHOLD = 50.0
SW_TOP_K = 25

# Perplexity evaluation
DO_EVAL = True
EVAL_DATASETS = ["c4", "wikitext2"]
EVAL_CTX_LENGTH = 2048

# Inference benchmark
DO_INFERENCE_BENCH = False
PREFILL_BATCH_SIZE = 32
PREFILL_SEQ_LENGTH = 512
DECODE_BATCH_SIZE = 64
DECODE_CONTEXT_LENGTH = 1
DECODE_NUM_TOKENS = 128

# ---------- Multi-GPU device map ----------

device_map = None
if NUM_GPUS > 1:
    config = AutoConfig.from_pretrained(MODEL)
    n_layers = config.num_hidden_layers
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": f"cuda:{NUM_GPUS - 1}",
        "model.rotary_emb": "cuda:0",
        "lm_head": f"cuda:{NUM_GPUS - 1}",
    }
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = f"cuda:{i * NUM_GPUS // n_layers}"
    logger.info("Device map: %d GPUs, %d layers", NUM_GPUS, n_layers)

# ---------- Optional: super weight detection ----------

fallback_layers = None
if DO_SUPER_WEIGHTS:
    logger.info("Detecting super weights...")
    sw_config = SuperWeightsConfig(
        include=SW_INCLUDE,
        spike_threshold=SW_SPIKE_THRESHOLD,
        top_k=SW_TOP_K,
    )
    fallback_layers = detect_fallback_layers(MODEL, config=sw_config, device_map="cpu")
    logger.info("Detected %d fallback layers", len(fallback_layers))

# ---------- Quantize + compress ----------

optimizer = SymmetricEntropyOptimizer(lr=LR, reg_param=LAMBDA)
optimizer_fb = WrappedAbsmaxOptimizer()

model = EntQuantModel.from_pretrained(
    MODEL,
    quantize=True,
    compress=True,
    save_dir=SAVE_DIR,
    device_map=device_map,
    weight_qtype=WEIGHT_QTYPE,
    dtype=DTYPE,
    fallback_layers=fallback_layers,
    optimizer=optimizer,
    optimizer_fallback=optimizer_fb,
)

# ---------- Compression stats ----------

GiB = 1024**3
stats = model.compression_stats()
logger.info(
    "Compression: %.2f GiB (%s) -> %.2f GiB (%.2fx) | avg entropy: %.2f bit",
    stats["injected_original_bytes"] / GiB,
    stats["original_dtype"],
    stats["injected_compressed_bytes"] / GiB,
    stats["injected_vs_original_ratio"],
    stats["avg_entropy"],
)

# ---------- Optional: reload from checkpoint ----------

if SAVE_DIR is not None:
    logger.info("Reloading checkpoint from %s ...", SAVE_DIR)
    model = EntQuantModel.from_pretrained(SAVE_DIR, compress=True)
    logger.info("Checkpoint loaded successfully")

# ---------- Optional: evaluation & inference benchmark ----------

tokenizer = None
if DO_EVAL or DO_INFERENCE_BENCH:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

if DO_EVAL:
    from entquant.eval.eval_ppl import PPLModelEvaluator

    evaluator = PPLModelEvaluator(
        tokenizer=tokenizer,
        dataset_names=EVAL_DATASETS,
        ctx_length=EVAL_CTX_LENGTH,
    )
    results = evaluator(model=model.model)
    for name, ppl in results.items():
        logger.info("  %s: %.2f", name, ppl)

if DO_INFERENCE_BENCH:
    from entquant.eval.eval_inference import BenchmarkConfig, EfficiencyModelEvaluator

    bench_config = BenchmarkConfig(
        prefill_batch_size=PREFILL_BATCH_SIZE,
        prefill_sequence_length=PREFILL_SEQ_LENGTH,
        decode_batch_size=DECODE_BATCH_SIZE,
        decode_context_length=DECODE_CONTEXT_LENGTH,
        decode_num_tokens_to_generate=DECODE_NUM_TOKENS,
    )
    bench_evaluator = EfficiencyModelEvaluator(tokenizer=tokenizer, config=bench_config)
    bench_results = bench_evaluator(model=model.model)
    logger.info("Inference Benchmark Results:")
    for key, val in bench_results.items():
        if key == "device_info":
            continue
        logger.info("  %s: %s", key, val)
