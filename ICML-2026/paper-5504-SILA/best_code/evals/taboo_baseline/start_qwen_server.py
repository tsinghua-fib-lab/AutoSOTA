#!/usr/bin/env python3
"""
Start vLLM as a persistent OpenAI-compatible server for Qwen models.

This server will stay running and handle incoming requests with automatic batching.
You only need to start it once, and it will serve all your inference requests.

Usage:
    python start_qwen_server.py --model-size 7b
    python start_qwen_server.py --model-size 72b --tensor-parallel-size 4
    python start_qwen_server.py --model Qwen/Qwen2.5-14B-Instruct --port 8080
"""
import argparse
import subprocess
import sys


# Model size to HuggingFace model name mapping
QWEN_MODELS = {
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}

# Recommended tensor parallel sizes for each model
RECOMMENDED_TP_SIZES = {
    "7b": 1,   # Fits on 1 GPU
    "14b": 1,  # Fits on 1 GPU (80GB) or 2 GPUs (40GB)
    "32b": 2,  # Needs 2+ GPUs
    "72b": 3,  # Needs 3 GPUs
}


def start_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    max_num_seqs: int = 256,
):
    """
    Start the vLLM OpenAI-compatible server.
    
    Args:
        model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        host: Host to bind to (0.0.0.0 for all interfaces)
        port: Port to bind to
        tensor_parallel_size: Number of GPUs to use (1 for single GPU, 2+ for multi-GPU)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum context length (None = use model default)
        max_num_seqs: Maximum concurrent sequences (default 256)
    """
    print("=" * 80)
    print("Starting vLLM Server for Qwen")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Host: {host}:{port}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Num Seqs: {max_num_seqs}")
    if max_model_len:
        print(f"Max Model Length: {max_model_len}")
    print("=" * 80)
    print("\nThis will take a few minutes to load the model...")
    print("Once started, the server will stay running until you stop it (Ctrl+C)")
    print("=" * 80 + "\n")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
        sys.exit(0)
    except FileNotFoundError:
        print("\n\nError: vLLM not found. Install it with: pip install vllm")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start vLLM server for Qwen models (Taboo baseline eval)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model identifier (overrides --model-size)"
    )
    parser.add_argument(
        "--model-size", type=str, default="7b",
        choices=list(QWEN_MODELS.keys()),
        help="Qwen model size (used if --model not specified)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Number of GPUs for tensor parallelism (default: auto based on model size)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Maximum context length (default: use model's default)"
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=256,
        help="Maximum concurrent sequences (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Determine model
    if args.model:
        model = args.model
        model_size = None
    else:
        model = QWEN_MODELS[args.model_size]
        model_size = args.model_size
    
    # Determine tensor parallel size
    if args.tensor_parallel_size is not None:
        tp_size = args.tensor_parallel_size
    elif model_size:
        tp_size = RECOMMENDED_TP_SIZES[model_size]
        print(f"Using recommended tensor parallel size for {model_size}: {tp_size}")
    else:
        tp_size = 1
    
    start_server(
        model=model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
