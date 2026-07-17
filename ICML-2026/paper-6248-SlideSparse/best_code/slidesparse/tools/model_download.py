#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Model Download Script

Batch download W8A8 quantized models for vLLM baseline testing.
Supports INT8 (quantized.w8a8) and FP8 (FP8-dynamic) formats.

Usage:
    python3 model_download.py [options]

Options:
    -a, --all           Download all models
    -i, --int8          Download INT8 models only
    -f, --fp8           Download FP8 models only
    -q, --qwen          Download Qwen2.5 series only
    -l, --llama         Download Llama3.2 series only
    -m, --model NAME    Download specific model (e.g., qwen2.5-7b-int8)
    -c, --check         Check downloaded model status
    -s, --size          Show estimated model sizes
    -h, --help          Show help message

Examples:
    python3 model_download.py --all                    # Download all models
    python3 model_download.py --int8 --qwen            # Download Qwen INT8 models
    python3 model_download.py --model qwen2.5-7b-fp8   # Download specific model
    python3 model_download.py --check                  # Check download status
"""

import sys
import argparse
from pathlib import Path

# Ensure slidesparse can be imported
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    model_registry,
    list_models,
    MODEL_SIZE_GB,
)
from slidesparse.tools.utils import (
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    CHECKPOINT_DIR,
    check_hf_cli,
    download_model,
    print_model_status,
)


def show_model_sizes():
    """Show estimated model sizes"""
    print_header("Estimated Model Sizes")
    
    print("Model size reference:")
    for size, gb in sorted(MODEL_SIZE_GB.items(), key=lambda x: x[1]):
        print(f"  - {size} model: ~{gb:.1f} GB")
    
    print()
    print("Estimated total size (all models):")
    
    int8_total = sum(
        MODEL_SIZE_GB.get(e.size.upper(), 0)
        for e in model_registry.list(quant="int8")
    )
    fp8_total = sum(
        MODEL_SIZE_GB.get(e.size.upper(), 0)
        for e in model_registry.list(quant="fp8")
    )
    
    print(f"  - INT8 total: ~{int8_total:.1f} GB")
    print(f"  - FP8 total:  ~{fp8_total:.1f} GB")
    print(f"  - Total:      ~{int8_total + fp8_total:.1f} GB")


def download_models(
    quant_filter: str | None = None,
    family_filter: str | None = None,
    specific_model: str | None = None,
):
    """
    Download models
    
    Args:
        quant_filter: Quantization type filter (int8, fp8)
        family_filter: Model family filter (qwen, llama)
        specific_model: Specific model key
    """
    # Check HF CLI
    if not check_hf_cli():
        print_error("HuggingFace CLI not installed")
        print_info("Please run: pip install -U huggingface_hub")
        sys.exit(1)
    
    # Ensure checkpoints directory exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine models to download
    if specific_model:
        # Download specific model
        entry = model_registry.get(specific_model)
        if entry is None:
            print_error(f"Model not found: {specific_model}")
            print_info(f"Available models: {', '.join(list_models())}")
            sys.exit(1)
        
        models_to_download = [entry]
    else:
        # Get model list by filter conditions
        models_to_download = model_registry.list(
            family=family_filter,
            quant=quant_filter,
        )
    
    if not models_to_download:
        print_warning("No matching models found")
        return
    
    # Show download plan
    total_gb = sum(e.estimated_gb for e in models_to_download)
    print_header(f"Preparing to download {len(models_to_download)} models (~{total_gb:.1f} GB)")
    
    for entry in models_to_download:
        print(f"  - {entry.local_name} ({entry.estimated_gb:.1f} GB)")
    print()
    
    # Start download
    success_count = 0
    failed_models = []
    
    for entry in models_to_download:
        print_header(f"Downloading: {entry.local_name}")
        print_info(f"HuggingFace: {entry.hf_path}")
        print_info(f"Local dir: {CHECKPOINT_DIR / entry.local_name}")
        print()
        
        success, msg = download_model(entry.key, CHECKPOINT_DIR)
        
        if success:
            print_success(msg)
            success_count += 1
        else:
            print_error(msg)
            failed_models.append(entry.key)
        
        print()
    
    # Show results
    print_header("Download completed")
    print(f"Success: {success_count}/{len(models_to_download)}")
    
    if failed_models:
        print_warning(f"Failed: {', '.join(failed_models)}")
    
    # Show final status
    print_model_status(CHECKPOINT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse Model Download Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:

  INT8 models (quantized.w8a8):
""" + "\n".join(f"    - {key}" for key in list_models(quant="int8")) + """

  FP8 models (FP8-dynamic):
""" + "\n".join(f"    - {key}" for key in list_models(quant="fp8")) + """

  BitNet models (BF16):
    - bitnet1.58-2b-bf16

Examples:
  %(prog)s --all                    # Download all models (INT8 + FP8)
  %(prog)s --int8 --qwen            # Download Qwen INT8 models
  %(prog)s --bitnet                 # Download BitNet BF16 model
  %(prog)s --model qwen2.5-7b-fp8   # Download specific model
  %(prog)s --check                  # Check download status
"""
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "-a", "--all", action="store_true",
        help="Download all models (INT8 + FP8)"
    )
    model_group.add_argument(
        "-i", "--int8", action="store_true",
        help="Download INT8 models only"
    )
    model_group.add_argument(
        "-f", "--fp8", action="store_true",
        help="Download FP8 models only"
    )
    model_group.add_argument(
        "-q", "--qwen", action="store_true",
        help="Download Qwen2.5 series only"
    )
    model_group.add_argument(
        "-l", "--llama", action="store_true",
        help="Download Llama3.2 series only"
    )
    model_group.add_argument(
        "-b", "--bitnet", action="store_true",
        help="Download BitNet BF16 model (microsoft)"
    )
    model_group.add_argument(
        "-m", "--model", type=str, metavar="NAME",
        help="Download specific model"
    )
    
    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "-c", "--check", action="store_true",
        help="Check downloaded model status"
    )
    other_group.add_argument(
        "-s", "--size", action="store_true",
        help="Show estimated model sizes"
    )
    
    args = parser.parse_args()
    
    # Show size info
    if args.size:
        show_model_sizes()
        return 0
    
    # Check mode
    if args.check:
        print_model_status(CHECKPOINT_DIR)
        return 0
    
    # Handle BitNet special case
    if args.bitnet:
        download_models(specific_model="bitnet1.58-2b-bf16")
        return 0
    
    # Determine filter conditions
    quant_filter = None
    family_filter = None
    
    if args.int8 and not args.fp8:
        quant_filter = "int8"
    elif args.fp8 and not args.int8:
        quant_filter = "fp8"
    elif args.all:
        quant_filter = None  # Download all
    elif not args.model:
        # No options specified, show help
        parser.print_help()
        return 0
    
    if args.qwen and not args.llama:
        family_filter = "qwen"
    elif args.llama and not args.qwen:
        family_filter = "llama"
    
    # Execute download
    download_models(
        quant_filter=quant_filter,
        family_filter=family_filter,
        specific_model=args.model,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
