"""Recompile the CUDA compression backend.

Clears the JIT cache for _compression_cuda and triggers a fresh build.
Requires NVCOMP_ROOT to be set and a CUDA-capable GPU environment.

Usage:
    uv run python scripts/compile_backend.py [--verbose]
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch.utils.cpp_extension


def find_cache_dir() -> Path | None:
    """Locate the JIT cache directory for _compression_cuda."""
    # torch caches under ~/.cache/torch_extensions/<py-version>_<cu-version>/
    base = Path(torch.utils.cpp_extension._get_build_directory("_compression_cuda", verbose=False))
    return base if base.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompile the CUDA compression backend")
    parser.add_argument("--verbose", action="store_true", help="Show compiler output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger(__name__)

    # Clear cached build
    cache_dir = find_cache_dir()
    if cache_dir:
        log.info(f"Clearing JIT cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    else:
        log.info("No existing JIT cache found")

    # Trigger recompilation via the existing backend loader
    log.info("Compiling CUDA backend...")
    from entquant.compression.backends import nvCOMPBackend

    nvCOMPBackend._load_cuda_backend()
    log.info("Done.")


if __name__ == "__main__":
    main()
