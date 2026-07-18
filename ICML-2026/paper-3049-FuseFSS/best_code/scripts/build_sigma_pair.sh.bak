#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ARCH="${GPU_ARCH:-120}"
NVCC="${NVCC:-nvcc}"
VARIANT="all"
BUILD_CORE="${BUILD_CORE:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/build_sigma_pair.sh [--variant sigma|fusefss|all]

Environment:
  GPU_ARCH=120        CUDA architecture passed as -arch=sm_${GPU_ARCH}
  NVCC=nvcc           nvcc executable
  BUILD_CORE=1        build FuseFSS core before the FuseFSS Sigma binary if needed

Outputs:
  build/gpu_mpc_upstream/sigma
  build/gpu_mpc_vendor/sigma
  ezpc_upstream/GPU-MPC/experiments/sigma/sigma
  third_party/EzPC_vendor/GPU-MPC/experiments/sigma/sigma
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$VARIANT" in
  sigma|fusefss|suf|all) ;;
  *)
    echo "--variant must be sigma, fusefss, or all" >&2
    exit 2
    ;;
esac
if [[ "$VARIANT" == "suf" ]]; then
  VARIANT="fusefss"
fi

ensure_fusefss_core() {
  if [[ -f "$ROOT_DIR/build/libfusefss_cuda.a" || -f "$ROOT_DIR/build/libsuf_cuda.a" ]]; then
    return
  fi
  if [[ "$BUILD_CORE" != "1" ]]; then
    echo "missing $ROOT_DIR/build/libfusefss_cuda.a; rerun with BUILD_CORE=1 or build FuseFSS core first" >&2
    exit 1
  fi
  cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" \
    -DFUSEFSS_ENABLE_CUDA=ON \
    -DFUSEFSS_ENABLE_TESTS=ON \
    -DFUSEFSS_ENABLE_BENCH=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH"
  cmake --build "$ROOT_DIR/build" -j"$(nproc)"
}

build_upstream_sigma() {
  local mpc_dir="$ROOT_DIR/ezpc_upstream/GPU-MPC"
  local out="$ROOT_DIR/build/gpu_mpc_upstream/sigma"
  mkdir -p "$(dirname "$out")"

  local include_flags=(
    "-I$mpc_dir"
    "-I$ROOT_DIR/include"
    "-I$ROOT_DIR/ezpc_upstream/SCI/extern"
    "-I$ROOT_DIR/ezpc_upstream/SCI/extern/eigen"
    "-I$mpc_dir/ext/sytorch/include"
    "-I$mpc_dir/ext/sytorch/include/eigen3"
    "-I$mpc_dir/ext/sytorch/ext/llama/include"
    "-I$mpc_dir/ext/sytorch/ext/cryptoTools"
    "-I$mpc_dir/ext/sytorch/ext/bitpack/include"
    "-I$mpc_dir/ext/cutlass/include"
    "-I$mpc_dir/ext/cutlass/tools/util/include"
  )
  local common_flags=(
    -std=c++17
    -O3
    "-arch=sm_${GPU_ARCH}"
    --expt-relaxed-constexpr
    -Xcompiler=-fopenmp
    -Xcompiler=-msse4.1
    -Xcompiler=-maes
    -Xcompiler=-mpclmul
  )
  local sources=(
    "$mpc_dir/experiments/sigma/sigma.cu"
    "$mpc_dir/utils/gpu_mem.cu"
    "$mpc_dir/utils/gpu_file_utils.cpp"
    "$mpc_dir/utils/sigma_comms.cpp"
    "$mpc_dir/ext/sytorch/src/sytorch/random.cpp"
    "$mpc_dir/ext/sytorch/src/sytorch/backend/cleartext.cpp"
    "$mpc_dir/ext/sytorch/ext/bitpack/src/bitpack/bitpack.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Common/Defines.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/AES.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/PRNG.cpp"
  )
  local llama_sources=(
    "$mpc_dir"/ext/sytorch/ext/llama/*.cpp
    "$mpc_dir"/ext/sytorch/ext/llama/src/llama/*.cpp
  )

  echo "building Sigma baseline -> $out"
  "$NVCC" "${common_flags[@]}" "${include_flags[@]}" \
    "${sources[@]}" "${llama_sources[@]}" \
    -lcublas -lcurand \
    -o "$out"
  cp "$out" "$mpc_dir/experiments/sigma/sigma"
}

build_fusefss_sigma() {
  ensure_fusefss_core

  local mpc_dir="$ROOT_DIR/third_party/EzPC_vendor/GPU-MPC"
  local out="$ROOT_DIR/build/gpu_mpc_vendor/sigma"
  mkdir -p "$(dirname "$out")"

  local include_flags=(
    "-I$mpc_dir"
    "-I$ROOT_DIR/include"
    "-I$ROOT_DIR/ezpc_upstream/SCI/extern"
    "-I$ROOT_DIR/ezpc_upstream/SCI/extern/eigen"
    "-I$mpc_dir/ext/sytorch/include"
    "-I$mpc_dir/ext/sytorch/include/eigen3"
    "-I$mpc_dir/ext/sytorch/ext/llama/include"
    "-I$mpc_dir/ext/sytorch/ext/cryptoTools"
    "-I$mpc_dir/ext/sytorch/ext/bitpack/include"
    "-I$mpc_dir/ext/cutlass/include"
    "-I$mpc_dir/ext/cutlass/tools/util/include"
  )
  local common_flags=(
    -std=c++17
    -O3
    "-arch=sm_${GPU_ARCH}"
    --expt-relaxed-constexpr
    -DSUF_HAVE_CUDA=1
    -Xcompiler=-fopenmp
    -Xcompiler=-msse4.1
    -Xcompiler=-maes
    -Xcompiler=-mpclmul
  )
  local sources=(
    "$mpc_dir/experiments/sigma/sigma.cu"
    "$ROOT_DIR/src/fusefss_sigma_bridge.cu"
    "$mpc_dir/utils/gpu_mem.cu"
    "$mpc_dir/utils/gpu_file_utils.cpp"
    "$mpc_dir/utils/sigma_comms.cpp"
    "$mpc_dir/ext/sytorch/src/sytorch/random.cpp"
    "$mpc_dir/ext/sytorch/src/sytorch/backend/cleartext.cpp"
    "$mpc_dir/ext/sytorch/ext/bitpack/src/bitpack/bitpack.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Common/Defines.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/AES.cpp"
    "$mpc_dir/ext/sytorch/ext/cryptoTools/cryptoTools/Crypto/PRNG.cpp"
  )
  local llama_sources=(
    "$mpc_dir"/ext/sytorch/ext/llama/*.cpp
    "$mpc_dir"/ext/sytorch/ext/llama/src/llama/*.cpp
  )

  echo "building FuseFSS Sigma -> $out"
  local fusefss_lib="$ROOT_DIR/build/libfusefss_cuda.a"
  if [[ ! -f "$fusefss_lib" ]]; then
    fusefss_lib="$ROOT_DIR/build/libsuf_cuda.a"
  fi
  "$NVCC" "${common_flags[@]}" "${include_flags[@]}" \
    "${sources[@]}" "${llama_sources[@]}" \
    "$fusefss_lib" \
    -lcublas -lcurand \
    -o "$out"
  cp "$out" "$mpc_dir/experiments/sigma/sigma"
}

if [[ "$VARIANT" == "sigma" || "$VARIANT" == "all" ]]; then
  build_upstream_sigma
fi
if [[ "$VARIANT" == "fusefss" || "$VARIANT" == "all" ]]; then
  build_fusefss_sigma
fi
