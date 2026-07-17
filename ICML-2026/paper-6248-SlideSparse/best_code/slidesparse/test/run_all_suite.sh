#!/bin/bash
# cd /root/vllmbench && bash slidesparse/test/run_all_suite.sh 2>&1

set -e

echo "===== Running INT8 Tests ====="
# 01
python3 slidesparse/test/INT8_vllm/test_01_bridge.py

# 02 - INT8
echo "--- INT8 02 Kernel: CUTLASS ---"
python3 slidesparse/test/INT8_vllm/test_02_kernel.py --model Llama3.2-1B-INT8 --use-cutlass
echo "--- INT8 02 Kernel: cuBLASLt ---"
python3 slidesparse/test/INT8_vllm/test_02_kernel.py --model Llama3.2-1B-INT8 --use-cublaslt
echo "--- INT8 02 Kernel: cuSPARSELt (2:10) ---"
python3 slidesparse/test/INT8_vllm/test_02_kernel.py --model Llama3.2-1B-INT8 --use-cusparselt --sparsity 2_10

# 03 - INT8
echo "--- INT8 03 Inference: CUTLASS ---"
python3 slidesparse/test/INT8_vllm/test_03_inference.py --model Llama3.2-1B-INT8 --use-cutlass
echo "--- INT8 03 Inference: cuBLASLt ---"
python3 slidesparse/test/INT8_vllm/test_03_inference.py --model Llama3.2-1B-INT8 --use-cublaslt
echo "--- INT8 03 Inference: cuSPARSELt (2:10) ---"
python3 slidesparse/test/INT8_vllm/test_03_inference.py --model Llama3.2-1B-INT8 --use-cusparselt --sparsity 2_10

# 04 - INT8
echo "--- INT8 04 Throughput: CUTLASS ---"
python3 slidesparse/test/INT8_vllm/test_04_throughput.py --model Llama3.2-1B-INT8 --use-cutlass --show-subprocess-output
echo "--- INT8 04 Throughput: cuBLASLt ---"
python3 slidesparse/test/INT8_vllm/test_04_throughput.py --model Llama3.2-1B-INT8 --use-cublaslt --show-subprocess-output
echo "--- INT8 04 Throughput: cuSPARSELt (2:10) ---"
python3 slidesparse/test/INT8_vllm/test_04_throughput.py --model Llama3.2-1B-INT8 --use-cusparselt --sparsity 2_10 --show-subprocess-output


echo "===== Running FP8 Tests ====="
# 01
python3 slidesparse/test/FP8_vllm/test_01_bridge.py

# 02
echo "--- 02 Kernel: CUTLASS ---"
python3 slidesparse/test/FP8_vllm/test_02_kernel.py --model Llama3.2-1B-FP8 --use-cutlass
echo "--- 02 Kernel: cuBLASLt ---"
python3 slidesparse/test/FP8_vllm/test_02_kernel.py --model Llama3.2-1B-FP8 --use-cublaslt
echo "--- 02 Kernel: cuSPARSELt (2:10) ---"
python3 slidesparse/test/FP8_vllm/test_02_kernel.py --model Llama3.2-1B-FP8 --use-cusparselt --sparsity 2_10

# 03 Inference
echo "--- 03 Inference: CUTLASS ---"
python3 slidesparse/test/FP8_vllm/test_03_inference.py --model Llama3.2-1B-FP8 --use-cutlass
echo "--- 03 Inference: cuBLASLt ---"
python3 slidesparse/test/FP8_vllm/test_03_inference.py --model Llama3.2-1B-FP8 --use-cublaslt
echo "--- 03 Inference: cuSPARSELt (2:10) ---"
python3 slidesparse/test/FP8_vllm/test_03_inference.py --model Llama3.2-1B-FP8 --use-cusparselt --sparsity 2_10

# 04 Throughput
echo "--- 04 Throughput: CUTLASS ---"
python3 slidesparse/test/FP8_vllm/test_04_throughput.py --model Llama3.2-1B-FP8 --use-cutlass --show-subprocess-output
echo "--- 04 Throughput: cuBLASLt ---"
python3 slidesparse/test/FP8_vllm/test_04_throughput.py --model Llama3.2-1B-FP8 --use-cublaslt --show-subprocess-output
echo "--- 04 Throughput: cuSPARSELt (2:10) ---"
python3 slidesparse/test/FP8_vllm/test_04_throughput.py --model Llama3.2-1B-FP8 --use-cusparselt --sparsity 2_10 --show-subprocess-output
