# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.1-405B-Instruct-FP8"
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL_NAME="meta-llama/Llama-3.1-70B"
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=2

vllm serve $MODEL_NAME \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 16384


