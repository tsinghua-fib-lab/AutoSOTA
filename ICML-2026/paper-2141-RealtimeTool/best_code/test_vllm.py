import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/opt/conda/lib/python3.10/site-packages/nvidia/cu13"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from vllm import LLM
print("Loading model...")
llm = LLM(
    model="/models/RT-Qwen2.5-0.5B",
    trust_remote_code=True,
    dtype="auto",
    gpu_memory_utilization=0.80,
    max_model_len=4096,
    enable_prefix_caching=True,
)
print("Model loaded successfully!")
