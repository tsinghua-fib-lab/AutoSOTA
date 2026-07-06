import os, sys

# Clear proxy env vars
for k in ["http_proxy","HTTP_PROXY","https_proxy","HTTPS_PROXY","all_proxy","ALL_PROXY","no_proxy","NO_PROXY"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/autosota_cache/hf"

import nltk
nltk.download("punkt", quiet=True)

class MockArgs:
    watermark_algorithm = "AliMark"
    watermark_model = "facebook/opt-1.3b"
    watermark_embedder = "all-mpnet-base-v2"
    watermark_embedding_dim = 768
    watermark_block_size = 8
    watermark_num_next_sentence_candidates = 64
    min_new_sentences = 12
    dataset_name = "c4"
    vllm_gpu_mem_util = 0.3
    device = "cuda"
    seed = 42
    watermark_rs_dropout = 0.0

sys.path.insert(0, "/repo")
from watermark.alimark import AliMark

print("Loading AliMark with vLLM...")
watermark = AliMark(MockArgs(), load_llm=True)
print("AliMark loaded OK!")

prompt = "The cat sat on the mat."
print(f"Generating unwatermarked text...")
try:
    text = watermark.generate_unwatermarked_text(prompt)
    print(f"Unwatermarked OK: {len(text)} chars")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print(f"Generating watermarked text...")
try:
    text = watermark.generate_watermarked_text(prompt)
    print(f"Watermarked OK: {len(text)} chars")
    print(f"First 200 chars: {text[:200]}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("All done!")
