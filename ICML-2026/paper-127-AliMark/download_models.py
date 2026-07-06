#!/usr/bin/env python3
"""Pre-download models needed for AliMark pipeline."""
import os, sys

for k in ["http_proxy","HTTP_PROXY","https_proxy","HTTPS_PROXY","all_proxy","ALL_PROXY","no_proxy","NO_PROXY"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["HF_HUB_CACHE"] = "/autosota_cache/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/autosota_cache/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/autosota_cache/hf"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/autosota_cache/hf"

os.makedirs("/autosota_cache/hf/hub", exist_ok=True)

# 1. Sentence embedder (420MB)
print("=== Downloading all-mpnet-base-v2 ===")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
print("OK: all-mpnet-base-v2")

# 2. OPT-1.3B tokenizer (small, for 2_attack.py)
print("=== Downloading OPT-1.3b tokenizer ===")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
print("OK: OPT-1.3b tokenizer")

# 3. Pegasus paraphraser (~2GB)
print("=== Downloading Pegasus paraphraser ===")
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
tok = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
print("OK: Pegasus tokenizer")
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
print("OK: Pegasus model")

# 4. Parrot paraphraser
print("=== Downloading Parrot paraphraser ===")
from parrot import Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
print("OK: Parrot")

print("=== All downloads complete ===")
