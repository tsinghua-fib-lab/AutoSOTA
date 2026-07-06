import sys; sys.path.insert(0, "/repo")
import os, numpy as np, torch
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from model.hooks import PatternCache, setup_pattern_hooks

def induction_score_simple(path, layer, head, n_seqs=100, half=32):
    OFFICIAL_MODEL_NAMES.append(path)
    MODEL_ALIASES[path] = ["local"]
    make_model_alias_map()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = HookedTransformer.from_pretrained_no_processing(path, dtype=torch.float32, tokenizer=tok, device="cuda:0")
    model.cfg.use_attn_result = False
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    inner = model
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, layer, pat_cache)
    rng = np.random.default_rng(12345)
    all_attn = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        pat_cache.clear()
        with torch.no_grad():
            _ = model(tokens)
        hp = pat_cache.pattern[0, head].cpu().numpy()
        for q in range(half, 2*half - 1):
            target_k = q - half + 1
            all_attn.append(float(hp[q, target_k]))
    inner.reset_hooks(hooks)
    return round(float(np.mean(all_attn)), 6)

for ckpt_path, label in [
    ("/models/pythia-14m-step2000", "step2000"),
    ("/models/pythia-14m-step1000", "step1000"),
]:
    try:
        score = induction_score_simple(ckpt_path, 3, 3)
        print(f"{label} L3H3: {score}")
    except Exception as e:
        print(f"{label}: ERROR - {e}")
