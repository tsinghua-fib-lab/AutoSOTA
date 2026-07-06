import sys; sys.path.insert(0, "/repo")
import os, numpy as np, torch
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from model.hooks import PatternCache, setup_pattern_hooks

def test_checkpoint(path, label):
    OFFICIAL_MODEL_NAMES.append(path)
    MODEL_ALIASES[path] = ["local"]
    make_model_alias_map()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = HookedTransformer.from_pretrained_no_processing(path, dtype=torch.float32, tokenizer=tokenizer, device="cuda:0")
    model.cfg.use_attn_result = False
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    inner = model

    results = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pat_cache = PatternCache()
            hooks = setup_pattern_hooks(inner, layer, pat_cache)
            rng = np.random.default_rng(12345)
            all_attn = []
            seq_half = 32
            for si in range(100):
                prefix = rng.integers(0, int(inner.cfg.d_vocab), size=seq_half)
                tokens_np = np.concatenate([prefix, prefix])
                tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
                pat_cache.clear()
                with torch.no_grad():
                    _ = model(tokens)
                hp = pat_cache.pattern[0, head].cpu().numpy()
                for q in range(seq_half, 2*seq_half - 1):
                    target_k = q - seq_half + 1
                    all_attn.append(float(hp[q, target_k]))
            inner.reset_hooks(hooks)
            results[f"L{layer}H{head}"] = round(float(np.mean(all_attn)), 6)
    print(f"{label}:")
    for k, v in sorted(results.items()):
        marker = " ***" if v > 0.1 else ""
        print(f"  {k}: {v:.6f}{marker}")

for ckpt_path, label in [
    ("/models/pythia-14m", "pythia-14m (final)"),
    ("/models/pythia-14m-step1000", "pythia-14m-step1000"),
    ("/models/pythia-14m-step2000", "pythia-14m-step2000"),
]:
    test_checkpoint(ckpt_path, label)
    print()
