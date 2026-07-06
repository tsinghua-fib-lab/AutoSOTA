import sys; sys.path.insert(0, "/repo")
import os, numpy as np, torch, json
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from model.hooks import PatternCache, setup_pattern_hooks

# Load model once
path = "/models/pythia-14m"
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

def _find_match_pos(tokens, t, induction_match, match_choice):
    """Returns (match_pos, target_pos) or (-1, -1)"""
    if induction_match == "previous":
        if t == 0: return -1, -1
        key = int(tokens[t - 1])
        left = tokens[:t - 1]
    else:
        key = int(tokens[t])
        left = tokens[:t]
    pos = np.where(left == key)[0]
    if len(pos) == 0:
        return -1, -1
    match_pos = int(pos[-1] if match_choice == "last" else pos[0])
    target_pos = match_pos + 1
    if target_pos >= len(tokens):
        return -1, -1
    return match_pos, target_pos

def compute_induction_v1(n_seqs=100, half=32):
    """Simple: hp[q, q-half+1]"""
    rng = np.random.default_rng(12345)
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, 3, pat_cache)
    all_attn = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        pat_cache.clear()
        with torch.no_grad():
            _ = model(tokens)
        hp = pat_cache.pattern[0, 3].cpu().numpy()
        for q in range(half, 2*half - 1):
            target_k = q - half + 1
            all_attn.append(float(hp[q, target_k]))
    inner.reset_hooks(hooks)
    return float(np.mean(all_attn))

def compute_induction_v2(n_seqs=100, half=32):
    """Correct find_match: for predicting token t, query is t-1, key = match_pos+1"""
    rng = np.random.default_rng(12345)
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, 3, pat_cache)
    all_attn = []
    all_baseline = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        pat_cache.clear()
        with torch.no_grad():
            _ = model(tokens)
        hp = pat_cache.pattern[0, 3].cpu().numpy()
        seq_len = 2 * half
        for t in range(1, seq_len):
            match_pos, target_key = _find_match_pos(tokens_np, t, "current", "last")
            if target_key != -1 and target_key < t:
                # Query position for predicting token t is t-1
                q = t - 1
                attn = float(hp[q, target_key])
                baseline = float(hp[q, :q+1].mean())
                all_attn.append(attn)
                all_baseline.append(baseline)
    inner.reset_hooks(hooks)
    s = np.array(all_attn)
    b = np.array(all_baseline)
    return float(s.mean()), float(b.mean()), float((s - b).mean())

def compute_induction_v3(n_seqs=100, half=32):
    """Attention from query q (predicting q+1) to the match_pos+1 for token q+1"""
    rng = np.random.default_rng(12345)
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, 3, pat_cache)
    all_attn = []
    all_baseline = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        pat_cache.clear()
        with torch.no_grad():
            _ = model(tokens)
        hp = pat_cache.pattern[0, 3].cpu().numpy()
        seq_len = 2 * half
        # For each query position q (0..seq_len-2), predict token q+1
        for q in range(seq_len - 1):
            t = q + 1  # token being predicted
            match_pos, target_key = _find_match_pos(tokens_np, t, "current", "last")
            if target_key != -1 and target_key <= q:
                attn = float(hp[q, target_key])
                baseline = float(hp[q, :q+1].mean())
                all_attn.append(attn)
                all_baseline.append(baseline)
    inner.reset_hooks(hooks)
    s = np.array(all_attn)
    b = np.array(all_baseline)
    return float(s.mean()), float(b.mean()), float((s - b).mean())

print("V1 (simple hp[q, q-half+1]):")
s1 = compute_induction_v1()
print(f"  Score: {s1:.6f}")

print("\nV2 (find_match: hp[t-1, match_pos+1]):")
s2_mean, s2_base, s2_diff = compute_induction_v2()
print(f"  Mean target attn: {s2_mean:.6f}")
print(f"  Mean baseline: {s2_base:.6f}")
print(f"  Diff: {s2_diff:.6f}")

print("\nV3 (find_match: hp[q, match_pos+1] for token q+1):")
s3_mean, s3_base, s3_diff = compute_induction_v3()
print(f"  Mean target attn: {s3_mean:.6f}")
print(f"  Mean baseline: {s3_base:.6f}")
print(f"  Diff: {s3_diff:.6f}")
