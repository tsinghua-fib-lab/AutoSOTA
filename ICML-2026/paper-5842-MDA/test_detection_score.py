import sys; sys.path.insert(0, "/repo")
import os, numpy as np, torch
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from transformer_lens.head_detector import get_induction_head_detection_pattern
from model.hooks import PatternCache, setup_pattern_hooks

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

# Test using transformer_lens built-in detection
from transformer_lens.head_detector import detect_head

def compute_induction_score_tl(n_seqs=100, half=32):
    rng = np.random.default_rng(12345)
    scores = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        
        result = detect_head(
            model, 
            seq=tokens,
            detection_pattern="induction_head",
            error_measure="mul"
        )
        scores.append(result[3, 3].item())  # L3H3
    return float(np.mean(scores))

# Also compute manually for verification
def compute_induction_manual(n_seqs=100, half=32):
    rng = np.random.default_rng(12345)
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, 3, pat_cache)
    all_scores = []
    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()
        
        det_pattern = get_induction_head_detection_pattern(tokens[0])
        
        pat_cache.clear()
        with torch.no_grad():
            _ = model(tokens)
        hp = pat_cache.pattern[0, 3]  # L3H3 attention [seq, seq]
        
        # mul error measure
        score = (hp * det_pattern.to(hp.device)).sum() / hp.sum()
        all_scores.append(score.item())
    inner.reset_hooks(hooks)
    return float(np.mean(all_scores))

print("Computing using transformer_lens detect_head...")
tl_score = compute_induction_score_tl(30)  # fewer seqs for speed
print(f"TL detect_head L3H3: {tl_score:.6f}")

print("\nComputing manually with detection pattern...")
manual_score = compute_induction_manual(100)
print(f"Manual L3H3: {manual_score:.6f}")

# Also report V1 (simple attention to induction target)
def compute_v1(n_seqs=100, half=32):
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

v1 = compute_v1(100)
print(f"\nV1 mean target attention: {v1:.6f}")
print(f"V1 * (half-1) / seq_len: {v1 * (half-1) / (2*half):.6f}")
print(f"V1 * 31 / 64: {v1 * 31 / 64:.6f}")
