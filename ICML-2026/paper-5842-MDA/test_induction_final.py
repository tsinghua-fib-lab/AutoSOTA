import sys; sys.path.insert(0, "/repo")
import os, numpy as np, torch, json
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

def compute_all(n_seqs=100, half=32):
    rng = np.random.default_rng(12345)
    all_results = {}
    for l in range(6):
        all_results[l] = {}
        for h in range(4):
            all_results[l][h] = []

    for si in range(n_seqs):
        prefix = rng.integers(0, int(inner.cfg.d_vocab), size=half)
        tokens_np = np.concatenate([prefix, prefix])
        tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()

        # Detection pattern needs CPU tensor
        det_pattern = get_induction_head_detection_pattern(tokens[0].cpu()).to(tokens.device)

        pat_caches = {}
        all_hooks = []
        for layer in range(6):
            pc = PatternCache()
            pat_caches[layer] = pc
            hooks = setup_pattern_hooks(inner, layer, pc)
            all_hooks.extend(hooks)

        with torch.no_grad():
            _ = model(tokens)

        for layer in range(6):
            hp = pat_caches[layer].pattern[0]
            for head in range(4):
                attn = hp[head]
                score = (attn * det_pattern).sum() / attn.sum()
                all_results[layer][head].append(score.item())

        try:
            inner.reset_hooks(all_hooks)
        except:
            pass

    print("Head       Mean       Std")
    print("-" * 30)
    for layer in range(6):
        for head in range(4):
            scores = all_results[layer][head]
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            marker = " ***" if mean_s > 0.1 else ""
            print(f"L{layer}H{head}:    {mean_s:.6f}  {std_s:.6f}{marker}")

    target_score = np.mean(all_results[layer][head])
    # Actually, compute L3H3
    target_score = np.mean(all_results[3][3])
    print(f"\nTarget L3H3: {target_score:.6f}")
    return target_score

score = compute_all(100)
print(f"\nFinal induction score (detection pattern mul): {score:.6f}")
print(f"Paper baseline: 0.432")
print(f"Ratio: {score / 0.432:.3f}")
