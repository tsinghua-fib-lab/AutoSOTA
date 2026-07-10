import argparse
import math
import os
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def hf_login_from_env():
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("[INFO] Hugging Face login successful via HF_TOKEN.")
        except Exception as e:
            print(f"[WARN] HF login failed: {e}")
    else:
        print("[WARN] HF_TOKEN env var not set (gated models will fail to load).")


def load_llama(model_id: str, cache_dir: str = "llm_weights"):
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def join_text(dataset, split: str) -> str:
    return "\n\n".join(dataset[split]["text"])


@torch.inference_mode()
def compute_perplexity(model, tokenizer, text: str, block_size: int = 2048) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, input_ids.size(0) - 1, block_size), desc="PPL blocks"):
        ids = input_ids[i:i+block_size].unsqueeze(0).to(device)
        labels = ids.clone()
        out = model(input_ids=ids, labels=labels, use_cache=False)
        n_tokens = ids.numel()
        total_nll += float(out.loss) * n_tokens
        total_tokens += n_tokens
        del ids, labels, out
        torch.cuda.empty_cache()
    return math.exp(total_nll / max(1, total_tokens))


def list_mlp_linears(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    mlps = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (".mlp." in name):
            if "lm_head" in name:
                continue
            mlps.append((name, mod))
    return mlps


class ActCollector:
    def __init__(self, modules: List[Tuple[str, nn.Linear]]):
        self.sq_sums: Dict[str, torch.Tensor] = {}
        self.hooks = []
        for name, mod in modules:
            self.sq_sums[name] = torch.zeros(mod.in_features, dtype=torch.float64)
            def make_hook(key):
                def _hook(m, inputs):
                    x = inputs[0].detach().to(torch.float32)
                    x = x.reshape(-1, x.shape[-1])
                    self.sq_sums[key] += (x.pow(2).sum(dim=0)).to("cpu", dtype=torch.float64)
                return _hook
            self.hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    def remove(self):
        for h in self.hooks:
            h.remove()
    def norms(self) -> Dict[str, torch.Tensor]:
        return {k: torch.sqrt(v).to(torch.float32) for k, v in self.sq_sums.items()}


def build_calib_loader(tokenizer, train_text: str, seq_len: int, num_samples: int):
    enc = tokenizer(train_text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, min(enc.size(0) - 1, num_samples * seq_len), seq_len):
        sl = enc[i:i+seq_len]
        if sl.numel() == seq_len:
            chunks.append(sl)
        if len(chunks) >= num_samples:
            break
    def gen():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for sl in chunks:
            yield sl.unsqueeze(0).to(device)
    return gen, len(chunks)


@torch.inference_mode()
def collect_activation_norms(model, tokenizer, train_text: str, seq_len: int = 512, num_samples: int = 128) -> Dict[str, torch.Tensor]:
    modules = list_mlp_linears(model)
    collector = ActCollector(modules)
    gen, got = build_calib_loader(tokenizer, train_text, seq_len, num_samples)
    print(f"[INFO] Calibrating on {got} sequences of length {seq_len}.")
    for batch in tqdm(gen(), total=got, desc="Calib pass"):
        _ = model(input_ids=batch, use_cache=False)
    collector.remove()
    return collector.norms()


# wanda / magnitude pruning with/without Neff

def wanda_per_row_prune(module: nn.Linear, x_norm: torch.Tensor, sparsity: float):
    """
    Wanda Algorithm 1 per-output/row pruning at given sparsity.
    score[i, j] = |W[i, j]| * x_norm[j]
    prune the lowest (sparsity * C_in) per row.
    """
    W = module.weight.data
    assert x_norm.numel() == W.size(1)
    metric = W.abs() * x_norm.to(W.device).unsqueeze(0)  # (C_out, C_in)
    k = int(W.size(1) * sparsity)
    if k <= 0:
        return
    sorted_idx = torch.argsort(metric, dim=1, descending=False)
    pruned_idx = sorted_idx[:, :k]
    with torch.no_grad():
        zero_src = torch.zeros_like(W)
        W.scatter_(dim=1, index=pruned_idx, src=zero_src)

    
def magnitude_prune(module: nn.Linear, sparsity: float):
    W = module.weight.data
    k = int(W.size(1) * sparsity)
    if k <= 0:
        return
    W_abs = W.abs()
    sorted_idx = torch.argsort(W_abs, dim=1, descending=False)
    pruned_idx = sorted_idx[:, :k]
    with torch.no_grad():
        zero_src = torch.zeros_like(W)
        W.scatter_(dim=1, index=pruned_idx, src=zero_src)


def wanda_neff(module: nn.Linear, x_norm: torch.Tensor, beta: float):
    W = module.weight.data
    assert x_norm.numel() == W.size(1)
    metric = W.abs() * x_norm.to(W.device).unsqueeze(0)  # (C_out, C_in)
    output_size, input_size = metric.shape
    m_norm = torch.abs(metric) / torch.sum(torch.abs(metric), dim=1, keepdim=True)
    neff = 1.0 / torch.sum(m_norm ** 2, dim=1, keepdim=True)
    r_neff = torch.floor(beta * neff)
    r_neff = torch.clamp(r_neff, min=1, max=metric.size(1))
    
    _, indices = torch.sort(m_norm, dim=1, descending=True)
    range_tensor = torch.arange(input_size, device=W.device).unsqueeze(0).expand(output_size, -1)
    sorted_mask = range_tensor < r_neff

    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, indices, sorted_mask)
    with torch.no_grad():
        W *= mask
        
        
def magnitude_neff(module: nn.Linear, beta: float):
    W = module.weight.data
    output_size, input_size = W.shape
    W_abs = torch.abs(W)
    m_norm = W_abs / torch.sum(W_abs, dim=1, keepdim=True)
    neff = 1.0 / torch.sum(m_norm ** 2, dim=1, keepdim=True)
    r_neff = torch.floor(beta * neff)
    r_neff = torch.clamp(r_neff, min=1, max=W.size(1))
    
    _, indices = torch.sort(W_abs, dim=1, descending=True)
    range_tensor = torch.arange(input_size, device=W.device).unsqueeze(0).expand(output_size, -1)
    sorted_mask = range_tensor < r_neff

    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, indices, sorted_mask)
    with torch.no_grad():
        W *= mask


def apply_pruning_mlp_only(
    model: nn.Module,
    variant: str,
    x_norms: Dict[str, torch.Tensor] = None,
    sparsity: float = 0.5,
    beta: float = 1.0
):
    modules = list_mlp_linears(model)
    print(f"[INFO] Pruning {len(modules)} Linear layers in MLP blocks with variant='{variant}'.")
    for name, lin in tqdm(modules, desc=f"Pruning ({variant})"):
        if variant == "wanda":
            assert x_norms is not None and name in x_norms
            wanda_per_row_prune(lin, x_norms[name], sparsity)

        elif variant == "magnitude":
            magnitude_prune(lin, sparsity)

        elif variant == "neff-wanda":
            assert x_norms is not None and name in x_norms
            wanda_neff(lin, x_norms[name], beta)
        elif variant == "neff-magnitude":
            magnitude_neff(lin, beta)
        else:
            raise ValueError("Unknown variant")
        

@dataclass
class SparsityStats:
    zeros: int
    total: int
    @property
    def sparsity(self) -> float:
        return 0.0 if self.total == 0 else self.zeros / self.total


def measure_mlp_weight_sparsity(model: nn.Module) -> SparsityStats:
    zeros = 0
    total = 0
    for _, lin in list_mlp_linears(model):
        W = lin.weight.data
        zeros += int((W == 0).sum().item())
        total += W.numel()
    return SparsityStats(zeros=zeros, total=total)


def _right_pad(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


@torch.inference_mode()
def batch_loglikelihood(model, tokenizer, prompts, completions, max_length=512, batch_size=8) -> torch.Tensor:
    assert len(prompts) == len(completions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="LL scores", leave=False):
        batch_prompts = prompts[i:i+batch_size]
        batch_comps   = completions[i:i+batch_size]

        full_ids_list, ctx_lens, seq_lens = [], [], []
        for p, c in zip(batch_prompts, batch_comps):
            ctx_ids  = tokenizer(p, add_special_tokens=False).input_ids
            comp_ids = tokenizer(c, add_special_tokens=False).input_ids
            full_ids = ctx_ids + comp_ids
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                ctx_ids  = ctx_ids[overflow:] if overflow < len(ctx_ids) else []
                full_ids = (ctx_ids + comp_ids)[-max_length:]
            ctx_len = len(ctx_ids)
            full_ids_list.append(full_ids)
            ctx_lens.append(ctx_len)
            seq_lens.append(len(full_ids))

        pad_id = tokenizer.pad_token_id
        input_ids = _right_pad(full_ids_list, pad_id=pad_id).to(device)
        attention_mask = (input_ids != pad_id).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        cont_masks = torch.zeros_like(shift_labels, dtype=torch.bool)
        for j, (cl, L) in enumerate(zip(ctx_lens, seq_lens)):
            start = max(0, cl)
            end   = max(0, L - 1)
            if start < end:
                cont_masks[j, start:end] = True

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp  = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        token_lp.masked_fill_(~cont_masks, 0.0)
        sample_sums = token_lp.sum(dim=1).detach().to("cpu")
        all_scores.append(sample_sums)

        del outputs, logits, shift_logits, shift_labels, token_lp
        torch.cuda.empty_cache()

    return torch.cat(all_scores, dim=0)


def mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold_indices, max_length=256, batch_size=8) -> float:
    flat_prompts, flat_comps, seg = [], [], []
    for ctx, opts in zip(contexts, options):
        for opt in opts:
            flat_prompts.append(ctx)
            flat_comps.append(" " + opt if not opt.startswith(" ") else opt)
        seg.append(len(opts))

    scores_flat = batch_loglikelihood(model, tokenizer, flat_prompts, flat_comps, max_length=max_length, batch_size=batch_size)

    preds, p = [], 0
    for n in seg:
        sc = scores_flat[p:p+n]
        pred = int(torch.argmax(sc).item())
        preds.append(pred); p += n

    correct = sum(int(a == b) for a, b in zip(preds, gold_indices))
    return correct / max(1, len(gold_indices))


def _limit(ds, n):
    if (n is None) or (n <= 0) or (n >= len(ds)):
        return ds
    return ds.select(range(n))


def eval_boolq(model, tokenizer, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("super_glue", "boolq", split="validation")
    ds = _limit(ds, max_samples)
    contexts = [f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:" for ex in ds]
    options  = [["yes", "no"] for _ in ds]
    # label True => "yes"(idx 0), False => "no"(idx 1)
    gold     = [0 if ex["label"] == 1 else 1 for ex in ds]
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def eval_rte(model, tokenizer, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("super_glue", "rte", split="validation")
    ds = _limit(ds, max_samples)
    contexts = [
        f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
        f"Does the premise entail the hypothesis? Answer yes or no:"
        for ex in ds
    ]
    options = [["yes", "no"] for _ in ds]
    gold = [0 if ex["label"] == 0 else 1 for ex in ds]  # 0=entailment
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def eval_hellaswag(model, tokenizer, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("hellaswag", split="validation")
    ds = _limit(ds, max_samples)
    contexts = [f"{ex['ctx_a']} {ex['ctx_b']}".strip() for ex in ds]
    options  = [ex["endings"] for ex in ds]
    gold     = [int(ex["label"]) for ex in ds]
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def eval_winogrande(model, tokenizer, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    ds = _limit(ds, max_samples)
    contexts = [ex["sentence"].split("_")[0].strip() for ex in ds]
    options  = [[ex["option1"], ex["option2"]] for ex in ds]
    gold     = [int(ex["answer"]) - 1 for ex in ds]
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def eval_arc(model, tokenizer, config: str, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("ai2_arc", config, split="test")
    if len(ds) == 0:
        ds = load_dataset("ai2_arc", config, split="validation")
    ds = _limit(ds, max_samples)
    contexts, options, gold = [], [], []
    for ex in ds:
        q = ex["question"]
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        contexts.append(f"Question: {q}\nAnswer:")
        options.append(texts)
        gold.append(labels.index(ex["answerKey"]))
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def eval_obqa(model, tokenizer, max_samples: int, bs: int, max_len: int) -> float:
    ds = load_dataset("openbookqa", "main", split="validation")
    if len(ds) == 0:
        ds = load_dataset("openbookqa", "main", split="test")
    ds = _limit(ds, max_samples)
    contexts, options, gold = [], [], []
    for ex in ds:
        q = ex.get("question_stem", ex.get("question", ""))
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        contexts.append(f"Question: {q}\nAnswer:")
        options.append(texts)
        gold.append(labels.index(ex["answerKey"]))
    return mcq_accuracy_by_ranking(model, tokenizer, contexts, options, gold, max_len, bs)


def run_zeroshot_suite(model, tokenizer, max_samples=0, batch_size=8, max_length=256) -> Dict[str, float]:
    ms = None if (max_samples is None or max_samples <= 0) else max_samples
    results = {}
    print("[INFO] Zero-shot eval (may take a while).")
    try:
        results["BoolQ"] = eval_boolq(model, tokenizer, ms, batch_size, max_length); print(f"  BoolQ        : {results['BoolQ']*100:.2f}%")
    except Exception as e: print(f"[WARN] BoolQ eval failed: {e}")
    try:
        results["RTE"] = eval_rte(model, tokenizer, ms, batch_size, max_length); print(f"  RTE          : {results['RTE']*100:.2f}%")
    except Exception as e: print(f"[WARN] RTE eval failed: {e}")
    try:
        results["HellaSwag"] = eval_hellaswag(model, tokenizer, ms, batch_size, max_length); print(f"  HellaSwag    : {results['HellaSwag']*100:.2f}%")
    except Exception as e: print(f"[WARN] HellaSwag eval failed: {e}")
    try:
        results["WinoGrande"] = eval_winogrande(model, tokenizer, ms, batch_size, max_length); print(f"  WinoGrande   : {results['WinoGrande']*100:.2f}%")
    except Exception as e: print(f"[WARN] WinoGrande eval failed: {e}")
    try:
        results["ARC-e"] = eval_arc(model, tokenizer, "ARC-Easy", ms, batch_size, max_length); print(f"  ARC-e        : {results['ARC-e']*100:.2f}%")
    except Exception as e: print(f"[WARN] ARC-e eval failed: {e}")
    try:
        results["ARC-c"] = eval_arc(model, tokenizer, "ARC-Challenge", ms, batch_size, max_length); print(f"  ARC-c        : {results['ARC-c']*100:.2f}%")
    except Exception as e: print(f"[WARN] ARC-c eval failed: {e}")
    try:
        results["OBQA"] = eval_obqa(model, tokenizer, ms, batch_size, max_length); print(f"  OBQA         : {results['OBQA']*100:.2f}%")
    except Exception as e: print(f"[WARN] OBQA eval failed: {e}")
    return results


def run_once(model_id, cache_dir, block_size_eval, calib_seq_len, calib_samples,
             variant=None, beta=1.0, do_zeroshot=False, zs_max_samples=0, zs_batch_size=8, zs_max_length=256):
    model, tokenizer = load_llama(model_id=model_id, cache_dir=cache_dir)
    print("[INFO] Loading WikiText-2...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
    test_text = join_text(wikitext, "test")

    zs_metrics = None
    if variant is None:
        ppl = compute_perplexity(model, tokenizer, test_text, block_size=block_size_eval)
        sparsity = measure_mlp_weight_sparsity(model).sparsity
        if do_zeroshot:
            zs_metrics = run_zeroshot_suite(model, tokenizer, zs_max_samples, zs_batch_size, zs_max_length)
    else:
        train_text = join_text(wikitext, "train")
        x_norms = None
        if variant in ("wanda", "neff-wanda"):
            x_norms = collect_activation_norms(model, tokenizer, train_text, seq_len=calib_seq_len, num_samples=calib_samples)

        apply_pruning_mlp_only(model, variant=variant, x_norms=x_norms, sparsity=0.5, beta=beta)
        sparsity = measure_mlp_weight_sparsity(model).sparsity
        ppl = compute_perplexity(model, tokenizer, test_text, block_size=block_size_eval)
        if do_zeroshot:
            zs_metrics = run_zeroshot_suite(model, tokenizer, zs_max_samples, zs_batch_size, zs_max_length)

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl, sparsity, zs_metrics


def print_zs_table(name: str, zs: Dict[str, float]):
    if not zs:
        return
    print(f"\n[ZERO-SHOT] {name}")
    print("-" * 70)
    keys = ["BoolQ", "RTE", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OBQA"]
    vals = []
    for k in keys:
        if k in zs and zs[k] is not None:
            vals.append(zs[k]); print(f"{k:<12}  {zs[k]*100:6.2f}%")
        else:
            print(f"{k:<12}      n/a")
    if vals:
        mean_acc = sum(vals) / len(vals)
        print("-" * 70)
        print(f"Mean accuracy over {len(vals)} tasks: {mean_acc*100:.2f}%")
    else:
        print("No tasks completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-13b-hf") # or LLama1 from huggyllama/llama-13b
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument("--block_size_eval", type=int, default=2048)
    parser.add_argument("--calib_seq_len", type=int, default=512)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)

    # zero-shot eval options
    parser.add_argument("--eval_zeroshot", action="store_true")
    parser.add_argument("--zs_max_samples", type=int, default=0, help="Max samples per task (<=0 => all).")
    parser.add_argument("--zs_batch_size", type=int, default=8)
    parser.add_argument("--zs_max_length", type=int, default=256)

    args = parser.parse_args()

    hf_login_from_env()
    print(f"[INFO] Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    print(f"[INFO] Loading {args.model_id}")

    # 1) Dense
    dense_ppl, dense_sprs, dense_zs = run_once(
        model_id=args.model_id, cache_dir=args.cache_dir, block_size_eval=args.block_size_eval,
        calib_seq_len=args.calib_seq_len, calib_samples=args.calib_samples,
        variant=None, do_zeroshot=args.eval_zeroshot, zs_max_samples=args.zs_max_samples,
        zs_batch_size=args.zs_batch_size, zs_max_length=args.zs_max_length
    )
    print(f"\n[RESULT] Dense PPL: {dense_ppl:.3f} | MLP sparsity: {dense_sprs*100:.2f}%")
    print_zs_table("Dense", dense_zs)

    # 2) Wanda
    wanda_ppl, wanda_sprs, wanda_zs = run_once(
        model_id=args.model_id, cache_dir=args.cache_dir, block_size_eval=args.block_size_eval,
        calib_seq_len=args.calib_seq_len, calib_samples=args.calib_samples,
        variant="wanda", do_zeroshot=args.eval_zeroshot, zs_max_samples=args.zs_max_samples,
        zs_batch_size=args.zs_batch_size, zs_max_length=args.zs_max_length
    )
    print(f"\n[RESULT] Wanda PPL: {wanda_ppl:.3f} | MLP sparsity: {wanda_sprs*100:.2f}%")
    print_zs_table("Wanda (MLP)", wanda_zs)
    
    # 3) Magnitude
    mag_ppl, mag_sprs, mag_zs = run_once(
        model_id=args.model_id, cache_dir=args.cache_dir, block_size_eval=args.block_size_eval,
        calib_seq_len=args.calib_seq_len, calib_samples=args.calib_samples,
        variant="magnitude", beta=args.beta, do_zeroshot=args.eval_zeroshot,
        zs_max_samples=args.zs_max_samples, zs_batch_size=args.zs_batch_size,
        zs_max_length=args.zs_max_length
    )
    print(f"\n[RESULT] Magnitude (|W|) PPL: {mag_ppl:.3f} | MLP sparsity: {mag_sprs*100:.2f}%")
    print_zs_table("Magnitude (|W|)", mag_zs)
    
    # 4) Neff with Wanda score
    neff_ws_ppl, neff_ws_sprs, neff_ws_zs = run_once(
        model_id=args.model_id, cache_dir=args.cache_dir, block_size_eval=args.block_size_eval,
        calib_seq_len=args.calib_seq_len, calib_samples=args.calib_samples,
        variant="neff-wanda", beta=args.beta, do_zeroshot=args.eval_zeroshot,
        zs_max_samples=args.zs_max_samples, zs_batch_size=args.zs_batch_size,
        zs_max_length=args.zs_max_length
    )
    print(f"\n[RESULT] Neff-Wanda (|W|*||X|| PPL: {neff_ws_ppl:.3f} | MLP sparsity: {neff_ws_sprs*100:.2f}%")
    print_zs_table(f"Neff-Wanda (|W|*||X||)", neff_ws_zs)

    # 5) Neff with magnitude
    neff_mag_ppl, neff_mag_sprs, neff_mag_zs = run_once(
        model_id=args.model_id, cache_dir=args.cache_dir, block_size_eval=args.block_size_eval,
        calib_seq_len=args.calib_seq_len, calib_samples=args.calib_samples,
        variant="neff-magnitude", beta=args.beta, do_zeroshot=args.eval_zeroshot,
        zs_max_samples=args.zs_max_samples, zs_batch_size=args.zs_batch_size,
        zs_max_length=args.zs_max_length
    )
    print(f"\n[RESULT] Neff-Magnitude (|W|) PPL: {neff_mag_ppl:.3f} | MLP sparsity: {neff_mag_sprs*100:.2f}%")
    print_zs_table("Neff-Magnitude (|W|)", neff_mag_zs)

    
    print("\n[SUMMARY]")
    print(f"  Dense PPL                 : {dense_ppl:.3f} | Sparsity: {dense_sprs*100:.2f}%")
    print(f"  Wanda                     : {wanda_ppl:.3f} | Sparsity: {wanda_sprs*100:.2f}%")
    print(f"  Magnitude                 : {mag_ppl:.3f} | Sparsity: {mag_sprs*100:.2f}%")
    print(f"  Neff-Wanda                : {neff_ws_ppl} | Sparsity: {neff_ws_sprs*100:.2f}%")
    print(f"  Neff-Magnitude            : {neff_mag_ppl} | Sparsity: {neff_mag_sprs*100:.2f}%")
if __name__ == "__main__":
    main()
