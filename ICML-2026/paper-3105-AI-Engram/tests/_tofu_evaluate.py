"""TOFU evaluation (14 metrics + Overall), ported VERBATIM from
examples/llm_tofu.ipynb (itself ported from open-unlearning).

Not a pytest test module (no test_ prefix) — imported by test_tofu_evaluate.py.
Exposes: QADataset, Collator, TEMPLATE, GEN_ARGS, IGNORE_INDEX,
compute_full_tofu(model, tok, D), evaluate_scores(exp, retain, ft).
"""

# ===== standalone TOFU evaluation (ported verbatim from open-unlearning) =====
import warnings, numpy as np, torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer

IGNORE_INDEX = -100
TEMPLATE = {"apply_chat_template": True, "system_prompt": "You are a helpful assistant.",
            "date_string": "10 Apr 2025"}
GEN_ARGS = {"do_sample": False, "max_new_tokens": 200, "use_cache": True}

# ---------- data pipeline (verbatim: data/utils.py preprocess_chat_instance, qa.py, collators.py) ----------
def preprocess_chat_instance(tok, tcfg, prompt_msgs, response_msgs, max_length, pwg=False):
    if isinstance(prompt_msgs, str): prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]
    chat = []
    if tcfg.get("system_prompt"): chat += [{"role": "system", "content": tcfg["system_prompt"]}]
    for p, r in zip(prompt_msgs, response_msgs):
        chat += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
    di = {"date_string": tcfg["date_string"]} if tcfg.get("date_string") is not None else {}
    chat_ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=False, **di)
    prompt_ids = tok.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **di)
    if chat_ids[-1] != tok.eos_token_id: chat_ids += [tok.eos_token_id]
    n = len(prompt_ids); it = {}
    if pwg: it["input_ids"] = prompt_ids; labels = chat_ids
    else:   it["input_ids"] = chat_ids;   labels = [IGNORE_INDEX]*n + chat_ids[n:]
    it["labels"] = labels; it["attention_mask"] = [1]*len(it["input_ids"])
    return {k: torch.tensor(v) for k, v in it.items()}

class QADataset(Dataset):
    def __init__(self, hf_split, tok, qkey="question", akey="answer", max_length=512, pwg=False):
        self.data, self.tok, self.qkey, self.akey, self.max_length, self.pwg = hf_split, tok, qkey, akey, max_length, pwg
    def __len__(self): return len(self.data)
    def _p(self, q, a, i):
        d = preprocess_chat_instance(self.tok, TEMPLATE, [q], [a], self.max_length, self.pwg)
        return {"input_ids": d["input_ids"], "labels": d["labels"], "attention_mask": d["attention_mask"], "index": i}
    def __getitem__(self, idx):
        q, a = self.data[idx][self.qkey], self.data[idx][self.akey]
        if isinstance(a, str): return self._p(q, a, idx)
        return {i: self._p(q, ans, idx) for i, ans in enumerate(a)}

class Collator:
    def __init__(self, tok, padding_side="right"): self.tok, self.padding_side = tok, padding_side
    def _pad(self, seqs, pad):
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)
        return torch.nn.utils.rnn.pad_sequence([torch.flip(s, [0]) for s in seqs], batch_first=True, padding_value=pad).flip([1])
    def __call__(self, ins):
        if "input_ids" not in ins[0]:
            return {k: self([x[k] for x in ins]) for k in ins[0].keys()}
        ids = self._pad([x["input_ids"] for x in ins], self.tok.pad_token_id)
        o = {"input_ids": ids, "attention_mask": ids.ne(self.tok.pad_token_id)}
        if "labels" in ins[0]: o["labels"] = self._pad([x["labels"] for x in ins], IGNORE_INDEX)
        o["index"] = torch.tensor([x["index"] for x in ins]); return o

# ---------- helpers (verbatim: evals/metrics/utils.py) ----------
def aggregate_to_1D(x): return np.mean(x, axis=tuple(range(1, x.ndim)))
def dict_transpose(ev):
    ii = list(ev.keys()); ix = list(ev[ii[0]].keys()); st = list(ev[ii[0]][ix[0]].keys())
    return {i: {s: [ev[j][i][s] for j in ii] for s in st} for i in ix}
def run_batchwise_evals(model, dl, fn, args, msg=""):
    ev = defaultdict(dict)
    for batch in dl:
        if "input_ids" in batch: batch = {"0": batch}
        for ii, mb in batch.items():
            di = mb.pop("index").cpu().numpy().tolist()
            ev[ii] |= dict(zip(di, fn(model=model, batch=mb, **args)))
    return next(iter(ev.values())) if len(ev) == 1 else dict_transpose(ev)
def evaluate_probability(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad(): out = model(**batch)
    sl = batch["labels"][..., 1:].contiguous(); lg = out.logits[..., :-1, :].contiguous()
    lf = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = lf(lg.transpose(-1, -2), sl).sum(-1)
    avg = losses / (batch["labels"] != IGNORE_INDEX).sum(-1)
    return [{"prob": p, "avg_loss": a} for p, a in zip(torch.exp(-avg).cpu().float().tolist(), avg.cpu().float().tolist())]
def tokenwise_vocab_logprobs(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad(): out = model(**batch)
    V = out.logits.shape[-1]; lp = torch.nn.functional.log_softmax(out.logits, -1)[:, :-1, :]
    LPB, LB = [], []
    for i in range(out.logits.shape[0]):
        labels = batch["labels"][i]; ai = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if len(ai) == 0: LB.append(torch.tensor([], device=labels.device)); LPB.append(torch.zeros(0, V, device=labels.device)); continue
        s, e = ai[0].item(), ai[-1].item(); LPB.append(lp[i, s-1:e]); LB.append(labels[ai])
    return LPB, LB
def eval_text_similarity(model, tokenizer, batch, generation_args):
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    ids, labels = batch["input_ids"], batch["labels"]
    in_txt = tokenizer.batch_decode(ids, skip_special_tokens=True)
    full = tokenizer.batch_decode([l[l != IGNORE_INDEX] for l in labels], skip_special_tokens=True)
    gts = [f.replace(i, "").strip() for i, f in zip(in_txt, full)]
    out = model.generate(ids, attention_mask=batch["attention_mask"], **generation_args, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.batch_decode(out[:, ids.shape[-1]:], skip_special_tokens=True)
    eos = tokenizer.decode([tokenizer.eos_token_id]); res = []
    for g in gen:
        if eos and eos in g: g = g.split(eos)[0]
        res.append(g.strip())
    return [{"rougeL_recall": sc.score(gt, g)["rougeL"].recall} for g, gt in zip(res, gts)]

# ---------- metric drivers ----------
def prob_sbi(model, ds, coll, bs=16):
    return run_batchwise_evals(model, DataLoader(ds, batch_size=bs, collate_fn=coll), evaluate_probability, {})
def agg_prob(sbi): return float(np.mean(aggregate_to_1D(np.array([e["prob"] for e in sbi.values() if e["prob"] is not None]))))
def metric_em(model, ds, coll, bs=16):
    def fn(model, batch):
        LPB, LB = tokenwise_vocab_logprobs(model, batch); r = []
        for lp, lb in zip(LPB, LB):
            r.append({"score": None if len(lb) == 0 else (torch.argmax(lp, -1) == lb).sum().item()/len(lb)})
        return r
    sbi = run_batchwise_evals(model, DataLoader(ds, batch_size=bs, collate_fn=coll), fn, {})
    return float(np.mean(aggregate_to_1D(np.array([e["score"] for e in sbi.values() if e["score"] is not None]))))
def metric_es(model, ds, coll, bs=16):
    def fn(model, batch):
        LPB, LB = tokenwise_vocab_logprobs(model, batch); r = []
        for lp, lb in zip(LPB, LB):
            n = len(lb)
            if n == 0: r.append({"score": 0}); continue
            preds = torch.argmax(lp, -1); k = n
            for kk in range(n):
                if torch.equal(preds[kk:], lb[kk:]): k = kk; break
            r.append({"score": 1 - k/n})
        return r
    sbi = run_batchwise_evals(model, DataLoader(ds, batch_size=bs, collate_fn=coll), fn, {})
    return float(np.mean(aggregate_to_1D(np.array([e["score"] for e in sbi.values()]))))
def metric_rouge(model, ds, coll, tok, bs=16):
    sbi = run_batchwise_evals(model, DataLoader(ds, batch_size=bs, collate_fn=coll), eval_text_similarity,
                              {"tokenizer": tok, "generation_args": GEN_ARGS})
    return float(np.mean(aggregate_to_1D(np.array([e["rougeL_recall"] for e in sbi.values()]))))
def truth_ratio(correct_sbi, wrong_sbi, aggregator):
    idx = list(correct_sbi.keys())
    ca = aggregate_to_1D(np.array([correct_sbi[i]["avg_loss"] for i in idx]))
    wa = aggregate_to_1D(np.array([wrong_sbi[i]["avg_loss"] for i in idx]))
    tr = np.exp(-wa) / (np.exp(-ca) + 1e-10)
    if aggregator == "closer_to_1_better": return float(np.mean(np.minimum(tr, 1/(tr+1e-10))))
    if aggregator == "true_better":        return float(np.mean(np.maximum(0, 1-tr)))
    if aggregator == "prob_mean":           return float(np.mean(tr))
    raise ValueError(aggregator)
def prob_w_options(correct_sbi, wrong_sbi):
    idx = list(correct_sbi.keys())
    correct = np.array([correct_sbi[i]["prob"] for i in idx])
    all_wrong = np.array([wrong_sbi[i]["prob"] for i in idx])
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))
    return float(np.mean(correct / (correct + wrong + 1e-10)))

# ---------- high-level: model_utility (HM of 9 sub-metrics) ----------
def compute_model_utility(model, tok, retain_pert, ra, wf, bs=16):
    import scipy.stats as st
    cr, cl = Collator(tok, "right"), Collator(tok, "left")
    QA = lambda d, ak, pwg=False: QADataset(d, tok, akey=ak, pwg=pwg)
    # retain
    rt_prob  = agg_prob(prob_sbi(model, QA(retain_pert, "answer"), cr, bs))
    rt_rouge = metric_rouge(model, QA(retain_pert, "answer", True), cl, tok, bs)
    rt_tr    = truth_ratio(prob_sbi(model, QA(retain_pert, "paraphrased_answer"), cr, bs),
                           prob_sbi(model, QA(retain_pert, "perturbed_answer"), cr, bs), "true_better")
    # real authors
    ra_probn = prob_w_options(prob_sbi(model, QA(ra, "answer"), cr, bs),
                              prob_sbi(model, QA(ra, "perturbed_answer"), cr, bs))
    ra_rouge = metric_rouge(model, QA(ra, "answer", True), cl, tok, bs)
    ra_tr    = truth_ratio(prob_sbi(model, QA(ra, "answer"), cr, bs),
                           prob_sbi(model, QA(ra, "perturbed_answer"), cr, bs), "true_better")
    # world facts
    wf_probn = prob_w_options(prob_sbi(model, QA(wf, "answer"), cr, bs),
                              prob_sbi(model, QA(wf, "perturbed_answer"), cr, bs))
    wf_rouge = metric_rouge(model, QA(wf, "answer", True), cl, tok, bs)
    wf_tr    = truth_ratio(prob_sbi(model, QA(wf, "answer"), cr, bs),
                           prob_sbi(model, QA(wf, "perturbed_answer"), cr, bs), "true_better")
    subs = {"retain_Q_A_Prob": rt_prob, "retain_Q_A_ROUGE": rt_rouge, "retain_Truth_Ratio": rt_tr,
            "ra_Q_A_Prob_normalised": ra_probn, "ra_Q_A_ROUGE": ra_rouge, "ra_Truth_Ratio": ra_tr,
            "wf_Q_A_Prob_normalised": wf_probn, "wf_Q_A_ROUGE": wf_rouge, "wf_Truth_Ratio": wf_tr}
    return float(st.hmean(list(subs.values()))), subs

# ---------- Stage 3: MIA attacks, gibberish, forget_quality ----------
import zlib as _zlib
from sklearn.metrics import roc_auc_score

def tokenwise_logprobs(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad(): out = model(**batch)
    lp = torch.nn.functional.log_softmax(out.logits, -1)[:, :-1, :]
    tlp = torch.gather(lp, 2, batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
    R = []
    for i in range(out.logits.shape[0]):
        labels = batch["labels"][i]; ai = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if len(ai) == 0: R.append(torch.tensor([], device=labels.device)); continue
        s, e = ai[0].item(), ai[-1].item(); R.append(tlp[i, s-1:e])
    return R
def extract_target_texts(tok, batch):
    return [tok.decode(l[l != IGNORE_INDEX].tolist(), skip_special_tokens=True) for l in batch["labels"]]

def mia_scores(model, ds, coll, attack, tok=None, k=0.4, bs=16):
    dl = DataLoader(ds, batch_size=bs, collate_fn=coll); sc = []
    for b in dl:
        b.pop("index")
        if attack == "loss":
            sc += [r["avg_loss"] for r in evaluate_probability(model, b)]
        elif attack == "zlib":
            ep = evaluate_probability(model, b); tx = extract_target_texts(tok, b)
            sc += [r["avg_loss"] / len(_zlib.compress(t.encode("utf-8"))) for r, t in zip(ep, tx)]
        elif attack == "min_k":
            for lp in tokenwise_logprobs(model, b):
                a = lp.float().cpu().numpy()
                if a.size == 0: sc.append(0); continue
                nk = max(1, int(len(a) * k)); sc.append(float(-np.mean(np.sort(a)[:nk])))
        elif attack == "min_k++":
            vlps = tokenwise_vocab_logprobs(model, b)[0]; tlps = tokenwise_logprobs(model, b)
            for vlp, tlp in zip(vlps, tlps):
                if len(tlp) == 0: sc.append(0); continue
                mu = (torch.exp(vlp) * vlp).sum(-1)
                sig = torch.clamp((torch.exp(vlp) * vlp**2).sum(-1) - mu**2, min=1e-6)
                s = (tlp.float().cpu().numpy() - mu.float().cpu().numpy()) / torch.sqrt(sig).float().cpu().numpy()
                nk = max(1, int(len(s) * k)); sc.append(float(-np.mean(sorted(s)[:nk])))
    return sc
def mia_auc(model, forget_ds, holdout_ds, coll, attack, tok=None, k=0.4, bs=16):
    fs = mia_scores(model, forget_ds, coll, attack, tok, k, bs)
    hs = mia_scores(model, holdout_ds, coll, attack, tok, k, bs)
    return float(roc_auc_score([0]*len(fs) + [1]*len(hs), fs + hs)), fs

def gen_texts(model, ds, coll, tok, bs=16):
    dl = DataLoader(ds, batch_size=bs, collate_fn=coll); gens = []
    eos = tok.decode([tok.eos_token_id])
    for b in dl:
        b.pop("index"); b = {k: v.to(model.device) for k, v in b.items()}
        out = model.generate(b["input_ids"], attention_mask=b["attention_mask"], **GEN_ARGS, pad_token_id=tok.eos_token_id)
        for g in tok.batch_decode(out[:, b["input_ids"].shape[-1]:], skip_special_tokens=True):
            gens.append(g.split(eos)[0].strip() if eos in g else g.strip())
    return gens
def gibberish_score(generations, device="cuda", bs=32, class_id=0):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cid = "madhurjindal/autonlp-Gibberish-Detector-492513457"
    ct = AutoTokenizer.from_pretrained(cid)
    cm = AutoModelForSequenceClassification.from_pretrained(cid).to(device).eval()
    probs = []
    for i in range(0, len(generations), bs):
        enc = ct(generations[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad(): logits = cm(**enc).logits
        probs += torch.softmax(logits, -1)[:, class_id].cpu().float().tolist()
    return float(np.mean(probs))

def forget_tr_per_index(correct_sbi, wrong_sbi):
    idx = list(correct_sbi.keys())
    ca = aggregate_to_1D(np.array([correct_sbi[i]["avg_loss"] for i in idx]))
    wa = aggregate_to_1D(np.array([wrong_sbi[i]["avg_loss"] for i in idx]))
    return np.exp(-wa) / (np.exp(-ca) + 1e-10)
def forget_quality(model_tr, retain_tr):
    from scipy.stats import ks_2samp
    return float(ks_2samp(model_tr, retain_tr).pvalue)

# ---------- full TOFU summary (all 14 metrics + per-index for rescaling) ----------
def compute_full_tofu(model, tok, D, bs=16):
    """D = {'fp':forget_perturbed,'ho':holdout,'rp':retain_perturbed,'ra':real_authors,'wf':world_facts}"""
    cr, cl = Collator(tok, "right"), Collator(tok, "left")
    QA = lambda d, ak, pwg=False: QADataset(d, tok, akey=ak, pwg=pwg)
    fp, ho = D["fp"], D["ho"]
    EM = metric_em(model, QA(fp, "answer"), cr, bs)
    ES = metric_es(model, QA(fp, "answer"), cr, bs)
    PP = agg_prob(prob_sbi(model, QA(fp, "answer"), cr, bs))
    RG = metric_rouge(model, QA(fp, "answer", True), cl, tok, bs)
    para = prob_sbi(model, QA(fp, "paraphrased_answer"), cr, bs)
    pert = prob_sbi(model, QA(fp, "perturbed_answer"), cr, bs)
    TR = truth_ratio(para, pert, "closer_to_1_better")
    ftr = forget_tr_per_index(para, pert)
    MU, _ = compute_model_utility(model, tok, D["rp"], D["ra"], D["wf"], bs)
    FF = gibberish_score(gen_texts(model, QA(fp, "answer", True), cl, tok, bs))
    mia, mia_fs = {}, {}
    for atk in ["loss", "zlib", "min_k", "min_k++"]:
        auc, fs = mia_auc(model, QA(fp, "answer"), QA(ho, "answer"), cr, atk, tok=tok, k=0.4, bs=bs)
        mia[atk] = auc; mia_fs[atk] = fs
    return {"exact_memorization": EM, "extraction_strength": ES, "forget_Q_A_Prob": PP,
            "forget_Q_A_ROUGE": RG, "forget_truth_ratio": TR, "model_utility": MU,
            "forget_Q_A_gibberish": FF, "mia_loss": mia["loss"], "mia_zlib": mia["zlib"],
            "mia_min_k": mia["min_k"], "mia_min_k_plus_plus": mia["min_k++"],
            "_forget_tr": ftr, "_mia_fs": mia_fs}

def evaluate_scores(exp, retain, ft):
    """Rescale the experiment's metrics against the base + retain gold -> Overall etc."""
    import scipy.stats as st
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    absr = lambda x, mn, mx: float(np.clip(abs(x - mn) / abs(mx - mn + 1e-12), 0, 1))
    divr = lambda x, mx: float(np.clip(x / mx, 0, 1))
    hm = lambda s: float(st.hmean([max(v, 1e-10) for v in s]))
    EM = absr(exp["exact_memorization"], retain["exact_memorization"], ft["exact_memorization"])
    ES = absr(exp["extraction_strength"], retain["extraction_strength"], ft["extraction_strength"])
    PP = absr(exp["forget_Q_A_Prob"], retain["forget_Q_A_Prob"], ft["forget_Q_A_Prob"])
    TR = absr(exp["forget_truth_ratio"], retain["forget_truth_ratio"], ft["forget_truth_ratio"])
    Mem = hm([1 - EM, 1 - ES, 1 - PP, 1 - TR])
    MU = divr(exp["model_utility"], ft["model_utility"])
    FF = divr(exp["forget_Q_A_gibberish"], ft["forget_Q_A_gibberish"])
    Util = hm([MU, FF])
    def indist(rv, u):
        y = np.r_[np.zeros(len(rv)), np.ones(len(u))]
        try: auc = roc_auc_score(y, np.r_[rv, u])
        except ValueError: return 1.0
        return float(1 - 2 * abs(auc - 0.5))
    Priv = hm([indist(retain["_mia_fs"][a], exp["_mia_fs"][a]) for a in ["loss", "zlib", "min_k", "min_k++"]])
    FQ = float(np.log10(max(ks_2samp(exp["_forget_tr"], retain["_forget_tr"]).pvalue, 1e-30)))
    return {"Overall": hm([Mem, Util, Priv]), "Memorization": Mem, "Utility": Util,
            "Privacy": Priv, "EM": EM, "FQ": FQ}
