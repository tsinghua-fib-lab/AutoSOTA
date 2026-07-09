#!/usr/bin/env python3
"""SimpleTool multi-head parallel decode — vLLM, v1/v2, external prompts
python 01_benchmark.py --version v2                    # v2 default model
python 01_benchmark.py --version v1                    # v1 default model
python 01_benchmark.py --version v2 --n-args 3         # fixed three arg heads 
python 01_benchmark.py --version v1 --model /my/model  # customed model path
"""
import argparse, json, time, os
from pathlib import Path

DIR = Path("./prompts")
HEADS = [("function","<function>","</function>")] + [(f"arg{i}",f"<arg{i}>",f"</arg{i}>") for i in range(1,7)]
STOPS = ["</function>"] + [f"</arg{i}>" for i in range(1,7)] + ["</content>","<|null|>","<|im_end|>"]
MODELS = {"v1":"./models/RT-Qwen3-4B-AWQ", "v2":"./models/RT-Qwen3-4B-AWQ-v2"}

def load_scenarios():
    scs = json.loads((DIR/"scenarios.json").read_text())
    for sc in scs:
        sc["tools"] = (DIR/sc["tools_file"]).read_text().strip()
    return scs

def max_tool_params(tools_str):
    m = 0
    for l in tools_str.strip().split("\n"):
        try: m = max(m, len(json.loads(l)["function"]["parameters"]["properties"]))
        except: pass
    return m

def build_prompt(sc, ver):
    t = sc["tools"]
    if ver == "v1":
        v1sys = (DIR/"v1_system.txt").read_text()
        return (f"<|im_start|>system\n{v1sys}\n## Available Tools:\n\n{t}<|im_end|>\n"
                f"<|im_start|>user\nenvironment: []\nhistory: {sc['history']}\n\n{sc['system']}\n\n{sc['query']}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return (f"<|im_start|>system\n{sc['system']}\n\n## Available Tools:\n\n{t}<|im_end|>\n"
            f"<|im_start|>user\nhistory: {sc['history']}\n\n{sc['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def clean(t):
    t = t.strip()
    return "<|null|>" if "<|null|>" in t or t == "" else t.split("</")[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--version", default="v2", choices=["v1","v2"])
    ap.add_argument("--n-args", default="auto")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()
    a.model = a.model or MODELS[a.version]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from vllm import LLM, SamplingParams

    SC = load_scenarios()
    print(f"\n{'='*60}\n  {a.version} | {a.model}\n{'='*60}")
    llm = LLM(model=a.model, trust_remote_code=True, dtype="auto", gpu_memory_utilization=0.80,
              max_model_len=a.max_model_len, max_num_seqs=8, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=128, stop=STOPS, include_stop_str_in_output=True)
    na = [min(max_tool_params(s["tools"]),6) if a.n_args=="auto" else max(1,min(6,int(a.n_args))) for s in SC]
    for s,n in zip(SC,na): print(f"  {s['name']:<35} heads={1+n}")

    def run(sc, n):
        hd = HEADS[:1+n]; base = build_prompt(sc, a.version)
        t0 = time.perf_counter()
        outs = llm.generate([base+op for _,op,_ in hd], sp)
        ms = (time.perf_counter()-t0)*1000
        raw, toks, full = {}, {}, {}
        for j,(nm,_,_) in enumerate(hd):
            if j<len(outs) and outs[j].outputs:
                o = outs[j].outputs[0]; full[nm]=o.text; raw[nm]=clean(o.text); toks[nm]=len(o.token_ids)
            else: raw[nm],toks[nm],full[nm] = "<|null|>",0,""
        return raw, toks, full, ms, hd

    # Cold
    print(f"\n{'='*60}\n  COLD START\n{'='*60}")
    cold = []
    for i,s in enumerate(SC): _,_,_,ms,_=run(s,na[i]); cold.append(ms); print(f"  {s['name']:<35} {ms:7.1f}ms")

    # Hot x3
    print(f"\n{'='*60}\n  HOT WARMUP (3 rounds)\n{'='*60}")
    hot = [[] for _ in SC]
    for r in range(3):
        for i,s in enumerate(SC): _,_,_,ms,_=run(s,na[i]); hot[i].append(ms)
        print(f"  Round {r+1}: "+"  ".join(f"{hot[j][-1]:6.1f}ms" for j in range(len(SC))))

    # Test
    print(f"\n{'='*60}\n  PARALLEL TEST ({a.version})\n{'='*60}\n")
    res = []
    for i,s in enumerate(SC):
        raw,toks,full,ms,hd = run(s,na[i]); mt=max(toks.values()) if toks else 0
        ok = raw.get("function","") == s["expected"]; res.append((s,raw,toks,full,ms,mt,hd,ok))
        print(f"─── {s['name']} ───\n{'PASS' if ok else 'FAIL'}  {s['desc']}")
        for nm,_,_ in hd:
            v,tc = raw.get(nm,""),toks.get(nm,0); d=v if len(v)<=43 else v[:43]+"…"
            st = ("OK" if ok else f"WRONG({v})") if nm=="function" else ("NULL" if v=="<|null|>" else "FILL")
            print(f"  {nm:<10} {d:<45} {tc:<4} {st}")
        print(f"  e2e={ms:.1f}ms  max_tok={mt}\n")

    # Summary
    N=len(res); np_=sum(r[7] for r in res); ae=sum(r[4] for r in res)/N; amt=sum(r[5] for r in res)/N
    print(f"{'='*60}\n  SUMMARY ({a.version})\n{'='*60}")
    print(f"  Accuracy       : {np_}/{N}\n  Cold start avg : {sum(cold)/N:.1f}ms\n  Hot prefill avg: {sum(sum(h) for h in hot)/sum(len(h) for h in hot):.1f}ms")
    print(f"  E2E avg (hot)  : {ae:.1f}ms\n  Max head tokens: {amt:.1f} avg\n  E2E / max_tok  : {ae/amt:.1f}ms/tok (decode bottleneck)\n")
    print(f"  {'Scenario':<35} {'Cold':>7} {'Hot':>7} {'E2E':>7} {'MaxTk':>6} {'ms/tk':>6}\n  {'─'*70}")
    for i,(s,_,_,_,ms,mt,_,_) in enumerate(res):
        print(f"  {s['name']:<35} {cold[i]:6.1f}  {sum(hot[i])/3:6.1f}  {ms:6.1f}  {mt:>5}  {ms/mt if mt else 0:5.1f}")

    # Example dump
    s,raw,toks,full,ms,mt,hd,ok = res[0]; base=build_prompt(s,a.version)
    print(f"\n{'='*60}\n  EXAMPLE ({a.version}): {s['name']}\n{'='*60}")
    print(f"\n┌─ Shared Prefix ({len(base)} chars) ────────────────────")
    for ln in base.split("\n"): print(f"│ {ln}")
    print(f"└──────────────────────────────────────────────────")
    print(f"\n┌─ Per-Head Trigger Tokens ─────────────────────────")
    for nm,op,_ in hd: print(f"│  {nm:<10} → {op}")
    print(f"└──────────────────────────────────────────────────")
    print(f"\n┌─ Decode Output (all tokens, incl. stop) ──────────")
    for nm,op,_ in hd: print(f"│  {nm:<10} [{toks.get(nm,0):>2} tok]  {op}{full.get(nm,'')}")
    print(f"└──────────────────────────────────────────────────")
    print(f"\n  Reconstructed multi-head response:")
    for nm,op,cl in hd:
        if raw.get(nm,"")=="<|null|>": print(f"    {op}<|null|>")
        else:
            ft=full.get(nm,""); print(f"    {op}{ft}" if any(ft.rstrip().endswith(x) for x in STOPS) else f"    {op}{ft}{cl}")
    print()

if __name__ == "__main__": main()
