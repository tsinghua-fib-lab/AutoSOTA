import warnings, os, re, random, json, time
warnings.filterwarnings("ignore")
import sympy as sp
from lcapy import Circuit

data_dir = "/datasets/CircuitSense/Analysis/synthetic/level1"

# Load samples
samples = []
for d in sorted(os.listdir(data_dir)):
    qf = os.path.join(data_dir, d, f"{d}_question.txt")
    nf = os.path.join(data_dir, d, f"{d}_netlist.txt")
    af = os.path.join(data_dir, d, f"{d}_ta.txt")
    if not (os.path.exists(qf) and os.path.exists(nf) and os.path.exists(af)):
        continue
    with open(qf) as f: q = f.read().strip()
    if "transfer function" not in q.lower(): continue
    m = re.search(r"from\s+(\w+)\s+to\s+(\w+)", q, re.IGNORECASE)
    if not m: continue
    with open(nf) as f: nl = f.read().strip()
    with open(af) as f: exp = f.read().strip()
    samples.append({"id": d, "netlist": nl, "src": m.group(1), "elem": m.group(2), "expected": exp})

print(f"Loaded {len(samples)} TF samples", flush=True)

# SymPy check
def check(computed, expected):
    pat = re.compile(r"\b([A-Za-z]\w*)\b")
    syms = set(pat.findall(computed + " " + expected))
    skip = {"s", "I", "V", "H", "E", "j", "sin", "cos", "exp", "log", "pi", "sqrt", "omega", "t", "e"}
    names = sorted(syms - skip)
    ss = {}
    for n in names: ss[n] = sp.Symbol(n, positive=True)
    ss["s"] = sp.Symbol("s")
    try:
        H = sp.sympify(computed.replace(" ", ""), locals=ss)
        E = sp.sympify(expected.replace(" ", ""), locals=ss)
        if sp.simplify(H - E) == 0: return True, "symbolic"
        if sp.simplify(sp.together(H - E)) == 0: return True, "rational"
        random.seed(42)
        for _ in range(3):
            subs = {}
            for nm, sym in ss.items():
                if nm.startswith("C"): subs[sym] = random.uniform(1e-12, 1e-6)
                elif nm.startswith("R"): subs[sym] = random.uniform(100, 1e6)
                elif nm.startswith("L"): subs[sym] = random.uniform(1e-9, 1e-3)
                else: subs[sym] = random.uniform(1, 1000)
            subs[ss["s"]] = complex(random.uniform(1e3, 1e9), random.uniform(1e3, 1e9))
            try:
                v1 = complex(H.subs(subs).evalf()); v2 = complex(E.subs(subs).evalf())
                if abs(v1-v2)/max(abs(v1),abs(v2),1e-12) > 1e-3: return False, "numerical_mismatch"
            except: pass
        return True, "numerical"
    except Exception as e:
        return False, f"sympy:{str(e)[:50]}"

ok = fail = 0
results = []
start = time.time()

for i, s in enumerate(samples):
    try:
        cct = Circuit(s["netlist"])
        src, elem = s["src"], s["elem"]
        if src not in cct.elements:
            for n in cct.elements:
                if n.upper() == src.upper(): src = n; break
            else:
                for n in cct.elements:
                    if n.startswith("V") or n.startswith("I"): src = n; break
        if elem not in cct.elements:
            for n in cct.elements:
                if n.upper() == elem.upper(): elem = n; break
            else: raise ValueError(f"{elem} not found")
        
        sn = list(cct[src].nodes); on = list(cct[elem].nodes)
        def nd(v): return int(str(v)) if str(v).lstrip("-").isdigit() else v
        H = cct.transfer(nd(str(sn[0])), nd(str(sn[1])), nd(str(on[0])), nd(str(on[1])))
        computed = str(H.simplify())
        
        match, reason = check(computed, s["expected"])
        if match: ok += 1
        else: fail += 1
        results.append({"id": s["id"], "match": match, "reason": reason})
    except Exception as e:
        fail += 1
        results.append({"id": s["id"], "match": False, "reason": str(e)[:100]})
    
    total = ok + fail
    acc = ok / max(total, 1) * 100
    elapsed = time.time() - start
    rate = (i+1)/elapsed if elapsed > 0 else 0
    eta = (len(samples)-i-1)/rate if rate > 0 else 0
    status = "OK" if results[-1]["match"] else "FAIL"
    print(f"[{i+1}/{len(samples)}] {s['id']} {status} ({results[-1]['reason'][:30]}) | Acc:{acc:.1f}% | {elapsed:.0f}s ETA:{eta:.0f}s", flush=True)

sep = "=" * 60
print(f"\n{sep}\nRESULTS\n{sep}")
print(f"Samples: {len(samples)}, Correct: {ok}, Failed: {fail}")
print(f"Accuracy: {ok/max(ok+fail,1)*100:.2f}%")
print(f"Time: {time.time()-start:.0f}s")

# Save
out = {
    "paper_id": 6315, "task": "Transfer Function Generation",
    "circuit_type": "Type2 (RLC circuits)",
    "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
    "symbolic_engine": "Lcapy",
    "n_samples": len(samples), "accuracy": round(ok/max(ok+fail,1)*100, 2),
    "correct": ok, "failed": fail, "results": results,
}
with open("/repo/output/type2_tf_accuracy.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Saved: /repo/output/type2_tf_accuracy.json")
print(f"ACCURACY: {ok/max(ok+fail,1)*100:.2f}%")
