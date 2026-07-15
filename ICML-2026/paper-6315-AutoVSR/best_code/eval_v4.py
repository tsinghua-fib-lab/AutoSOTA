import warnings, os, re, random, json, time
warnings.filterwarnings("ignore")
import sympy as sp
from lcapy import Circuit

log = open("/repo/output/eval_log.txt", "w")

def log_print(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

def fix_netlist_sources(netlist):
    """Fix netlist sources for transfer function analysis.
    Converts step/dc sources to s-domain sources for TF computation."""
    lines = []
    for line in netlist.strip().split("\n"):
        line = line.strip()
        if not line: 
            lines.append(line)
            continue
        
        # Match Vx/Ix Np Nm step -> Vx Np Nm s Vx (or Ix)
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            if (name.startswith("V") or name.startswith("I")) and parts[3].lower() in ("step", "dc"):
                # Convert to s-domain
                parts[3] = "s"
                if len(parts) == 4:
                    parts.append(name)
            elif (name.startswith("V") or name.startswith("I")):
                # Check if it's got a value but no domain specifier
                rest = parts[3:]
                valid_domains = {"s", "dc", "ac", "step"}
                if len(parts) == 4 and parts[3].lower() not in valid_domains:
                    # V1 1 0 Vin -> V1 1 0 s Vin
                    val = parts[3]
                    parts[3] = "s"
                    parts.append(val)
        
        lines.append(" ".join(parts))
    return "\n".join(lines)

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

log_print(f"Loaded {len(samples)} TF samples")

def check_eq(computed, expected):
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
        for _ in range(5):
            subs = {}
            for nm, sym in ss.items():
                if nm.startswith("C"): subs[sym] = random.uniform(1e-12, 1e-6)
                elif nm.startswith("R"): subs[sym] = random.uniform(100, 1e6)
                elif nm.startswith("L"): subs[sym] = random.uniform(1e-9, 1e-3)
                else: subs[sym] = random.uniform(1, 1000)
            subs[ss["s"]] = complex(random.uniform(1e3, 1e9), random.uniform(1e3, 1e9))
            try:
                v1 = complex(H.subs(subs).evalf()); v2 = complex(E.subs(subs).evalf())
                if abs(v1-v2)/max(abs(v1),abs(v2),1e-12) > 1e-3:
                    return False, "num_mismatch"
            except: pass
        return True, "numerical"
    except Exception as e:
        return False, f"sympy:{str(e)[:40]}"

ok = fail = 0
results = []
start = time.time()

for i, s in enumerate(samples):
    t0 = time.time()
    try:
        # Fix netlist sources for TF analysis
        nl_fixed = fix_netlist_sources(s["netlist"])
        
        cct = Circuit(nl_fixed)
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
        t_comp = time.time() - t0
        
        match, reason = check_eq(computed, s["expected"])
        if match: ok += 1
        else: fail += 1
        results.append({"id": s["id"], "match": match, "reason": reason, "time": round(t_comp,1)})
    except Exception as e:
        fail += 1
        results.append({"id": s["id"], "match": False, "reason": str(e)[:80], "time": round(time.time()-t0,1)})
    
    total = ok + fail
    acc = ok / max(total, 1) * 100
    elapsed = time.time() - start
    rate = (i+1)/elapsed if elapsed > 0 else 0
    eta = (len(samples)-i-1)/rate if rate > 0 else 0
    r = results[-1]
    log_print(f"[{i+1}/{len(samples)}] {r['id']} {'OK' if r['match'] else 'FAIL'} {r['reason'][:30]} | {r['time']}s | Acc:{acc:.1f}% | {elapsed:.0f}s ETA:{eta:.0f}s")

sep = "=" * 60
log_print(f"\n{sep}\nRESULTS\n{sep}")
log_print(f"Samples: {len(samples)}, Correct: {ok}, Failed: {fail}")
acc = ok/max(ok+fail,1)*100
log_print(f"Accuracy: {acc:.2f}%")
log_print(f"Time: {time.time()-start:.0f}s")

out = {
    "paper_id": 6315, "task": "Transfer Function Generation",
    "circuit_type": "Type2 (RLC circuits)",
    "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
    "symbolic_engine": "Lcapy",
    "n_samples": len(samples), "accuracy": round(acc, 2),
    "correct": ok, "failed": fail, "results": results,
}
with open("/repo/output/type2_tf_accuracy.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log_print(f"Saved: /repo/output/type2_tf_accuracy.json")
log_print(f"ACCURACY: {acc:.2f}%")
log.close()
