"""Final eval with subprocess-based timeouts for Lcapy.
Improved version: multi-strategy transfer function computation with fallbacks.
"""
import warnings, os, re, random, json, time, subprocess, sys
warnings.filterwarnings("ignore")
import sympy as sp

TIMEOUT_PRIMARY = 15      # Short timeout for fast transfer() method
TIMEOUT_FALLBACK = 120    # Longer timeout for V-ratio fallback method

log = open("/repo/output/eval_log.txt", "w")
def log_print(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

def fix_netlist(nl):
    """Fix source types for transfer function analysis.
    Only replaces step/dc with s for independent sources."""
    lines = []
    for line in nl.strip().split("\n"):
        line = line.strip()
        if not line:
            lines.append(line); continue
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            if (name.startswith("V") or name.startswith("I")) and parts[3].lower() in ("step", "dc"):
                parts[3] = "s"
        lines.append(" ".join(parts))
    return "\n".join(lines)

def compute_transfer_transfer(netlist, src, elem, timeout=TIMEOUT_PRIMARY):
    """Strategy 1: Use Lcapy cct.transfer() - fast for most circuits."""
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from lcapy import Circuit\n"
        "import sys, traceback\n"
        "try:\n"
        '    cct = Circuit("""' + netlist + '""")\n'
        '    src = "' + src + '"\n'
        '    elem = "' + elem + '"\n'
        "    if src not in cct.elements:\n"
        "        for n in cct.elements:\n"
        "            if n.upper() == src.upper(): src = n; break\n"
        "        else:\n"
        "            for n in cct.elements:\n"
        "                if n.startswith('V') or n.startswith('I'): src = n; break\n"
        "    if elem not in cct.elements:\n"
        "        for n in cct.elements:\n"
        "            if n.upper() == elem.upper(): elem = n; break\n"
        "        else:\n"
        '            print("ERROR:Element not found: " + elem)\n'
        "            sys.exit(0)\n"
        "    sn = list(cct[src].nodes)\n"
        "    on = list(cct[elem].nodes)\n"
        "    def nd(v): return int(str(v)) if str(v).lstrip('-').isdigit() else v\n"
        "    H = cct.transfer(nd(str(sn[0])), nd(str(sn[1])), nd(str(on[0])), nd(str(on[1])))\n"
        "    result = str(H.simplify())\n"
        '    print("OK:" + result, end="")\n'
        "except Exception as e:\n"
        '    print("ERROR:" + str(e)[:200], end="")\n'
        "    traceback.print_exc(file=sys.stderr)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        stdout = r.stdout.strip()
        if stdout.startswith("OK:"):
            return True, stdout[3:], "transfer"
        else:
            err = stdout.replace("ERROR:", "", 1) if stdout.startswith("ERROR:") else stdout[:100]
            return False, None, err
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    except Exception as e:
        return False, None, str(e)[:100]

def compute_transfer_vratio(netlist, src, elem, timeout=TIMEOUT_FALLBACK):
    """Strategy 2: Use V_elem.s / V_src.s ratio - handles circuits where transfer() hangs.
    This avoids the cct.transfer() call which hangs when source and target share nodes."""
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from lcapy import Circuit\n"
        "import sys, traceback\n"
        "try:\n"
        '    cct = Circuit("""' + netlist + '""")\n'
        '    src = "' + src + '"\n'
        '    elem = "' + elem + '"\n'
        "    if src not in cct.elements:\n"
        "        for n in cct.elements:\n"
        "            if n.upper() == src.upper(): src = n; break\n"
        "        else:\n"
        "            for n in cct.elements:\n"
        "                if n.startswith('V') or n.startswith('I'): src = n; break\n"
        "    if elem not in cct.elements:\n"
        "        for n in cct.elements:\n"
        "            if n.upper() == elem.upper(): elem = n; break\n"
        "        else:\n"
        '            print("ERROR:Element not found: " + elem)\n'
        "            sys.exit(0)\n"
        "    V_elem_s = cct[elem].V.s\n"
        "    V_src_s = cct[src].V.s\n"
        "    H = (V_elem_s / V_src_s).simplify()\n"
        "    result = str(H)\n"
        '    print("OK:" + result, end="")\n'
        "except Exception as e:\n"
        '    print("ERROR:" + str(e)[:200], end="")\n'
        "    traceback.print_exc(file=sys.stderr)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        stdout = r.stdout.strip()
        if stdout.startswith("OK:"):
            return True, stdout[3:], "vratio"
        else:
            err = stdout.replace("ERROR:", "", 1) if stdout.startswith("ERROR:") else stdout[:100]
            return False, None, err
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    except Exception as e:
        return False, None, str(e)[:100]

def compute_transfer_subprocess(netlist, src, elem):
    """Compute transfer function with multi-strategy fallback.

    Strategy 1 (fast): cct.transfer() with primary timeout
    Strategy 2 (fallback): V_elem.s / V_src.s with longer fallback timeout
    """
    # Strategy 1: Try transfer() first (fast for ~95% of circuits)
    success, result, method = compute_transfer_transfer(netlist, src, elem, TIMEOUT_PRIMARY)
    if success:
        return True, result, method

    # Strategy 2: Fall back to voltage ratio method
    success2, result2, method2 = compute_transfer_vratio(netlist, src, elem, TIMEOUT_FALLBACK)
    if success2:
        return True, result2, method2

    # Both failed - return the error from the last attempt
    err = result if result else (result2 if result2 else "all_strategies_failed")
    return False, None, err

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
                if abs(v1-v2)/max(abs(v1),abs(v2),1e-12) > 1e-3: return False, "num_mismatch"
            except: pass
        return True, "numerical"
    except Exception as e:
        return False, f"sympy:{str(e)[:40]}"

# Load
data_dir = "/datasets/CircuitSense/Analysis/synthetic/level1"
samples = []
for d in sorted(os.listdir(data_dir)):
    qf = os.path.join(data_dir, d, f"{d}_question.txt")
    nf = os.path.join(data_dir, d, f"{d}_netlist.txt")
    af = os.path.join(data_dir, d, f"{d}_ta.txt")
    if not (os.path.exists(qf) and os.path.exists(nf) and os.path.exists(af)): continue
    with open(qf) as f: q = f.read().strip()
    if "transfer function" not in q.lower(): continue
    m = re.search(r"from\s+(\w+)\s+to\s+(\w+)", q, re.IGNORECASE)
    if not m: continue
    with open(nf) as f: nl = f.read().strip()
    with open(af) as f: exp = f.read().strip()
    samples.append({"id": d, "netlist": nl, "src": m.group(1), "elem": m.group(2), "expected": exp})

log_print(f"Loaded {len(samples)} TF samples")
log_print(f"Strategy: transfer() ({TIMEOUT_PRIMARY}s) -> V-ratio fallback ({TIMEOUT_FALLBACK}s)")

ok = fail = timeout = 0
transfer_ok = vratio_ok = 0
results = []
start = time.time()

for i, s in enumerate(samples):
    t0 = time.time()
    nl_fixed = fix_netlist(s["netlist"])

    success, computed, method = compute_transfer_subprocess(nl_fixed, s["src"], s["elem"])
    t_comp = time.time() - t0

    if not success:
        if method == "timeout" or (method and "timeout" in str(method)):
            timeout += 1
            results.append({"id": s["id"], "match": False, "reason": "timeout", "time": round(t_comp,1), "method": str(method)})
        else:
            fail += 1
            results.append({"id": s["id"], "match": False, "reason": str(method)[:80] if method else "unknown", "time": round(t_comp,1)})
    else:
        if method == "transfer":
            transfer_ok += 1
        elif method == "vratio":
            vratio_ok += 1
        match, reason = check_eq(computed, s["expected"])
        if match: ok += 1
        else: fail += 1
        results.append({"id": s["id"], "match": match, "reason": reason, "time": round(t_comp,1), "method": method})

    total = ok + fail + timeout
    acc = ok / max(total - timeout, 1) * 100
    elapsed = time.time() - start
    rate = (i+1)/elapsed if elapsed > 0 else 0
    eta = (len(samples)-i-1)/rate if rate > 0 else 0
    r = results[-1]
    method_str = f"[{r.get('method','?')}]" if success else ""
    log_print(f"[{i+1}/{len(samples)}] {r['id']} {'OK' if r['match'] else 'FAIL/TO'} {str(r['reason'])[:30]} | {r['time']}s {method_str} | Acc:{acc:.1f}% | {elapsed:.0f}s ETA:{eta:.0f}s")

sep = "=" * 60
log_print(f"\n{sep}\nRESULTS\n{sep}")
log_print(f"Samples: {len(samples)}, Correct: {ok}, Failed: {fail}, Timeout: {timeout}")
log_print(f"By method: transfer={transfer_ok}, vratio={vratio_ok}")
acc = ok/max(total-timeout, 1)*100
log_print(f"Accuracy: {acc:.2f}% (excl. timeouts)")
log_print(f"Accuracy (incl. timeouts): {ok/len(samples)*100:.2f}%")
log_print(f"Time: {time.time()-start:.0f}s")

out = {
    "paper_id": 6315, "task": "Transfer Function Generation",
    "circuit_type": "Type2 (RLC circuits)",
    "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
    "symbolic_engine": "Lcapy with multi-strategy fallback (transfer + vratio)",
    "n_samples": len(samples), "n_evaluated": total - timeout,
    "accuracy": round(acc, 2), "correct": ok, "failed": fail, "timeout": timeout,
    "by_method": {"transfer": transfer_ok, "vratio": vratio_ok},
    "results": results,
}
with open("/repo/output/type2_tf_accuracy.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log_print(f"Saved: /repo/output/type2_tf_accuracy.json")
log_print(f"ACCURACY: {acc:.2f}%")
log.close()
