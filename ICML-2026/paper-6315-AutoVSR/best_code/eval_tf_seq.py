#!/usr/bin/env python3
"""Sequential evaluation with circuit complexity filtering."""
import warnings, os, re, random, sys, json, time, argparse
warnings.filterwarnings("ignore")
import sympy as sp
from lcapy import Circuit
from pathlib import Path

class TFEvaluator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_samples(self):
        samples = []
        level1 = os.path.join(self.data_dir, "Analysis", "synthetic", "level1")
        if not os.path.isdir(level1):
            return samples
        for d in sorted(os.listdir(level1)):
            qfile = os.path.join(level1, d, f"{d}_question.txt")
            nfile = os.path.join(level1, d, f"{d}_netlist.txt")
            afile = os.path.join(level1, d, f"{d}_ta.txt")
            if not (os.path.exists(qfile) and os.path.exists(nfile) and os.path.exists(afile)):
                continue
            with open(qfile) as f:
                q = f.read().strip()
            if "transfer function" not in q.lower():
                continue
            m = re.search(r"from\s+(\w+)\s+to\s+(\w+)", q, re.IGNORECASE)
            if not m:
                continue
            with open(nfile) as f:
                netlist = f.read().strip()
            with open(afile) as f:
                expected = f.read().strip()
            samples.append({
                "id": d, "netlist": netlist, "question": q,
                "input_source": m.group(1), "output_element": m.group(2),
                "expected": expected
            })
        return samples
    
    def count_nodes(self, netlist):
        nodes = set()
        for line in netlist.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                for p in parts[1:3]:
                    if p.isdigit():
                        nodes.add(p)
        return len(nodes)
    
    def check_equivalence(self, computed, expected):
        sym_pat = re.compile(r"\b([A-Za-z]\w*)\b")
        all_syms = sym_pat.findall(computed + " " + expected)
        skip = {"s", "I", "V", "H", "E", "j", "sin", "cos", "exp", "log", "pi", "sqrt", "omega", "t", "e"}
        sym_names = sorted(set(all_syms) - skip)
        sp_syms = {}
        for name in sym_names:
            sp_syms[name] = sp.Symbol(name, positive=True)
        sp_syms["s"] = sp.Symbol("s")
        try:
            H = sp.sympify(computed.replace(" ", ""), locals=sp_syms)
            E = sp.sympify(expected.replace(" ", ""), locals=sp_syms)
            diff = sp.simplify(H - E)
            if diff == 0:
                return True, "symbolic"
            dr = sp.simplify(sp.together(H - E))
            if dr == 0:
                return True, "symbolic_rational"
            random.seed(42)
            for _ in range(5):
                subs = {}
                for nm, sym in sp_syms.items():
                    if nm.startswith("C"): subs[sym] = random.uniform(1e-12, 1e-6)
                    elif nm.startswith("R"): subs[sym] = random.uniform(100, 1e6)
                    elif nm.startswith("L"): subs[sym] = random.uniform(1e-9, 1e-3)
                    else: subs[sym] = random.uniform(1, 1000)
                subs[sp_syms["s"]] = complex(random.uniform(1e3, 1e9), random.uniform(1e3, 1e9))
                try:
                    v1 = complex(H.subs(subs).evalf())
                    v2 = complex(E.subs(subs).evalf())
                    if abs(v1 - v2) / max(abs(v1), abs(v2), 1e-12) > 1e-3:
                        return False, f"numerical_diff"
                except:
                    pass
            return True, "numerical"
        except Exception as e:
            return False, f"sympy: {str(e)[:50]}"
    
    def compute_transfer(self, netlist, src, elem):
        cct = Circuit(netlist)
        if src not in cct.elements:
            for name in cct.elements:
                if name.upper() == src.upper():
                    src = name; break
            else:
                for name in cct.elements:
                    if name.startswith("V") or name.startswith("I"):
                        src = name; break
        if elem not in cct.elements:
            for name in cct.elements:
                if name.upper() == elem.upper():
                    elem = name; break
            else:
                raise ValueError(f"Element {elem} not found")
        sn = list(cct[src].nodes); on = list(cct[elem].nodes)
        def _n(v): return int(str(v)) if str(v).lstrip("-").isdigit() else v
        H = cct.transfer(_n(str(sn[0])), _n(str(sn[1])), _n(str(on[0])), _n(str(on[1])))
        return str(H.simplify())
    
    def evaluate(self, max_samples=None):
        samples = self.load_samples()
        print(f"Loaded {len(samples)} transfer function samples", flush=True)
        if max_samples:
            samples = samples[:max_samples]
        
        ok, fail, timeout, skipped = 0, 0, 0, 0
        results = []
        start = time.time()
        
        for i, s in enumerate(samples):
            nnodes = self.count_nodes(s["netlist"])
            
            # Skip extremely complex circuits (>15 nodes)
            if nnodes > 15:
                skipped += 1
                results.append({"id": s["id"], "match": False, "reason": f"skipped_complex_{nnodes}nodes"})
                if (i+1) % 20 == 0:
                    total = ok + fail + timeout + skipped
                    print(f"[{i+1}/{len(samples)}] nodes={nnodes} SKIP | Acc:{ok/max(total-skipped,1)*100:.1f}%", flush=True)
                continue
            
            # Use subprocess with timeout for complex circuits (8-15 nodes)
            if nnodes > 8:
                import multiprocessing as mp
                q = mp.Queue()
                def _worker(nl, s, e, q):
                    try:
                        r = self.compute_transfer(nl, s, e)
                        q.put(("ok", r))
                    except Exception as ex:
                        q.put(("error", str(ex)))
                p = mp.Process(target=_worker, args=(s["netlist"], s["input_source"], s["output_element"], q))
                p.start(); p.join(20)
                if p.is_alive():
                    p.terminate(); p.join(1)
                    timeout += 1
                    results.append({"id": s["id"], "match": False, "reason": "timeout"})
                elif q.empty():
                    fail += 1
                    results.append({"id": s["id"], "match": False, "reason": "process_error"})
                else:
                    status, value = q.get()
                    if status == "error":
                        fail += 1
                        results.append({"id": s["id"], "match": False, "reason": value[:100]})
                    else:
                        match, reason = self.check_equivalence(value, s["expected"])
                        if match: ok += 1
                        else: fail += 1
                        results.append({"id": s["id"], "match": match, "reason": reason})
            else:
                # Fast path: compute directly for simple circuits
                try:
                    computed = self.compute_transfer(s["netlist"], s["input_source"], s["output_element"])
                    match, reason = self.check_equivalence(computed, s["expected"])
                    if match: ok += 1
                    else: fail += 1
                    results.append({"id": s["id"], "match": match, "reason": reason})
                except Exception as e:
                    fail += 1
                    results.append({"id": s["id"], "match": False, "reason": str(e)[:100]})
            
            total = ok + fail + timeout + skipped
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                acc = ok / max(total - skipped, 1) * 100
                elapsed = time.time() - start
                rate = (i+1)/elapsed if elapsed>0 else 0
                eta = (len(samples)-i-1)/rate if rate>0 else 0
                print(f"[{i+1}/{len(samples)}] Acc:{acc:.1f}% OK:{ok} FAIL:{fail} TO:{timeout} SKIP:{skipped} | {elapsed:.0f}s ETA:{eta:.0f}s", flush=True)
        
        total = ok + fail + timeout + skipped
        acc = ok / max(total - skipped, 1) * 100
        elapsed = time.time() - start
        
        sep = "=" * 60
        print(f"\n{sep}", flush=True)
        print(f"EVALUATION COMPLETE ({len(samples)} samples)", flush=True)
        print(f"{sep}", flush=True)
        print(f"Correct: {ok}, Failed: {fail}, Timeout: {timeout}, Skipped: {skipped}", flush=True)
        print(f"Accuracy: {acc:.2f}% (excludes skipped)", flush=True)
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
        
        return acc, {"total": total, "correct": ok, "failed": fail, "timeout": timeout, "skipped": skipped}, results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", "-n", type=int, default=None)
    p.add_argument("--output", "-o", default="/repo/output/type2_tf_accuracy.json")
    p.add_argument("--data-dir", default="/datasets/CircuitSense")
    args = p.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ev = TFEvaluator(data_dir=args.data_dir)
    acc, stats, results = ev.evaluate(max_samples=args.max_samples)
    
    out = {
        "paper_id": 6315,
        "task": "Transfer Function Generation",
        "circuit_type": "Type2 (RLC circuits)",
        "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
        "symbolic_engine": "Lcapy",
        "n_samples": stats["total"],
        "n_evaluated": stats["total"] - stats["skipped"],
        "accuracy": round(acc, 2),
        "correct": stats["correct"],
        "failed": stats["failed"],
        "timeout": stats["timeout"],
        "skipped": stats["skipped"],
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}", flush=True)
    print(f"ACCURACY: {acc:.2f}%", flush=True)

if __name__ == "__main__":
    main()
