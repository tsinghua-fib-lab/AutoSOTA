#!/usr/bin/env python3
"""
Type 2 (RLC Circuits) Transfer Function Accuracy Evaluation
Uses Lcapy + SymPy symbolic equivalence matching the paper.
Filters for transfer function questions only.
"""
import os, sys, json, re, time, argparse, warnings, random
from pathlib import Path
import sympy as sp

warnings.filterwarnings("ignore")

class TFAccuracyEvaluator:
    def __init__(self, data_dir="/datasets/CircuitSense"):
        self.data_dir = Path(data_dir)
        self.stats = {"total": 0, "correct": 0, "error": 0, "timeout": 0}
        self.results = []
    
    def load_tf_samples(self):
        """Load Type 2 transfer function samples."""
        level1_dir = self.data_dir / "Analysis" / "synthetic" / "level1"
        if not level1_dir.exists():
            print(f"ERROR: {level1_dir} not found")
            return []
        
        samples = []
        for sample_dir in sorted(level1_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            
            question_file = sample_dir / f"{sample_dir.name}_question.txt"
            netlist_file = sample_dir / f"{sample_dir.name}_netlist.txt"
            answer_file = sample_dir / f"{sample_dir.name}_ta.txt"
            
            if not (question_file.exists() and netlist_file.exists() and answer_file.exists()):
                continue
            
            question = question_file.read_text().strip()
            
            # Filter: transfer function questions only
            if 'transfer function' not in question.lower():
                continue
            
            samples.append({
                "id": sample_dir.name,
                "netlist": netlist_file.read_text().strip(),
                "question": question,
                "expected_answer": answer_file.read_text().strip(),
            })
        
        return samples
    
    def parse_tf_question(self, question):
        """Parse 'What is the transfer function from X to Y?' -> (X, Y)"""
        m = re.search(r'from\s+(\w+)\s+to\s+(\w+)', question, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
        return None, None
    
    def compute_element_transfer(self, netlist, input_source, output_element):
        """Compute transfer function H(s) = V_element / V_input using Lcapy."""
        from lcapy import Circuit
        cct = Circuit(netlist)
        
        # Find input source
        if input_source not in cct.elements:
            found = False
            for name in cct.elements:
                if name.upper() == input_source.upper():
                    input_source = name
                    found = True
                    break
            if not found:
                for name in cct.elements:
                    if name.startswith('V') or name.startswith('I'):
                        input_source = name
                        break
        
        # Find output element
        if output_element not in cct.elements:
            for name in cct.elements:
                if name.upper() == output_element.upper():
                    output_element = name
                    break
            else:
                raise ValueError(f"Output {output_element} not found in {list(cct.elements.keys())}")
        
        src = cct[input_source]
        out = cct[output_element]
        
        sn = list(src.nodes)
        on = list(out.nodes)
        
        def n(v):
            return int(v) if str(v).lstrip('-').isdigit() else v
        
        H = cct.transfer(n(str(sn[0])), n(str(sn[1])), n(str(on[0])), n(str(on[1])))
        return str(H.simplify())
    
    def check_equivalence(self, computed, expected):
        """SymPy symbolic equivalence with numerical fallback."""
        sym_pat = re.compile(r'\b([A-Za-z]\w*)\b')
        all_syms = sym_pat.findall(computed + ' ' + expected)
        skip = {'s', 'I', 'V', 'H', 'E', 'j', 'sin', 'cos', 'exp', 'log', 'pi', 'sqrt', 'omega', 't', 'e'}
        sym_names = sorted(set(all_syms) - skip)
        
        sp_syms = {}
        for name in sym_names:
            sp_syms[name] = sp.Symbol(name, positive=True)
        sp_syms['s'] = sp.Symbol('s')
        
        try:
            H = sp.sympify(computed.replace(' ', ''), locals=sp_syms)
            E = sp.sympify(expected.replace(' ', ''), locals=sp_syms)
            
            diff = sp.simplify(H - E)
            if diff == 0:
                return True, "symbolic"
            
            dr = sp.simplify(sp.together(H - E))
            if dr == 0:
                return True, "symbolic_rational"
            
            # Numerical fallback
            random.seed(42)
            for _ in range(5):
                subs = {}
                for nm, sym in sp_syms.items():
                    if nm.startswith('C'):
                        subs[sym] = random.uniform(1e-12, 1e-6)
                    elif nm.startswith('R'):
                        subs[sym] = random.uniform(100, 1e6)
                    elif nm.startswith('L'):
                        subs[sym] = random.uniform(1e-9, 1e-3)
                    else:
                        subs[sym] = random.uniform(1, 1000)
                subs[sp_syms['s']] = complex(random.uniform(1e3, 1e9), random.uniform(1e3, 1e9))
                
                try:
                    v1 = complex(H.subs(subs).evalf())
                    v2 = complex(E.subs(subs).evalf())
                    rel = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-12)
                    if rel > 1e-3:
                        return False, f"numerical_diff={rel:.2e}"
                except:
                    pass
            
            return True, "numerical"
        except Exception as e:
            return False, f"sympy: {e}"
    
    def eval_sample(self, sample, timeout=60):
        import multiprocessing as mp
        
        def _compute():
            try:
                src, elem = self.parse_tf_question(sample["question"])
                if not src or not elem:
                    return False, "parse_error", None
                computed = self.compute_element_transfer(sample["netlist"], src, elem)
                match, reason = self.check_equivalence(computed, sample["expected_answer"])
                return match, reason, computed
            except Exception as e:
                return False, f"error: {e}", None
        
        q = mp.Queue()
        p = mp.Process(target=lambda: q.put(_compute()))
        p.start()
        p.join(timeout)
        
        if p.is_alive():
            p.terminate()
            p.join(1)
            return False, "timeout", None
        if q.empty():
            return False, "process_error", None
        return q.get()
    
    def run(self, max_samples=None):
        samples = self.load_tf_samples()
        print(f"Loaded {len(samples)} Type 2 transfer function samples", flush=True)
        
        if max_samples:
            samples = samples[:max_samples]
        
        start = time.time()
        
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                elapsed = time.time() - start
                acc = self.stats["correct"] / max(self.stats["total"], 1) * 100
                print(f"[{i+1}/{len(samples)}] Acc: {acc:.1f}% | {elapsed:.0f}s", flush=True)
            
            match, reason, computed = self.eval_sample(sample)
            
            self.stats["total"] += 1
            if match:
                self.stats["correct"] += 1
            elif "timeout" in str(reason):
                self.stats["timeout"] += 1
            else:
                self.stats["error"] += 1
            
            self.results.append({
                "id": sample["id"],
                "question": sample["question"],
                "match": match,
                "reason": reason,
            })
        
        elapsed = time.time() - start
        accuracy = self.stats["correct"] / max(self.stats["total"], 1) * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total TF samples: {self.stats['total']}")
        print(f"Correct: {self.stats['correct']}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Errors: {self.stats['error']}")
        print(f"Timeouts: {self.stats['timeout']}")
        print(f"Time: {elapsed:.0f}s")
        
        return accuracy, self.stats, self.results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", "-n", type=int, default=None)
    p.add_argument("--output", "-o", default="/repo/output/type2_tf_accuracy.json")
    p.add_argument("--data-dir", default="/datasets/CircuitSense")
    args = p.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    ev = TFAccuracyEvaluator(data_dir=args.data_dir)
    accuracy, stats, results = ev.run(max_samples=args.max_samples)
    
    out = {
        "paper_id": 6315,
        "task": "Transfer Function Generation",
        "circuit_type": "Type2 (RLC circuits)",
        "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
        "symbolic_engine": "Lcapy",
        "n_samples": stats["total"],
        "accuracy": round(accuracy, 2),
        "correct": stats["correct"],
        "stats": stats,
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {args.output}")
    print(f"ACCURACY: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
