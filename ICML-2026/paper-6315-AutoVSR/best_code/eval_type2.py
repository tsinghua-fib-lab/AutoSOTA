#!/usr/bin/env python3
"""
Type 2 (RLC Circuits) Transfer Function Evaluation
Computes accuracy using Lcapy + SymPy symbolic equivalence,
matching the paper's evaluation protocol.
"""
import os, sys, json, re, time, argparse, warnings, traceback, random
from pathlib import Path
import sympy as sp

warnings.filterwarnings("ignore")

class LcapyEvaluator:
    def __init__(self, data_dir="/datasets/CircuitSense"):
        self.data_dir = Path(data_dir)
        self.stats = {"total": 0, "correct": 0, "lcapy_error": 0, "sympy_error": 0, "timeout": 0}
        self.results = []
    
    def load_samples(self):
        level1_dir = self.data_dir / "Analysis" / "synthetic" / "level1"
        if not level1_dir.exists():
            print(f"ERROR: Data directory not found: {level1_dir}")
            return []
        
        samples = []
        for sample_dir in sorted(level1_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            
            netlist_file = None
            question_file = None
            answer_file = None
            
            for f in sample_dir.iterdir():
                if f.name.endswith("_netlist.txt"):
                    netlist_file = f
                elif f.name.endswith("_question.txt"):
                    question_file = f
                elif f.name.endswith("_ta.txt"):
                    answer_file = f
            
            if netlist_file and question_file and answer_file:
                sample_id = sample_dir.name
                samples.append({
                    "id": sample_id,
                    "netlist": netlist_file.read_text().strip(),
                    "question": question_file.read_text().strip(),
                    "expected_answer": answer_file.read_text().strip(),
                })
        
        return samples
    
    def parse_question(self, question):
        m = re.search(r'from\s+(\w+)\s+to\s+(\w+)', question, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
        m = re.search(r'(\w+)\s*/\s*(\w+)', question)
        if m:
            return m.group(2), m.group(1)
        return None, None
    
    def compute_transfer(self, netlist, input_source, output_element):
        from lcapy import Circuit
        cct = Circuit(netlist)
        
        if input_source not in cct.elements:
            for name in cct.elements:
                if name.upper() == input_source.upper():
                    input_source = name
                    break
            else:
                for name in cct.elements:
                    if name.startswith('V') or name.startswith('I'):
                        input_source = name
                        break
        
        if output_element not in cct.elements:
            for name in cct.elements:
                if name.upper() == output_element.upper():
                    output_element = name
                    break
            else:
                raise ValueError(f"Output element {output_element} not found")
        
        src_nodes = list(cct[input_source].nodes)
        n1p, n1m = str(src_nodes[0]), str(src_nodes[1])
        out_nodes = list(cct[output_element].nodes)
        n2p, n2m = str(out_nodes[0]), str(out_nodes[1])
        
        def _node(v):
            return int(v) if v.isdigit() else v
        
        H = cct.transfer(_node(n1p), _node(n1m), _node(n2p), _node(n2m))
        return str(H.simplify())
    
    def check_equivalence(self, computed, expected):
        sym_pat = re.compile(r'\b([A-Za-z]\w*)\b')
        sym_c = set(sym_pat.findall(computed))
        sym_e = set(sym_pat.findall(expected))
        all_syms = sym_c | sym_e
        skip = {'s', 'I', 'V', 'H', 'E', 'j', 'sin', 'cos', 'exp', 'log', 'pi', 'sqrt', 'omega', 't'}
        sym_names = sorted(all_syms - skip)
        
        sp_syms = {}
        for name in sym_names:
            sp_syms[name] = sp.Symbol(name, positive=True)
        sp_syms['s'] = sp.Symbol('s')
        
        computed_c = computed.replace(' ', '')
        expected_c = expected.replace(' ', '')
        
        try:
            H = sp.sympify(computed_c, locals=sp_syms)
            E = sp.sympify(expected_c, locals=sp_syms)
            
            diff = sp.simplify(H - E)
            if diff == 0:
                return True, "symbolic"
            
            diff2 = sp.simplify(sp.together(H - E))
            if diff2 == 0:
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
                    elif nm.startswith('gm') or nm.startswith('G'):
                        subs[sym] = random.uniform(1e-6, 1e-1)
                    else:
                        subs[sym] = random.uniform(1, 1000)
                subs[sp_syms['s']] = complex(random.uniform(1e3, 1e9), random.uniform(1e3, 1e9))
                
                try:
                    v1 = complex(H.subs(subs).evalf())
                    v2 = complex(E.subs(subs).evalf())
                    rel = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-12)
                    if rel > 1e-3:
                        return False, f"numerical_diff={rel:.2e}"
                except Exception as e:
                    return False, f"eval_error: {e}"
            
            return True, "numerical"
        except Exception as e:
            return False, f"sympy: {e}"
    
    def evaluate_sample(self, sample, timeout=60):
        import multiprocessing as mp
        
        def _compute():
            try:
                src, elem = self.parse_question(sample["question"])
                if not src or not elem:
                    return False, "parse_error", None
                computed = self.compute_transfer(sample["netlist"], src, elem)
                match, reason = self.check_equivalence(computed, sample["expected_answer"])
                return match, reason, computed
            except Exception as e:
                return False, f"error: {e}", None
        
        result_queue = mp.Queue()
        p = mp.Process(target=lambda: result_queue.put(_compute()))
        p.start()
        p.join(timeout)
        
        if p.is_alive():
            p.terminate()
            p.join(1)
            return False, "timeout", None
        
        if result_queue.empty():
            return False, "process_error", None
        
        return result_queue.get()
    
    def run(self, max_samples=None):
        samples = self.load_samples()
        print(f"Loaded {len(samples)} Type 2 samples", flush=True)
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"Limited to {max_samples} samples", flush=True)
        
        start = time.time()
        
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(samples) - i - 1) / rate if rate > 0 else 0
                acc = self.stats["correct"] / max(self.stats["total"], 1) * 100
                print(f"[{i+1}/{len(samples)}] Acc: {acc:.1f}% | {elapsed:.0f}s ETA:{eta:.0f}s", flush=True)
            
            match, reason, computed = self.evaluate_sample(sample)
            
            self.stats["total"] += 1
            if match:
                self.stats["correct"] += 1
            elif "lcapy" in str(reason).lower():
                self.stats["lcapy_error"] += 1
            elif "sympy" in str(reason).lower():
                self.stats["sympy_error"] += 1
            elif "timeout" in str(reason).lower():
                self.stats["timeout"] += 1
            
            self.results.append({
                "id": sample["id"],
                "question": sample["question"],
                "expected": sample["expected_answer"],
                "computed": computed,
                "match": match,
                "reason": reason,
            })
        
        elapsed = time.time() - start
        accuracy = self.stats["correct"] / max(self.stats["total"], 1) * 100
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total: {self.stats['total']}")
        print(f"Correct: {self.stats['correct']}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Lcapy Errors: {self.stats['lcapy_error']}")
        print(f"SymPy Errors: {self.stats['sympy_error']}")
        print(f"Timeouts: {self.stats['timeout']}")
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        
        return accuracy, self.stats, self.results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", "-n", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default="/repo/output/type2_eval_results.json")
    parser.add_argument("--data-dir", type=str, default="/datasets/CircuitSense")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    evaluator = LcapyEvaluator(data_dir=args.data_dir)
    accuracy, stats, results = evaluator.run(max_samples=args.max_samples)
    
    output = {
        "paper_id": 6315,
        "task": "Transfer Function Generation",
        "circuit_type": "Type 2 (RLC circuits)",
        "evaluation_protocol": "SymPy symbolic equivalence with numerical substitution fallback",
        "symbolic_engine": "Lcapy",
        "n_samples": stats["total"],
        "accuracy": round(accuracy, 2),
        "correct": stats["correct"],
        "stats": stats,
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    print(f"ACCURACY: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
