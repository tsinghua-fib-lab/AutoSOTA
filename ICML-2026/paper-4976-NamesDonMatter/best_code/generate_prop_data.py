"""Generate propositional logic dataset for paper 4976 reproduction."""
import spot
import os
import random
import time
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from pebble import ProcessPool
from pebble.common import SLEEP_UNIT
from functools import partial

# Propositional logic priorities: no temporal operators
# Operators: not, and, or, equiv, xor, True, False
PROP_PRIORITIES = 'false=1,true=1,not=1,F=0,G=0,X=0,equiv=1,implies=0,xor=1,R=0,U=0,W=0,M=0,and=1,or=1'

AP_COUNT = 5
AP_LIST = list(map(chr, range(97, 97 + AP_COUNT)))  # a, b, c, d, e
TREE_SIZE_MIN = 1
TREE_SIZE_MAX = 35
TOTAL_TARGET = 120000  # Generate ~120K to get ~100K SAT
SEED = 42
OUTPUT_DIR = 'data/na-5-ts-35-nf-120k-lbt-sat-prop'
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def spot_get_trace(formula_str):
    """Find a satisfying trace for a formula using spot."""
    try:
        spot_formula = spot.formula(formula_str)
        automaton = spot_formula.translate()
        automaton.merge_edges()
        acc_run = automaton.accepting_run()
        if acc_run is None:
            return None
        trace = spot.twa_word(acc_run)
        return str(trace)
    except Exception:
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    # Formula generator with propositional logic priorities
    formula_gen = spot.randltl(
        AP_LIST,
        seed=SEED,
        tree_size=(TREE_SIZE_MIN, TREE_SIZE_MAX),
        ltl_priorities=PROP_PRIORITIES,
        simplify=0,
    )

    cpus = cpu_count()
    print(f"Using {cpus} CPUs")

    samples = []
    unsat_count = 0
    error_count = 0

    start_time = time.time()

    with tqdm(total=TOTAL_TARGET, desc="SAT formulas") as pbar:
        with ProcessPool(cpus) as pool:
            futures = {}

            while len(samples) < TOTAL_TARGET:
                # Submit new formulas up to queue limit
                while len(futures) < 500:
                    try:
                        formula_spot = next(formula_gen)
                    except StopIteration:
                        print("Formula generator exhausted")
                        break
                    formula_str = formula_spot.to_str()
                    fut = pool.schedule(
                        spot_get_trace, args=(formula_str,), timeout=30
                    )
                    futures[fut] = formula_str

                if not futures:
                    break

                # Collect completed futures
                done_futures = []
                for fut, formula_str in list(futures.items()):
                    if fut.done():
                        done_futures.append(fut)
                        try:
                            result = fut.result()
                            if result is not None:
                                samples.append((formula_str, result))
                                pbar.update(1)
                            else:
                                unsat_count += 1
                        except Exception:
                            error_count += 1

                for fut in done_futures:
                    del futures[fut]

                time.sleep(SLEEP_UNIT)

    elapsed = time.time() - start_time
    total_target = len(samples)
    total_generated = total_target + unsat_count + error_count

    print(f"\nGenerated {total_target} SAT formulas in {elapsed:.1f}s")
    print(f"SAT rate: {total_target / max(total_generated, 1) * 100:.1f}%")
    print(f"UNSAT: {unsat_count}, Errors: {error_count}")

    # Shuffle and split
    random.shuffle(samples)
    train_end = int(TRAIN_FRAC * total_target)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * total_target)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_data in splits.items():
        path = os.path.join(OUTPUT_DIR, split_name + ".txt")
        with open(path, "w") as f:
            for formula, trace in split_data:
                f.write(f"{formula}\n{trace}\n")
        print(f"{split_name}: {len(split_data)} pairs -> {path}")

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "info.txt"), "w") as f:
        f.write(f"Generated: {total_generated}\n")
        f.write(f"SAT: {total_target}\n")
        f.write(f"UNSAT: {unsat_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"APs: {AP_LIST}\n")
        f.write(f"Tree size: {TREE_SIZE_MIN}-{TREE_SIZE_MAX}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Priorities: {PROP_PRIORITIES}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")

    print(f"Dataset saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
