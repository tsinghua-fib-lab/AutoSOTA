import json
import os
import numpy as np

def audit_3sat():
    print("--- 3-SAT AUDIT ---")
    results_dir = 'c:/Users/valno/Dev/geometry-of-reason/data/results/rebuttal/'
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found.")
        return
    files = [x for x in os.listdir(results_dir) if 'sat' in x.lower() or '3sat' in x.lower()]
    print('Files found:', files)
    if not files:
        print("RESULT: No 3-SAT specific JSON found in the directory.")
        return
    
    with open(os.path.join(results_dir, files[0])) as f:
        d = json.load(f)
    print(f'Type: {type(d)}, Len: {len(d) if isinstance(d,list) else "dict"}')
    if isinstance(d, list) and len(d) > 0:
        valid = [x for x in d if x.get('is_valid') or x.get('correct')]
        invalid = [x for x in d if not (x.get('is_valid') or x.get('correct'))]
        print(f'Valid items: {len(valid)}, Invalid items: {len(invalid)}')
        print('Sample keys:', list(d[0].keys()))
        if 'hfer' in d[0]:
            print('Sample HFER:', d[0]['hfer'])
    elif isinstance(d, dict):
        print('Keys:', list(d.keys()))
        if 'summary' in d:
            print('Summary:', d['summary'])

def audit_bon():
    print("\n--- BEST-OF-N AUDIT ---")
    results_dir = 'c:/Users/valno/Dev/geometry-of-reason/data/results/rebuttal/'
    files = [x for x in os.listdir(results_dir) if 'best' in x.lower() or 'bon' in x.lower()]
    print('Files found:', files)
    for fn in files:
        print(f"\nAudit for {fn}:")
        with open(os.path.join(results_dir, fn)) as f:
            d = json.load(f)
        if isinstance(d, list) and len(d) > 0:
            print(f'List length: {len(d)}')
            print('Sample item keys:', list(d[0].keys()))
            if 'best_correct' in d[0]:
                best_corrects = [x.get('best_correct', False) for x in d]
                print(f"Accuracy (best_correct): {sum(best_corrects)/len(best_corrects):.4f}")
        elif isinstance(d, dict):
            print('Keys:', list(d.keys()))
            if 'summary' in d:
                print('Summary:', d['summary'])
            if 'Random Selection Pass@1' in d:
                print('Pass@1 Metrics:', d)

def audit_steering():
    print("\n--- STEERING AUDIT ---")
    results_dir = 'c:/Users/valno/Dev/geometry-of-reason/data/results/rebuttal/'
    files = [x for x in os.listdir(results_dir) if 'steer' in x.lower() or 'repe' in x.lower() or 'gsm' in x.lower()]
    print('Files found:', files)
    for fn in files:
        print(f"\nAudit for {fn}:")
        with open(os.path.join(results_dir, fn)) as f:
            d = json.load(f)
        if isinstance(d, list) and len(d) > 0:
            print(f"List of {len(d)} items.")
            print("Sample item keys:", list(d[0].keys()))
            print("Sample:", json.dumps(d[0], indent=2)[:300])
        else:
            print(json.dumps(d, indent=2)[:500])

if __name__ == "__main__":
    audit_3sat()
    audit_bon()
    audit_steering()
