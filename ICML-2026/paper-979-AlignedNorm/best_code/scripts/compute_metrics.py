import re
import numpy as np
import os

base_dir = "/repo/output/base2new/train_base/eurosat/shots_16/ALIGNEDNORM/vit_b16"
new_dir = "/repo/output/base2new/test_new/eurosat/shots_16/ALIGNEDNORM/vit_b16"

base_results = []
new_results = []
for seed in [1, 2, 3]:
    with open(os.path.join(base_dir, f"seed{seed}", "log.txt")) as f:
        for line in f:
            m = re.search(r"\* accuracy: ([\d.]+)%", line)
            if m:
                base_acc = float(m.group(1))
    with open(os.path.join(new_dir, f"seed{seed}", "log.txt")) as f:
        for line in f:
            m = re.search(r"\* accuracy: ([\d.]+)%", line)
            if m:
                new_acc = float(m.group(1))
    base_results.append(base_acc)
    new_results.append(new_acc)
    hm = 2 * base_acc * new_acc / (base_acc + new_acc)
    print(f"Seed {seed}: Base={base_acc:.2f}%, New={new_acc:.2f}%, HM={hm:.2f}%")

avg_base = np.mean(base_results)
avg_new = np.mean(new_results)
hms = [2 * b * n / (b + n) for b, n in zip(base_results, new_results)]
avg_hm = np.mean(hms)
print()
print("=== FINAL RESULTS (3-seed average) ===")
print(f"Base: {avg_base:.2f}%")
print(f"New:  {avg_new:.2f}%")
print(f"HM:   {avg_hm:.2f}%")
print()
print("=> result")
