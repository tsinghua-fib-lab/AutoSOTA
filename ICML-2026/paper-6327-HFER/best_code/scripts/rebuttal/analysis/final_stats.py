import json, numpy as np
with open('data/results/rebuttal/steering_results.json') as f:
    d = json.load(f)
v = [x for x in d if x['label'] == 'valid']
i = [x for x in d if x['label'] == 'invalid']
print(f"Steer Valid: {np.mean([x['hfer_base'] for x in v]):.4f} -> {np.mean([x['hfer_steered'] for x in v]):.4f}")
print(f"Steer Invalid: {np.mean([x['hfer_base'] for x in i]):.4f} -> {np.mean([x['hfer_steered'] for x in i]):.4f}")

with open('data/results/rebuttal/best_of_n.json') as f:
    bon = json.load(f)
for n in ['4', '8', '16']:
    p = bon[n]
    print(f"BoN N={n}: Random {np.mean([x['random_hfer'] for x in p]):.4f} Lowest {np.mean([x['lowest_hfer'] for x in p]):.4f}")
