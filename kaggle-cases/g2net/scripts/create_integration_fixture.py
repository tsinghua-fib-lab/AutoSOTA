"""Small random dataset used only to check the training and submission pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "experiments/integration_fixture/packed"
destination.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2021)
for split, size in [("train", 50), ("test", 10)]:
    waves = rng.normal(0, 0.1, (size, 3, 4096)).astype(np.float32)
    table = pd.DataFrame({"id": [f"{i:010x}" for i in range(size)], "target": np.arange(size) % 2})
    if split == "train":
        for seed in [2021, 48]:
            table[f"fold_{seed}"] = -1
            for fold, (_, indices) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(table, table.target)):
                table.loc[indices, f"fold_{seed}"] = fold
    np.save(destination / f"{split}.npy", waves)
    table.to_csv(destination / f"{split}.csv", index=False)
(destination / "metadata.json").write_text(json.dumps({"fixture": True, "amplitude_scale": 1e20, "competition_score": None}))
print(destination)
