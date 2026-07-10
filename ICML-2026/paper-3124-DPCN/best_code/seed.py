import os
import sys
import json
import datetime

import torch
import numpy
import pcx
import random

SEEDS = json.loads(os.environ.get("PC_SEEDS", "[0, 1, 2, 3, 4]"))
SAVE_INTERVAL = 10


class run:
    def __init__(self, fn):
        self._seeds = SEEDS

        def wrap_fn(*args, **kwargs):
            best_per_seed = []
            accuracies_per_seed = []
            best_per_seed5 = []
            accuracies_per_seed5 = []
            for seed_idx, seed in enumerate(self._seeds):
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)
                pcx.RKG.seed(seed)
                save_model = (seed == 0)

                best, best5, accuracies, accuracies5 = fn(
                    *args, save_model=save_model,
                    savepath=self._savepath, seed_idx=seed_idx, **kwargs
                )
                best_per_seed.append(best)
                accuracies_per_seed.append(accuracies)
                best_per_seed5.append(best5)
                accuracies_per_seed5.append(accuracies5)

                data = {
                    "status": "running",
                    "completed_seeds": seed_idx + 1,
                    "total_seeds": len(self._seeds),
                    "accuracies": accuracies_per_seed,
                    "best_per_seed": best_per_seed,
                    "avg": float(numpy.mean(best_per_seed)),
                    "std": float(numpy.std(best_per_seed)) if len(best_per_seed) > 1 else 0.0,
                    "accuracies_5": accuracies_per_seed5,
                    "best_per_seed5": best_per_seed5,
                    "avg_5": float(numpy.mean(best_per_seed5)),
                    "std_5": float(numpy.std(best_per_seed5)) if len(best_per_seed5) > 1 else 0.0,
                }
                with open(self._savepath, "w") as f:
                    json.dump(data, f, indent=4)

            return best_per_seed, accuracies_per_seed, best_per_seed5, accuracies_per_seed5

        self._fn = wrap_fn

    def __call__(self, *args, **kwargs):
        self._savepath = f"{sys.argv[1]}_accuracy.json"
        if os.path.exists(self._savepath):
            timestamp = datetime.datetime.now().timestamp()
            self._savepath = f"{sys.argv[1]}_accuracy_{timestamp}.json"

        with open(self._savepath, "w") as f:
            json.dump({"status": "running", "completed_seeds": 0}, f, indent=4)

        best, accuracies, best5, accuracies5 = self._fn(*args, **kwargs)

        with open(self._savepath, "w") as f:
            json.dump({
                "status": "done",
                "accuracies": accuracies,
                "avg": float(numpy.mean(best)),
                "std": float(numpy.std(best)),
                "accuracies_5": accuracies5,
                "avg_5": float(numpy.mean(best5)),
                "std_5": float(numpy.std(best5)),
            }, f, indent=4)

        print(f"{self._savepath} Done")
