"""Exercise the winner's actual signal/noise generator with a tiny output size."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/kdl_generation"
sys.path.insert(0, str(SOURCE))
import noise
import signals


def main():
    destination = ROOT / "data/synthetic_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    # noise.py loads the three supplied PSD files relative to cwd.
    os.chdir(SOURCE)
    started = time.time()
    noise.make_noise_ls(0, catalog=str(destination / "noise"), chunks=4)
    successes = 0
    attempts = []
    for seed in range(6):
        ok = bool(signals.generate_signal_spin(seed, catalog=str(destination / "signals")))
        attempts.append({"seed": seed, "success": ok})
        successes += int(ok)
        if successes >= 2:
            break
    files = []
    for path in sorted(destination.rglob("*.npy")):
        wave = np.load(path, allow_pickle=False)
        if wave.shape != (3, 4096) or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError(f"Invalid generated waveform: {path}")
        files.append({"path": str(path.relative_to(ROOT)), "shape": list(wave.shape),
                      "dtype": str(wave.dtype), "rms": np.sqrt(np.mean(wave ** 2, axis=1)).tolist()})
    result = {"passed": successes >= 2, "signal_attempts": attempts,
              "files": files, "seconds": round(time.time() - started, 2),
              "test_kind": "winner's basic generator; not a full synthetic-pretraining reproduction",
              "competition_auc": None}
    (ROOT / "reports/generator_smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
