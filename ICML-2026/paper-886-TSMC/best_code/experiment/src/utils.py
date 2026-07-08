from typing import Any
import json
import sys
import os


class OptionalWANDBLogger:
    def __init__(self, wandb_service=None):
        self.wandb_service = wandb_service
        self._file = None
        if wandb_service is None:
            log_dir = os.environ.get("METRICS_LOG_DIR", "/tmp")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "metrics.jsonl")
            self._file = open(log_path, "a")

    def log(self, data: dict[str, Any]):
        if self.wandb_service is not None:
            self.wandb_service.log(data)
        elif self._file is not None:
            filtered = {k: v for k, v in data.items()
                       if "eval" in k or "return" in k or "loss" in k or "step" in k or "value" in k}
            if filtered:
                line = json.dumps(filtered, default=str)
                self._file.write(line + "\n")
                self._file.flush()
            for k, v in data.items():
                if "return" in k.lower() or "episode" in k.lower():
                    print("  [{}] {}: {}".format(data.get("step", "?"), k, v), file=sys.stderr)
