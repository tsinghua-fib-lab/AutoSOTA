from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json

from rich.console import Console

from .config import Config


@dataclass
class ExperimentRun:
    run_id: str
    run_dir: Path
    config: Config
    console: Console = field(default_factory=Console)

    @classmethod
    def create(cls, config: Config) -> "ExperimentRun":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{config.experiment.name}_{timestamp}"

        experiments_dir = Path(config.output.experiments_dir)
        run_dir = experiments_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config.to_yaml(run_dir / "config.yaml")

        return cls(run_id=run_id, run_dir=run_dir, config=config)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        metrics_path = self.run_dir / filename
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_results(self, results: List[Dict[str, Any]], filename: str = "results.json") -> None:
        results_path = self.run_dir / filename
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    def get_checkpoint_dir(self) -> Path:
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir

    def log(self, message: str, style: str = "") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"

        log_path = self.run_dir / "run.log"
        with open(log_path, "a") as f:
            f.write(log_line)

        if style:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            self.console.print(message)
