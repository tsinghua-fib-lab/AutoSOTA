from typing import List, Dict, Callable, Any, Iterator
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from tqdm import tqdm  # type: ignore
import os
import csv
from datetime import datetime

# Assuming these have their own correct type hints in your project
from algos.algo import Algorithm
from feedback.feedback import Feedback

import os
import numpy as np
import random

class Experiment:
    def __init__(self,
                 name: str,
                 num_samples: int,
                 true_ranking: List[int],
                 generators: List[Iterator[Feedback]],  # Now accepts a list directly
                 baselines_factory: Callable[[], List[Algorithm]],
                 loss: Callable[[List[int], List[int], int], float],
                 k: int,
                 seed: int,
                 num_recordings: int = 200,
                 initial_samples: int = 5) -> None:
        
        self.seed = seed

        self.name: str = name
        self.num_samples: int = num_samples
        self.true_ranking: List[int] = true_ranking
        
        # Store the list of generators directly
        self.generators: List[Iterator[Feedback]] = generators 
        self.baselines_factory: Callable[[], List[Algorithm]] = baselines_factory
        
        self.loss: Callable[[List[int], List[int], int], float] = loss
        self.k: int = k
        self.initial_samples: int = initial_samples
        
        # Infer num_runs directly from the generators list
        self.num_runs: int = len(self.generators)
        
        # LOG-SPACING LOGIC:
        geom_points: Any = np.geomspace(initial_samples, num_samples, num=num_recordings)
        rounded_points: List[int] = [int(p) for p in np.round(geom_points)]
        
        self.recording_points: List[int] = sorted(list(set(
            p for p in rounded_points if initial_samples <= p <= num_samples
        )))
        
        self.results: Dict[str, List[List[float]]] = {}
        self.sample_points: List[int] = []

    def run(self) -> None:
        """Executes the simulation across the provided list of generators."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        # Initialize results dictionary based on a test instantiation
        test_baselines: List[Algorithm] = self.baselines_factory()
        self.results = {algo.name: [] for algo in test_baselines}
        
        total_steps: int = self.num_runs * self.num_samples
        
        with tqdm(total=total_steps, desc=f"Running: {self.name}") as pbar:
            # Iterate directly through the provided generators
            for run_idx, generator in enumerate(self.generators):
                
                # We STILL need fresh algorithms for each generator so they don't carry over memory
                baselines: Dict[str, Algorithm] = {algo.name: algo for algo in self.baselines_factory()}
                
                run_losses: Dict[str, List[float]] = {name: [] for name in baselines.keys()}
                sample_points: List[int] = []
                recording_idx: int = 0
                num_points: int = len(self.recording_points)
                
                for i in range(1, self.num_samples + 1):
                    try:
                        # Pull from the current generator in the list
                        feedback: Feedback = next(generator)
                    except StopIteration:
                        pbar.update(self.num_samples - i + 1)
                        break
                    
                    for algo in baselines.values():
                        algo.feedback(feedback) 
                    
                    if recording_idx < num_points and i == self.recording_points[recording_idx]:
                        sample_points.append(i)
                        for name, algo in baselines.items():
                            pred_result: List[int] = algo.predict() 
                            loss_val: float = self.loss(self.true_ranking, pred_result, self.k)
                            run_losses[name].append(loss_val)
                        recording_idx += 1
                        
                    pbar.update(1)
                
                if run_idx == 0:
                    self.sample_points = sample_points
                    
                for name in baselines.keys():
                    self.results[name].append(run_losses[name])

    def plot(self, ax: Axes) -> None:
        for name, runs_data in self.results.items():
            if not runs_data: continue
            
            data_matrix = np.array(runs_data)
            mean_losses = np.mean(data_matrix, axis=0)
            std_losses = np.std(data_matrix, axis=0)
            
            # Calculate 95% CI
            # n = self.num_runs
            n = data_matrix.shape[0]
            confidence_interval = 1.96 * (std_losses / np.sqrt(n))
            
            line = ax.semilogx(self.sample_points, mean_losses, label=name, alpha=0.9, linewidth=2)[0]
            ax.fill_between(self.sample_points, 
                            mean_losses - confidence_interval, 
                            mean_losses + confidence_interval, 
                            color=line.get_color(), 
                            alpha=0.2)
        
        ax.set_xlabel("Number of Samples (Log Scale)")
        ax.set_ylabel(f"Top-{self.k} Kendall Tau Loss")
        ax.set_title(self.name)
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

class ExperimentBatch:
    def __init__(self, name: str, experiments: List[Experiment]) -> None:
        self.name: str = name
        self.experiments: List[Experiment] = experiments

    def run(self) -> None:
        """Executes all experiments in the batch sequence."""
        for exp in self.experiments:
            exp.run()
    
    def plot(self) -> None:
        """Saves cleaned-up plots and CSV data to the output directory."""
        output_dir: str = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir_name: str = f"{self.name.lower().replace(' ', '_')}_{timestamp}"
        batch_output_dir: str = os.path.join(output_dir, batch_dir_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        for exp in self.experiments:
            plot_name: str = exp.name.lower().replace(" ", "_")
            
            # Explicit typing for fig/ax to satisfy MyPy
            fig: Any
            ax: Any
            fig, ax = plt.subplots(figsize=(10, 6))
            
            exp.plot(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_output_dir, f"{plot_name}.png"), dpi=300)
            plt.close(fig)
            
            # CSV export logging mean and std for external analysis
            csv_path: str = os.path.join(batch_output_dir, f"{plot_name}.csv")
            with open(csv_path, mode='w', newline='') as f:
                writer: Any = csv.writer(f)
                baselines: List[str] = list(exp.results.keys())
                
                # 1. Create headers with _mean and _std suffixes
                header: List[str] = ['Number of Samples']
                for baseline in baselines:
                    header.extend([f"{baseline}_mean", f"{baseline}_std"])
                writer.writerow(header)
                
                # 2. Calculate and write the values row by row
                for i, sample_point in enumerate(exp.sample_points):
                    row: List[Any] = [sample_point]
                    for baseline in baselines:
                        # Convert the list of runs into a 2D numpy array
                        data_matrix: Any = np.array(exp.results[baseline])
                        
                        # Calculate mean and std across runs (axis=0) for the i-th sample point
                        mean_val: float = float(np.mean(data_matrix, axis=0)[i])
                        std_val: float = float(np.std(data_matrix, axis=0)[i])
                        row.extend([mean_val, std_val])
                        
                    writer.writerow(row)
