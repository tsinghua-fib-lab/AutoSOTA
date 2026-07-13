"""Data splitting utilities for experiment-based train/test splits.

The split is determined by the maximum calibration size used in training:
- Calibration indices: random subset of size max_cal (from shuffled indices)
- Test indices: remaining indices after calibration

This ensures no overlap between calibration and test data for file-based tasks
(light_tunnel, wind_tunnel) where observations come from fixed datasets.

The split uses a seeded random permutation for reproducibility.

Benchmark Protocol:
- n_pool: Calibration pool size (default: 2000)
- n_test: Test pool size (default: 5000, must be >= 5000 for reliable metrics)
- calibration_sizes: Subsample sizes from n_pool (e.g., [10, 50, 200, 1000])
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Default benchmark protocol values
DEFAULT_N_POOL = 2000
DEFAULT_N_TEST = 5000
DEFAULT_CALIBRATION_SIZES = [10, 50, 200, 1000]


@dataclass
class DataSplit:
    """Stores split configuration for an experiment.

    Uses a random permutation of indices to split data into calibration and test sets.
    The permutation is deterministic given the seed for reproducibility.
    """

    max_cal: int  # Maximum calibration size (determines split point)
    total_available: int  # Total available samples in dataset
    seed: int = 42  # Seed for random permutation
    _permutation: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate the random permutation on initialization."""
        if self._permutation is None:
            rng = np.random.default_rng(seed=self.seed)
            self._permutation = rng.permutation(self.total_available)

    @property
    def calibration_pool(self) -> np.ndarray:
        """Get the full pool of calibration indices (first max_cal from permutation)."""
        return self._permutation[:self.max_cal].copy()

    @property
    def test_pool(self) -> np.ndarray:
        """Get the full pool of test indices (remaining after max_cal)."""
        return self._permutation[self.max_cal:].copy()

    def get_calibration_indices(self, n: int) -> np.ndarray:
        """Get indices for calibration data.

        Args:
            n: Number of calibration samples needed (must be <= max_cal)

        Returns:
            Array of n random indices from the calibration pool
        """
        if n > self.max_cal:
            raise ValueError(
                f"Requested {n} calibration samples but max_cal is {self.max_cal}. "
                f"Increase data.calibration_sizes or reduce requested samples."
            )
        return self._permutation[:n].copy()

    def get_test_indices(self, n: int) -> np.ndarray:
        """Get indices for test data.

        Args:
            n: Number of test samples needed

        Returns:
            Array of n random indices from the test pool
        """
        available_test = self.total_available - self.max_cal
        if n > available_test:
            raise ValueError(
                f"Requested {n} test samples but only {available_test} available "
                f"(total={self.total_available}, max_cal={self.max_cal})."
            )
        return self._permutation[self.max_cal:self.max_cal + n].copy()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_cal": self.max_cal,
            "total_available": self.total_available,
            "seed": self.seed,
            "permutation": self._permutation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataSplit":
        """Create from dictionary."""
        split = cls(
            max_cal=d["max_cal"],
            total_available=d["total_available"],
            seed=d.get("seed", 42),
            _permutation=np.array(d["permutation"]) if "permutation" in d else None,
        )
        return split


@dataclass
class BenchmarkDataSplit:
    """Data split for benchmark experiments with explicit n_pool and n_test.

    This follows the benchmark protocol:
    - n_pool: Calibration pool size (samples available for calibration subsampling)
    - n_test: Test pool size (must be >= 5000 for reliable metrics)
    - calibration_sizes: Subsample sizes from n_pool (e.g., [10, 50, 200, 1000])

    For file-based tasks (wind_tunnel, light_tunnel, js):
        - n_tot_calib: Total available (theta, y) pairs from files
        - n_pool + n_test <= n_tot_calib

    For generative tasks (pendulum, gaussian, ou_process):
        - Data is generated on-the-fly
        - n_sim: Additional simulation data for NPE/FMPE training
    """

    n_pool: int  # Calibration pool size
    n_test: int  # Test pool size
    seed: int = 42  # Seed for random permutation
    calibration_sizes: List[int] = field(default_factory=lambda: DEFAULT_CALIBRATION_SIZES.copy())
    _permutation: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate and generate the random permutation."""
        if self.n_test < 5000:
            import warnings
            warnings.warn(
                f"n_test={self.n_test} is less than recommended minimum of 5000. "
                "Metrics may be unreliable."
            )

        if max(self.calibration_sizes) > self.n_pool:
            raise ValueError(
                f"Max calibration size ({max(self.calibration_sizes)}) exceeds "
                f"n_pool ({self.n_pool}). Reduce calibration_sizes or increase n_pool."
            )

        total = self.n_pool + self.n_test
        if self._permutation is None:
            rng = np.random.default_rng(seed=self.seed)
            self._permutation = rng.permutation(total)

    @property
    def total_available(self) -> int:
        """Total samples needed (n_pool + n_test)."""
        return self.n_pool + self.n_test

    @property
    def calibration_pool_indices(self) -> np.ndarray:
        """Get indices for the calibration pool (first n_pool from permutation)."""
        return self._permutation[:self.n_pool].copy()

    @property
    def test_pool_indices(self) -> np.ndarray:
        """Get indices for the test pool (remaining n_test from permutation)."""
        return self._permutation[self.n_pool:].copy()

    def get_calibration_indices(self, ncal: int, subsample_seed: Optional[int] = None) -> np.ndarray:
        """Get indices for calibration subsampling.

        IMPORTANT: Returns indices into the calibration pool (0 to n_pool-1),
        NOT into the original dataset. Use these to index calibration_pool.pt.

        Args:
            ncal: Number of calibration samples (must be in calibration_sizes)
            subsample_seed: Optional seed for subsampling. If provided, creates
                           a different subsample of size ncal from the pool.

        Returns:
            Array of ncal indices into the calibration pool (values 0 to n_pool-1)
        """
        if ncal not in self.calibration_sizes:
            raise ValueError(
                f"ncal={ncal} not in calibration_sizes={self.calibration_sizes}"
            )

        # Generate indices into the calibration pool (0 to n_pool-1)
        pool_indices = np.arange(self.n_pool)

        if subsample_seed is None:
            # Default: use first ncal indices
            return pool_indices[:ncal].copy()
        else:
            # Use seed for reproducible subsampling
            rng = np.random.default_rng(seed=subsample_seed)
            return rng.choice(pool_indices, size=ncal, replace=False)

    def get_test_indices(self, n: Optional[int] = None) -> np.ndarray:
        """Get indices for test data.

        Args:
            n: Number of test samples. If None, returns all n_test indices.

        Returns:
            Array of test indices
        """
        if n is None:
            return self.test_pool_indices
        if n > self.n_test:
            raise ValueError(
                f"Requested {n} test samples but n_test is {self.n_test}"
            )
        return self._permutation[self.n_pool:self.n_pool + n].copy()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_pool": self.n_pool,
            "n_test": self.n_test,
            "seed": self.seed,
            "calibration_sizes": self.calibration_sizes,
            "permutation": self._permutation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkDataSplit":
        """Create from dictionary."""
        return cls(
            n_pool=d["n_pool"],
            n_test=d["n_test"],
            seed=d.get("seed", 42),
            calibration_sizes=d.get("calibration_sizes", DEFAULT_CALIBRATION_SIZES.copy()),
            _permutation=np.array(d["permutation"]) if "permutation" in d else None,
        )

    @classmethod
    def from_config(cls, cfg) -> "BenchmarkDataSplit":
        """Create from Hydra config.

        Args:
            cfg: Config with data.n_pool, data.n_test, data.calibration_sizes

        Returns:
            BenchmarkDataSplit instance
        """
        data_cfg = cfg.get("data", {})
        return cls(
            n_pool=data_cfg.get("n_pool", DEFAULT_N_POOL),
            n_test=data_cfg.get("n_test", DEFAULT_N_TEST),
            seed=data_cfg.get("seed", 42),
            calibration_sizes=list(data_cfg.get("calibration_sizes", DEFAULT_CALIBRATION_SIZES)),
        )


class DataSplitter:
    """Manages data splits for an experiment.

    The split point is determined by max(data.calibration_sizes) from the config.
    This ensures that:
    1. All calibration training uses random indices from calibration pool
    2. All evaluation uses random indices from test pool
    3. No overlap between calibration and test data

    The split uses a seeded random permutation stored per-experiment for
    reproducibility across runs.

    Usage:
        # In training script
        splitter = DataSplitter.from_config(cfg)
        split = splitter.get_or_create_split(experiment_path, total_available=5000)
        cal_indices = split.get_calibration_indices(ncal)

        # In evaluation script
        splitter = DataSplitter.from_config(cfg)
        split = splitter.load_split(experiment_path)
        test_indices = split.get_test_indices(num_test)
    """

    def __init__(self, num_cal_values: List[int], seed: int = 42):
        """Initialize splitter.

        Args:
            num_cal_values: List of calibration sizes from config (e.g., [10, 50, 200, 1000])
            seed: Random seed for reproducibility
        """
        self.num_cal_values = sorted(num_cal_values)
        self.max_cal = max(num_cal_values)
        self.seed = seed

    @classmethod
    def from_config(cls, cfg) -> "DataSplitter":
        """Create splitter from Hydra config.

        Args:
            cfg: Hydra DictConfig with data.calibration_sizes

        Returns:
            DataSplitter instance
        """
        # Support both new (data.calibration_sizes) and legacy (data.calibration_sizes) paths
        if "data" in cfg and "calibration_sizes" in cfg.data:
            num_cal = list(cfg.data.calibration_sizes)
        elif "training" in cfg and "num_cal" in cfg.training:
            num_cal = list(cfg.data.calibration_sizes)
        else:
            num_cal = DEFAULT_CALIBRATION_SIZES
        seed = cfg.get("seed", 42)
        return cls(num_cal, seed)

    def create_split(self, total_available: int) -> DataSplit:
        """Create a new data split.

        Args:
            total_available: Total number of samples available in dataset

        Returns:
            DataSplit instance

        Raises:
            ValueError: If total_available is less than max_cal
        """
        if total_available < self.max_cal:
            raise ValueError(
                f"Dataset has {total_available} samples but max_cal requires "
                f"{self.max_cal}. Reduce data.calibration_sizes or use a larger dataset."
            )
        return DataSplit(
            max_cal=self.max_cal,
            total_available=total_available,
            seed=self.seed,
        )

    def save_split(self, split: DataSplit, experiment_path: Path) -> None:
        """Save split configuration to experiment directory.

        Args:
            split: DataSplit to save
            experiment_path: Path to experiment directory
        """
        experiment_path.mkdir(parents=True, exist_ok=True)
        split_path = experiment_path / "data_split.json"
        with open(split_path, "w") as f:
            json.dump(split.to_dict(), f, indent=2)
        print(f"Saved data split to {split_path}")
        print(f"  max_cal: {split.max_cal}, total: {split.total_available}, seed: {split.seed}")

    def load_split(self, experiment_path: Path) -> Optional[DataSplit]:
        """Load split configuration from experiment directory.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            DataSplit if exists, None otherwise
        """
        split_path = experiment_path / "data_split.json"
        if not split_path.exists():
            return None
        with open(split_path, "r") as f:
            split = DataSplit.from_dict(json.load(f))
        print(f"Loaded data split from {split_path}")
        print(f"  max_cal: {split.max_cal}, total: {split.total_available}, seed: {split.seed}")
        return split

    def get_or_create_split(
        self,
        experiment_path: Path,
        total_available: int,
    ) -> DataSplit:
        """Load existing split or create new one.

        Args:
            experiment_path: Path to experiment directory
            total_available: Total samples available (required for creating new split,
                           used for validation if loading existing split)

        Returns:
            DataSplit instance

        Raises:
            ValueError: If existing split is incompatible with current config
        """
        existing = self.load_split(experiment_path)
        if existing is not None:
            # Validate that existing split is compatible
            if existing.max_cal < self.max_cal:
                raise ValueError(
                    f"Existing split has max_cal={existing.max_cal} but config "
                    f"requires max_cal={self.max_cal}. Delete {experiment_path}/data_split.json "
                    f"or reduce data.calibration_sizes."
                )
            if existing.total_available != total_available:
                print(
                    f"  WARNING: Split was created with {existing.total_available} samples "
                    f"but current dataset has {total_available}. Using existing split."
                )
            return existing

        # Create and save new split
        split = self.create_split(total_available)
        self.save_split(split, experiment_path)
        return split


class BenchmarkSplitter:
    """Manages data splits for benchmark experiments.

    Uses the benchmark protocol with explicit n_pool and n_test sizes.
    Saves splits to shared/data/{task}/ directory.

    Usage:
        splitter = BenchmarkSplitter.from_config(cfg)
        split = splitter.get_or_create_split(data_path, task="pendulum")
        cal_indices = split.get_calibration_indices(ncal=200, subsample_seed=seed)
        test_indices = split.get_test_indices()
    """

    def __init__(
        self,
        n_pool: int = DEFAULT_N_POOL,
        n_test: int = DEFAULT_N_TEST,
        calibration_sizes: Optional[List[int]] = None,
        seed: int = 42,
    ):
        """Initialize splitter.

        Args:
            n_pool: Calibration pool size
            n_test: Test pool size (must be >= 5000 for reliable metrics)
            calibration_sizes: Subsample sizes from n_pool
            seed: Random seed for reproducibility
        """
        self.n_pool = n_pool
        self.n_test = n_test
        self.calibration_sizes = calibration_sizes or DEFAULT_CALIBRATION_SIZES.copy()
        self.seed = seed

    @classmethod
    def from_config(cls, cfg) -> "BenchmarkSplitter":
        """Create splitter from Hydra config.

        Args:
            cfg: Config with data.n_pool, data.n_test, data.calibration_sizes

        Returns:
            BenchmarkSplitter instance
        """
        data_cfg = cfg.get("data", {})
        return cls(
            n_pool=data_cfg.get("n_pool", DEFAULT_N_POOL),
            n_test=data_cfg.get("n_test", DEFAULT_N_TEST),
            calibration_sizes=list(data_cfg.get("calibration_sizes", DEFAULT_CALIBRATION_SIZES)),
            seed=data_cfg.get("seed", 42),
        )

    def create_split(self) -> BenchmarkDataSplit:
        """Create a new benchmark data split."""
        return BenchmarkDataSplit(
            n_pool=self.n_pool,
            n_test=self.n_test,
            seed=self.seed,
            calibration_sizes=self.calibration_sizes,
        )

    def save_split(self, split: BenchmarkDataSplit, data_path: Path) -> Path:
        """Save split configuration to data directory.

        Args:
            split: BenchmarkDataSplit to save
            data_path: Path to shared/data/{task}/ directory

        Returns:
            Path to saved split file
        """
        data_path.mkdir(parents=True, exist_ok=True)
        split_path = data_path / "data_split.json"
        with open(split_path, "w") as f:
            json.dump(split.to_dict(), f, indent=2)
        print(f"Saved benchmark data split to {split_path}")
        print(f"  n_pool: {split.n_pool}, n_test: {split.n_test}, seed: {split.seed}")
        print(f"  calibration_sizes: {split.calibration_sizes}")
        return split_path

    def load_split(self, data_path: Path) -> Optional[BenchmarkDataSplit]:
        """Load split configuration from data directory.

        Args:
            data_path: Path to shared/data/{task}/ directory

        Returns:
            BenchmarkDataSplit if exists, None otherwise
        """
        split_path = data_path / "data_split.json"
        if not split_path.exists():
            return None
        with open(split_path, "r") as f:
            split = BenchmarkDataSplit.from_dict(json.load(f))
        return split

    def get_or_create_split(self, data_path: Path) -> BenchmarkDataSplit:
        """Load existing split or create new one.

        Args:
            data_path: Path to shared/data/{task}/ directory

        Returns:
            BenchmarkDataSplit instance

        Raises:
            ValueError: If existing split is incompatible with current config
        """
        existing = self.load_split(data_path)
        if existing is not None:
            # Validate compatibility
            if existing.n_pool != self.n_pool:
                raise ValueError(
                    f"Existing split has n_pool={existing.n_pool} but config "
                    f"requires n_pool={self.n_pool}. Delete {data_path}/data_split.json "
                    f"to recreate."
                )
            if existing.n_test != self.n_test:
                raise ValueError(
                    f"Existing split has n_test={existing.n_test} but config "
                    f"requires n_test={self.n_test}. Delete {data_path}/data_split.json "
                    f"to recreate."
                )
            return existing

        # Create and save new split
        split = self.create_split()
        self.save_split(split, data_path)
        return split


def get_split_indices_for_mode(
    split: DataSplit,
    n: int,
    mode: str,
) -> np.ndarray:
    """Get indices for a specific mode.

    Args:
        split: DataSplit configuration
        n: Number of samples needed
        mode: Either "training" (calibration) or "testing"

    Returns:
        Array of randomly selected indices (deterministic based on split seed)
    """
    if mode == "training":
        return split.get_calibration_indices(n)
    elif mode == "testing":
        return split.get_test_indices(n)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'training' or 'testing'.")
