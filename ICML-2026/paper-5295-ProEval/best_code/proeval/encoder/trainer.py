# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder training for cross-benchmark BQ prior.

Provides :class:`EncoderTrainer` which orchestrates the full training
pipeline: load benchmarks → prepare target split → train encoder →
save checkpoint.

Supports three settings via the ``setting`` parameter:
    - ``"new_pair"``: Encoder trains on all benchmarks including the target
      benchmark (with the target model's scores removed).
    - ``"new_benchmark"``: Encoder trains on non-target benchmarks only.
      The target benchmark is completely excluded from training.
    - ``"new_model"``: Same as ``"new_pair"`` for encoder training — other
      models' scores on the target benchmark are used to learn embeddings.

Migrated from ``colabs/training_utils.py`` and ``colabs/train_encoder.py``.

Example::

    from proeval.encoder import EncoderTrainer

    trainer = EncoderTrainer(
        train_benchmarks=["gsm8k", "strategyqa"],
        target_benchmark="svamp",
        target_model="gemini25_flash",
        setting="new_benchmark",
        hidden_dim=16,
        kernel_type="matern",
        epochs=200,
    )
    encoder_path = trainer.train(data_dir="data", output_dir="results")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from proeval.encoder.data import (
    get_model_index_by_name,
    load_all_benchmarks,
    split_train_and_target,
)
from proeval.encoder.nn_utils import (
    QuestionEncoder,
    compute_gp_loss,
    save_encoder,
)


# Mini-batch utilities (from training_utils.py)


def split_benchmark_data(
    benchmark_data: Dict,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Split benchmark data into training and validation sets.

    Args:
        benchmark_data: Dict of benchmark data.
        val_ratio: Fraction for validation (default 10%).
        seed: Random seed.

    Returns:
        ``(train_data, val_data)``
    """
    rng = np.random.RandomState(seed)
    train_data = {}
    val_data = {}

    for name, data in benchmark_data.items():
        n = data["embeddings"].shape[0]
        n_val = max(1, int(n * val_ratio))
        indices = rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_data[name] = {
            "embeddings": data["embeddings"][train_idx],
            "labels": data["labels"][train_idx],
            "model_names": data["model_names"],
        }
        val_data[name] = {
            "embeddings": data["embeddings"][val_idx],
            "labels": data["labels"][val_idx],
            "model_names": data["model_names"],
        }

    return train_data, val_data


def create_minibatch_dataset(
    benchmark_data: Dict,
    batch_size: int = 100,
) -> List[Tuple[str, int, np.ndarray, np.ndarray]]:
    """Create list of ``(benchmark, model_idx, embeddings, labels)`` tuples.

    Each entry has a random sample of questions from one (benchmark, model)
    pair.
    """
    dataset = []
    for name, data in benchmark_data.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        n_questions = embeddings.shape[0]
        n_models = labels.shape[1]

        for model_idx in range(n_models):
            sample_size = min(batch_size, n_questions)
            idx = np.random.choice(n_questions, size=sample_size, replace=False)
            dataset.append((
                name, model_idx,
                embeddings[idx],
                labels[idx, model_idx:model_idx + 1],
            ))
    return dataset


def get_minibatch_iterator(
    benchmark_data: Dict,
    data_batch_size: int = 100,
    num_pairs_per_batch: int = 8,
    shuffle: bool = True,
):
    """Yield mini-batches of ``(benchmark, model)`` pairs.

    Each iteration creates NEW random samples from the data.
    """
    dataset = create_minibatch_dataset(benchmark_data, batch_size=data_batch_size)
    if shuffle:
        np.random.shuffle(dataset)

    for start in range(0, len(dataset), num_pairs_per_batch):
        yield dataset[start : start + num_pairs_per_batch]


def train_encoder_minibatch(
    encoder,
    optimizer,
    benchmark_data: Dict,
    data_batch_size: int = 100,
    num_pairs_per_batch: int = 8,
    num_epochs: int = 50,
    device: torch.device = None,
    verbose: bool = True,
    val_ratio: float = 0.1,
    checkpoint_interval: int = 0,
) -> Tuple[List[float], List[float]]:
    """Train encoder using mini-batches of random (benchmark, model) pairs.

    When *checkpoint_interval* > 0 the function saves in-memory checkpoints
    every *checkpoint_interval* epochs and automatically restores the
    checkpoint with the lowest validation loss at the end of training.

    Returns ``(train_loss_history, val_loss_history)``.
    """
    import copy

    if device is None:
        device = next(encoder.parameters()).device

    train_data, val_data = split_benchmark_data(benchmark_data, val_ratio=val_ratio)

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    # In-memory checkpoint tracking
    checkpoints: Dict[int, dict] = {}  # epoch -> {val_loss, state_dict}
    best_val_loss = float("inf")
    best_epoch = 0

    total_pairs = sum(d["labels"].shape[1] for d in train_data.values())
    if verbose:
        print(f"Training on {total_pairs} (benchmark, model) pairs")
        print(f"Data batch size: {data_batch_size} questions/pair")
        print(f"Pairs per batch: {num_pairs_per_batch}")
        print(f"Validation ratio: {val_ratio:.0%}")
        if checkpoint_interval > 0:
            print(f"Checkpoint interval: every {checkpoint_interval} epochs (auto-select best)")

    for epoch in range(num_epochs):
        encoder.train()
        epoch_losses = []

        batch_iter = get_minibatch_iterator(
            train_data,
            data_batch_size=data_batch_size,
            num_pairs_per_batch=num_pairs_per_batch,
            shuffle=True,
        )

        for batch in batch_iter:
            batch_loss = torch.tensor(0.0, device=device)
            valid_items = 0
            for _, _, emb_batch, label_batch in batch:
                emb_t = torch.from_numpy(emb_batch).float().to(device)
                lab_t = torch.from_numpy(label_batch).float().to(device)
                loss_i = compute_gp_loss(encoder, emb_t, lab_t)
                if torch.isfinite(loss_i):
                    batch_loss = batch_loss + loss_i
                    valid_items += 1

            if valid_items == 0:
                # Entire batch is NaN — skip without updating parameters
                continue

            batch_loss = batch_loss / valid_items

            optimizer.zero_grad()
            batch_loss.backward()
            # Gradient clipping to prevent parameter explosion
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(batch_loss.item())

        avg_train = float(np.mean(epoch_losses))
        train_loss_history.append(avg_train)

        # Validation
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            for _, data in val_data.items():
                emb = data["embeddings"]
                labs = data["labels"]
                for mi in range(labs.shape[1]):
                    emb_t = torch.from_numpy(emb).float().to(device)
                    lab_t = torch.from_numpy(labs[:, mi:mi + 1]).float().to(device)
                    vl = compute_gp_loss(encoder, emb_t, lab_t).item()
                    if np.isfinite(vl):
                        val_losses.append(vl)

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        val_loss_history.append(avg_val)

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch + 1

        # Save in-memory checkpoint at intervals
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoints[epoch + 1] = {
                "val_loss": avg_val,
                "train_loss": avg_train,
                "state_dict": copy.deepcopy(encoder.state_dict()),
            }
            marker = " *** BEST ***" if is_best else ""
            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train: {avg_train:.4f}, Val: {avg_val:.4f} "
                    f"[CHECKPOINT]{marker}"
                )
        elif verbose and (epoch + 1) % 10 == 0:
            marker = " *" if is_best else ""
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train: {avg_train:.4f}, Val: {avg_val:.4f}{marker}"
            )

    # Restore best checkpoint if available
    if checkpoints:
        best_ckpt_epoch = min(checkpoints, key=lambda e: checkpoints[e]["val_loss"])
        best_ckpt = checkpoints[best_ckpt_epoch]
        encoder.load_state_dict(best_ckpt["state_dict"])
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"BEST CHECKPOINT: Epoch {best_ckpt_epoch}")
            print(f"  Validation Loss: {best_ckpt['val_loss']:.4f}")
            print(f"  Training Loss:   {best_ckpt['train_loss']:.4f}")
            print(f"{'=' * 60}")

    return train_loss_history, val_loss_history


# EncoderTrainer (high-level API)

# Valid settings
VALID_SETTINGS = ("new_pair", "new_benchmark", "new_model")


class EncoderTrainer:
    """End-to-end encoder training for cross-benchmark transfer.

    Trains a neural encoder that maps prompt embeddings to a learned space
    for GP-based Bayesian Quadrature (BQ).

    Settings:
        - ``"new_pair"``: Both the target model and benchmark are known.
          Other models' data on the target benchmark IS included in training.
          Only the specific (target_model, target_benchmark) pair is held out.
        - ``"new_benchmark"``: The target benchmark is new — no models have
          been evaluated on it. The target benchmark is fully excluded from
          encoder training.
        - ``"new_model"``: The target model is new but the benchmark is known.
          Equivalent to ``"new_pair"`` for encoder training (other models'
          data on the target benchmark is available).

    Args:
        train_benchmarks: Benchmarks to train on (e.g., ``["gsm8k", "strategyqa"]``).
        target_benchmark: Benchmark to evaluate on (e.g., ``"svamp"``).
        target_model: Model to evaluate on the target benchmark.
        setting: One of ``"new_pair"``, ``"new_benchmark"``, ``"new_model"``.
        hidden_dim: Encoder hidden dimension (default 16).
        learning_rate: Adam learning rate (default 0.01).
        epochs: Number of training epochs (default 200).
        init_var: Initial noise variance (default 0.3).
        kernel_type: ``"linear"``, ``"matern"``, or ``"rbf"`` (default ``"matern"``).
        init_lengthscale: Initial lengthscale for matern/rbf kernels (default 1.0).
        matern_nu: Matérn smoothness parameter (default 2.5).
        data_batch_size: Questions per (benchmark, model) pair per batch.
        num_pairs_per_batch: Number of pairs per training batch.
        embedding_model: Embedding model name for file lookup.
        exclude_models: Models to exclude from training.
        checkpoint_interval: Save in-memory checkpoint every N epochs.

    Example::

        trainer = EncoderTrainer(
            train_benchmarks=["gsm8k", "strategyqa"],
            target_benchmark="svamp",
            target_model="gemini25_flash",
            setting="new_benchmark",
        )
        path = trainer.train(data_dir="data", output_dir="results")
    """

    def __init__(
        self,
        train_benchmarks: List[str],
        target_benchmark: str,
        target_model: str = "gemini25_flash",
        setting: str = "new_benchmark",
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 200,
        init_var: float = 0.3,
        steepness: float = 1.0,
        kernel_type: str = "matern",
        init_lengthscale: float = 1.0,
        matern_nu: float = 2.5,
        data_batch_size: int = 1000,
        num_pairs_per_batch: int = 1000,
        embedding_model: str = "text-embedding-3-large",
        exclude_models: Optional[List[str]] = None,
        include_models: Optional[List[str]] = None,
        checkpoint_interval: int = 50,
    ):

        if setting not in VALID_SETTINGS:
            raise ValueError(
                f"Invalid setting '{setting}'. Must be one of {VALID_SETTINGS}"
            )

        self.train_benchmarks = train_benchmarks
        self.target_benchmark = target_benchmark
        self.target_model = target_model
        self.setting = setting
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.init_var = init_var
        self.steepness = steepness
        self.kernel_type = kernel_type
        self.init_lengthscale = init_lengthscale
        self.matern_nu = matern_nu
        self.data_batch_size = data_batch_size
        self.num_pairs_per_batch = num_pairs_per_batch
        self.embedding_model = embedding_model
        self.exclude_models = exclude_models
        self.include_models = include_models
        self.checkpoint_interval = checkpoint_interval

    @property
    def _include_target_in_training(self) -> bool:
        """Whether to include other models' data on target benchmark."""
        return self.setting in ("new_pair", "new_model")

    def train(
        self,
        data_dir: str = "data",
        output_dir: str = "results",
        checkpoint_path: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Run the full training pipeline.

        Args:
            data_dir: Directory containing benchmark CSVs and embeddings.
            output_dir: Directory to save encoder checkpoint (ignored if
                *checkpoint_path* is set).
            checkpoint_path: Exact path to save the encoder ``.pth`` file.
                If ``None``, auto-generates path under *output_dir*.
            seed: Random seed.

        Returns:
            Path to the saved encoder ``.pth`` file.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        setting_label = self.setting.replace("_", " ").title()
        print("=" * 70)
        print(f"Encoder Training — {setting_label}")
        print("=" * 70)

        # Load benchmarks
        if self._include_target_in_training:
            # New Pair / New Model: include target benchmark for loading
            all_benchmarks = list(set(self.train_benchmarks + [self.target_benchmark]))
        else:
            # New Benchmark: only load training benchmarks
            all_benchmarks = list(set(self.train_benchmarks))
            # Ensure target benchmark is NOT in the list
            all_benchmarks = [b for b in all_benchmarks if b != self.target_benchmark]

        print(f"\nSetting: {setting_label}")
        print(f"Target: ({self.target_model}, {self.target_benchmark})")
        print(f"Loading benchmarks: {sorted(all_benchmarks)}")

        # Resolve include_models → exclude_models
        effective_exclude = self.exclude_models
        if self.include_models is not None:
            print(f"Include models (GMM-selected): {self.include_models}")
            # Load benchmark briefly to discover all available model names
            _temp_data = load_all_benchmarks(
                all_benchmarks[:1], data_dir=data_dir,
                embedding_model=self.embedding_model,
            )
            _first = next(iter(_temp_data.values()))
            all_model_names = _first["model_names"]
            # Exclude everything NOT in include_models (and not the target)
            models_to_exclude = [
                m for m in all_model_names
                if m not in self.include_models and m != self.target_model
            ]
            if effective_exclude:
                models_to_exclude = list(set(models_to_exclude) | set(effective_exclude))
            effective_exclude = models_to_exclude
            print(f"Effective exclude models: {effective_exclude}")

        benchmark_data = load_all_benchmarks(
            all_benchmarks, data_dir=data_dir,
            embedding_model=self.embedding_model,
            exclude_models=effective_exclude,
        )

        # Prepare training split
        train_data, target_info = split_train_and_target(
            benchmark_data, self.target_benchmark, self.target_model,
            setting=self.setting,
        )

        # Determine embedding dimension
        first = next(iter(train_data.values()))
        embedding_dim = first["embeddings"].shape[1]

        # Print config
        print(f"\nEncoder Config:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim:    {self.hidden_dim}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs:        {self.epochs}")
        print(f"  Kernel:        {self.kernel_type}")
        if self.kernel_type in ("matern", "rbf"):
            print(f"  Lengthscale:   {self.init_lengthscale}")
            if self.kernel_type == "matern":
                print(f"  Matérn ν:      {self.matern_nu}")

        # Initialize encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = QuestionEncoder(
            input_dim=embedding_dim,
            hidden_dim=self.hidden_dim,
            init_var=self.init_var,
            steepness=self.steepness,
            kernel_type=self.kernel_type,
            init_lengthscale=self.init_lengthscale,
            matern_nu=self.matern_nu,
        ).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)

        print(f"  Device:        {device}")
        print(f"  Parameters:    {sum(p.numel() for p in encoder.parameters()):,}")

        # Print training data summary
        total_q = sum(d["embeddings"].shape[0] for d in train_data.values())
        total_m = sum(d["labels"].shape[1] for d in train_data.values())
        print(f"\nTraining Data:")
        print(f"  Benchmarks: {list(train_data.keys())}")
        print(f"  Questions:  {total_q}")
        print(f"  Models:     {total_m}")

        # Train
        print(f"\nTraining encoder...")
        train_loss, val_loss = train_encoder_minibatch(
            encoder, optimizer, train_data,
            data_batch_size=self.data_batch_size,
            num_pairs_per_batch=self.num_pairs_per_batch,
            num_epochs=self.epochs,
            device=device,
            verbose=True,
            checkpoint_interval=self.checkpoint_interval,
        )

        # Determine save path
        if checkpoint_path:
            encoder_path = Path(checkpoint_path)
            results_dir = encoder_path.parent
        else:
            bench_str = "_".join(sorted(self.train_benchmarks))
            dirname = (
                f"{self.setting}_train_{bench_str}_target_{self.target_benchmark}"
                f"_hdim{self.hidden_dim}_lr{self.learning_rate}"
                f"_kernel_{self.kernel_type}_epoch{self.epochs}"
            )
            results_dir = Path(output_dir) / dirname
            encoder_filename = (
                f"encoder_{self.setting}_train_{bench_str}"
                f"_target_{self.target_benchmark}_epoch{self.epochs}.pth"
            )
            encoder_path = results_dir / encoder_filename
        results_dir.mkdir(parents=True, exist_ok=True)
        save_encoder(
            encoder, str(encoder_path),
            embedding_dim=embedding_dim,
            var=self.init_var,
            loss_history=train_loss,
            val_loss_history=val_loss,
            train_benchmarks=list(train_data.keys()),
            target_model=self.target_model,
            hidden_dim=self.hidden_dim,
            steepness=self.steepness,
            kernel_type=self.kernel_type,
        )
        print(f"\nEncoder saved to: {encoder_path}")

        # Save config
        import json
        config = {
            "setting": self.setting,
            "train_benchmarks": self.train_benchmarks,
            "target_benchmark": self.target_benchmark,
            "target_model": self.target_model,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "init_var": self.init_var,
            "steepness": self.steepness,
            "kernel_type": self.kernel_type,
            "init_lengthscale": self.init_lengthscale,
            "matern_nu": self.matern_nu,
            "embedding_dim": embedding_dim,
            "final_train_loss": train_loss[-1] if train_loss else None,
            "final_val_loss": val_loss[-1] if val_loss else None,
        }
        with open(results_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n{'=' * 70}")
        print("Training complete!")
        print(f"{'=' * 70}")

        return str(encoder_path)
