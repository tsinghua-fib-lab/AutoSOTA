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

"""Unified CSV manager for multi-model predictions with resume and fix-error.

Provides:

- :class:`UnifiedCSVManager` — Multi-model CSV storage with checkpoint resume
  and error-fixing capabilities.
- :func:`load_dataset_data` — HuggingFace/local dataset loading for all
  supported datasets.
- :func:`save_predictions_to_csv` / :func:`load_predictions_from_csv` — Simple
  single-model CSV I/O.

Example — full evaluation with resume::

    from proeval.evaluator import LLMPredictor, DATASET_CONFIGS
    from proeval.evaluator.csv_manager import UnifiedCSVManager

    csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
    csv_mgr.load_or_create(questions, ground_truths)

    predictor = LLMPredictor(model="google/gemma-3-27b-it")
    csv_mgr.run_evaluation(
        predictor, "gemma3_27b", DATASET_CONFIGS["gsm8k"],
        questions, ground_truths, parallel=True, workers=10,
    )

Example — fix errors in existing CSV::

    csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
    csv_mgr.load_or_create(questions, ground_truths)
    csv_mgr.fix_errors(predictor, "gemma3_27b", DATASET_CONFIGS["gsm8k"],
                       questions, ground_truths)
"""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Numpy JSON serialisation helper


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(e) for e in obj]
        return type(obj)(converted)
    return obj


# UnifiedCSVManager


class UnifiedCSVManager:
    """Multi-model CSV storage with checkpoint resume and fix-error support.

    Each dataset has **one** CSV file with columns::

        index, question, ground_truth,
        prediction_<model>, label_<model>, raw_response_<model>, ...

    Args:
        dataset_name: e.g. ``"gsm8k"``
        output_dir: directory for CSV and checkpoint files
    """

    def __init__(self, dataset_name: str, output_dir: str = "."):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
        self.df: Optional[pd.DataFrame] = None

    # ── load / create ─────────────────────────────────────────────────

    def load_or_create(
        self, questions: List[str], ground_truths: List[Any]
    ) -> pd.DataFrame:
        """Load existing CSV or create a new DataFrame.

        Returns the DataFrame (also stored as ``self.df``).
        """
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded existing CSV: {self.csv_path} ({len(self.df)} rows)")
            if len(self.df) != len(questions):
                raise ValueError(
                    f"Row count mismatch: CSV has {len(self.df)}, "
                    f"but {len(questions)} questions provided"
                )
        else:
            self.df = pd.DataFrame(
                {"index": range(len(questions)), "question": questions, "ground_truth": ground_truths}
            )
            print(f"Created new CSV structure: {self.csv_path}")
        return self.df

    # ── model column helpers ──────────────────────────────────────────

    def has_model(self, model_name: str) -> bool:
        """Check if prediction + label columns exist for *model_name*."""
        if self.df is None:
            return False
        return (
            f"prediction_{model_name}" in self.df.columns
            and f"label_{model_name}" in self.df.columns
        )

    def add_model_predictions(
        self,
        model_name: str,
        predictions: List[Any],
        labels: List[float],
        raw_responses: Optional[List[str]] = None,
        rerun: bool = False,
    ) -> None:
        """Add or overwrite model prediction columns."""
        self._check_init()
        if self.has_model(model_name) and not rerun:
            print(f"Model '{model_name}' already evaluated. Use rerun=True to re-evaluate.")
            return
        self.df[f"prediction_{model_name}"] = predictions
        self.df[f"label_{model_name}"] = labels
        if raw_responses is not None:
            self.df[f"raw_response_{model_name}"] = raw_responses

    def get_model_accuracy(self, model_name: str) -> Optional[float]:
        """Return accuracy for *model_name* (``1 − mean(labels)``), or ``None``."""
        if not self.has_model(model_name):
            return None
        labels = self.df[f"label_{model_name}"]
        valid = labels.dropna()
        return float(1.0 - valid.mean()) if len(valid) > 0 else None

    # ── error detection & fixing ──────────────────────────────────────

    ERROR_SENTINELS = {"SKIPPED", "ERROR", "RATE_LIMITED", "PARSE_ERROR"}

    def get_error_indices(self, model_name: str) -> List[int]:
        """Return row indices with error predictions or NaN labels."""
        if not self.has_model(model_name):
            return []
        pred_col = f"prediction_{model_name}"
        label_col = f"label_{model_name}"
        mask = self.df[pred_col].isin(self.ERROR_SENTINELS) | self.df[label_col].isna()
        return self.df.index[mask].tolist()

    def update_predictions_at_indices(
        self,
        model_name: str,
        indices: List[int],
        predictions: List[Any],
        labels: List[float],
        raw_responses: Optional[List[str]] = None,
    ) -> None:
        """Update specific rows with new results."""
        self._check_init()
        for i, idx in enumerate(indices):
            self.df.at[idx, f"prediction_{model_name}"] = predictions[i]
            self.df.at[idx, f"label_{model_name}"] = labels[i]
            if raw_responses is not None:
                self.df.at[idx, f"raw_response_{model_name}"] = raw_responses[i]
        print(f"Updated {len(indices)} predictions for model: {model_name}")

    # ── save ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save DataFrame to CSV."""
        self._check_init()
        os.makedirs(self.output_dir, exist_ok=True)
        self.df.to_csv(self.csv_path, index=False)
        print(f"Saved predictions to: {self.csv_path}")

    # ── high-level run & fix ──────────────────────────────────────────

    def run_evaluation(
        self,
        predictor,
        model_name: str,
        dataset_config,
        questions: List[str],
        ground_truths: List[Any],
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 5,
        skip_error: bool = False,
        rerun: bool = False,
        checkpoint_interval: int = 50,
    ) -> None:
        """Run full evaluation with checkpoint resume support.

        Saves a ``.checkpoint_*.json`` file every *checkpoint_interval* items.
        If a checkpoint exists from a previous interrupted run, evaluation
        resumes from where it left off.

        Args:
            predictor: :class:`LLMPredictor` instance.
            model_name: Friendly name (used as column suffix).
            dataset_config: :class:`DatasetConfig`.
            questions: Full question list.
            ground_truths: Full ground-truth list.
            parallel: Use ``predict_batch_parallel`` (default True).
            workers: Thread count for parallel mode.
            max_parse_retries: Retries per item.
            skip_error: Mark parse errors as NaN instead of 1.0.
            rerun: Force re-evaluation even if columns exist.
            checkpoint_interval: Save checkpoint every N items (sequential).
        """
        self._check_init()

        # Skip if already done
        if self.has_model(model_name) and not rerun:
            acc = self.get_model_accuracy(model_name)
            errs = len(self.get_error_indices(model_name))
            print(f"Model '{model_name}' already evaluated (acc={acc:.2%}, {errs} errors).")
            print("Use rerun=True to force, or fix_errors() to fix failures.")
            return

        ckpt_path = os.path.join(
            self.output_dir, f".checkpoint_{self.dataset_name}_{model_name}.json"
        )
        start_idx = 0
        completed: List[Tuple] = []

        # Resume from checkpoint
        if os.path.exists(ckpt_path) and not rerun:
            try:
                with open(ckpt_path) as f:
                    ckpt = json.load(f)
                start_idx = ckpt.get("last_completed_idx", 0) + 1
                completed = [tuple(r) for r in ckpt.get("results", [])]
                print(f"Resuming from checkpoint: {start_idx}/{len(questions)}")
            except Exception as e:
                print(f"Warning: could not load checkpoint ({e}), starting fresh")
                start_idx, completed = 0, []

        all_results = list(completed)
        skipped = 0

        try:
            if parallel:
                remaining_q = questions[start_idx:]
                remaining_gt = ground_truths[start_idx:]
                if remaining_q:
                    batch_results = predictor.predict_batch_parallel(
                        remaining_q, remaining_gt, dataset_config,
                        max_workers=workers,
                        max_parse_retries=max_parse_retries,
                        skip_error=skip_error,
                    )
                    all_results.extend(batch_results)
                    skipped += sum(1 for r in batch_results if r[3] in self.ERROR_SENTINELS)
            else:
                for idx in range(start_idx, len(questions)):
                    raw, pred, score = predictor.evaluate(
                        questions[idx], ground_truths[idx], dataset_config,
                        max_parse_retries=max_parse_retries,
                    )
                    if pred is None:
                        skipped += 1
                        pred = "SKIPPED"
                        score = float("nan") if skip_error else 1.0
                    all_results.append((questions[idx], ground_truths[idx], raw, pred, score))

                    # Save checkpoint periodically
                    if (idx + 1) % checkpoint_interval == 0:
                        self._save_checkpoint(ckpt_path, idx, all_results, model_name)

                # Final checkpoint
                self._save_checkpoint(ckpt_path, len(questions) - 1, all_results, model_name)

        except Exception:
            if all_results:
                last = start_idx + len(all_results) - 1 - len(completed)
                self._save_checkpoint(ckpt_path, last, all_results, model_name)
                print(f"Error! Progress saved ({len(all_results)} items). Re-run to resume.")
            raise

        # Store into CSV
        predictions = [r[3] for r in all_results]
        labels = [r[4] for r in all_results]
        raw_responses = [r[2] for r in all_results]
        self.add_model_predictions(model_name, predictions, labels, raw_responses, rerun=rerun)
        self.save()

        # Clean up checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # Report
        valid = [l for l in labels if not (isinstance(l, float) and np.isnan(l))]
        acc = 1 - (sum(valid) / len(valid)) if valid else 0
        print(f"\nModel: {model_name} | Evaluated: {len(valid)} | Skipped: {skipped} | Accuracy: {acc:.2%}")

    def fix_errors(
        self,
        predictor,
        model_name: str,
        dataset_config,
        questions: List[str],
        ground_truths: List[Any],
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 5,
        skip_error: bool = False,
    ) -> None:
        """Re-run only failed predictions (SKIPPED / ERROR / RATE_LIMITED).

        Args:
            predictor: :class:`LLMPredictor` instance.
            model_name: Friendly model name.
            dataset_config: :class:`DatasetConfig`.
            questions: Full question list (same as original run).
            ground_truths: Full ground-truth list.
        """
        self._check_init()
        if not self.has_model(model_name):
            print(f"Model '{model_name}' not found in CSV. Run evaluation first.")
            return

        error_idx = self.get_error_indices(model_name)
        if not error_idx:
            acc = self.get_model_accuracy(model_name)
            print(f"No errors for '{model_name}'. Accuracy: {acc:.2%}")
            return

        print(f"Fixing {len(error_idx)} errors for '{model_name}'...")
        err_q = [questions[i] for i in error_idx]
        err_gt = [ground_truths[i] for i in error_idx]

        if parallel:
            fix_results = predictor.predict_batch_parallel(
                err_q, err_gt, dataset_config,
                max_workers=workers,
                max_parse_retries=max_parse_retries,
                skip_error=skip_error,
            )
        else:
            fix_results = []
            for q, gt in tqdm(zip(err_q, err_gt), total=len(err_q), desc="Fixing errors"):
                raw, pred, score = predictor.evaluate(q, gt, dataset_config, max_parse_retries)
                if pred is None:
                    pred = "SKIPPED"
                    score = float("nan") if skip_error else 1.0
                fix_results.append((q, gt, raw, pred, score))

        preds = [r[3] for r in fix_results]
        labels = [r[4] for r in fix_results]
        raws = [r[2] for r in fix_results]
        self.update_predictions_at_indices(model_name, error_idx, preds, labels, raws)
        self.save()

        still_bad = sum(1 for p in preds if p in self.ERROR_SENTINELS)
        print(f"Fixed: {len(error_idx) - still_bad} | Still failing: {still_bad}")
        acc = self.get_model_accuracy(model_name)
        if acc is not None:
            print(f"New accuracy: {acc:.2%}")

    # ── internals ─────────────────────────────────────────────────────

    def _check_init(self):
        if self.df is None:
            raise ValueError("DataFrame not initialised. Call load_or_create() first.")

    def _save_checkpoint(self, path, last_idx, results, model_name):
        data = convert_numpy_types({
            "last_completed_idx": last_idx,
            "results": results,
            "model_name": model_name,
            "task": self.dataset_name,
        })
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"💾 Checkpoint saved at {last_idx + 1}")


# Simple single-model CSV I/O (legacy compat)


def save_predictions_to_csv(
    results: List[Tuple], output_path: str, task: str = "generic"
) -> None:
    """Save prediction results to a simple CSV.

    Each row: ``index, question, ground_truth, raw_response, prediction, correct``.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "question", "ground_truth", "raw_response", "prediction", "correct"])
        for i, row in enumerate(results):
            w.writerow([i, *row])
    print(f"Saved {len(results)} predictions to {output_path}")


def load_predictions_from_csv(csv_path: str) -> Dict[str, List]:
    """Load predictions from a simple CSV.

    Returns dict with keys: ``questions``, ``ground_truths``, ``predictions``,
    ``correct_labels``.
    """
    questions, gts, preds, labels = [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(row["question"])
            gt = row["ground_truth"]
            gts.append(gt.lower() == "true" if gt.lower() in ("true", "false") else gt)
            p = row["prediction"]
            preds.append(p.lower() == "true" if p.lower() in ("true", "false") else p)
            labels.append(float(row["correct"]))
    return {"questions": questions, "ground_truths": gts, "predictions": preds, "correct_labels": labels}


# HuggingFace / local dataset loading


def load_dataset_data(task: str) -> Tuple[List[str], List[Any]]:
    """Load a full dataset from HuggingFace (or local files for GQA / DICES-T2I).

    Supported tasks: ``strategyqa``, ``gsm8k``, ``svamp``, ``mmlu``,
    ``mmlu_professionallaw``, ``jigsaw``, ``toxicchat``, ``gqa``, ``dices``,
    ``dices_t2i``.

    Returns ``(questions, ground_truths)``.
    """
    from datasets import load_dataset  # lazy import

    questions: List = []
    ground_truths: List = []

    if task == "strategyqa":
        ds = load_dataset("ChilleD/StrategyQA", split="train")
        for ex in ds:
            questions.append(ex["question"])
            ground_truths.append(ex["answer"])

    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            questions.append(ex["question"])
            ground_truths.append(ex["answer"])

    elif task == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="train")
        for ex in ds:
            questions.append(ex["Body"] + " " + ex["Question"])
            ground_truths.append(ex["Answer"])

    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
        df = ds.to_pandas()
        questions = df.to_dict(orient="records")
        ground_truths = df["answer"].tolist()

    elif task == "mmlu_professionallaw":
        ds = load_dataset("cais/mmlu", "professional_law", split="test")
        df = ds.to_pandas()
        questions = df.to_dict(orient="records")
        ground_truths = df["answer"].tolist()

    elif task == "jigsaw":
        ds = load_dataset("google/civil_comments", split="train")
        df = ds.to_pandas()
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)
        questions = df["text"].tolist()
        ground_truths = df["toxicity"].tolist()

    elif task == "toxicchat":
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        df = ds.to_pandas()
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)
        questions = df["user_input"].tolist()
        ground_truths = df["toxicity"].tolist()

    elif task == "gqa":
        gqa_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gqa_raw.csv")
        df = pd.read_csv(gqa_path)
        questions = df[["question", "image_id"]].to_dict(orient="records")
        ground_truths = df["ground_truth"].tolist()

    elif task == "dices":
        import tensorflow_datasets as tfds
        ds = tfds.load("dices/350", split="train", shuffle_files=True)
        for ex in tfds.as_numpy(ds.take(1500)):
            questions.append({"context": ex["context"].decode(), "response": ex["response"].decode()})
            ground_truths.append((float(ex["Q_overall"]) - 1.0) / 4.0)

    elif task == "dices_t2i":
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "dices_t2i_data",
            "latest_dv-t2i-full-filtered.jsonl",
        )
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompt = d.get("text_prompt", d.get("prompt", d.get("text", "")))
                rating = d.get("how_harmful", d.get("human_rating"))
                if not prompt or rating is None or rating == "unsure":
                    continue
                questions.append(prompt)
                ground_truths.append(float(rating) / 4.0)
        if len(questions) > 1500:
            idx = np.random.RandomState(42).choice(len(questions), 1500, replace=False)
            questions = [questions[i] for i in idx]
            ground_truths = [ground_truths[i] for i in idx]

    else:
        raise ValueError(f"Unknown task: {task}")

    return questions, ground_truths
