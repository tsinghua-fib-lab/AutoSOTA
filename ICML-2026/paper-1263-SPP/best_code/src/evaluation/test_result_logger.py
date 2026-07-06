"""
Test result logging module
Stores detailed test results, including pruning-related information
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


class TestResultLogger:
    """
    Test result logger
    Stores detailed test results to facilitate later analysis and debugging
    """

    def __init__(
        self,
        output_dir: str = "outputs/test_results",
        save_format: str = "json"  # "json" or "csv"
    ):
        """
        Initialize the test result logger

        Args:
            output_dir: output directory
            save_format: save format (json or csv)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format
        self.incremental_save = True  # Enable incremental saving by default
        self.incremental_save_interval = 10  # Save once every 10 results by default
        self.incremental_filename = "test_results_incremental"

        # When starting a new process, delete the old incremental save files to avoid leftover incorrect results
        if self.incremental_save:
            incremental_file = self.output_dir / f"{self.incremental_filename}.json"
            if incremental_file.exists():
                incremental_file.unlink()
                logger.info(f"Deleted old incremental save file: {incremental_file}")
            incremental_file_csv = self.output_dir / f"{self.incremental_filename}.csv"
            if incremental_file_csv.exists():
                incremental_file_csv.unlink()
                logger.info(f"Deleted old incremental save file: {incremental_file_csv}")

        # Store all test results
        self.results: List[Dict[str, Any]] = []

        # Statistics
        self.stats: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Test result logger initialized: {output_dir}, format={save_format}, incremental save=enabled (interval={self.incremental_save_interval})")
    
    def log_result(
        self,
        scenario_id: str,
        turn: int,
        task_type: str,
        prompt: str,
        expected_answer: str,
        test_answer: str,
        acc: float,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        domain_similarity_probs: Optional[np.ndarray] = None,
        selected_heads_count: Optional[int] = None,
        pruning_ratio: Optional[float] = None,
        session_breadth: Optional[float] = None,
        inferred_domains: Optional[List[int]] = None,
        dataset_type: Optional[str] = None,  # single-domain, out-of-domain, cross-domain, dataset1, dataset2
        layer_head_counts: Optional[Dict[int, int]] = None,
        pruning_mask: Optional[Dict[int, List[int]]] = None,
        head_scores: Optional[Dict[int, np.ndarray]] = None,
        inference_time: Optional[float] = None,  # Inference time (seconds) - counts only pure inference time (the time of model.generate())
        pruning_time: Optional[float] = None,  # Pruning time (seconds) - recorded only on the first turn of each scenario
        **kwargs  # Other custom fields
    ):
        """
        Log a test result

        Args:
            scenario_id: scenario ID
            turn: turn number
            task_type: question type (multiple_choice, factual, code, reasoning)
            prompt: input prompt
            expected_answer: expected answer
            test_answer: test answer
            acc: accuracy
            evaluation_metrics: other evaluation metrics
            domain_similarity_probs: domain similarity probabilities
            selected_heads_count: number of selected heads
            pruning_ratio: pruning ratio
            session_breadth: session breadth
            inferred_domains: list of inferred domains
            dataset_type: dataset type
            layer_head_counts: number of heads per layer
            pruning_mask: pruning mask (optional, may be too large)
            head_scores: head scores (optional)
            inference_time: inference time (seconds) - counts only pure inference time (the time of model.generate())
            pruning_time: pruning time (seconds) - recorded only on the first turn of each scenario
            **kwargs: other custom fields
        """
        result = {
            "scenario_id": scenario_id,
            "turn": turn,
            "task_type": task_type,
            "prompt": prompt,
            "expected_answer": expected_answer,
            "test_answer": test_answer,
            "acc": acc,
            "evaluation_metrics": evaluation_metrics or {},
            "domain_similarity_probs": domain_similarity_probs.tolist() if isinstance(domain_similarity_probs, np.ndarray) else domain_similarity_probs,
            "selected_heads_count": selected_heads_count,
            "pruning_ratio": pruning_ratio,
            "session_breadth": session_breadth,
            "inferred_domains": inferred_domains,
            "dataset_type": dataset_type,
            "layer_head_counts": layer_head_counts,
            "inference_time": inference_time,  # Inference time (seconds) - counts only pure inference time (the time of model.generate())
            "pruning_time": pruning_time,  # Pruning time (seconds) - recorded only on the first turn of each scenario
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }

        # Optional fields (not saved if too large)
        if pruning_mask is not None:
            # Save only the number of heads per layer, not the full mask
            result["pruning_mask_summary"] = {
                layer_idx: len(heads) for layer_idx, heads in pruning_mask.items()
            }

        if head_scores is not None:
            # Save only the score statistics per layer, not the full scores
            result["head_scores_summary"] = {
                layer_idx: {
                    "mean": float(np.mean(scores)),
                    "max": float(np.max(scores)),
                    "min": float(np.min(scores)),
                    "std": float(np.std(scores))
                }
                for layer_idx, scores in head_scores.items()
            }
        
        self.results.append(result)

        # Incremental saving: save once every N results to avoid data loss
        if self.incremental_save and len(self.results) % self.incremental_save_interval == 0:
            self._incremental_save()

        logger.debug(f"Logged test result: scenario={scenario_id}, turn={turn}, acc={acc:.3f}")
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save the test results

        Args:
            filename: file name (if None, use a timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}"

        if self.save_format == "json":
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Test results saved: {filepath} ({len(self.results)} records)")

        elif self.save_format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            if not self.results:
                logger.warning("No test results to save")
                return

            # Get all fields
            fieldnames = set()
            for result in self.results:
                fieldnames.update(result.keys())
            fieldnames = sorted(fieldnames)

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    # Handle nested dicts and lists
                    row = {}
                    for key, value in result.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            row[key] = value
                    writer.writerow(row)

            logger.info(f"Test results saved: {filepath} ({len(self.results)} records)")
    
    def _incremental_save(self):
        """Incremental save (internal method, does not change the final file name)"""
        if self.save_format == "json":
            filepath = self.output_dir / f"{self.incremental_filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.debug(f"Incremental save: {filepath} ({len(self.results)} records)")
        elif self.save_format == "csv":
            filepath = self.output_dir / f"{self.incremental_filename}.csv"
            if not self.results:
                return
            
            fieldnames = set()
            for result in self.results:
                fieldnames.update(result.keys())
            fieldnames = sorted(fieldnames)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = {}
                    for key, value in result.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            row[key] = value
                    writer.writerow(row)
            logger.debug(f"Incremental save: {filepath} ({len(self.results)} records)")

    def compute_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics

        Returns:
            A statistics dictionary, grouped by dataset_type and task_type
        """
        if not self.results:
            logger.warning("No test results; cannot compute statistics")
            return {}

        stats = {}

        # Group by dataset_type
        for result in self.results:
            dataset_type = result.get("dataset_type", "unknown")
            task_type = result.get("task_type", "unknown")
            
            if dataset_type not in stats:
                stats[dataset_type] = {
                    "total_count": 0,
                    "task_types": {},
                    "acc_sum": 0.0,
                    "pruning_ratio_sum": 0.0,
                    "pruning_ratio_count": 0,
                    "selected_heads_count_sum": 0,
                    "selected_heads_count_count": 0,
                    "session_breadth_sum": 0.0,
                    "session_breadth_count": 0,
                    "domain_similarity_probs_sum": None,
                    "domain_similarity_probs_count": 0,
                    "inference_time_sum": 0.0,
                    "inference_time_count": 0
                }
            
            dataset_stats = stats[dataset_type]
            dataset_stats["total_count"] += 1
            dataset_stats["acc_sum"] += result.get("acc", 0.0)

            # Aggregate by task_type
            if task_type not in dataset_stats["task_types"]:
                dataset_stats["task_types"][task_type] = {
                    "count": 0,
                    "acc_sum": 0.0
                }
            
            task_stats = dataset_stats["task_types"][task_type]
            task_stats["count"] += 1
            task_stats["acc_sum"] += result.get("acc", 0.0)

            # Pruning-related statistics
            if result.get("pruning_ratio") is not None:
                dataset_stats["pruning_ratio_sum"] += result["pruning_ratio"]
                dataset_stats["pruning_ratio_count"] += 1

            if result.get("selected_heads_count") is not None:
                dataset_stats["selected_heads_count_sum"] += result["selected_heads_count"]
                dataset_stats["selected_heads_count_count"] += 1

            if result.get("session_breadth") is not None:
                dataset_stats["session_breadth_sum"] += result["session_breadth"]
                dataset_stats["session_breadth_count"] += 1

            # Domain similarity statistics
            if result.get("domain_similarity_probs") is not None:
                probs = np.array(result["domain_similarity_probs"])
                if dataset_stats["domain_similarity_probs_sum"] is None:
                    dataset_stats["domain_similarity_probs_sum"] = np.zeros_like(probs)
                dataset_stats["domain_similarity_probs_sum"] += probs
                dataset_stats["domain_similarity_probs_count"] += 1

            # Inference time statistics
            if result.get("inference_time") is not None:
                dataset_stats["inference_time_sum"] += result["inference_time"]
                dataset_stats["inference_time_count"] += 1

        # Compute averages
        for dataset_type, dataset_stats in stats.items():
            if dataset_stats["total_count"] > 0:
                dataset_stats["avg_acc"] = dataset_stats["acc_sum"] / dataset_stats["total_count"]
            
            if dataset_stats["pruning_ratio_count"] > 0:
                dataset_stats["avg_pruning_ratio"] = dataset_stats["pruning_ratio_sum"] / dataset_stats["pruning_ratio_count"]
            
            if dataset_stats["selected_heads_count_count"] > 0:
                dataset_stats["avg_selected_heads_count"] = dataset_stats["selected_heads_count_sum"] / dataset_stats["selected_heads_count_count"]
            
            if dataset_stats["session_breadth_count"] > 0:
                dataset_stats["avg_session_breadth"] = dataset_stats["session_breadth_sum"] / dataset_stats["session_breadth_count"]
            
            if dataset_stats["domain_similarity_probs_count"] > 0:
                dataset_stats["avg_domain_similarity_probs"] = (
                    dataset_stats["domain_similarity_probs_sum"] / dataset_stats["domain_similarity_probs_count"]
                ).tolist()
            
            if dataset_stats["inference_time_count"] > 0:
                dataset_stats["avg_inference_time"] = dataset_stats["inference_time_sum"] / dataset_stats["inference_time_count"]
            
            # Compute the per-task_type averages
            for task_type, task_stats in dataset_stats["task_types"].items():
                if task_stats["count"] > 0:
                    task_stats["avg_acc"] = task_stats["acc_sum"] / task_stats["count"]
        
        self.stats = stats
        return stats
    
    def save_statistics(self, filename: Optional[str] = None):
        """
        Save the statistics

        Args:
            filename: file name (if None, use a timestamp)
        """
        stats = self.compute_statistics()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_statistics_{timestamp}"

        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Statistics saved: {filepath}")

        # Print a summary of the statistics
        self._print_statistics_summary(stats)

    def _print_statistics_summary(self, stats: Dict[str, Dict[str, Any]]):
        """
        Print a summary of the statistics
        """
        logger.info("=" * 80)
        logger.info("Test result statistics summary")
        logger.info("=" * 80)

        for dataset_type, dataset_stats in stats.items():
            logger.info(f"\nDataset type: {dataset_type}")
            logger.info(f"  Total samples: {dataset_stats['total_count']}")
            logger.info(f"  Average accuracy: {dataset_stats.get('avg_acc', 0.0):.3f}")

            if dataset_stats.get("avg_pruning_ratio") is not None:
                logger.info(f"  Average pruning ratio: {dataset_stats['avg_pruning_ratio']:.3f}")

            if dataset_stats.get("avg_selected_heads_count") is not None:
                logger.info(f"  Average number of selected heads: {dataset_stats['avg_selected_heads_count']:.1f}")

            if dataset_stats.get("avg_session_breadth") is not None:
                logger.info(f"  Average session breadth: {dataset_stats['avg_session_breadth']:.3f}")

            if dataset_stats.get("avg_domain_similarity_probs") is not None:
                probs = dataset_stats["avg_domain_similarity_probs"]
                logger.info(f"  Average domain similarity probabilities: {[f'{p:.3f}' for p in probs]}")

            if dataset_stats.get("avg_inference_time") is not None:
                logger.info(f"  Average inference time: {dataset_stats['avg_inference_time']:.3f}s")

            # Aggregate by task_type
            if dataset_stats.get("task_types"):
                logger.info(f"  Statistics by question type:")
                for task_type, task_stats in dataset_stats["task_types"].items():
                    logger.info(f"    {task_type}: {task_stats['count']} samples, average accuracy={task_stats.get('avg_acc', 0.0):.3f}")

        logger.info("=" * 80)

