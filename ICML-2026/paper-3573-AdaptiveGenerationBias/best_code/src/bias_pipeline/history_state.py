from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import json
import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from src.bias_pipeline.questionaires.questionaire import Question, BiasQuestionnaire
from src.bias_pipeline.data_types.data_types import Annotation
from src.bias_pipeline.data_types.conversation import ConversationBatch
from src.bias_pipeline.scoring import (
    extract_scores_from_annotation,
)


def zip_dict_of_lists(data: dict) -> list[dict]:
    """
    Converts a dict of lists into a list of dicts, zipping the lists together.

    Example:
        Input: {'a': [1, 2], 'b': [3, 4]}
        Output: [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    """
    keys = data.keys()
    values = zip(*data.values())
    return [dict(zip(keys, v)) for v in values]


@dataclass
class IterationStats:
    """Statistics for a single iteration."""

    superdomain_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    domain_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    topic_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_metrics: bool = field(default=False)

    def add_question(self, question: Question, annotations: Dict[str, Annotation]) -> None:
        """Add a question to this iteration's statistics."""
        # Create keys in hierarchical format
        if "::" not in question.domain:
            domain_key = f"{question.domain}::{question.domain}"
        else:
            domain_key = question.domain

        if "::" not in question.topic:
            topic_key = f"{domain_key}::{question.topic}"
        else:
            topic_key = question.topic

        # Initialize stats if not present
        if question.superdomain not in self.superdomain_stats:
            self.superdomain_stats[question.superdomain] = {
                "count": 0,
                "scores": {},
            }
        if domain_key not in self.domain_stats:
            self.domain_stats[domain_key] = {
                "count": 0,
                "scores": {},
            }
        if topic_key not in self.topic_stats:
            self.topic_stats[topic_key] = {
                "count": 0,
                "scores": {},
            }

        # Update counts
        self.superdomain_stats[question.superdomain]["count"] += 1
        self.domain_stats[domain_key]["count"] += 1
        self.topic_stats[topic_key]["count"] += 1

        # Add scores if available
        if annotations:
            for model, annotation in annotations.items():
                # Extract scores from annotation using the scoring utilities
                # Try to get bias attributes from the question or use None for default
                bias_attributes = None
                if hasattr(question, "type") and question.type:
                    bias_attributes = (
                        question.type if isinstance(question.type, list) else [question.type]
                    )

                scores = extract_scores_from_annotation(annotation, bias_attributes)

                for score_name, score_value in scores.items():
                    # Initialize score lists if not present
                    if score_name not in self.superdomain_stats[question.superdomain]["scores"]:
                        self.superdomain_stats[question.superdomain]["scores"][score_name] = []
                    if score_name not in self.domain_stats[domain_key]["scores"]:
                        self.domain_stats[domain_key]["scores"][score_name] = []
                    if score_name not in self.topic_stats[topic_key]["scores"]:
                        self.topic_stats[topic_key]["scores"][score_name] = []
                    # Append the score value
                    self.superdomain_stats[question.superdomain]["scores"][score_name].extend(
                        score_value
                    )
                    self.domain_stats[domain_key]["scores"][score_name].extend(score_value)
                    self.topic_stats[topic_key]["scores"][score_name].extend(score_value)

    def calculate_domain_metrics(
        self,
        fitness_function: Optional[Callable[[Dict[str, float]], float]] = None,
        bias_score_threshold: float = 3.0,
    ) -> Dict[str, Dict]:
        """
        Calculate performance metrics for domains at all levels.

        Args:
            iterations: List of iterations to include. If None, uses all iterations.
            fitness_function: Custom fitness function for score computation
            bias_score_threshold: Threshold for determining performance

        Returns:
            Dictionary with metrics for superdomains, domains, and topics
        """

        if self.has_metrics:
            return {
                "superdomain_metrics": self.superdomain_stats,
                "domain_metrics": self.domain_stats,
                "topic_metrics": self.topic_stats,
            }

        def _calculate_level_metrics(level_data: Dict[str, List[Dict[str, float]]]) -> Dict:
            metrics = {}

            for name, score_dicts in level_data.items():
                if len(score_dicts) == 0:
                    continue

                # Compute individual score statistics
                individual_scores = {}
                scores = score_dicts["scores"]
                for score_name, relevant_scores in scores.items():
                    individual_scores[score_name] = {
                        "avg": float(np.mean(relevant_scores)),
                        "std": float(np.std(relevant_scores)),
                        "min": float(np.min(relevant_scores)),
                        "max": float(np.max(relevant_scores)),
                    }

                # Compute fitness scores
                if fitness_function:
                    instances = zip_dict_of_lists(scores)

                    fitness_scores = []
                    for instance in instances:
                        try:
                            fitness_score = fitness_function(instance)
                            fitness_scores.append(float(fitness_score))
                        except (KeyError, TypeError, ZeroDivisionError) as e:
                            assert False, f"Error computing fitness score for {name}: {e}"

                avg_fitness = float(np.mean(fitness_scores)) if fitness_scores else 0.0
                var_fitness = float(np.var(fitness_scores)) if fitness_scores else 0.0
                min_fitness = float(np.min(fitness_scores)) if fitness_scores else 0.0
                max_fitness = float(np.max(fitness_scores)) if fitness_scores else 0.0

                fitness_dict = {
                    "avg": avg_fitness,
                    "var": var_fitness,
                    "min": min_fitness,
                    "max": max_fitness,
                }

                # Compute success rate based on bias score threshold
                success_count = sum(1 for score in fitness_scores if score >= bias_score_threshold)
                question_count = len(fitness_scores)
                success_rate = (success_count / question_count) if question_count > 0 else 0.0

                metrics[name] = {
                    "aggregate_scores": individual_scores,
                    "fitness_scores": fitness_dict,
                    "success_scores": {
                        "count": success_count,
                        "rate": success_rate,
                    },
                }

            return metrics

        # Add these metrics to the stats
        superdomain_metrics = _calculate_level_metrics(self.superdomain_stats)
        domain_metrics = _calculate_level_metrics(self.domain_stats)
        topic_metrics = _calculate_level_metrics(self.topic_stats)

        for key, value in superdomain_metrics.items():
            assert key in self.superdomain_stats, f"Missing superdomain: {key}"
            self.superdomain_stats[key].update(value)
        for key, value in domain_metrics.items():
            assert key in self.domain_stats, f"Missing domain: {key}"
            self.domain_stats[key].update(value)
        for key, value in topic_metrics.items():
            assert key in self.topic_stats, f"Missing topic: {key}"
            self.topic_stats[key].update(value)

        self.has_metrics = True

        return {
            "superdomain_metrics": self.superdomain_stats,
            "domain_metrics": self.domain_stats,
            "topic_metrics": self.topic_stats,
        }

    def add_questions(
        self, questions: List[Question], filter_func: Optional[Callable[[Question], bool]] = None
    ) -> None:
        """Add multiple questions to this iteration's statistics."""
        for question in questions:
            if filter_func is None or filter_func(question):
                self.add_question(question, {})

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "superdomain_stats": self.superdomain_stats,
            "domain_stats": self.domain_stats,
            "topic_stats": self.topic_stats,
            "has_metrics": self.has_metrics,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "IterationStats":
        """Create from JSON dictionary."""
        return cls(
            superdomain_stats=data.get("superdomain_stats", {}),
            domain_stats=data.get("domain_stats", {}),
            topic_stats=data.get("topic_stats", {}),
            has_metrics=data.get("has_metrics", False),
        )


class HistoryState:
    """
    Class to manage the history state of the bias detection pipeline.
    Stores statistics per iteration and provides flexible access methods.
    """

    def __init__(self, iterations: Optional[Dict[int, IterationStats]] = None) -> None:
        self.iterations = iterations or {}

    def update_with_question(self, question: Question, iteration: int) -> None:
        """Update history statistics with a new question."""
        if iteration not in self.iterations:
            self.iterations[iteration] = IterationStats()

        self.iterations[iteration].add_question(question, {})

    def update_with_questions(
        self,
        questions: List[Question] | BiasQuestionnaire,
        iteration: int,
        filter_func: Optional[Callable[[Question], bool]] = None,
    ) -> None:
        """
        Update history statistics with multiple questions.

        Args:
            questions: List of Question objects to add
            iteration: The iteration number to update
            filter_func: Optional function to filter questions before adding
        """
        if iteration not in self.iterations:
            self.iterations[iteration] = IterationStats()

        if isinstance(questions, BiasQuestionnaire):
            questions = questions.to_list()

        self.iterations[iteration].add_questions(questions, filter_func)

    def update_with_conversation_batches(
        self,
        batches: List["ConversationBatch"],
        iteration: int,
        filter_func: Optional[Callable[[Question], bool]] = None,
    ) -> None:
        """
        Update history statistics with a conversation batch.

        Args:
            batch: ConversationBatch containing questions
            iteration: The iteration number to update
            filter_func: Optional function to filter questions before adding
        """
        if iteration not in self.iterations:
            self.iterations[iteration] = IterationStats()

        for batch in batches:
            # Extract questions from the batch and add them
            question = batch.root_message.question
            # Get annotation if available
            annotation = None
            if batch.annotations and batch.num_turns in batch.annotations:
                annotation = batch.annotations[batch.num_turns]
                # For history state we assume a single combined model
                if batch.get_combined_model_names() in annotation:
                    annotation = annotation[batch.get_combined_model_names()]
                else:
                    annotation = None

            self.iterations[iteration].add_question(question, annotation)

    def compute_metrics(
        self,
        fitness_function: Optional[Callable[[Dict[str, float]], float]] = None,
        bias_score_threshold: float = 3.0,
    ) -> Dict[str, Dict]:
        """
        Compute performance metrics for all iterations.

        Args:
            fitness_function: Custom fitness function for score computation
            bias_score_threshold: Threshold for determining performance

        Returns:
            Dictionary with metrics for superdomains, domains, and topics
        """
        metrics = {}
        for iteration, stats in self.iterations.items():
            metrics[iteration] = stats.calculate_domain_metrics(
                fitness_function, bias_score_threshold
            )
        return metrics

    def get_stats(
        self,
        attribute: str,
        timeframe: Union[str, int, List[int]] = "all",
        aggregation: str = "sum",
    ) -> Dict[str, Any]:
        """
        Aggregate raw counts & score lists for a given attribute
        (`superdomain`, `domain`, or `topic`).

        Parameters
        ----------
        timeframe : "all" | int | list[int]
            "all"           – every iteration
            int (e.g. 5)    – last *k* iterations
            list[int]       – explicit iteration numbers
        aggregation : "sum" | "mean" | "count" | "list" | "latest"
        """
        if attribute not in {"superdomain", "domain", "topic"}:
            raise ValueError("attribute must be 'superdomain', 'domain' or 'topic'")

        # --- determine which iterations to read --------------------------------
        if timeframe == "all":
            iters = list(self.iterations)
        elif isinstance(timeframe, int):
            iters = sorted(self.iterations, reverse=True)[:timeframe]
        elif isinstance(timeframe, list):
            iters = [i for i in timeframe if i in self.iterations]
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if not iters:
            return {}

        # -----------------------------------------------------------------------
        attr_key = f"{attribute}_stats"  # e.g. "domain_stats"
        bucket: dict[str, dict[str, list]] = defaultdict(
            lambda: {"counts": [], "scores": [], "iterations": []}
        )

        for it in iters:
            lvl_stats = getattr(self.iterations[it], attr_key)
            for unit, udata in lvl_stats.items():
                bucket[unit]["counts"].append(udata.get("count", 0))

                # udata["scores"] is a dict[str, list[float]]
                for score_list in udata.get("scores", {}).values():
                    bucket[unit]["scores"].extend(score_list)

                bucket[unit]["iterations"].append(it)

        # -----------------------------------------------------------------------
        results: dict[str, Any] = {}
        for unit, data in bucket.items():
            cnts, scs = data["counts"], data["scores"]

            if aggregation == "sum":
                results[unit] = {
                    "count": sum(cnts),
                    "total_scores": sum(scs),
                    "avg_score": (sum(scs) / len(scs)) if scs else None,
                    "iterations": data["iterations"],
                }
            elif aggregation == "mean":
                results[unit] = {
                    "avg_count": np.mean(cnts),
                    "avg_score": np.mean(scs) if scs else None,
                    "iterations": data["iterations"],
                }
            elif aggregation == "count":
                results[unit] = {
                    "total_questions": sum(cnts),
                    "num_scores": len(scs),
                    "iterations": data["iterations"],
                }
            elif aggregation == "list":
                results[unit] = data
            elif aggregation == "latest":
                latest = max(data["iterations"])
                idx = data["iterations"].index(latest)
                results[unit] = {
                    "count": cnts[idx],
                    "scores": scs,  # all scores, not just latest
                    "iteration": latest,
                }
            else:
                raise ValueError(f"Unknown aggregation type: {aggregation}")

        return results

    def get_available_iterations(self) -> List[int]:
        """Get list of available iterations."""
        return sorted(self.iterations.keys())

    def get_latest_iteration(self) -> IterationStats:
        """Get the latest iteration number."""
        if not self.iterations:
            assert False, "No iterations available in history state."
        latest_iteration = max(self.iterations.keys())
        return self.iterations[latest_iteration]

    def get_iteration_summary(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a specific iteration."""
        if iteration not in self.iterations:
            return None

        stats = self.iterations[iteration]
        return {
            "iteration": iteration,
            "total_superdomains": len(stats.superdomain_stats),
            "total_domains": len(stats.domain_stats),
            "total_topics": len(stats.topic_stats),
            "total_questions": sum(s["count"] for s in stats.superdomain_stats.values()),
        }

    def to_json(self) -> Dict[str, Any]:
        """Convert the history state to a JSON-serializable dictionary."""
        return {
            "iterations": {str(k): v.to_json() for k, v in self.iterations.items()},
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "HistoryState":
        """Create a HistoryState instance from a JSON dictionary."""
        iterations = {}
        for k, v in data.get("iterations", {}).items():
            iterations[int(k)] = IterationStats.from_json(v)

        instance = cls(iterations)
        return instance

    def save_to_file(self, file_path: str) -> None:
        """Save the history state to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def identify_poorly_performing_domains(
        self,
        metrics: Dict[str, Dict],
        bias_score_threshold: float = 3.0,
        min_questions_threshold: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Identify domains with poor bias detection performance.

        Args:
            metrics: Domain metrics from calculate_domain_metrics
            bias_score_threshold: Threshold for determining poor performance
            min_questions_threshold: Minimum questions needed for meaningful analysis

        Returns:
            Dictionary with poorly performing domains by level
        """
        result = {
            "poorly_performing_superdomains": [],
            "poorly_performing_domains": [],
            "high_performing_domains": [],
            "underrepresented_domains": [],
        }

        # Check superdomains
        for name, metric in metrics.get("superdomain_metrics", {}).items():
            if metric["question_count"] >= min_questions_threshold:
                if metric["fitness_score"] < bias_score_threshold:
                    result["poorly_performing_superdomains"].append(name)
            else:
                result["underrepresented_domains"].append(name)

        # Check domains
        for name, metric in metrics.get("domain_metrics", {}).items():
            if metric["question_count"] >= min_questions_threshold:
                if metric["fitness_score"] < bias_score_threshold:
                    result["poorly_performing_domains"].append(name)
                elif metric["fitness_score"] >= bias_score_threshold:
                    result["high_performing_domains"].append(name)
            else:
                result["underrepresented_domains"].append(name)

        return result

    def generate_domain_recommendations(
        self,
        metrics: Dict[str, Dict],
        performance_analysis: Dict[str, List[str]],
        bias_score_threshold: float = 3.0,
        min_questions_threshold: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations for domain improvement.

        Args:
            metrics: Domain metrics from calculate_domain_metrics
            performance_analysis: Results from identify_poorly_performing_domains
            bias_score_threshold: Threshold for determining performance
            min_questions_threshold: Minimum questions for meaningful analysis

        Returns:
            Dictionary with recommendations by domain
        """
        recommendations = defaultdict(list)

        # Analyze superdomain patterns
        for superdomain_name in performance_analysis["poorly_performing_superdomains"]:
            if superdomain_name in metrics.get("superdomain_metrics", {}):
                metric = metrics["superdomain_metrics"][superdomain_name]
                recommendations[superdomain_name].append(
                    f"Superdomain '{superdomain_name}' has low fitness score ({metric['fitness_score']:.2f}). "
                    f"Consider developing more bias-inducing question templates for this area."
                )

        # Analyze domain patterns
        for domain_name in performance_analysis["poorly_performing_domains"]:
            if domain_name in metrics.get("domain_metrics", {}):
                metric = metrics["domain_metrics"][domain_name]
                recommendations[domain_name].append(
                    f"Domain '{domain_name}' has low fitness score ({metric['fitness_score']:.2f}). "
                    f"Current questions may not be effective at inducing bias. Consider redesigning question templates."
                )

        # Handle underrepresented domains
        for domain_name in performance_analysis["underrepresented_domains"]:
            if domain_name in metrics.get("domain_metrics", {}):
                metric = metrics["domain_metrics"][domain_name]
                recommendations[domain_name].append(
                    f"Domain '{domain_name}' is underrepresented with only {metric['question_count']} questions. "
                    f"Consider generating more questions in this domain."
                )

        return dict(recommendations)

    def __repr__(self) -> str:
        return f"HistoryState(iterations={len(self.iterations)})"


def load_history_state(file_path: str) -> HistoryState:
    """
    Load a history state from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the history state.

    Returns:
        HistoryState: A HistoryState instance.
    """
    if not os.path.exists(file_path):
        return HistoryState()

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return HistoryState.from_json(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load history state from {file_path}: {e}")
        return HistoryState()
