"""
Online inference with budget-constrained head pruning (PSP, online stage).

For each test scenario the pipeline:
  1. loads the frozen base model and the offline artifacts
     (linear probes, head importance I_{l,h,k}, domain-invariant whitelist,
     and optional temperature calibration);
  2. diagnoses the scenario domain mixture from the first turn(s) via the probes;
  3. compiles a budget-constrained binary head-pruning mask once per scenario
     (whitelist heads are always kept) and reuses it for every turn;
  4. runs single-step inference under the pruned mask;
  5. scores the answer with exact-match keyword overlap (no semantic similarity,
     no LLM judge).

The same `pruning_strength` sweep is supported as in the paper: larger values
prune more heads. History (multi-turn conversation context) is OFF by default.

Example:
    python scripts/run_inference.py --model Qwen2.5-7B-Instruct --gpu 0 \
        --pruning_strengths 0.2 0.4 0.6 --num_samples 20
"""

import sys
import json
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Project root (this file lives in <root>/scripts/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import BaseModel
from src.reasoning.baseline_reasoning import BaselineReasoning
from src.probe.session_pruning import SessionPruner
from src.probe.domain_inference import DomainInference
from src.probe.head_importance import HeadImportanceCalculator
from src.probe.whitelist_identification import HeadWhitelistIdentifier
from src.preorientation.linear_probe import MultiLayerProbe
from src.preorientation.probe_calibration import MultiProbeSystem
from src.evaluation.answer_evaluator import evaluate_answer_accuracy
from src.evaluation.test_result_logger import TestResultLogger
from src.utils import get_logger
from src.utils.model_utils import get_output_dir_for_model

logger = get_logger(__name__)

# Default test categories (sub-directories under data/test/).
DEFAULT_DATASETS = ["selected_domain", "out_of_domain", "cross_domain"]


def sample_scenarios(data_dir: Path, num_samples: Optional[int], seed: int = 42):
    """Sample scenarios from a directory.

    If num_samples is None or >= the number of available scenarios, all
    scenarios are returned; otherwise a random subset of size num_samples.
    """
    random.seed(seed)
    np.random.seed(seed)

    json_files = sorted(data_dir.glob("*.json"))
    if len(json_files) == 0:
        return []

    all_scenarios = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for scenario in data:
                    scenario["source_file"] = json_file.name
                    all_scenarios.append(scenario)
            else:
                data["source_file"] = json_file.name
                all_scenarios.append(data)
        except Exception as e:
            logger.error(f"Failed to load file {json_file}: {e}")

    if num_samples is None or len(all_scenarios) <= num_samples:
        logger.info(f"  Returning all {len(all_scenarios)} scenarios")
        return all_scenarios

    selected = random.sample(all_scenarios, num_samples)
    logger.info(f"  Sampled {len(selected)} of {len(all_scenarios)} scenarios")
    return selected


def load_test_scenarios(datasets: List[str], num_per_dir: Dict[str, Optional[int]]):
    """Load scenarios from each requested test category under data/test/."""
    all_scenarios = []
    for dataset_type in datasets:
        data_dir = project_root / "data" / "test" / dataset_type
        if not data_dir.exists():
            logger.warning(f"Directory does not exist, skipping: {data_dir}")
            continue
        num_samples = num_per_dir.get(dataset_type)
        scenarios = sample_scenarios(data_dir, num_samples=num_samples)
        for scenario in scenarios:
            scenario["dataset_type"] = dataset_type
        all_scenarios.extend(scenarios)
        logger.info(f"Loaded {len(scenarios)} scenarios from '{dataset_type}'")
    logger.info(f"Loaded {len(all_scenarios)} scenarios in total")
    return all_scenarios


def _clear_mask_hooks(base_model):
    """Remove any active pruning hooks (prevents a mask leaking across scenarios)."""
    if getattr(base_model, "mask_hooks", None):
        for hook in base_model.mask_hooks:
            hook.remove()
        base_model.mask_hooks = []
    if hasattr(base_model, "current_mask"):
        del base_model.current_mask


def run_inference(
    model_name: str = "Qwen2.5-7B-Instruct",
    pruning_strengths: List[float] = [0.2, 0.4, 0.6],
    datasets: List[str] = DEFAULT_DATASETS,
    num_per_dir: Optional[Dict[str, Optional[int]]] = None,
    output_subdir: str = "pruning_strength",
    use_history: bool = False,
    use_calibration: bool = False,
    max_new_tokens: int = 80,
    temperature: float = 0.3,
):
    """Run the online pruning + inference + evaluation loop over a pruning_strength sweep.

    Args:
        model_name: model directory name under models/.
        pruning_strengths: list of pruning strengths to sweep (higher prunes more).
        datasets: test categories (sub-dirs of data/test/) to evaluate.
        num_per_dir: per-category sample count; a value of None means "use all".
        output_subdir: results sub-directory under outputs/<model>/.
        use_history: include previous turns as conversation context (default False).
        use_calibration: use the temperature-calibrated probe system (artifacts shipped).
    """
    logger.info("=" * 80)
    logger.info(f"Online pruning + inference - {model_name}")
    logger.info(f"Pruning strengths: {pruning_strengths}")
    logger.info(f"use_history={use_history}  use_calibration={use_calibration}")
    logger.info("=" * 80)

    # --- Load the frozen base model ---
    model_path = project_root / "models" / model_name
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        logger.error("Place the HuggingFace model under models/<model_name>/.")
        return

    logger.info(f"Loading model: {model_path}")
    base_model = BaseModel(
        model_name=str(model_path),
        quantization="none",
        torch_dtype="float16",
    )

    # --- Locate offline artifacts: outputs/<model>/ppd_pipeline/ ---
    ppd_output_dir = get_output_dir_for_model(
        base_output_dir=str(project_root / "outputs"),
        model_path=str(model_path),
        subdir="ppd_pipeline",
    )

    # Base linear probes
    probe1_path = ppd_output_dir / "probe1_base.pt"
    if not probe1_path.exists():
        logger.error(f"Probe file not found: {probe1_path}")
        logger.error("Run scripts/train_offline.py first, or use a shipped artifact set.")
        return
    logger.info(f"Loading base probes: {probe1_path}")
    base_probe = MultiLayerProbe.load(str(probe1_path), device=base_model.device)

    # Optional temperature-calibrated probe system
    multi_probe_system = None
    if use_calibration:
        try:
            multi_probe_system = MultiProbeSystem.load(ppd_output_dir, base_probe)
            logger.info("Loaded calibrated probe system")
        except Exception as e:
            logger.warning(f"Could not load calibrated probe system: {e}")
            logger.warning("Falling back to the uncalibrated base probes")

    # Head importance I_{l,h,k}
    head_importance_path = None
    for path in (
        ppd_output_dir / "head_importance.pt",
        ppd_output_dir / "head_importance_probe1.pt",
        ppd_output_dir / "backup_importance" / "head_importance_probe1.pt",
    ):
        if path.exists():
            head_importance_path = path
            break
    if head_importance_path is None:
        logger.error(f"Head importance file not found under {ppd_output_dir}")
        return
    logger.info(f"Loading head importance: {head_importance_path}")
    head_importance = HeadImportanceCalculator.load(str(head_importance_path))

    # Domain-invariant whitelist (always-kept heads)
    whitelist = []
    whitelist_path = ppd_output_dir / "whitelist.json"
    if whitelist_path.exists():
        try:
            whitelist = HeadWhitelistIdentifier.load_whitelist(str(whitelist_path))
            logger.info(f"Loaded whitelist: {len(whitelist)} heads")
        except Exception as e:
            logger.warning(f"Failed to load whitelist: {e}")
    else:
        logger.warning(f"Whitelist file not found: {whitelist_path}")

    # Probe-index domains are integer indices [0, num_domains); names are not
    # needed online because pruning is index-based and internally consistent
    # with the head importance computed in the same offline run.
    selected_domains = list(range(base_probe.num_domains))
    domain_inference = DomainInference(
        num_domains=base_probe.num_domains,
        min_probability_threshold=0.05,
    )

    # --- Load test scenarios ---
    if num_per_dir is None:
        num_per_dir = {d: None for d in datasets}
    test_scenarios = load_test_scenarios(datasets, num_per_dir)
    if not test_scenarios:
        logger.error("No test scenarios loaded; aborting.")
        return

    # --- Inference engine (history OFF by default) ---
    reasoning = BaselineReasoning(
        base_model=base_model,
        max_steps=1,
        use_history=use_history,
        max_history_turns=2 if use_history else 0,
    )

    output_base_dir = get_output_dir_for_model(
        base_output_dir=str(project_root / "outputs"),
        model_path=str(model_path),
        subdir=output_subdir,
    )

    layer_probes = (
        multi_probe_system.probe1
        if (use_calibration and multi_probe_system is not None)
        else base_probe
    )

    all_results_summary = {}

    for pruning_strength in pruning_strengths:
        logger.info("\n" + "=" * 80)
        logger.info(f"pruning_strength = {pruning_strength}")
        logger.info("=" * 80)

        result_logger = TestResultLogger(
            output_dir=str(output_base_dir / f"pruning_strength_{pruning_strength}"),
            save_format="json",
        )

        total_turns = 0
        total_accuracy = 0.0
        total_correct = 0
        dataset_stats: Dict[str, Dict] = {}
        task_type_stats: Dict[str, Dict] = {}
        pruning_ratios: List[float] = []

        for scenario_idx, scenario in enumerate(
            tqdm(test_scenarios, desc=f"pruning_strength={pruning_strength}")
        ):
            scenario_id = scenario.get("scenario_id", f"scenario_{scenario_idx}")
            dataset_type = scenario.get("dataset_type", "unknown")
            turns = scenario.get("turns", [])

            session_pruner = SessionPruner(
                layer_probes=layer_probes,
                domain_inference=domain_inference,
                selected_domains=selected_domains,
                retention_rate=0.6,  # kept for compatibility; pruning_strength governs the budget
                num_heads_per_layer=base_model.model.config.num_attention_heads,
                pruning_strength=pruning_strength,
                head_importance=head_importance,
                whitelist=whitelist,
            )

            dataset_stats.setdefault(
                dataset_type,
                {"total_turns": 0, "total_correct": 0, "total_accuracy": 0.0, "scenarios": 0},
            )

            reasoning.reset_history()
            _clear_mask_hooks(base_model)

            is_first_turn = True
            pruning_mask = None
            pruning_metadata = None

            for turn_idx, turn in enumerate(turns):
                prompt = turn.get("prompt", "")
                expected_answer = turn.get("answer", "")
                task_type = turn.get("task_type", "factual")
                if not prompt or not expected_answer:
                    continue

                pruning_time = None
                if is_first_turn:
                    # Compile the budget-constrained pruning mask once for the scenario.
                    pruning_start_time = time.time()
                    session_data = {
                        "description": scenario.get("topic_description", ""),
                        "turns": turns,
                    }
                    pruning_mask, pruning_metadata = session_pruner.prune_for_session(
                        session_data=session_data,
                        base_model=base_model,
                        is_first_turn=True,
                        return_metadata=True,
                    )
                    pruning_time = time.time() - pruning_start_time
                    if pruning_metadata and "pruning_ratio" in pruning_metadata:
                        pruning_ratios.append(pruning_metadata["pruning_ratio"])

                    _clear_mask_hooks(base_model)
                    base_model._apply_mask_hooks(pruning_mask)
                    is_first_turn = False
                else:
                    # Subsequent turns reuse the first turn's mask.
                    if pruning_mask is not None and not getattr(base_model, "mask_hooks", None):
                        base_model._apply_mask_hooks(pruning_mask)

                if isinstance(expected_answer, list):
                    expected_answer = expected_answer[0] if expected_answer else ""

                result = reasoning.reason(
                    prompt=prompt,
                    task_type=task_type,
                    is_first_turn=(turn_idx == 0),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                predicted_answer = result["result"]

                eval_result = evaluate_answer_accuracy(
                    predicted_answer=predicted_answer,
                    expected_answer=expected_answer,
                    task_type=task_type,
                )
                accuracy = eval_result.get("accuracy", 0.0)

                common_kwargs = dict(
                    scenario_id=scenario_id,
                    turn=turn_idx + 1,
                    task_type=task_type,
                    prompt=prompt,
                    expected_answer=expected_answer,
                    test_answer=predicted_answer,
                    acc=accuracy,
                    dataset_type=dataset_type,
                    inference_time=result.get("inference_time", 0.0),
                    wall_time=result.get("wall_time", 0.0),
                    peak_memory_mb_allocated=result.get("peak_memory_mb_allocated"),
                    peak_memory_mb_reserved=result.get("peak_memory_mb_reserved"),
                    use_calibration=use_calibration,
                    pruning_strength=pruning_strength,
                    use_history=use_history,
                )
                if turn_idx == 0:
                    layer_head_counts = {
                        layer_idx: len(heads) for layer_idx, heads in pruning_mask.items()
                    }
                    result_logger.log_result(
                        **common_kwargs,
                        domain_similarity_probs=pruning_metadata.get("domain_similarity_probs") if pruning_metadata else None,
                        selected_heads_count=pruning_metadata.get("selected_heads_count") if pruning_metadata else None,
                        pruning_ratio=pruning_metadata.get("pruning_ratio") if pruning_metadata else None,
                        session_breadth=pruning_metadata.get("session_breadth") if pruning_metadata else None,
                        inferred_domains=pruning_metadata.get("inferred_domains") if pruning_metadata else None,
                        layer_head_counts=layer_head_counts,
                        pruning_time=pruning_time,
                    )
                else:
                    result_logger.log_result(
                        **common_kwargs,
                        pruning_ratio=pruning_metadata.get("pruning_ratio") if pruning_metadata else None,
                        inferred_domains=pruning_metadata.get("inferred_domains") if pruning_metadata else None,
                        pruning_time=None,
                    )

                total_turns += 1
                total_accuracy += accuracy
                if accuracy >= 0.8:
                    total_correct += 1

                ds = dataset_stats[dataset_type]
                ds["total_turns"] += 1
                ds["total_accuracy"] += accuracy
                if accuracy >= 0.8:
                    ds["total_correct"] += 1

                tt = task_type_stats.setdefault(
                    task_type, {"total_turns": 0, "total_correct": 0, "total_accuracy": 0.0}
                )
                tt["total_turns"] += 1
                tt["total_accuracy"] += accuracy
                if accuracy >= 0.8:
                    tt["total_correct"] += 1

            _clear_mask_hooks(base_model)

        result_logger.save_results(f"pruning_strength_{pruning_strength}_results")

        avg_pruning_ratio = float(np.mean(pruning_ratios)) if pruning_ratios else None
        avg_accuracy = total_accuracy / total_turns if total_turns > 0 else 0.0
        match_rate = total_correct / total_turns * 100 if total_turns > 0 else 0.0

        all_results_summary[pruning_strength] = {
            "avg_accuracy": avg_accuracy,
            "match_rate": match_rate,
            "total_turns": total_turns,
            "total_correct": total_correct,
            "avg_pruning_ratio": avg_pruning_ratio,
            "dataset_stats": {
                dt: {
                    "avg_accuracy": s["total_accuracy"] / s["total_turns"] if s["total_turns"] else 0.0,
                    "match_rate": s["total_correct"] / s["total_turns"] * 100 if s["total_turns"] else 0.0,
                    "total_turns": s["total_turns"],
                }
                for dt, s in dataset_stats.items()
            },
            "task_type_stats": {
                tt: {
                    "avg_accuracy": s["total_accuracy"] / s["total_turns"] if s["total_turns"] else 0.0,
                    "match_rate": s["total_correct"] / s["total_turns"] * 100 if s["total_turns"] else 0.0,
                    "total_turns": s["total_turns"],
                }
                for tt, s in task_type_stats.items()
            },
        }

        logger.info(f"\npruning_strength={pruning_strength} results:")
        logger.info(f"  total turns: {total_turns}")
        logger.info(f"  avg accuracy (EM): {avg_accuracy:.3f}")
        logger.info(f"  match rate (acc>=0.8): {total_correct}/{total_turns} = {match_rate:.1f}%")
        if avg_pruning_ratio is not None:
            logger.info(f"  avg pruned-head fraction: {avg_pruning_ratio:.2%}")

    summary_file = output_base_dir / "pruning_strength_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved to: {summary_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Pruning-strength comparison")
    logger.info("=" * 80)
    logger.info(f"{'Pruning Strength':<20}{'Avg Pruned Frac':<20}{'Avg Accuracy':<15}{'Match Rate':<15}")
    logger.info("-" * 70)
    for pruning_strength in sorted(pruning_strengths):
        summary = all_results_summary[pruning_strength]
        apr = summary["avg_pruning_ratio"]
        apr_str = f"{apr:.2%}" if apr is not None else "N/A"
        logger.info(
            f"{pruning_strength:<20}{apr_str:<20}{summary['avg_accuracy']:<15.3f}{summary['match_rate']:<15.1f}"
        )
    logger.info("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Online pruning + inference + evaluation")
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct",
                        help="Model directory name under models/")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA device index (sets CUDA_VISIBLE_DEVICES). Omit to respect the environment.")
    parser.add_argument("--pruning_strengths", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6],
                        help="Pruning strengths to sweep (higher prunes more heads)")
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS,
                        help="Test categories under data/test/ to evaluate")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Scenarios sampled per category (use -1 for all)")
    parser.add_argument("--use_history", type=str, default="false",
                        help="Include previous turns as context (true/false; default false)")
    parser.add_argument("--use_calibration", type=str, default="false",
                        help="Use the temperature-calibrated probe system (true/false; default false)")
    parser.add_argument("--output_subdir", type=str, default="pruning_strength",
                        help="Results sub-directory under outputs/<model>/")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n = None if args.num_samples is not None and args.num_samples < 0 else args.num_samples
    num_per_dir = {d: n for d in args.datasets}

    def _as_bool(s: str) -> bool:
        return s.lower() in ("true", "1", "yes", "on")

    run_inference(
        model_name=args.model,
        pruning_strengths=args.pruning_strengths,
        datasets=args.datasets,
        num_per_dir=num_per_dir,
        output_subdir=args.output_subdir,
        use_history=_as_bool(args.use_history),
        use_calibration=_as_bool(args.use_calibration),
    )
