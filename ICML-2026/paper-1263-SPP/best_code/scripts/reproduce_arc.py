"""
Reproduction script for paper 1263 — ARC evaluation with Qwen2.5-7B-Instruct.

Runs both pruned (eta=0.5) and dense baselines, reports:
- Token-level Recall (keyword match ratio * 100)
- Speedup (dense_time / pruned_time)
- Memory (GB)
- Retention ((recall_pruned / recall_dense) * speedup)
"""

import sys
import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

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
from src.utils import get_logger
from src.utils.model_utils import get_output_dir_for_model

logger = get_logger(__name__)


def _clear_mask_hooks(base_model):
    if getattr(base_model, "mask_hooks", None):
        for hook in base_model.mask_hooks:
            hook.remove()
        base_model.mask_hooks = []
    if hasattr(base_model, "current_mask"):
        del base_model.current_mask


def load_arc_scenarios(data_dir: Path, num_samples: Optional[int] = None, seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

    json_files = sorted(data_dir.glob("*.json"))
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
            logger.error(f"Failed to load {json_file}: {e}")

    if num_samples is None or len(all_scenarios) <= num_samples:
        logger.info(f"  Using all {len(all_scenarios)} ARC scenarios")
        return all_scenarios

    selected = random.sample(all_scenarios, num_samples)
    logger.info(f"  Sampled {len(selected)} of {len(all_scenarios)} ARC scenarios")
    return selected


def evaluate_scenarios(
    base_model,
    reasoning,
    scenarios,
    pruning_strength=None,
    layer_probes=None,
    domain_inference=None,
    selected_domains=None,
    head_importance=None,
    whitelist=None,
    num_heads_per_layer=28,
):
    """
    Evaluate scenarios. If pruning_strength is None, runs dense (no pruning).
    """
    total_turns = 0
    total_accuracy = 0.0
    total_inference_time = 0.0
    total_wall_time = 0.0
    total_pruning_time = 0.0
    peak_memory_mb = 0.0
    pruning_ratios = []
    accuracies = []

    for scenario_idx, scenario in enumerate(tqdm(scenarios, desc=f"Eval (prune={pruning_strength})")):
        turns = scenario.get("turns", [])
        reasoning.reset_history()
        _clear_mask_hooks(base_model)

        is_first_turn = True
        pruning_mask = None

        for turn_idx, turn in enumerate(turns):
            prompt = turn.get("prompt", "")
            expected_answer = turn.get("answer", "")
            task_type = turn.get("task_type", "factual")
            if not prompt or not expected_answer:
                continue

            # Compile pruning mask on first turn (if pruning)
            if is_first_turn and pruning_strength is not None:
                session_pruner = SessionPruner(
                    layer_probes=layer_probes,
                    domain_inference=domain_inference,
                    selected_domains=selected_domains,
                    retention_rate=0.6,
                    num_heads_per_layer=num_heads_per_layer,
                    pruning_strength=pruning_strength,
                    head_importance=head_importance,
                    whitelist=whitelist,
                )
                pruning_start = time.time()
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
                total_pruning_time += time.time() - pruning_start
                if pruning_metadata and "pruning_ratio" in pruning_metadata:
                    pruning_ratios.append(pruning_metadata["pruning_ratio"])

                _clear_mask_hooks(base_model)
                base_model._apply_mask_hooks(pruning_mask)
                is_first_turn = False
            elif not is_first_turn and pruning_mask is not None:
                if not getattr(base_model, "mask_hooks", None):
                    base_model._apply_mask_hooks(pruning_mask)

            if isinstance(expected_answer, list):
                expected_answer = expected_answer[0] if expected_answer else ""

            result = reasoning.reason(
                prompt=prompt,
                task_type=task_type,
                is_first_turn=(turn_idx == 0),
            )
            predicted_answer = result["result"]

            eval_result = evaluate_answer_accuracy(
                predicted_answer=predicted_answer,
                expected_answer=expected_answer,
                task_type=task_type,
            )
            accuracy = eval_result.get("accuracy", 0.0)

            total_turns += 1
            total_accuracy += accuracy
            accuracies.append(accuracy)
            total_inference_time += result.get("inference_time", 0.0)
            total_wall_time += result.get("wall_time", 0.0)
            peak_memory_mb = max(peak_memory_mb, result.get("peak_memory_mb_allocated", 0.0))

            if (scenario_idx + 1) % 100 == 0:
                avg_acc = total_accuracy / total_turns
                logger.info(f"  [{scenario_idx+1}/{len(scenarios)}] avg_accuracy={avg_acc:.4f} ({avg_acc*100:.2f}%)")

        _clear_mask_hooks(base_model)

    avg_accuracy = total_accuracy / total_turns if total_turns > 0 else 0.0
    avg_pruning_ratio = float(np.mean(pruning_ratios)) if pruning_ratios else None
    avg_inference_time = total_inference_time / total_turns if total_turns > 0 else 0.0
    avg_wall_time = total_wall_time / total_turns if total_turns > 0 else 0.0

    return {
        "total_turns": total_turns,
        "avg_accuracy": avg_accuracy,
        "token_level_recall": avg_accuracy * 100.0,
        "avg_inference_time": avg_inference_time,
        "avg_wall_time": avg_wall_time,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_gb": peak_memory_mb / 1024.0,
        "avg_pruning_ratio": avg_pruning_ratio,
        "total_pruning_time": total_pruning_time,
        "accuracies": accuracies,
        "pruning_ratios": pruning_ratios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--pruning_strength", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_pruned", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info("=" * 80)
    logger.info(f"Paper 1263 Reproduction — ARC Evaluation")
    logger.info(f"Model: {args.model_name}, pruning_strength={args.pruning_strength}")
    logger.info(f"ARC samples: {args.num_samples}")
    logger.info("=" * 80)

    # Load model
    model_path = project_root / "models" / args.model_name
    logger.info(f"Loading model from: {model_path}")
    base_model = BaseModel(
        model_name=str(model_path),
        quantization="none",
        torch_dtype="bfloat16",
    )

    # Load offline artifacts
    ppd_output_dir = get_output_dir_for_model(
        base_output_dir=str(project_root / "outputs"),
        model_path=str(model_path),
        subdir="ppd_pipeline",
    )
    logger.info(f"Loading artifacts from: {ppd_output_dir}")

    probe1_path = ppd_output_dir / "probe1_base.pt"
    base_probe = MultiLayerProbe.load(str(probe1_path), device=base_model.device)
    logger.info(f"Loaded base probes: {base_probe.num_domains} domains, {len(base_probe.probes)} layers")

    head_importance_path = ppd_output_dir / "head_importance.pt"
    head_importance = HeadImportanceCalculator.load(str(head_importance_path))

    whitelist = []
    whitelist_path = ppd_output_dir / "whitelist.json"
    if whitelist_path.exists():
        whitelist = HeadWhitelistIdentifier.load_whitelist(str(whitelist_path))
        logger.info(f"Loaded whitelist: {len(whitelist)} heads")

    selected_domains = list(range(base_probe.num_domains))
    domain_inference = DomainInference(
        num_domains=base_probe.num_domains,
        min_probability_threshold=0.05,
    )

    num_heads = base_model.model.config.num_attention_heads
    num_layers = base_model.get_num_layers()
    logger.info(f"Model: {num_layers} layers, {num_heads} heads/layer, {num_layers * num_heads} total heads")

    # Load ARC scenarios
    arc_dir = project_root / "data" / "test" / "arc"
    scenarios = load_arc_scenarios(arc_dir, num_samples=args.num_samples)
    logger.info(f"Loaded {len(scenarios)} ARC scenarios")

    reasoning = BaselineReasoning(
        base_model=base_model,
        max_steps=1,
        use_history=False,
        max_history_turns=0,
    )

    results = {}

    # --- DENSE (no pruning) ---
    if not args.skip_dense:
        logger.info("\n" + "=" * 80)
        logger.info("DENSE BASELINE (no pruning)")
        logger.info("=" * 80)
        _clear_mask_hooks(base_model)
        dense_result = evaluate_scenarios(
            base_model=base_model,
            reasoning=reasoning,
            scenarios=scenarios,
            pruning_strength=None,
        )
        results["dense"] = dense_result
        logger.info(f"Dense result: Token-level Recall = {dense_result['token_level_recall']:.2f}%")
        logger.info(f"Dense: avg_inference_time = {dense_result['avg_inference_time']:.4f}s")
        logger.info(f"Dense: peak_memory = {dense_result['peak_memory_gb']:.2f} GB")

    # --- PRUNED (eta=0.5) ---
    if not args.skip_pruned:
        logger.info("\n" + "=" * 80)
        logger.info(f"PRUNED (pruning_strength={args.pruning_strength})")
        logger.info("=" * 80)
        _clear_mask_hooks(base_model)
        pruned_result = evaluate_scenarios(
            base_model=base_model,
            reasoning=reasoning,
            scenarios=scenarios,
            pruning_strength=args.pruning_strength,
            layer_probes=base_probe,
            domain_inference=domain_inference,
            selected_domains=selected_domains,
            head_importance=head_importance,
            whitelist=whitelist,
            num_heads_per_layer=num_heads,
        )
        results["pruned"] = pruned_result
        logger.info(f"Pruned result: Token-level Recall = {pruned_result['token_level_recall']:.2f}%")
        logger.info(f"Pruned: avg_inference_time = {pruned_result['avg_inference_time']:.4f}s")
        logger.info(f"Pruned: peak_memory = {pruned_result['peak_memory_gb']:.2f} GB")
        logger.info(f"Pruned: avg_pruning_ratio = {pruned_result['avg_pruning_ratio']}")
        if pruned_result['avg_pruning_ratio'] is not None:
            logger.info(f"Pruned: head_retention = {(1 - pruned_result['avg_pruning_ratio']) * 100:.1f}%")

    # --- Compute derived metrics ---
    if "dense" in results and "pruned" in results:
        d = results["dense"]
        p = results["pruned"]

        speedup = d["avg_inference_time"] / p["avg_inference_time"] if p["avg_inference_time"] > 0 else 0.0
        retention = (p["token_level_recall"] / d["token_level_recall"]) * speedup if d["token_level_recall"] > 0 else 0.0

        logger.info("\n" + "=" * 80)
        logger.info("DERIVED METRICS")
        logger.info("=" * 80)
        logger.info(f"Token-level Recall (pruned): {p['token_level_recall']:.2f}%")
        logger.info(f"Token-level Recall (dense):  {d['token_level_recall']:.2f}%")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Memory (pruned): {p['peak_memory_gb']:.2f} GB")
        logger.info(f"Memory (dense):  {d['peak_memory_gb']:.2f} GB")
        logger.info(f"Retention: {retention:.2f}")
        if p.get("avg_pruning_ratio") is not None:
            logger.info(f"Head retention: {(1 - p['avg_pruning_ratio']) * 100:.1f}%")
        logger.info(f"Total turns evaluated: {p['total_turns']}")

        # Compute SEM
        if len(p.get("accuracies", [])) > 1:
            sem = np.std(p["accuracies"]) / np.sqrt(len(p["accuracies"]))
            logger.info(f"SEM (pruned recall): {sem * 100:.2f}")

    # Save results
    output_file = project_root / "outputs" / args.model_name / "reproduction_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        sv = dict(v)
        if "accuracies" in sv:
            sv["accuracies"] = [float(a) for a in sv["accuracies"]]
        if "pruning_ratios" in sv:
            sv["pruning_ratios"] = [float(r) for r in sv["pruning_ratios"]]
        serializable[k] = sv

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
