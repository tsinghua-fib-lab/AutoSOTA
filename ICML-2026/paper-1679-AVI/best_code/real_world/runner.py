

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RealModeConfig
from .environment import LLMComparisonEnvironment

logger = logging.getLogger(__name__)


def load_checkpoint_from_csv(
    csv_path: str,
    model_names: list[str],
    output_dir: str = None
) -> Optional[Dict[str, Any]]:

    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        if output_dir:
            csv_file = Path(output_dir) / csv_file.name
            if not csv_file.exists():
                logger.warning(f"Checkpoint file not found: {csv_path}")
                logger.warning(f"Also not found in output directory: {csv_file}")
                logger.info("Starting experiment from scratch")
                return None
            else:
                logger.info(f"✅ Found checkpoint in output directory: {csv_file}")
        else:
            logger.info(f"Checkpoint file not found: {csv_path}, starting experiment from scratch")
            return None
    
    csv_file = csv_file.absolute()
    
    try:
        logger.info(f"Loading checkpoint from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning("CSV file is empty, starting experiment from scratch")
            return None
        
        m = len(model_names)
        model_to_idx = {name: idx for idx, name in enumerate(model_names)}
        
        win_records = np.zeros((m, m))
        total_comparisons = np.zeros((m, m))
        
        interactions = []
        for _, row in df.iterrows():
            model_j_name = row['model_j']
            model_i_name = row['model_i']
            winner = row['winner']
            
            if model_j_name not in model_to_idx or model_i_name not in model_to_idx:
                logger.warning(f"Skipping unknown model record: {model_j_name} vs {model_i_name}")
                continue
            
            j_idx = model_to_idx[model_j_name]
            i_idx = model_to_idx[model_i_name]
            
            if winner == 'model_j':
                win_records[j_idx, i_idx] += 1
            elif winner == 'model_i':
                win_records[i_idx, j_idx] += 1
            
            total_comparisons[j_idx, i_idx] += 1
            total_comparisons[i_idx, j_idx] += 1
            
            interactions.append(row.to_dict())
        
        last_t = int(df['time_index'].max())
        
        logger.info(f"✅ Successfully loaded checkpoint:")
        logger.info(f"   - Completed time steps: {last_t}")
        logger.info(f"   - Interactions count: {len(interactions)}")
        logger.info(f"   - Total comparisons: {int(total_comparisons.sum() / 2)}")
        
        timestamp_match = csv_file.stem.replace('interactions_', '')
        
        return {
            'last_t': last_t,
            'win_records': win_records,
            'total_comparisons': total_comparisons,
            'interactions': interactions,
            'checkpoint_csv_path': str(csv_file),
            'checkpoint_timestamp': timestamp_match,
            'checkpoint_dir': str(csv_file.parent),
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.warning("Starting experiment from scratch")
        return None


def run_real_mode_experiment(
    config: RealModeConfig,
    checkpoint_csv: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run SERPANT experiment in real mode.

    Args:
        config: RealModeConfig configuration object
        checkpoint_csv: Checkpoint CSV file path (for resume)

    Returns:
        Dictionary containing experiment results
    """
    if config.use_covariates:
        from core import serpant_algorithm_covariate as _serpant_algo
        logger.info("=== Starting real mode experiment (COVARIATE mode) ===")
    else:
        from core import serpant_algorithm as _serpant_algo
        logger.info("=== Starting real mode experiment ===")
    logger.info(f"Model count: {len(config.models)}")
    logger.info(f"Sampling method: {config.sampling_method}")
    logger.info(f"Max steps: {config.max_t}")
    logger.info(f"Alpha: {config.alpha}")
    if config.top_k:
        logger.info(f"Top-K: {config.top_k}")

    env = LLMComparisonEnvironment(config)
    m = len(config.models)
    model_names = [client.name for client in env.model_clients]
    
    checkpoint_data = None
    serpant_checkpoint = None
    resume_mode = False
    checkpoint_timestamp = None
    checkpoint_output_dir = None
    
    if checkpoint_csv:
        checkpoint_data = load_checkpoint_from_csv(
            checkpoint_csv, 
            model_names,
            output_dir=config.output.dir
        )
        if checkpoint_data:
            env.interactions = checkpoint_data['interactions']
            
            from core.e_value import compute_e_value
            from core.transitivity import propagate_transitivity
            
            K = m * (m - 1) / (2 * config.alpha)
            R = np.zeros((m, m), dtype=bool)
            
            win_records = checkpoint_data['win_records']
            total_comparisons = checkpoint_data['total_comparisons']
            
            for j in range(m):
                for i in range(m):
                    if i != j and total_comparisons[j, i] > 0:
                        e_value = compute_e_value(win_records[j, i], total_comparisons[j, i])
                        if e_value >= K:
                            R[j, i] = True
            
            direct_rejected = int(R.sum())
            logger.info(f"   - Direct comparisons restored: {direct_rejected} rejected hypotheses")
            
            prev_count = direct_rejected
            while True:
                T = propagate_transitivity(R, np.zeros((m, m), dtype=bool), m)
                R = R | T
                
                current_count = int(R.sum())
                if current_count == prev_count: 
                    break
                prev_count = current_count
            
            transitive_rejected = int(R.sum()) - direct_rejected
            total_rejected = int(R.sum())
            
            if transitive_rejected > 0:
                logger.info(f"   - Transitive derivation added: {transitive_rejected} hypotheses")
            logger.info(f"   - Total restored R matrix: {total_rejected} rejected hypotheses")
            
            serpant_checkpoint = {
                'start_t': checkpoint_data['last_t'],
                'successes': checkpoint_data['win_records'],
                'trials': checkpoint_data['total_comparisons'],
                'R': R
            }
            
            resume_mode = True
            checkpoint_timestamp = checkpoint_data['checkpoint_timestamp']
            checkpoint_output_dir = checkpoint_data['checkpoint_dir']
            
            logger.info(f"✅ Resume mode")
            logger.info(f"   - Continue from time step t={checkpoint_data['last_t'] + 1}")
            logger.info(f"   - Output will be appended to original file: {checkpoint_data['checkpoint_csv_path']}")
            logger.info(f"   - Timestamp: {checkpoint_timestamp}")

    def observation_fn(j: int, i: int, t: Optional[int] = None) -> int:

        result = env.compare(j, i, t=t, sampling_method=config.sampling_method)
        obs = result["obs"]
        
        if config.verbose and t is not None and t % 10 == 0:
            logger.info(
                f"[t={t}] {result['model_j']} vs {result['model_i']}: "
                f"winner={result['winner']} (obs={obs})"
            )
        
        return obs

    logger.info("\nRunning SERPANT algorithm...")
    start_time = time.time()

    serpant_kwargs = dict(
        m=m,
        alpha=config.alpha,
        true_probs=observation_fn,
        max_t=config.max_t,
        sampling_method=config.sampling_method,
        verbose=config.verbose,
        max_tournament_samples=config.max_tournament_samples,
        top_k=config.top_k,
    )
    if serpant_checkpoint is not None:
        serpant_kwargs['checkpoint'] = serpant_checkpoint
    if config.use_covariates:
        # Build covariate info from model parameter counts
        MODEL_PARAMS = {
            "Qwen/Qwen2.5-1.5B-Instruct": 1.54e9,
            "Qwen/Qwen2.5-3B-Instruct": 3.09e9,
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.10e9,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 1.54e9,
            "openai-community/gpt2": 0.124e9,
        }
        param_counts = []
        for client in env.model_clients:
            params = MODEL_PARAMS.get(client.name, 1e9)
            param_counts.append(params)
        covariate_info = {"X": np.array(param_counts)}
        serpant_kwargs["covariate_info"] = covariate_info
        serpant_kwargs["theta_update_interval"] = config.theta_update_interval
    results = _serpant_algo(**serpant_kwargs)

    elapsed_time = time.time() - start_time
    logger.info(f"\nAlgorithm completed, time elapsed: {elapsed_time:.2f} seconds")

    from core.algorithm import update_model_strength
    win_records = np.zeros((m, m))
    total_comparisons = np.zeros((m, m))
    
    for interaction in env.interactions:
        j_idx = next(i for i, client in enumerate(env.model_clients) if client.name == interaction['model_j'])
        i_idx = next(i for i, client in enumerate(env.model_clients) if client.name == interaction['model_i'])
        
        if interaction['winner'] == 'model_j':
            win_records[j_idx, i_idx] += 1
        elif interaction['winner'] == 'model_i':
            win_records[i_idx, j_idx] += 1
        
        total_comparisons[j_idx, i_idx] += 1
        total_comparisons[i_idx, j_idx] += 1
    
    final_strengths = update_model_strength(win_records, total_comparisons, m)
    results['final_strengths'] = final_strengths
    results['win_records'] = win_records
    results['total_comparisons'] = total_comparisons
    
    rejected_hypotheses = []
    if 'final_rejected' in results:
        rejected_pairs = np.argwhere(results['final_rejected'])
        rejected_hypotheses = [(int(i), int(j)) for i, j in rejected_pairs]
    results['rejected_hypotheses'] = rejected_hypotheses

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_csv and checkpoint_data:
        checkpoint_csv_path = Path(checkpoint_csv)
        start_t = checkpoint_data['last_t']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Resume mode: append new records to original file")
        logger.info(f"  - Original file: {checkpoint_csv_path.name}")
        logger.info(f"  - Existing records: t=1 to t={start_t}")
        logger.info(f"  - New records: t={start_t+1} to t={results.get('t', 'N/A')}")
        logger.info(f"{'='*70}\n")
        
        if config.output.save_csv:
            csv_path = env.save_interactions_csv(
                output_dir, 
                append_to_file=str(checkpoint_csv_path),
                start_index=start_t
            )
            logger.info(f"✅ CSV records appended to: {csv_path}")
        
        if config.output.save_interactions:
            jsonl_filename = checkpoint_csv_path.stem + '.jsonl'
            jsonl_path = checkpoint_csv_path.parent / jsonl_filename
            interactions_path = env.save_interactions(
                output_dir,
                append_to_file=str(jsonl_path),
                start_index=start_t
            )
            logger.info(f"✅ JSONL records appended to: {interactions_path}")
    else:
        if config.output.save_interactions:
            interactions_path = env.save_interactions(output_dir)
            logger.info(f"Interactions saved (JSONL): {interactions_path}")

        if config.output.save_csv:
            csv_path = env.save_interactions_csv(output_dir)
            logger.info(f"Interactions saved (CSV): {csv_path}")

    if rejected_hypotheses:
        dag_path = env.save_partial_order_graph(output_dir, rejected_hypotheses, format='png')
        if dag_path:
            logger.info(f"Partial order DAG saved: {dag_path}")
        
        dot_path = env.save_partial_order_graph(output_dir, rejected_hypotheses, format='dot')
        if dot_path:
            logger.info(f"Partial order DOT saved: {dot_path}")

    if config.output.save_config_snapshot:
        config_path = env.save_config_snapshot(output_dir)
        logger.info(f"Config snapshot saved: {config_path}")

    if config.output.save_serpant_trace:
        serpant_path = _save_serpant_results(results, output_dir, config)
        logger.info(f"SERPANT results saved: {serpant_path}")

    _print_final_ranking(results, env)

    return {
        "serpant_results": results,
        "environment_summary": env.summary(),
        "elapsed_time": elapsed_time,
        "output_dir": str(output_dir),
    }


def _convert_to_serializable(obj):
    """Recursive conversion of object to JSON serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, type(None))):
        return obj
    else:
        return str(obj)


def _save_serpant_results(
    results: Dict[str, Any],
    output_dir: Path,
    config: RealModeConfig,
) -> Path:
    """Save SERPANT algorithm results."""
    import json
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"serpant_results_{timestamp}.json"

    serializable_results = _convert_to_serializable(results)

    model_names = [m.name for m in config.models]
    serializable_results["model_names"] = model_names
    
    if "rejected_hypotheses" in serializable_results:
        partial_order_readable = []
        for i, j in serializable_results["rejected_hypotheses"]:
            name_i_full = model_names[i]
            name_j_full = model_names[j]
            name_i_short = name_i_full.split('/')[-1]
            name_j_short = name_j_full.split('/')[-1]
            
            partial_order_readable.append({
                "index_stronger": i,
                "index_weaker": j,
                "model_stronger": name_i_full,
                "model_weaker": name_j_full,
                "model_stronger_short": name_i_short,
                "model_weaker_short": name_j_short,
                "relation": f"{name_i_short} > {name_j_short}",
                "relation_full": f"{name_i_full} > {name_j_full}",
                "description": f"{name_i_short} significantly better than {name_j_short}"
            })
        serializable_results["partial_order_readable"] = partial_order_readable
    
    m = len(model_names)
    total_possible = m * (m - 1) // 2
    discovered = len(serializable_results.get("rejected_hypotheses", []))
    serializable_results["summary"] = {
        "num_models": m,
        "total_possible_pairs": total_possible,
        "discovered_partial_order_pairs": discovered,
        "discovery_rate": f"{discovered/total_possible*100:.1f}%",
        "alpha": float(config.alpha),
        "fwer_guarantee": f"≤ {config.alpha:.2f}",
        "note": "Partial order relations have statistical guarantees, heuristic ranking仅供参考"
    }

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    return file_path


def _print_final_ranking(results: Dict[str, Any], env: LLMComparisonEnvironment) -> None:
    """Print final partial order relation results."""
    logger.info("\n" + "=" * 70)
    logger.info("=== SERPANT algorithm results: discovered partial order relations ===")
    logger.info("=" * 70)
    
    model_names = [client.name for client in env.model_clients]
    m = len(model_names)
    
    if "rejected_hypotheses" in results:
        rejected = results["rejected_hypotheses"]
        logger.info(f"\n✅ Discovered partial order pairs: {len(rejected)} / {m*(m-1)//2} ({len(rejected)/(m*(m-1)//2)*100:.1f}%)")
        logger.info(f"   (FWER ≤ {results.get('alpha', 0.1):.2f} statistical guarantee)\n")
        
        if rejected:
            logger.info("Partial order relation list (format: model A > model B, means A significantly better than B):")
            logger.info("-" * 70)
            for idx, (i, j) in enumerate(rejected, 1):
                name_i = model_names[i].split('/')[-1]
                name_j = model_names[j].split('/')[-1]
                logger.info(f"  {idx:3d}. {name_i:35s} > {name_j:35s}  (model {i} > model {j})")
            logger.info("-" * 70)
            logger.info("Note: The above partial order relations have statistical guarantees (FWER controlled)")
        else:
            logger.info("⚠️  No partial order pairs with statistical significance found")
            logger.info("Increase max_t or check if the models actually exist differences")
    

    if "final_rejected" in results:
        R = results["final_rejected"]
        direct_comparisons = np.sum(results.get('trials', np.zeros_like(R)) > 0)
        total_pairs = np.sum(R)
        inferred_pairs = total_pairs - len(results.get("rejected_hypotheses", []))
        
        logger.info(f"\n📊 Partial order discovery statistics:")
        logger.info(f"   - Number of model pairs compared: {direct_comparisons}")
        logger.info(f"   - Directly discovered partial order pairs: {len(results.get('rejected_hypotheses', []))}")
        logger.info(f"   - Transitive derived partial order pairs: {inferred_pairs}")
        logger.info(f"   - Total partial order pairs: {total_pairs}")
    
    if "topk_set" in results and results.get("top_k"):
        topk_set = results["topk_set"]
        k = results["top_k"]
        logger.info(f"\n🏆 Top-{k} confidence set:")
        logger.info(f"   - Confidence set size: {len(topk_set)}")
        if len(topk_set) == k:
            logger.info("   ✅ Perfect recognition! Confidence set size = k")
        elif len(topk_set) > k:
            logger.info(f"   ⚠️  Confidence set contains {len(topk_set)} models (greater than k={k})")
            logger.info("   More comparisons are needed to accurately determine Top-K")
        
        logger.info(f"\n   Top-{k} candidate models:")
        for idx in topk_set:
            name = model_names[idx].split('/')[-1]
            logger.info(f"   - {name}")
    
    if "final_strengths" in results:
        logger.info("\n" + "=" * 70)
        logger.info("📋 Reference: Heuristic ranking (based on Bradley-Terry scores, no statistical guarantee)")
        logger.info("=" * 70)
        strengths = results["final_strengths"]
        ranking = np.argsort(-strengths)
        
        for rank, idx in enumerate(ranking, 1):
            name = model_names[idx].split('/')[-1]
            logger.info(f"  #{rank}: {name:30s} (score: {strengths[idx]:.4f})")
        
        logger.info("\n⚠️  Note: This ranking is heuristic.")
        logger.info("   Reliable conclusions should be based on the above partial order relations.")
    
    logger.info("\n" + "=" * 70)


def quick_test_real_mode(
    model_names: list[str],
    questions: list[str],
    alpha: float = 0.1,
    max_t: int = 50,
    sampling_method: str = "random_pair",
    judge_type: str = "heuristic",
) -> Dict[str, Any]:

    from .config import RealModeConfig, ModelConfig, JudgeConfig, OutputConfig

    models = [
        ModelConfig(name=name, provider="stub")
        for name in model_names
    ]

    config = RealModeConfig(
        alpha=alpha,
        max_t=max_t,
        sampling_method=sampling_method,
        questions=questions,
        models=models,
        judge=JudgeConfig(type=judge_type),
        output=OutputConfig(dir="real_results/quick_test"),
        verbose=True,
    )

    return run_real_mode_experiment(config)

