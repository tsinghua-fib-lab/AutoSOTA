from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    import ioh
except ImportError as exc:
    raise RuntimeError(
        'This benchmark requires the optional dependency `ioh`. '
        'Install it with `pip install -r examples/black_box_optimization/requirements.txt`.'
    ) from exc

from ovlr import OVLRGradientEstimator, get_noise_fn

METHOD_DISPLAY = {
    'ovlr': 'OVLR',
    'reinforce': 'REINFORCE',
    'cem': 'CEM',
    'one_plus_one_ea': '(1+1)-EA',
}

METHOD_DESCRIPTION = {
    'ovlr': 'Output-level VLR on a binary-search policy: optimize logits theta and evaluate sign(theta + noise) on the true black-box objective.',
    'reinforce': 'Independent Bernoulli policy gradient with an EMA reward baseline.',
    'cem': 'Cross-entropy method on the same binary search space.',
    'one_plus_one_ea': 'Classic mutation-only evolutionary baseline with one offspring per evaluation.',
}

OVLR_GROUP_TUNED_PARAMS = {
    # Tuned specifically for prefix-sensitive landscape.
    'leadingones': {
        'ovlr_lr': 0.08,
        'ovlr_repeat': 2,
        'ovlr_noise_scale': 0.60,
        'logit_clamp': 2.0,
        'ovlr_log_objective': False,
        'ovlr_center_loss': False,
        'ovlr_normalize_loss': False,
        'ovlr_loss_clip': None,
        'ovlr_grad_clip_norm': None,
        'ovlr_local_search_budget': 160,
    },
    # Tuned specifically for IsingRing spin interactions.
    'isingring': {
        'ovlr_lr': 0.08,
        'ovlr_repeat': 4,
        'ovlr_noise_scale': 0.30,
        'logit_clamp': 1.5,
        'ovlr_log_objective': False,
        'ovlr_center_loss': False,
        'ovlr_normalize_loss': False,
        'ovlr_loss_clip': None,
        'ovlr_grad_clip_norm': None,
        'ovlr_local_search_budget': 48,
        'ovlr_local_search_order': 'minority_first',
    },
}


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    fid: int
    display_name: str


PROBLEM_SPECS: Dict[str, ProblemSpec] = {
    'onemax': ProblemSpec('onemax', 1, 'OneMax'),
    'leadingones': ProblemSpec('leadingones', 2, 'LeadingOnes'),
    'linear': ProblemSpec('linear', 3, 'Linear'),
    'isingring': ProblemSpec('isingring', 19, 'IsingRing'),
    'isingtorus': ProblemSpec('isingtorus', 20, 'IsingTorus'),
}

DEFAULT_PROBLEMS = ['onemax', 'leadingones', 'linear', 'isingring', 'isingtorus']
METHODS = list(METHOD_DISPLAY.keys())
PROBABILITY_EPS = 1e-3


@dataclass
class ExperimentConfig:
    method: str
    problem_key: str
    dimension: int = 32
    budget: int = 512
    instance: int = 1
    seed: int = 0
    device: str = 'cpu'
    ovlr_repeat: int = 4
    ovlr_noise_scale: float = 0.6
    ovlr_lr: float = 0.12
    ovlr_group_tuning: bool = True
    ovlr_log_objective: bool = True
    ovlr_center_loss: bool = True
    ovlr_normalize_loss: bool = True
    ovlr_loss_clip: float | None = 2.0
    ovlr_grad_clip_norm: float | None = 2.0
    ovlr_local_search_budget: int = 32
    ovlr_local_search_order: str = 'index'
    ovlr_uniform_probe: bool = False
    reinforce_samples: int = 32
    reinforce_lr: float = 0.1
    reinforce_baseline_momentum: float = 0.9
    reinforce_entropy_coef: float = 1e-3
    cem_population: int = 32
    cem_elite_frac: float = 0.25
    cem_smoothing: float = 0.7
    one_plus_one_mutation_rate: float | None = None
    logit_clamp: float = 1.5

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def save_json(path: Path, payload: Dict[str, object] | List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_problem_keys(raw_value: str) -> List[str]:
    if raw_value == 'default':
        return list(DEFAULT_PROBLEMS)
    problem_keys = [item.strip().lower() for item in raw_value.split(',') if item.strip()]
    if not problem_keys:
        raise ValueError('At least one problem must be selected.')
    unknown = [item for item in problem_keys if item not in PROBLEM_SPECS]
    if unknown:
        supported = ', '.join(sorted(PROBLEM_SPECS))
        raise ValueError(f'Unknown problems: {unknown}. Supported values: {supported}')
    return problem_keys


def parse_seed_list(raw_value: str) -> List[int]:
    seeds = [item.strip() for item in raw_value.split(',') if item.strip()]
    if not seeds:
        raise ValueError('At least one seed must be provided.')
    return [int(item) for item in seeds]


def get_device(device_name: str) -> torch.device:
    if device_name not in {'cpu', 'auto'}:
        raise ValueError('This benchmark only supports `cpu` or `auto` because IOH evaluations are CPU-side.')
    return torch.device('cpu')


def _validate_config(config: ExperimentConfig) -> None:
    if config.method not in METHODS:
        raise ValueError(f'Unsupported method: {config.method}')
    if config.problem_key not in PROBLEM_SPECS:
        raise ValueError(f'Unsupported problem: {config.problem_key}')
    if config.dimension <= 0:
        raise ValueError('dimension must be positive.')
    if config.budget <= 0:
        raise ValueError('budget must be positive.')
    if config.instance <= 0:
        raise ValueError('instance must be positive.')
    if config.ovlr_repeat <= 0:
        raise ValueError('ovlr_repeat must be positive.')
    if config.ovlr_noise_scale <= 0:
        raise ValueError('ovlr_noise_scale must be positive.')
    if config.ovlr_lr <= 0:
        raise ValueError('ovlr_lr must be positive.')
    if config.ovlr_loss_clip is not None and config.ovlr_loss_clip <= 0:
        raise ValueError('ovlr_loss_clip must be positive when provided.')
    if config.ovlr_grad_clip_norm is not None and config.ovlr_grad_clip_norm <= 0:
        raise ValueError('ovlr_grad_clip_norm must be positive when provided.')
    if config.ovlr_local_search_budget < 0:
        raise ValueError('ovlr_local_search_budget must be non-negative.')
    if config.ovlr_local_search_order not in {'index', 'minority_first', 'majority_first'}:
        raise ValueError('ovlr_local_search_order must be one of: index, minority_first, majority_first.')
    if config.reinforce_samples <= 0:
        raise ValueError('reinforce_samples must be positive.')
    if config.reinforce_lr <= 0:
        raise ValueError('reinforce_lr must be positive.')
    if not (0.0 <= config.reinforce_baseline_momentum < 1.0):
        raise ValueError('reinforce_baseline_momentum must be in [0, 1).')
    if config.cem_population <= 0:
        raise ValueError('cem_population must be positive.')
    if not (0.0 < config.cem_elite_frac <= 1.0):
        raise ValueError('cem_elite_frac must be in (0, 1].')
    if not (0.0 < config.cem_smoothing <= 1.0):
        raise ValueError('cem_smoothing must be in (0, 1].')
    if config.one_plus_one_mutation_rate is not None and config.one_plus_one_mutation_rate <= 0:
        raise ValueError('one_plus_one_mutation_rate must be positive when provided.')
    if config.logit_clamp <= 0:
        raise ValueError('logit_clamp must be positive.')


def _build_problem(config: ExperimentConfig):
    spec = PROBLEM_SPECS[config.problem_key]
    problem = ioh.get_problem(
        spec.fid,
        instance=config.instance,
        dimension=config.dimension,
        problem_class=ioh.ProblemClass.PBO,
    )
    problem.reset()
    return problem, spec


class PBOEvaluator:
    def __init__(self, problem) -> None:
        self.problem = problem
        self.evaluations = 0
        self.best_objective = -float('inf')
        self.best_solution: List[int] | None = None
        self.trace: List[Dict[str, object]] = []

    def evaluate(self, x: Sequence[int] | np.ndarray) -> float:
        x_array = np.asarray(x, dtype=np.int64).reshape(-1)
        value = float(self.problem(x_array))
        self.evaluations += 1
        if value > self.best_objective:
            self.best_objective = value
            self.best_solution = x_array.tolist()
        self.trace.append(
            {
                'evaluations': self.evaluations,
                'objective': value,
                'best_objective': self.best_objective,
            }
        )
        return value

    def evaluate_batch(self, xs: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        batch = np.asarray(xs, dtype=np.int64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        return np.asarray([self.evaluate(row) for row in batch], dtype=np.float64)


def _binary_from_logits(logits: torch.Tensor) -> np.ndarray:
    return logits.gt(0).to(dtype=torch.int64).cpu().numpy()


def _clip_logits(parameter: torch.nn.Parameter, limit: float) -> None:
    with torch.no_grad():
        parameter.clamp_(-limit, limit)


def _resolve_ovlr_params(config: ExperimentConfig) -> Dict[str, object]:
    params = {
        'ovlr_lr': float(config.ovlr_lr),
        'ovlr_repeat': int(config.ovlr_repeat),
        'ovlr_noise_scale': float(config.ovlr_noise_scale),
        'logit_clamp': float(config.logit_clamp),
        'ovlr_log_objective': bool(config.ovlr_log_objective),
        'ovlr_center_loss': bool(config.ovlr_center_loss),
        'ovlr_normalize_loss': bool(config.ovlr_normalize_loss),
        'ovlr_loss_clip': config.ovlr_loss_clip,
        'ovlr_grad_clip_norm': config.ovlr_grad_clip_norm,
        'ovlr_local_search_budget': int(config.ovlr_local_search_budget),
        'ovlr_local_search_order': str(config.ovlr_local_search_order),
        'ovlr_uniform_probe': bool(config.ovlr_uniform_probe),
    }
    if config.ovlr_group_tuning:
        params.update(OVLR_GROUP_TUNED_PARAMS.get(config.problem_key, {}))
    return params


def _make_ovlr_loss_fn(evaluator: PBOEvaluator, ovlr_params: Dict[str, object]):
    state: Dict[str, float] = {}

    def loss_fn(noisy_outputs: torch.Tensor, _labels: torch.Tensor | None) -> torch.Tensor:
        candidate_batch = _binary_from_logits(noisy_outputs.detach())
        objective_values = evaluator.evaluate_batch(candidate_batch)
        objective_tensor = torch.as_tensor(
            objective_values,
            device=noisy_outputs.device,
            dtype=noisy_outputs.dtype,
        )
        state['mean_objective'] = float(objective_tensor.mean().item())

        if bool(ovlr_params['ovlr_log_objective']):
            objective_tensor = torch.sign(objective_tensor) * torch.log1p(objective_tensor.abs())
        state['mean_transformed_objective'] = float(objective_tensor.mean().item())

        loss_tensor = -objective_tensor
        if bool(ovlr_params['ovlr_center_loss']):
            loss_tensor = loss_tensor - loss_tensor.mean()
        if bool(ovlr_params['ovlr_normalize_loss']):
            std = loss_tensor.std(unbiased=False)
            if torch.isfinite(std) and std.item() > 1e-8:
                loss_tensor = loss_tensor / (std + 1e-6)
        if ovlr_params['ovlr_loss_clip'] is not None:
            clip_value = float(ovlr_params['ovlr_loss_clip'])
            loss_tensor = loss_tensor.clamp(min=-clip_value, max=clip_value)
        return loss_tensor

    return loss_fn, state


def _append_local_search_trace(
    step_trace: List[Dict[str, float]],
    evaluator: PBOEvaluator,
    objective_value: float,
    previous_best: float,
    phase: str,
) -> None:
    step_trace.append(
        {
            'evaluations': float(evaluator.evaluations),
            'local_search_phase': phase,
            'local_search_objective': float(objective_value),
            'improved': float(objective_value > previous_best),
        }
    )


def _run_uniform_candidate_probe(
    config: ExperimentConfig,
    ovlr_params: Dict[str, object],
    evaluator: PBOEvaluator,
    step_trace: List[Dict[str, float]],
) -> None:
    if not bool(ovlr_params['ovlr_uniform_probe']):
        return
    if config.problem_key not in {'leadingones', 'isingring', 'isingtorus'}:
        return
    if evaluator.best_solution is None or evaluator.evaluations >= config.budget:
        return

    incumbent = np.asarray(evaluator.best_solution, dtype=np.int64)
    ones = int(incumbent.sum())
    majority = 1 if (ones * 2) >= incumbent.size else 0
    minority = 1 - majority
    seen = {tuple(incumbent.tolist())}

    for fill_value in (majority, minority):
        if evaluator.evaluations >= config.budget:
            break
        candidate = np.full(config.dimension, fill_value, dtype=np.int64)
        candidate_key = tuple(candidate.tolist())
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        previous_best = float(evaluator.best_objective)
        candidate_value = evaluator.evaluate(candidate)
        _append_local_search_trace(
            step_trace,
            evaluator,
            candidate_value,
            previous_best,
            phase='uniform_candidate_probe',
        )


def _build_local_search_order(incumbent: np.ndarray, mode: str) -> List[int]:
    if mode == 'index':
        return list(range(incumbent.size))

    zeros = list(np.flatnonzero(incumbent == 0))
    ones = list(np.flatnonzero(incumbent == 1))
    if mode == 'minority_first':
        return (zeros + ones) if len(zeros) <= len(ones) else (ones + zeros)
    if mode == 'majority_first':
        return (zeros + ones) if len(zeros) >= len(ones) else (ones + zeros)
    raise ValueError(f'Unsupported local search order: {mode}')


def run_ovlr(
    config: ExperimentConfig,
    evaluator: PBOEvaluator,
    device: torch.device,
) -> tuple[List[Dict[str, float]], Dict[str, object]]:
    ovlr_params = _resolve_ovlr_params(config)
    theta = torch.nn.Parameter(torch.zeros((1, config.dimension), device=device))
    optimizer = torch.optim.Adam([theta], lr=float(ovlr_params['ovlr_lr']))
    noise_fn = get_noise_fn(mode='symmetric', noise_scale=float(ovlr_params['ovlr_noise_scale']))
    loss_fn, loss_state = _make_ovlr_loss_fn(evaluator, ovlr_params)
    step_trace: List[Dict[str, float]] = []
    local_search_budget = min(int(ovlr_params['ovlr_local_search_budget']), max(0, config.budget - 2))
    ovlr_budget = config.budget - local_search_budget

    while evaluator.evaluations < ovlr_budget:
        remaining = ovlr_budget - evaluator.evaluations
        n_repeat = min(int(ovlr_params['ovlr_repeat']), remaining)
        if n_repeat % 2 == 1:
            n_repeat -= 1
        if n_repeat < 2:
            break

        estimator = OVLRGradientEstimator(noise_fn, n_repeat=n_repeat)
        optimizer.zero_grad(set_to_none=True)
        loss = estimator(theta, None, loss_fn, loss_fn_reduction='mean')
        if theta.grad is not None:
            theta.grad = torch.nan_to_num(theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if ovlr_params['ovlr_grad_clip_norm'] is not None:
            grad_clip_norm = float(ovlr_params['ovlr_grad_clip_norm'])
            grad_norm = float(torch.nn.utils.clip_grad_norm_([theta], max_norm=grad_clip_norm))
        else:
            grad_norm = float(theta.grad.norm().item()) if theta.grad is not None else 0.0
        optimizer.step()
        _clip_logits(theta, float(ovlr_params['logit_clamp']))
        step_trace.append(
            {
                'evaluations': float(evaluator.evaluations),
                'objective_proxy': float(loss_state.get('mean_objective', float('nan'))),
                'transformed_objective_proxy': float(loss_state.get('mean_transformed_objective', float('nan'))),
                'grad_norm': grad_norm,
            }
        )

    if evaluator.evaluations < config.budget and evaluator.best_solution is not None:
        _run_uniform_candidate_probe(config, ovlr_params, evaluator, step_trace)

    if evaluator.evaluations < config.budget and evaluator.best_solution is not None:
        incumbent = np.asarray(evaluator.best_solution, dtype=np.int64)
        incumbent_value = float(evaluator.best_objective)
        while evaluator.evaluations < config.budget:
            improved = False
            for coord in _build_local_search_order(
                incumbent,
                str(ovlr_params['ovlr_local_search_order']),
            ):
                if evaluator.evaluations >= config.budget:
                    break
                candidate = incumbent.copy()
                candidate[coord] = 1 - candidate[coord]
                previous_best = float(evaluator.best_objective)
                candidate_value = evaluator.evaluate(candidate)
                if candidate_value >= incumbent_value:
                    incumbent = candidate
                    incumbent_value = candidate_value
                    _append_local_search_trace(
                        step_trace,
                        evaluator,
                        candidate_value,
                        previous_best,
                        phase='single_bit_local_search',
                    )
                    improved = True
                    break
            if not improved:
                break

    return step_trace, ovlr_params


def run_reinforce(
    config: ExperimentConfig,
    evaluator: PBOEvaluator,
    device: torch.device,
) -> List[Dict[str, float]]:
    theta = torch.nn.Parameter(torch.zeros(config.dimension, device=device))
    optimizer = torch.optim.Adam([theta], lr=config.reinforce_lr)
    baseline_ema: float | None = None
    step_trace: List[Dict[str, float]] = []

    while evaluator.evaluations < config.budget:
        sample_count = min(config.reinforce_samples, config.budget - evaluator.evaluations)
        if sample_count <= 0:
            break

        optimizer.zero_grad(set_to_none=True)
        expanded_logits = theta.unsqueeze(0).expand(sample_count, -1)
        dist = torch.distributions.Bernoulli(logits=expanded_logits)
        samples = dist.sample()
        log_probs = dist.log_prob(samples).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()

        objective_values = evaluator.evaluate_batch(samples.to(dtype=torch.int64).cpu().numpy())
        rewards = torch.as_tensor(objective_values, device=device, dtype=theta.dtype)
        reward_mean = rewards.mean().detach()

        baseline_value = 0.0 if baseline_ema is None else baseline_ema
        baseline = torch.as_tensor(baseline_value, device=device, dtype=theta.dtype)
        advantages = rewards - baseline
        surrogate = (advantages.detach() * log_probs).mean()
        loss = -(surrogate + (config.reinforce_entropy_coef * entropy))
        loss.backward()
        optimizer.step()
        _clip_logits(theta, config.logit_clamp)

        baseline_ema = (
            reward_mean.item()
            if baseline_ema is None
            else (config.reinforce_baseline_momentum * baseline_ema)
            + ((1.0 - config.reinforce_baseline_momentum) * reward_mean.item())
        )
        step_trace.append(
            {
                'evaluations': float(evaluator.evaluations),
                'mean_reward': float(reward_mean.item()),
            }
        )

    return step_trace


def run_cem(config: ExperimentConfig, evaluator: PBOEvaluator) -> List[Dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    probabilities = np.full(config.dimension, 0.5, dtype=np.float64)
    step_trace: List[Dict[str, float]] = []

    while evaluator.evaluations < config.budget:
        population = min(config.cem_population, config.budget - evaluator.evaluations)
        if population <= 0:
            break

        samples = (rng.random((population, config.dimension)) < probabilities).astype(np.int64)
        objective_values = evaluator.evaluate_batch(samples)
        elite_count = max(1, int(math.ceil(population * config.cem_elite_frac)))
        elite_indices = np.argsort(objective_values)[-elite_count:]
        elite_mean = samples[elite_indices].mean(axis=0)
        probabilities = ((1.0 - config.cem_smoothing) * probabilities) + (config.cem_smoothing * elite_mean)
        probabilities = np.clip(probabilities, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)

        step_trace.append(
            {
                'evaluations': float(evaluator.evaluations),
                'mean_reward': float(np.mean(objective_values)),
            }
        )

    return step_trace


def run_one_plus_one_ea(config: ExperimentConfig, evaluator: PBOEvaluator) -> List[Dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    mutation_rate = config.one_plus_one_mutation_rate
    if mutation_rate is None:
        mutation_rate = 1.0 / config.dimension

    incumbent = rng.integers(0, 2, size=config.dimension, dtype=np.int64)
    incumbent_value = evaluator.evaluate(incumbent)
    step_trace = [
        {
            'evaluations': float(evaluator.evaluations),
            'incumbent_objective': float(incumbent_value),
        }
    ]

    while evaluator.evaluations < config.budget:
        mask = rng.random(config.dimension) < mutation_rate
        if not np.any(mask):
            mask[rng.integers(0, config.dimension)] = True
        candidate = incumbent.copy()
        candidate[mask] = 1 - candidate[mask]
        candidate_value = evaluator.evaluate(candidate)
        if candidate_value >= incumbent_value:
            incumbent = candidate
            incumbent_value = candidate_value

        step_trace.append(
            {
                'evaluations': float(evaluator.evaluations),
                'incumbent_objective': float(incumbent_value),
            }
        )

    return step_trace


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    _validate_config(config)
    set_seed(config.seed)
    device = get_device(config.device)
    problem, spec = _build_problem(config)
    evaluator = PBOEvaluator(problem)
    effective_ovlr_params: Dict[str, object] | None = None

    start_time = time.perf_counter()
    if config.method == 'ovlr':
        step_trace, effective_ovlr_params = run_ovlr(config, evaluator, device)
    elif config.method == 'reinforce':
        step_trace = run_reinforce(config, evaluator, device)
    elif config.method == 'cem':
        step_trace = run_cem(config, evaluator)
    elif config.method == 'one_plus_one_ea':
        step_trace = run_one_plus_one_ea(config, evaluator)
    else:
        raise ValueError(f'Unsupported method: {config.method}')
    wall_time = time.perf_counter() - start_time

    optimum = float(problem.optimum.y)
    best_objective = float(evaluator.best_objective)
    objective_gap = optimum - best_objective if math.isfinite(optimum) else None
    normalized_best = (
        best_objective / optimum
        if math.isfinite(optimum) and optimum != 0.0
        else None
    )

    return {
        'method': config.method,
        'method_display': METHOD_DISPLAY[config.method],
        'method_description': METHOD_DESCRIPTION[config.method],
        'problem_key': config.problem_key,
        'problem_display': spec.display_name,
        'dimension': config.dimension,
        'budget': config.budget,
        'instance': config.instance,
        'seed': config.seed,
        'device': str(device),
        'config': config.to_dict(),
        'effective_ovlr_params': effective_ovlr_params,
        'optimum_objective': optimum,
        'best_objective': best_objective,
        'objective_gap': objective_gap,
        'normalized_best': normalized_best,
        'hit_optimum': bool(math.isfinite(optimum) and abs(best_objective - optimum) < 1e-12),
        'evaluations_used': evaluator.evaluations,
        'runtime_seconds': wall_time,
        'best_solution': evaluator.best_solution,
        'evaluation_trace': evaluator.trace,
        'step_trace': step_trace,
    }


def aggregate_results(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    by_problem_method: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        problem_key = str(row['problem_key'])
        method = str(row['method'])
        by_problem_method.setdefault((problem_key, method), []).append(row)
        by_method.setdefault(method, []).append(row)

    problem_aggregate: List[Dict[str, object]] = []
    for (problem_key, method), items in sorted(by_problem_method.items()):
        normalized_values = [float(item['normalized_best']) for item in items if item['normalized_best'] is not None]
        gaps = [float(item['objective_gap']) for item in items if item['objective_gap'] is not None]
        problem_aggregate.append(
            {
                'problem_key': problem_key,
                'problem_display': items[0]['problem_display'],
                'method': method,
                'method_display': items[0]['method_display'],
                'mean_best_objective': mean(float(item['best_objective']) for item in items),
                'mean_normalized_best': mean(normalized_values) if normalized_values else None,
                'mean_objective_gap': mean(gaps) if gaps else None,
                'success_rate': mean(1.0 if bool(item['hit_optimum']) else 0.0 for item in items),
                'mean_runtime_seconds': mean(float(item['runtime_seconds']) for item in items),
                'num_runs': len(items),
            }
        )

    overall_aggregate: List[Dict[str, object]] = []
    for method, items in sorted(by_method.items()):
        normalized_values = [float(item['normalized_best']) for item in items if item['normalized_best'] is not None]
        gaps = [float(item['objective_gap']) for item in items if item['objective_gap'] is not None]
        overall_aggregate.append(
            {
                'method': method,
                'method_display': items[0]['method_display'],
                'mean_best_objective': mean(float(item['best_objective']) for item in items),
                'mean_normalized_best': mean(normalized_values) if normalized_values else None,
                'mean_objective_gap': mean(gaps) if gaps else None,
                'success_rate': mean(1.0 if bool(item['hit_optimum']) else 0.0 for item in items),
                'mean_runtime_seconds': mean(float(item['runtime_seconds']) for item in items),
                'num_runs': len(items),
            }
        )

    return {
        'problem_aggregate': problem_aggregate,
        'overall_aggregate': overall_aggregate,
    }


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'problem_key',
        'problem_display',
        'method',
        'method_display',
        'seed',
        'dimension',
        'budget',
        'optimum_objective',
        'best_objective',
        'objective_gap',
        'normalized_best',
        'hit_optimum',
        'evaluations_used',
        'runtime_seconds',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_aggregate_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(
    path: Path,
    rows: Sequence[Dict[str, object]],
    aggregate: Dict[str, List[Dict[str, object]]],
    problem_keys: Iterable[str],
    seeds: Sequence[int],
) -> None:
    if not rows:
        path.write_text('# IOH PBO Benchmark\n\nNo runs were recorded.\n', encoding='utf-8')
        return

    config = rows[0]['config']
    problem_names = ', '.join(PROBLEM_SPECS[key].display_name for key in problem_keys)
    overall_rows = aggregate['overall_aggregate']
    best_overall = None
    if overall_rows:
        best_overall = max(
            overall_rows,
            key=lambda item: float('-inf') if item['mean_normalized_best'] is None else float(item['mean_normalized_best']),
        )

    lines = [
        '# IOH PBO Discrete Black-Box Optimization Benchmark',
        '',
        '## Setup',
        '',
        f"- Problems: {problem_names}",
        f"- Dimension: {config['dimension']}",
        f"- Budget: {config['budget']} true black-box evaluations per run",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Device: {rows[0]['device']}",
        '- Objective direction: maximize',
        '- Objective type: discrete/integer-valued PBO functions from IOH',
        '',
        '## Methods',
        '',
    ]

    for method in METHODS:
        if any(row['method'] == method for row in rows):
            lines.append(f"- `{method}`: {METHOD_DESCRIPTION[method]}")

    lines.extend(
        [
            '',
            '## Per-Problem Aggregate',
            '',
            '| Problem | Method | Mean Best | Mean Normalized | Mean Gap | Success Rate | Mean Time (s) |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
        ]
    )

    for row in aggregate['problem_aggregate']:
        normalized = 'n/a' if row['mean_normalized_best'] is None else f"{row['mean_normalized_best']:.4f}"
        gap = 'n/a' if row['mean_objective_gap'] is None else f"{row['mean_objective_gap']:.4f}"
        lines.append(
            f"| {row['problem_display']} | {row['method_display']} | {row['mean_best_objective']:.4f} | "
            f"{normalized} | {gap} | {100.0 * row['success_rate']:.1f}% | {row['mean_runtime_seconds']:.3f} |"
        )

    lines.extend(
        [
            '',
            '## Overall Aggregate',
            '',
            '| Method | Mean Best | Mean Normalized | Mean Gap | Success Rate | Mean Time (s) | Runs |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for row in overall_rows:
        normalized = 'n/a' if row['mean_normalized_best'] is None else f"{row['mean_normalized_best']:.4f}"
        gap = 'n/a' if row['mean_objective_gap'] is None else f"{row['mean_objective_gap']:.4f}"
        lines.append(
            f"| {row['method_display']} | {row['mean_best_objective']:.4f} | {normalized} | "
            f"{gap} | {100.0 * row['success_rate']:.1f}% | {row['mean_runtime_seconds']:.3f} | {row['num_runs']} |"
        )

    if best_overall is not None:
        lines.extend(
            [
                '',
                '## Headline',
                '',
                f"- Best overall normalized score: {best_overall['method_display']} ({best_overall['mean_normalized_best']:.4f})",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
