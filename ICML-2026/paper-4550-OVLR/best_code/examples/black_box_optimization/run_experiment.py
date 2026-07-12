from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    METHODS,
    METHOD_DISPLAY,
    ExperimentConfig,
    aggregate_results,
    parse_problem_keys,
    parse_seed_list,
    run_experiment,
    save_json,
    write_aggregate_csv,
    write_markdown_report,
    write_summary_csv,
)

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / 'reported' / 'discrete'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run true black-box optimization on discrete IOH PBO functions with OVLR and baseline methods.',
    )
    parser.add_argument('--method', choices=METHODS + ['all'], default='all')
    parser.add_argument('--problems', type=str, default='default')
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--instance', type=int, default=1)
    parser.add_argument('--seeds', type=str, default='0,1,2')
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    parser.add_argument('--ovlr-repeat', type=int, default=4)
    parser.add_argument('--ovlr-noise-scale', type=float, default=0.6)
    parser.add_argument('--ovlr-lr', type=float, default=0.12)
    parser.add_argument(
        '--ovlr-log-objective',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply signed log1p to OVLR objectives before loss construction.',
    )
    parser.add_argument(
        '--ovlr-center-loss',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Subtract batch mean from OVLR losses.',
    )
    parser.add_argument(
        '--ovlr-normalize-loss',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Normalize OVLR losses by batch standard deviation.',
    )
    parser.add_argument('--ovlr-loss-clip', type=float, default=2.0)
    parser.add_argument('--ovlr-grad-clip-norm', type=float, default=2.0)
    parser.add_argument(
        '--ovlr-local-search-budget',
        type=int,
        default=32,
        help='Reserve this many final evaluations for deterministic single-bit local search around the OVLR incumbent.',
    )
    parser.add_argument(
        '--ovlr-local-search-order',
        choices=['index', 'minority_first', 'majority_first'],
        default='index',
        help='Coordinate ordering used by OVLR local search.',
    )
    parser.add_argument(
        '--ovlr-uniform-probe',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable benchmark-specific all-zeros/all-ones probe before local search. Use for ablation, not main reported results.',
    )
    parser.add_argument(
        '--ovlr-group-tuning',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable per-problem OVLR parameter overrides for leadingones and isingring.',
    )
    parser.add_argument('--reinforce-samples', type=int, default=32)
    parser.add_argument('--reinforce-lr', type=float, default=0.1)
    parser.add_argument('--reinforce-baseline-momentum', type=float, default=0.9)
    parser.add_argument('--reinforce-entropy-coef', type=float, default=1e-3)
    parser.add_argument('--cem-population', type=int, default=32)
    parser.add_argument('--cem-elite-frac', type=float, default=0.25)
    parser.add_argument('--cem-smoothing', type=float, default=0.7)
    parser.add_argument('--one-plus-one-mutation-rate', type=float, default=None)
    parser.add_argument('--logit-clamp', type=float, default=1.5)
    parser.add_argument('--save-dir', type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def build_config(args, method: str, problem_key: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        method=method,
        problem_key=problem_key,
        dimension=args.dimension,
        budget=args.budget,
        instance=args.instance,
        seed=seed,
        device=args.device,
        ovlr_repeat=args.ovlr_repeat,
        ovlr_noise_scale=args.ovlr_noise_scale,
        ovlr_lr=args.ovlr_lr,
        ovlr_log_objective=args.ovlr_log_objective,
        ovlr_center_loss=args.ovlr_center_loss,
        ovlr_normalize_loss=args.ovlr_normalize_loss,
        ovlr_loss_clip=(None if args.ovlr_loss_clip <= 0 else args.ovlr_loss_clip),
        ovlr_grad_clip_norm=(None if args.ovlr_grad_clip_norm <= 0 else args.ovlr_grad_clip_norm),
        ovlr_local_search_budget=max(0, args.ovlr_local_search_budget),
        ovlr_local_search_order=args.ovlr_local_search_order,
        ovlr_uniform_probe=args.ovlr_uniform_probe,
        ovlr_group_tuning=args.ovlr_group_tuning,
        reinforce_samples=args.reinforce_samples,
        reinforce_lr=args.reinforce_lr,
        reinforce_baseline_momentum=args.reinforce_baseline_momentum,
        reinforce_entropy_coef=args.reinforce_entropy_coef,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        cem_smoothing=args.cem_smoothing,
        one_plus_one_mutation_rate=args.one_plus_one_mutation_rate,
        logit_clamp=args.logit_clamp,
    )


def run_path(save_dir: Path, problem_key: str, method: str, seed: int) -> Path:
    return save_dir / 'runs' / f'{problem_key}__{method}__seed{seed}.json'


def main():
    args = parse_args()
    methods = METHODS if args.method == 'all' else [args.method]
    problem_keys = parse_problem_keys(args.problems)
    seeds = parse_seed_list(args.seeds)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for problem_key in problem_keys:
        for seed in seeds:
            for method in methods:
                config = build_config(args, method=method, problem_key=problem_key, seed=seed)
                payload = run_experiment(config)
                save_json(run_path(args.save_dir, problem_key, method, seed), payload)
                all_results.append(payload)
                print(
                    f"[{METHOD_DISPLAY[method]}] "
                    f"problem={payload['problem_display']} seed={seed} "
                    f"best={payload['best_objective']:.4f}/{payload['optimum_objective']:.4f} "
                    f"evals={payload['evaluations_used']} time={payload['runtime_seconds']:.3f}s"
                )

    summary_rows = []
    for row in all_results:
        summary_rows.append(
            {
                'problem_key': row['problem_key'],
                'problem_display': row['problem_display'],
                'method': row['method'],
                'method_display': row['method_display'],
                'seed': row['seed'],
                'dimension': row['dimension'],
                'budget': row['budget'],
                'optimum_objective': row['optimum_objective'],
                'best_objective': row['best_objective'],
                'objective_gap': row['objective_gap'],
                'normalized_best': row['normalized_best'],
                'hit_optimum': row['hit_optimum'],
                'evaluations_used': row['evaluations_used'],
                'runtime_seconds': row['runtime_seconds'],
            }
        )

    aggregate = aggregate_results(all_results)
    save_json(args.save_dir / 'summary.json', summary_rows)
    save_json(args.save_dir / 'aggregate.json', aggregate)
    write_summary_csv(args.save_dir / 'summary.csv', summary_rows)
    write_aggregate_csv(args.save_dir / 'aggregate_problem.csv', aggregate['problem_aggregate'])
    write_aggregate_csv(args.save_dir / 'aggregate_overall.csv', aggregate['overall_aggregate'])
    write_markdown_report(args.save_dir / 'report.md', all_results, aggregate, problem_keys, seeds)


if __name__ == '__main__':
    main()
