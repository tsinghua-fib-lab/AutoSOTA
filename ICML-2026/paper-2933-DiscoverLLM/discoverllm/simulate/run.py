#!/usr/bin/env python3
"""
CLI for the simulator.

Two modes — both run the same user-simulator-driven multi-turn loop, but
differ in what happens each turn:

* ``--mode best_of_1`` (default) — for each (artifact, assistant) pair, run
  ONE independent conversation. The assistant generates a single response
  per turn. Output: ``<output_dir>/<artifact_id>/<assistant_id>.json``
  (one file per pair). Used for evaluation.

* ``--mode best_of_n`` — for each artifact, run ONE shared conversation
  where every assistant in ``--assistant-configs-file`` generates a
  candidate per turn. The highest-reward candidate is committed; every
  candidate (with score) is recorded. Output:
  ``<output_dir>/<artifact_id>/best_of_n.json`` (one file per artifact).
  Used for synthesising training data.

Examples:
    # Evaluation: compare 3 assistants on the same artifacts.
    python -m discoverllm.simulate.run artifacts.json results/ \\
        -a assistants.json -u user.json -r reward.json --mode best_of_1

    # Synthesis: generate best-of-N training data with N candidates per turn.
    python -m discoverllm.simulate.run artifacts.json results/ \\
        -a assistants.json -u user.json -r reward.json --mode best_of_n
"""

from __future__ import annotations

import argparse
import os
import sys

from discoverllm.simulate.config import (
    MODE_BEST_OF_1,
    MODE_BEST_OF_N,
    ExperimentConfig,
)
from discoverllm.simulate.io import load_assistant_configs, load_user_config
from discoverllm.simulate.logging_utils import close_error_log, log_error
from discoverllm.simulate.runner import run_experiment


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("artifacts_file",
                   help="JSON file with a list of {artifact_type, text} objects.")
    p.add_argument("output_dir", help="Directory to save experiment results.")
    p.add_argument("--assistant-configs-file", "-a", required=True,
                   help="JSON file containing the assistant configs to run.")
    p.add_argument("--user-config-file", "-u", required=True,
                   help="JSON file containing the user-simulator config.")
    p.add_argument("--reward-assistant-config-file", "-r", default=None,
                   help="Optional JSON file for the reward-judge assistant. "
                        "Defaults to the first --assistant-configs-file entry.")
    p.add_argument("--mode", choices=[MODE_BEST_OF_1, MODE_BEST_OF_N], default=MODE_BEST_OF_1,
                   help="best_of_1 = one assistant per turn (evaluation). "
                        "best_of_n = all assistants compete per turn (synthesis). "
                        f"Default: {MODE_BEST_OF_1}.")
    p.add_argument("--max-turns", type=int, default=8,
                   help="Maximum conversation turns per artifact (default: 8).")
    p.add_argument("--window-size", type=int, default=0,
                   help="Future-turn lookahead used by the reward formula (default: 0).")
    p.add_argument("--parallel-workers", type=int, default=4,
                   help="Number of conversations to run concurrently (default: 4).")
    p.add_argument("--num-trials", type=int, default=1,
                   help="best_of_1 only: trials per (artifact, assistant) pair. "
                        "Auto-skips trials already on disk. Default: 1.")
    p.add_argument("--max-artifacts", type=int, default=None,
                   help="Cap on artifacts to process (default: all).")
    p.add_argument("--verbose", action="store_true", help="Verbose output.")
    return p.parse_args()


def _load_configs(args: argparse.Namespace):
    """Validate inputs and load the three config objects (assistants, user, reward)."""
    for path, label in [
        (args.artifacts_file, "Artifacts file"),
        (args.assistant_configs_file, "Assistant configs file"),
        (args.user_config_file, "User config file"),
    ]:
        if not os.path.exists(path):
            print(f"❌ Error: {label} not found: {path}")
            sys.exit(1)

    try:
        assistant_configs = load_assistant_configs(args.assistant_configs_file)
        if not assistant_configs:
            print("❌ Error: no valid assistant configs found")
            sys.exit(1)
        print(f"✅ Loaded {len(assistant_configs)} assistant config(s)")
    except Exception as e:
        log_error(message=f"Error loading assistant configs: {e}", exception=e, include_traceback=True)
        close_error_log()
        sys.exit(1)

    try:
        user_config = load_user_config(args.user_config_file)
        print("✅ Loaded user config")
    except Exception as e:
        log_error(message=f"Error loading user config: {e}", exception=e, include_traceback=True)
        close_error_log()
        sys.exit(1)

    if args.reward_assistant_config_file:
        try:
            reward_assistant_config = load_assistant_configs(args.reward_assistant_config_file)[0]
            print("✅ Loaded reward assistant config")
        except Exception as e:
            log_error(message=f"Error loading reward assistant config: {e}",
                      exception=e, include_traceback=True)
            close_error_log()
            sys.exit(1)
    else:
        reward_assistant_config = assistant_configs[0]
        print("ℹ️  No reward assistant config provided; using first assistant config.")

    return assistant_configs, user_config, reward_assistant_config


def main() -> None:
    args = _parse_args()
    assistant_configs, user_config, reward_assistant_config = _load_configs(args)

    # num_trials only applies to best_of_1.
    num_trials = args.num_trials if args.mode == MODE_BEST_OF_1 else 1
    if args.num_trials > 1 and args.mode == MODE_BEST_OF_N:
        print("⚠️  --num-trials is ignored in best_of_n mode")

    config = ExperimentConfig(
        artifacts_file=args.artifacts_file,
        output_dir=args.output_dir,
        assistant_configs=assistant_configs,
        reward_assistant_config=reward_assistant_config,
        user_config=user_config,
        max_turns=args.max_turns,
        window_size=args.window_size,
        parallel_workers=args.parallel_workers,
        mode=args.mode,
        verbose=args.verbose,
        num_trials=num_trials,
        max_artifacts=args.max_artifacts,
    )

    print(f"\n🔄 Running {args.mode.upper()} experiment")
    try:
        summary = run_experiment(config)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT SUMMARY ({args.mode})")
        print("=" * 80)
        print(f"Artifacts: {summary['experiment_config']['num_artifacts']}")
        print(f"Assistants: {len(config.assistant_configs)}")
        print(f"\n✅ Done. Results in: {config.output_dir}")
    except Exception as e:
        log_error(message=f"Experiment failed: {e}", exception=e, include_traceback=True)
        close_error_log()
        if args.verbose:
            import traceback as _tb
            _tb.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
