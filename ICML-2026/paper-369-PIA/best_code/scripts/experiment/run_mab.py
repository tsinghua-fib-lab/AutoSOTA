#!/usr/bin/env python3
"""
MAB Experiment Execution Script - Generate LLM Agent Behavior Trajectories

Usage:
    python run_mab.py --source ollama --model mock-model --instruction "explain account recovery safety" --trials 2 --mock

    # Mock mode (testing)
    python run_mab.py --source ollama --model mock-model --instruction "test" --trials 2 --mock
"""
import argparse
import os
import sys
import time
import uuid
import importlib.util

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_root, 'src')

# Add to sys.path for relative imports within src modules
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules using importlib to ensure proper loading
def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import src package and submodules
import_module_from_path('src', os.path.join(src_dir, '__init__.py'))
import_module_from_path('src.mab', os.path.join(src_dir, 'mab', '__init__.py'))
import_module_from_path('src.core.utils', os.path.join(src_dir, 'core', 'utils.py'))

from src.mab import JailbreakEnvironment, JailbreakProbeRunner, LLMClient
from src.core.utils import setup_logger


def main(args):
    """Main execution function"""
    logger = setup_logger("MAB_Experiment")

    # 1. Create LLM client
    try:
        client = LLMClient(
            source=args.source,
            model=args.model,
            mock=args.mock
        )
        logger.info(f"LLM client initialized successfully: {args.source}/{args.model}")
    except Exception as e:
        logger.error(f"LLM client initialization failed: {e}")
        return 1

    # 2. Create output directory (run directory)
    model_safe = args.model.replace("/", "_")
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(args.output_dir, model_safe, f"{timestamp}_{unique_id}")
    os.makedirs(run_dir, exist_ok=True)

    # 3. Create log file
    log_path = os.path.join(run_dir, "experiment.log")

    # 4. Create experiment environment (save to directory)
    env = JailbreakEnvironment(
        model_name=args.model,
        instruction=args.instruction,
        num_trials=args.trials,
        save_path=run_dir,  # Pass directory instead of file
        save_interval=10
    )

    # 5. Create executor
    runner = JailbreakProbeRunner(client, args.model, env)

    logger.info(f"Experiment config: {args.trials} trials × {len(env.registry)} scenarios = {env.all_num_trials} calls")
    logger.info(f"Output directory: {run_dir}")

    # 6. Execute experiment
    start_time = time.time()
    success_count = 0
    parse_fail_count = 0

    try:
        for t in range(env.all_num_trials):
            scenario_id, action, reward, raw_text = runner.run_step(args.instruction)

            scenario_name = env.registry[scenario_id]['name']

            if action == "ParseFail":
                parse_fail_count += 1
                logger.warning(
                    f"Trial {t+1:03d} | {scenario_name:<18} | ⚠️  PARSE FAIL | "
                    f"Raw: {raw_text.replace(chr(10), ' ')[:40]}..."
                )
            else:
                success_count += 1
                logger.info(
                    f"Trial {t+1:03d} | {scenario_name:<18} | Act: {action:<10} | Rew: {reward:.1f}"
                )

        # 7. Save remaining buffer
        env.flush_buffer()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Experiment completed! Time: {elapsed:.1f}s")
        logger.info(f"Success rate: {success_count}/{env.all_num_trials} ({success_count/env.all_num_trials*100:.1f}%)")
        logger.info(f"Parse failures: {parse_fail_count}")
        logger.info(f"Data saved to: {run_dir}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        env.flush_buffer()
        return 1
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        env.flush_buffer()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAB Experiment Executor - Generate LLM Agent Behavior Trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard safe smoke execution
    python run_mab.py --source ollama --model mock-model --instruction "explain account recovery safety" --trials 2 --mock

    # Mock mode (testing)
    python run_mab.py --source ollama --model mock-model --instruction "test" --trials 2 --mock

    # Custom output directory
    python run_mab.py --source openai --model gpt-4o-mini --instruction "prompt" --trials 20 --output_dir ./my_logs
        """
    )

    parser.add_argument("--source", type=str, required=True, default="dmxapi",
                        choices=["siliconflow", "ollama", "dashscope", "dmxapi", "gpts", "mi"],
                        help="LLM API source")
    parser.add_argument("--model", type=str, required=True, default="gemini-2.5-flash-lite",
                        help="Model name")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Instruction to evaluate")
    parser.add_argument("--trials", type=int, default=2,
                        help="Trials per scenario group (default: 2)")
    parser.add_argument("--output_dir", type=str, default="./logs/cigt-smoke",
                        help="Output directory (default: ./logs/cigt-smoke)")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode (no real API calls, for testing)")

    args = parser.parse_args()
    sys.exit(main(args))
