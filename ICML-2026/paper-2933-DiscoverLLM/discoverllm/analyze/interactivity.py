"""
Script to evaluate conversation interactivity for synthesized evaluation conversations.

This mirrors the workflow of `evaluate_artifact_quality.py`:
1. Scans `data/evaluations/<X>/artifact_*/assistant_*.json`
2. Calls the interactivity evaluator prompt on each conversation
3. Writes results to `interactivity_evaluations/` (parallel directory tree)
4. Produces a summary JSON with basic statistics
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from discoverllm.analyze._base import AnalyzerSpec, run_analyzer
from discoverllm.analyze._helpers import (
    load_conversation_file,
    save_conversation_file,
)
from discoverllm.core.generate import generate_and_process_json
from discoverllm.core.prompts import load_prompt
from discoverllm.simulate.logging_utils import log_error, log_warning
from discoverllm.utils import format_chat_history

DEFAULT_MODEL = "gpt-5.1"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3


def validate_interactivity_output(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Interactivity output is not a JSON object")
    if "interactivity" not in obj:
        raise ValueError("Missing 'interactivity' in output JSON")
    if obj["interactivity"] is None:
        raise ValueError("'interactivity' is None")


def evaluate_interactivity(
    conversation: List[Dict[str, str]],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Run the interactivity evaluator on a conversation.
    Returns parsed JSON and raw model output.
    """
    system_prompt_template = load_prompt("interactivity_evaluation.yaml")
    chat_history_str = format_chat_history(conversation)
    system_prompt = system_prompt_template.format(
        chat_history=chat_history_str,
        A=3,
        B=2,
        C=1,
    )

    user_prompt = "Please return the JSON evaluation as specified."

    parsed, raw_output = generate_and_process_json(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        validation_function=validate_interactivity_output,
    )
    return parsed, raw_output








# ---------- Core processing ----------

def process_conversation_file(
    file_path: Path,
    source_dir: Path,
    target_dir: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    verbose: bool = False,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    data = load_conversation_file(file_path)
    if data is None:
        return file_path, None

    try:
        relative_path = file_path.relative_to(source_dir)
    except ValueError:
        relative_path = Path(file_path.name)

    target_file_path = target_dir / relative_path

    # Skip if already evaluated with a score
    if target_file_path.exists():
        existing = load_conversation_file(target_file_path)
        if (
            existing
            and "interactivity_evaluation" in existing
            and existing["interactivity_evaluation"].get("score") is not None
        ):
            print(f"  ⏭️  Skipping {file_path.name} - already evaluated")
            return target_file_path, existing["interactivity_evaluation"]

    conversation = data.get("conversation", [])[:-1] # exclude the last message, which is a user turn
    if not conversation:
        log_warning("No conversation found", artifact_id=str(file_path.parent), assistant_id=file_path.name)
        return target_file_path, None

    try:
        parsed, raw_output = evaluate_interactivity(
            conversation=conversation,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            verbose=verbose,
        )

        result = {
            "score": parsed.get("interactivity"),
            "thought": parsed.get("thought"),
            "raw_output": raw_output,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        data["interactivity_evaluation"] = result
        save_conversation_file(target_file_path, data)
        return target_file_path, result

    except Exception as e:
        log_error(
            f"Error processing {file_path}: {e}",
            artifact_id=str(file_path.parent.name),
            assistant_id=file_path.name,
            exception=e,
        )
        error_result = {
            "error": str(e),
            "score": None,
            "timestamp": datetime.now().isoformat(),
        }
        data["interactivity_evaluation"] = error_result
        save_conversation_file(target_file_path, data)
        return target_file_path, error_result




def run_evaluation(
    results_dir: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    parallel_workers: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run interactivity evaluation over every conversation in ``results_dir``.

    Each conversation is rated 0-4 by the judge LLM on how interactive the
    assistant's behaviour was. Results are written under
    ``<results_dir>/interactivity_evaluations/<artifact>/<assistant>.json``
    and rolled up into a summary JSON.
    """
    spec = AnalyzerSpec(
        display_name="INTERACTIVITY",
        output_subdir="interactivity_evaluations",
        summary_filename="interactivity_summary.json",
        score_buckets={
            "≤1": (0, 1),
            "1-2": (1.0001, 2),  # > 1 and <= 2
            "2-3": (2.0001, 3),
            ">3": (3.0001, 1e9),
        },
        extra_summary={
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )

    def process_one(file_path: Path, source_dir: Path, output_dir: Path):
        return process_conversation_file(
            file_path, source_dir, output_dir,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            verbose=verbose,
        )

    return run_analyzer(spec, results_dir, process_one, parallel_workers=parallel_workers)


def main():
    parser = argparse.ArgumentParser(description="Evaluate interactivity of conversations")
    parser.add_argument(
        "results_dir",
        help="Path to the evaluation results directory (e.g., data/evaluations/0120_stories)",
    )
    parser.add_argument(
        "--evaluator-model",
        default=DEFAULT_MODEL,
        help="Model to use for interactivity evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries for generation/parsing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        run_evaluation(
            results_dir=args.results_dir,
            model_name=args.evaluator_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            parallel_workers=args.workers,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()


