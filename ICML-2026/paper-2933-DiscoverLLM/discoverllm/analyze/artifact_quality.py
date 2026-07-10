"""
Script to evaluate artifact quality for synthesized evaluation conversations.

This script:
1. Takes synthesized conversation files
2. Replaces the last user message with a default artifact generation request
3. Uses the assistant simulator (with same configurations) to generate an artifact
4. Evaluates the quality of the generated artifact on the original criteria

Usage:
    python -m discoverllm.analyze.artifact_quality <results_dir> [options]

Examples:
    python -m discoverllm.analyze.artifact_quality data/evaluations/0113_stories_20
    python -m discoverllm.analyze.artifact_quality data/evaluations/0113_stories_20 --workers 4
    python -m discoverllm.analyze.artifact_quality data/evaluations/0113_stories_20 --evaluator-model gpt-5.1-2025-11-13
"""

import argparse
import copy
import statistics
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
from discoverllm.pipeline.assistant_simulator import AssistantSimulator
from discoverllm.simulate.logging_utils import log_error

# Default prompt to request artifact generation
DEFAULT_ARTIFACT_REQUEST = "okay now generate a complete output artifact by considering the conversation up to now. only return the artifact without any other text or explanation. strictly follow this request"


def extract_original_requirements(criteria_history: List[List[Dict[str, Any]]]) -> Optional[str]:
    """Extract the original criterion text from criteria_history[0]."""
    if not criteria_history or len(criteria_history) == 0:
        return None
    first_criteria = criteria_history[0]
    if not first_criteria or len(first_criteria) == 0:
        return None
    first_criterion = first_criteria[0]
    abstractions = first_criterion.get('abstractions', [])
    if not abstractions or len(abstractions) == 0:
        return None
    original_abstraction = abstractions[-1]
    original_requirements = original_abstraction.get('checklist', [])
    if not original_requirements or len(original_requirements) == 0:
        return None
    return original_requirements


def prepare_conversation_for_artifact_generation(
    conversation: List[Dict[str, str]],
    artifact_request: str = DEFAULT_ARTIFACT_REQUEST
) -> List[Dict[str, str]]:
    """
    Prepare conversation by replacing the last user message with the artifact request.

    The conversation should end with a user message (the last one before artifact generation).
    We replace this to ask for the full artifact.
    """
    if not conversation:
        return [{"role": "user", "content": artifact_request}]

    # Make a copy to avoid modifying the original
    modified_conv = copy.deepcopy(conversation)

    # If last message is user message, replace it with the artifact request
    if modified_conv[-1].get('role') == 'user':
        modified_conv[-1]['content'] = artifact_request
        return modified_conv

    # If no user message found, append the request
    modified_conv.append({"role": "user", "content": artifact_request})
    return modified_conv


def generate_artifact(
    conversation: List[Dict[str, str]],
    model_name: str,
    system_prompt: str,
    user_prompt: Optional[str] = None,
    temperature: float = 0.7,
    artifact_request: str = DEFAULT_ARTIFACT_REQUEST,
    verbose: bool = False
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Generate an artifact using the assistant simulator.

    Returns:
        Tuple of (generated_artifact, full_conversation_with_artifact)
    """
    # Prepare conversation by replacing last user message
    modified_conv = prepare_conversation_for_artifact_generation(
        conversation, artifact_request
    )

    # Remove the last message (the artifact request) as the simulator will add it
    conv_without_last = modified_conv[:-1] if modified_conv else []

    # Create assistant simulator with the conversation history
    assistant = AssistantSimulator(
        initial_chat_history=conv_without_last,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=temperature,
        verbose=verbose
    )

    # Generate the artifact response
    artifact_response = assistant(artifact_request)

    return artifact_response, assistant.chat_history


def evaluate_artifact_quality(
    requirements: List[str],
    artifact: str,
    conversation: List[Dict[str, str]],
    evaluator_model: str = "gpt-5.1",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    verbose: bool = False,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated artifact against the original requirements or constraints.

    Returns:
        Dictionary containing the evaluation results
    """
    system_prompt = load_prompt("evaluate_artifact_quality.yaml")

    requirements_str = "\n".join([f"{i+1}. {requirement}" for i, requirement in enumerate(requirements)])

    user_prompt = f"""# Artifact

{artifact}

# Requirements or Constraints

{requirements_str}

Please evaluate how well the artifact satisfies the requirements or constraints."""

    try:
        parsed_json, raw_output = generate_and_process_json(
            model_name=evaluator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            verbose=verbose
        )
        if not parsed_json or not parsed_json.get('evaluations'):
            return {
                "error": "No evaluations found",
                "score": None,
                "evaluations": []
            }
        evaluations = parsed_json.get('evaluations', [])
        average_score = statistics.mean([evaluation.get('score', 0) for evaluation in evaluations])
        return {
            "score": average_score,
            "evaluations": evaluations
        }
    except Exception as e:
        return {
            "error": str(e),
            "score": None,
            "evaluations": []
        }


def process_conversation_file(
    file_path: Path,
    source_dir: Path,
    target_dir: Path,
    evaluator_model: str,
    artifact_request: str,
    verbose: bool = False
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """
    Process a single conversation file to add artifact quality evaluation.

    Args:
        file_path: Path to the source conversation file
        source_dir: Root directory of source files (to compute relative path)
        target_dir: Root directory where updated files should be saved
        evaluator_model: Model to use for evaluation
        artifact_request: Prompt to request artifact generation
        verbose: Enable verbose output

    Returns:
        Tuple of (target_file_path, result_dict or None if error)
    """
    # Load the conversation data
    data = load_conversation_file(file_path)
    if data is None:
        return file_path, None

    # Compute the relative path from source_dir to file_path
    try:
        relative_path = file_path.relative_to(source_dir)
    except ValueError:
        # If file_path is not relative to source_dir, use the file name
        relative_path = Path(file_path.name)

    # Compute the target file path
    target_file_path = target_dir / relative_path

    # Skip if already evaluated in target location
    if target_file_path.exists():
        existing_data = load_conversation_file(target_file_path)
        if existing_data and 'artifact_quality' in existing_data and existing_data['artifact_quality'].get('score') is not None:
            print(f"  ⏭️  Skipping {file_path.name} - already evaluated")
            return target_file_path, existing_data.get('artifact_quality')

    # Extract required fields
    conversation = data.get('conversation', [])
    criteria_history = data.get('criteria_history', [])
    # model_name = "gpt-5-mini" # TODO: REPLACE WITH LINE BELOW WHEN USING REAL MODELS
    model_name = data.get('model_name', None) or (data.get('assistant_configs', [{}])[0].get('model_name') if data.get('assistant_configs') else None)

    if model_name is None:
        raise ValueError(f"No model name found in {file_path.name}")

    system_prompt = data.get('system_prompt', '') or (data.get('assistant_configs', [{}])[0].get('system_prompt', '') if data.get('assistant_configs') else '')

    # Get the original requirements or constraints
    requirements = extract_original_requirements(criteria_history)
    if requirements is None:
        print(f"  ⚠️  Warning: No requirements or constraints found in {file_path.name}")
        return target_file_path, None

    try:
        # Step 1: Generate the artifact
        artifact, full_conversation = generate_artifact(
            conversation=conversation,
            model_name=model_name,
            system_prompt=system_prompt,
            artifact_request=artifact_request,
            verbose=verbose
        )

        # Step 2: Evaluate the artifact quality
        evaluation = evaluate_artifact_quality(
            requirements=requirements,
            artifact=artifact,
            conversation=full_conversation,
            evaluator_model=evaluator_model,
            verbose=verbose
        )

        # Construct the result
        result = {
            'artifact_request': artifact_request,
            'generated_artifact': artifact,
            'requirements_used': requirements,
            'evaluation': evaluation.get('evaluations', []),
            'score': evaluation.get('score'),
            'evaluator_model': evaluator_model,
            'timestamp': datetime.now().isoformat()
        }

        # Update the data and save to target location
        data['artifact_quality'] = result

        # Ensure the target directory exists
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        save_conversation_file(target_file_path, data)

        return target_file_path, result

    except Exception as e:
        # Bubble up; the orchestrator in run_evaluation collects per-file
        # errors and decides whether to write a stub or skip.
        log_error(f"Error processing {file_path}: {e}", exception=e, include_traceback=True)
        raise








def run_evaluation(
    results_dir: str,
    evaluator_model: str = "gpt-5-mini",
    artifact_request: str = DEFAULT_ARTIFACT_REQUEST,
    parallel_workers: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run artifact-quality evaluation over every conversation in ``results_dir``.

    Each conversation gets re-played through its assistant to produce a final
    artifact, which is then scored by the evaluator LLM against the user's
    original requirements. Results are written under
    ``<results_dir>/quality_evaluations/<artifact>/<assistant>.json`` and
    rolled up into a single summary JSON.
    """
    spec = AnalyzerSpec(
        display_name="ARTIFACT QUALITY",
        output_subdir="quality_evaluations",
        summary_filename="artifact_quality_summary.json",
        score_buckets={
            "0-2": (0, 2),
            "3-4": (3, 4),
            "5-6": (5, 6),
            "7-8": (7, 8),
            "9-10": (9, 10),
        },
        extra_summary={
            "evaluator_model": evaluator_model,
            "artifact_request": artifact_request,
        },
    )

    def process_one(file_path: Path, source_dir: Path, output_dir: Path):
        return process_conversation_file(
            file_path, source_dir, output_dir,
            evaluator_model=evaluator_model,
            artifact_request=artifact_request,
            verbose=verbose,
        )

    return run_analyzer(spec, results_dir, process_one, parallel_workers=parallel_workers)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate artifact quality for synthesized evaluation conversations"
    )
    parser.add_argument(
        "results_dir",
        help="Path to the evaluation results directory"
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-5.1",
        help="Model to use for artifact quality evaluation (default: gpt-5.1-2025-11-13)"
    )
    parser.add_argument(
        "--artifact-request",
        default=DEFAULT_ARTIFACT_REQUEST,
        help=f"Prompt to request artifact generation (default: '{DEFAULT_ARTIFACT_REQUEST}')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        run_evaluation(
            results_dir=args.results_dir,
            evaluator_model=args.evaluator_model,
            artifact_request=args.artifact_request,
            parallel_workers=args.workers,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
