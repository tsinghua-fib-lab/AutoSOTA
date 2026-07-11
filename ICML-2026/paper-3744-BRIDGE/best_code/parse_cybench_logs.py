import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"


def extract_task_id_from_path(challenge_path: str) -> str:
    """
    Extract task ID from challenge path.

    Example:
        'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic'
        -> 'hackthebox_cyber-apocalypse-2024_crypto_dynastic'
    """
    path = challenge_path.replace('benchmark/', '')
    parts = path.split('/')

    # Clean up the last part (task name) - remove difficulty markers
    if len(parts) > 0:
        task_name = parts[-1]
        task_name = re.sub(r'\[.*?\]\s*', '', task_name)
        parts[-1] = task_name.strip()

    task_id = '_'.join(parts)
    task_id = task_id.lower().replace(' ', '-')

    return task_id


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to standard format."""
    if 'claude-3-5-sonnet' in model_name or 'claude-3.5-sonnet' in model_name:
        return 'claude-3.5-sonnet'
    elif 'claude-3-opus' in model_name:
        return 'claude-3-opus'
    elif 'gpt-4o-mini' in model_name:
        return 'gpt-4o-mini'
    elif 'gpt-4o-2024-05-13' in model_name:
        return 'gpt-4o-2024-05-13'
    elif 'gpt-4o-2024-08-06' in model_name:
        return 'gpt-4o-2024-08-06'
    elif 'gpt-4o' in model_name:
        return 'gpt-4o'
    elif 'o1-preview' in model_name:
        return 'o1-preview'
    elif 'llama-3-70b' in model_name:
        return 'llama-3-70b-chat'
    elif 'llama-3.1-405b' in model_name:
        return 'llama-3.1-405b-instruct'
    elif 'mixtral' in model_name:
        return 'mixtral-8x22b-instruct'
    elif 'gemini-1.5-pro' in model_name:
        return 'gemini-1.5-pro'
    elif 'qwen3coder30b' in model_name:
        return 'qwen3-coder-30b-a3b-instruct'
    else:
        return model_name


def parse_filename(filename: str) -> Tuple[str, bool, bool]:
    """
    Parse log filename to extract model name, success status, and subtask mode.

    Returns:
        (model_name, is_success, is_subtask_mode)
    """
    parts = filename.replace('_run_log.json', '').split('_')

    # Find where success/fail is
    success_idx = -1
    for i, part in enumerate(parts):
        if part in ['success', 'fail']:
            success_idx = i
            break

    if success_idx == -1:
        raise ValueError(f"Could not find success/fail in filename: {filename}")

    model_name = '_'.join(parts[:success_idx])
    model_name = normalize_model_name(model_name)

    is_success = parts[success_idx] == 'success'
    is_subtask = success_idx + 1 < len(parts) and parts[success_idx + 1] == 'subtask'

    return model_name, is_success, is_subtask


def extract_difficulty_label(challenge_name: str) -> str:
    """
    Extract difficulty label from challenge name.

    Examples:
        '[Very Easy] Dynastic' -> 'very_easy'
        '[Easy] Crushing' -> 'easy'
        '[Medium] Data Siege' -> 'medium'
        '[Hard] Permuted' -> 'hard'
        'chunky' -> 'unlabelled'
    """
    match = re.search(r'\[(.*?)\]', challenge_name)
    if match:
        label = match.group(1).lower().replace(' ', '_')
        return label
    return 'unlabelled'


def parse_log_file(log_path: Path) -> Dict:
    """Parse a single log file and extract relevant information."""
    with open(log_path, 'r') as f:
        data = json.load(f)

    # Handle both 'challenge' and 'task' keys
    challenge_data = data.get('challenge') or data.get('task')
    if challenge_data is None:
        raise ValueError("No 'challenge' or 'task' key found")

    challenge_name = challenge_data.get('name', '')
    difficulty_label = extract_difficulty_label(challenge_name)

    return {
        'challenge_path': challenge_data['path'],
        'challenge_name': challenge_name,
        'difficulty_label': difficulty_label,
        'status': data.get('status'),
        'num_correct_subtasks': data.get('num_correct_subtasks', 0),
        'num_subtasks': data.get('num_subtasks', 0),
    }


def load_fst_data(fst_path: Path) -> Dict[str, float]:
    """Load FST data and return task_id -> human_minutes mapping."""
    fst_mapping = {}
    if fst_path.exists():
        with open(fst_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                fst_mapping[record['task_id']] = record['human_minutes']
    return fst_mapping


def parse_all_logs(data_dir: Path, require_subtask_completion: bool, verbose: bool) -> Dict[Tuple[str, str], Dict]:
    """
    Parse all log files in data_by_challenges directory.

    Args:
        data_dir: Directory containing log files grouped by challenge.
        require_subtask_completion: If True, guided successes require all subtasks correct.
        verbose: If True, print warnings for parsing errors.

    Returns:
        Dict mapping (task_id, model) -> {score, challenge_path, difficulty_label, ...}
    """
    results = {}

    for challenge_dir in sorted(data_dir.iterdir()):
        if not challenge_dir.is_dir():
            continue

        for log_file in challenge_dir.glob('*.json'):
            try:
                # Parse filename
                model_name, is_success, is_subtask = parse_filename(log_file.name)

                # Skip subtask (guided) mode - only use unguided
                if is_subtask:
                    continue

                # Parse log content
                log_data = parse_log_file(log_file)

                # Extract task ID from challenge path
                task_id = extract_task_id_from_path(log_data['challenge_path'])

                # Determine score
                if is_success:
                    if require_subtask_completion and log_data['num_subtasks'] > 0:
                        score = 1.0 if log_data['num_correct_subtasks'] == log_data['num_subtasks'] else 0.0
                    else:
                        score = 1.0
                else:
                    score = 0.0

                # Store result (keep best score for each task/model pair)
                key = (task_id, model_name)

                if key not in results:
                    results[key] = {
                        'task_id': task_id,
                        'model': model_name,
                        'challenge_path': log_data['challenge_path'],
                        'difficulty_label': log_data['difficulty_label'],
                        'score': None,
                    }

                # Keep best score
                if results[key]['score'] is None or score > results[key]['score']:
                    results[key]['score'] = score

            except Exception as e:
                if verbose:
                    print(f"Warning: Error parsing {log_file}: {e}")
                continue

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse Cybench evaluation logs')
    parser.add_argument('--require-subtask-completion', action='store_true',
                        help='When set, a guided run marked success must have all subtasks correct to count as success')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = DATA_DIR / 'cybench_data_by_challenges'
    fst_path = DATA_DIR / 'cybench_fst_master.jsonl'
    output_path = DATA_DIR / 'cybench_normalized_results.jsonl'
    human_minutes_output = DATA_DIR / 'cybench_human_minutes_by_task.jsonl'

    if args.verbose:
        print("=" * 80)
        print("Parsing Cybench Evaluation Logs")
        print("=" * 80)

    # Parse all logs (unguided mode only)
    if args.verbose:
        print(f"\n1. Parsing logs from {data_dir}")
    parsed_results = parse_all_logs(data_dir, args.require_subtask_completion, args.verbose)
    if args.verbose:
        print(f"   Found {len(parsed_results)} (task_id, model) evaluation pairs")

    # Get unique tasks and models
    task_ids = sorted(set(task_id for task_id, _ in parsed_results.keys()))
    models = sorted(set(model for _, model in parsed_results.keys()))
    if args.verbose:
        print(f"   Unique tasks with evaluations: {len(task_ids)}")
        print(f"   Unique models: {len(models)}")
        for m in models:
            print(f"      - {m}")

    # Filter out tasks where no model succeeded (exclude-no-success always true)
    tasks_with_success = set()
    for (task_id, model), data in parsed_results.items():
        if data['score'] is not None and data['score'] >= 1.0:
            tasks_with_success.add(task_id)

    if args.verbose:
        all_tasks = set(tid for (tid, _) in parsed_results.keys())
        excluded_tasks = all_tasks - tasks_with_success
        print("\n   Filtering tasks with no successes:")
        print(f"   Tasks with at least one success: {len(tasks_with_success)}/{len(all_tasks)}")
        if excluded_tasks:
            print(f"   Excluded (0 successes): {len(excluded_tasks)}")

    # Load existing FST data to get human_minutes
    if args.verbose:
        print(f"\n2. Loading FST data from {fst_path}")
    fst_mapping = load_fst_data(fst_path)
    if args.verbose:
        print(f"   Loaded FST for {len(fst_mapping)} tasks")

    # Match FST to our tasks
    matched_fst = {}
    unmatched_tasks = []
    for task_id in task_ids:
        if task_id in fst_mapping:
            matched_fst[task_id] = fst_mapping[task_id]
        else:
            # Try fuzzy match
            found = False
            for fst_task_id, minutes in fst_mapping.items():
                # Check if task names are similar (last part of the ID)
                task_name = task_id.split('_')[-1].replace('-', '')
                fst_name = fst_task_id.split('_')[-1].replace('-', '')
                if task_name == fst_name:
                    matched_fst[task_id] = minutes
                    found = True
                    break
            if not found:
                unmatched_tasks.append(task_id)

    if args.verbose:
        print(f"   Matched FST for {len(matched_fst)}/{len(task_ids)} tasks")
        if unmatched_tasks:
            print(f"   Tasks without FST match: {len(unmatched_tasks)}")
            for t in unmatched_tasks[:5]:
                print(f"      - {t}")

    # Generate normalized results - ONLY for tasks with actual evaluations
    if args.verbose:
        print("\n3. Generating normalized results")
        print("   Using unguided mode only")

    # Count unique tasks for weighting (all tasks with valid scores, before filtering)
    all_tasks_with_scores = set(
        task_id for (task_id, model), data in parsed_results.items()
        if data['score'] is not None
    )
    total_task_instances = len(all_tasks_with_scores)

    records = []
    for (task_id, model), data in sorted(parsed_results.items()):
        # Skip tasks where no model succeeded
        if task_id not in tasks_with_success:
            continue

        # Skip if no score data
        if data['score'] is None:
            continue

        human_minutes = matched_fst.get(task_id)

        record = {
            "task_id": f"{task_id}_unguided",
            "task_id_base": task_id,
            "run_id": f"cybench_{model}_unguided",
            "alias": model,
            "model": model,
            "score_cont": data['score'],
            "score_binarized": int(data['score']),
            "fatal_error_from": None,
            "human_minutes": human_minutes,
            "human_score": 1.0 if human_minutes else None,
            "human_source": "fst" if human_minutes else None,
            "task_source": "Cybench",
            "difficulty_label": data['difficulty_label'],
            "eval_mode": "unguided",
            "generation_cost": 0.0,
            "human_cost": 0.0,
            "time_limit": None,
            "started_at": None,
            "completed_at": None,
            "task_version": None,
            "equal_task_weight": 1.0 / total_task_instances if total_task_instances else 1.0,
            "invsqrt_task_weight": 1.0 / (total_task_instances ** 0.5) if total_task_instances else 1.0,
        }
        records.append(record)

    # Write normalized results
    if args.verbose:
        print(f"\n4. Writing {len(records)} records to {output_path}")
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    # Update human_minutes file to only include the tasks with evaluations
    unique_task_ids_with_mode = set(record['task_id'] for record in records)
    if args.verbose:
        print(f"\n5. Updating human_minutes file for {len(unique_task_ids_with_mode)} task instances")
    with open(human_minutes_output, 'w') as f:
        for full_task_id in sorted(unique_task_ids_with_mode):
            # Extract base task_id (remove _unguided suffix)
            base_task_id = full_task_id.rsplit('_', 1)[0] if full_task_id.endswith('_unguided') else full_task_id
            if base_task_id in matched_fst:
                record = {"task_id": full_task_id, "human_minutes": matched_fst[base_task_id]}
                f.write(json.dumps(record) + '\n')

    tasks_with_fst = sum(1 for tid in unique_task_ids_with_mode
                         if tid.rsplit('_', 1)[0] in matched_fst or tid in matched_fst)
    if args.verbose:
        print(f"   Written {tasks_with_fst} task instances with FST data")

    # Print statistics
    if args.verbose:
        print("\n" + "=" * 80)
        print("Statistics")
        print("=" * 80)

        print(f"\nTotal evaluation records: {len(records)}")
        print(f"Unique base tasks: {len(tasks_with_success)}")
        print(f"Task instances with FST: {tasks_with_fst}")

        if records:
            # Stats for unguided mode
            success_count = sum(1 for r in records if r['score_binarized'] == 1)
            unique_tasks = len(set(r['task_id_base'] for r in records))
            print("\nUNGUIDED mode:")
            print(f"  Records: {len(records)}, Unique tasks: {unique_tasks}")
            print(f"  Successes: {success_count}/{len(records)} ({100*success_count/len(records):.1f}%)")

            # Count by model
            print("\nSuccess rate by model:")
            model_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            for record in records:
                model = record['model']
                model_stats[model]['total'] += 1
                if record['score_binarized'] == 1:
                    model_stats[model]['success'] += 1

            for model in sorted(model_stats.keys()):
                stats = model_stats[model]
                rate = 100.0 * stats['success'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {model}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

        print(f"\nResults saved to {output_path}")
        print(f"Human minutes saved to {human_minutes_output}")


if __name__ == "__main__":
    main()
