import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimeEstimation:
    """Stores the time estimation result for a single task."""
    instance_id: str
    repo: str
    estimated_minutes: float
    justification: str
    time_breakdown: dict
    complexity_indicators: dict
    model_used: str
    raw_response: str


# ============================================================================
# Prompt Construction
# ============================================================================

def build_estimation_prompt(problem_statement: str,
                            repo: str,
                            hints_text: str = "",
                            patch_info: Optional[str] = None) -> str:
    """
    Build the meta-prompt for time estimation.

    Args:
        problem_statement: The GitHub issue description
        repo: Repository name
        hints_text: Optional hints about the issue
        patch_info: Optional information about the patch (can be used to provide context)

    Returns:
        Complete prompt string for the LLM
    """

    prompt = f"""# Task Time Estimation

You are an expert software engineering analyst. Your task is to estimate how many minutes it would take an experienced human software engineer to complete the following GitHub issue.

## Context

**Repository:** {repo}
**Task Type:** Bug fix / Feature implementation based on GitHub issue

## Important Assumptions

When estimating time, assume the human engineer:
1. Has moderate familiarity with the repository (not a first-time contributor, but not the original author)
2. Has access to standard development tools and documentation
3. Will write proper tests and follow repository conventions
4. Needs to understand the existing code before making changes
5. Will need to verify the fix doesn't introduce regressions
6. Is working without AI assistance (traditional development workflow)

## The GitHub Issue

<problem_statement>
{problem_statement}
</problem_statement>
"""

    if hints_text and hints_text.strip():
        prompt += f"""
## Additional Hints/Context

<hints>
{hints_text}
</hints>
"""

    if patch_info:
        prompt += f"""
## Patch Size Information

{patch_info}

Note: This is provided for context about the scope of changes. A larger patch doesn't always mean more time if the changes are mechanical.
"""

    prompt += """
## Your Task

Analyze the problem statement and estimate the total time in minutes. Your response MUST be in the following JSON format:

```json
{
    "estimated_minutes": <number - your point estimate in minutes>,
    "justification": "<A detailed 2-4 sentence explanation of why this time estimate is appropriate>",
    "time_breakdown": {
        "understanding_and_investigation": <minutes for reading code, understanding the issue, debugging>,
        "implementation": <minutes for writing the actual fix>,
        "testing_and_verification": <minutes for writing tests, running tests, verifying the fix>
    },
    "complexity_indicators": {
        "code_investigation_needed": "<none/minimal/moderate/extensive>",
        "number_of_files_likely_affected": "<1/2-3/4-6/7+>",
        "testing_complexity": "<trivial/simple/moderate/complex>",
        "domain_knowledge_required": "<none/basic/intermediate/expert>",
        "debugging_difficulty": "<straightforward/moderate/challenging/very_challenging>"
    }
}
```

## Estimation Guidelines

Consider these factors when making your estimate:

1. **Problem Clarity**: Is the issue well-defined or vague? Vague issues require more investigation time.

2. **Scope of Changes**: How many files/components are likely affected? Check for mentions of specific files, modules, or broad behavioral changes.

3. **Debugging Complexity**: Is this a straightforward bug with clear reproduction steps, or a subtle issue requiring deep investigation?

4. **Testing Requirements**: How much testing will be needed to verify the fix? Are there existing tests to update or new tests to write?

5. **Domain Knowledge**: Does the issue require specialized knowledge (e.g., astronomy calculations, machine learning algorithms, complex parsing)?

6. **Code Familiarity**: How much time will be spent understanding existing code patterns and architecture?

7. **Edge Cases**: Does the issue mention or imply multiple edge cases that need handling?

8. **Integration Points**: Does the fix touch integration points with external systems or APIs?

## Time Reference Points

Use these as calibration points for your estimates:
- **5-15 minutes**: Trivial fixes like typos, simple one-line changes, obvious bugs with clear solutions
- **15-60 minutes**: Tasks requiring some investigation, changes to 1-3 files, moderate debugging
- **60-240 minutes (1-4 hours)**: Complex tasks requiring understanding of multiple components, significant testing
- **240+ minutes (4+ hours)**: Very complex tasks requiring deep codebase knowledge, architectural changes, extensive debugging

## Important Notes

- Be realistic and consider all phases: understanding, implementing, testing, and reviewing
- A "simple" one-line fix might still require significant investigation time
- Consider the overhead of understanding unfamiliar code patterns
- Err slightly toward higher estimates when uncertain (engineers typically underestimate)
- The repository context matters: popular, well-maintained repos often have complex conventions
- Provide your estimate as a single number (e.g., 45, 90, 180), not a range

Provide your estimation now:
"""

    return prompt


def extract_patch_summary(patch: str) -> str:
    """Extract a summary of patch characteristics without revealing the solution."""
    lines = patch.split('\n')

    # Count files changed
    files_changed = sum(1 for line in lines if line.startswith('diff --git'))

    # Count additions and deletions
    additions = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))

    return f"Files changed: {files_changed}, Lines added: ~{additions}, Lines removed: ~{deletions}"


# ============================================================================
# LLM API Clients
# ============================================================================

def query_openai(prompt: str, model: str = "gpt-4o", reasoning_effort: str = "medium") -> str:
    """
    Query OpenAI API with the given prompt.

    Args:
        prompt: The complete prompt
        model: OpenAI model to use
        reasoning_effort: Reasoning effort for o1/o3/gpt-5 models ("low", "medium", "high")

    Returns:
        Model response text
    """
    from openai import OpenAI

    client = OpenAI()  # Uses OPENAI_API_KEY environment variable

    # Check if this is a reasoning model (o1, o3, gpt-5 series)
    is_reasoning_model = any(
        model.startswith(prefix) or model.startswith(f"openai/{prefix}")
        for prefix in ("o1", "o3", "gpt-5")
    )

    system_message = (
        "You are an expert software engineering analyst specializing in "
        "estimating task complexity and completion time. Always respond with "
        "valid JSON as specified in the prompt."
    )

    if is_reasoning_model:
        # Reasoning models (o1, o3, gpt-5) don't support system messages or temperature
        # Prepend system instructions to the user prompt instead
        combined_prompt = f"Instructions: {system_message}\n\n{prompt}"

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_prompt}],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=32000
        )
    else:
        # Standard models (gpt-4o, gpt-4-turbo, etc.)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=32000
        )

    return response.choices[0].message.content


def query_google(prompt: str, model: str = "gemini-2.5-pro-preview-05-06") -> str:
    """
    Query Google Gemini API with the given prompt.

    Args:
        prompt: The complete prompt
        model: Gemini model to use

    Returns:
        Model response text
    """
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 32000,
    }

    # Configure safety settings to be less restrictive for code analysis
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction="You are an expert software engineering analyst specializing in "
                          "estimating task complexity and completion time. Always respond with "
                          "valid JSON as specified in the prompt."
    )

    response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

    # Map finish reason codes to names for better error messages
    # See: https://ai.google.dev/api/generate-content#finishreason
    FINISH_REASONS = {
        0: "FINISH_REASON_UNSPECIFIED",
        1: "STOP",
        2: "MAX_TOKENS",
        3: "SAFETY",
        4: "RECITATION",
        5: "OTHER",
        6: "BLOCKLIST",
        7: "PROHIBITED_CONTENT",
        8: "SPII",
    }

    # Handle missing candidates
    if not response.candidates:
        # Check if there's a prompt feedback block reason
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'unknown')
            raise ValueError(f"Prompt blocked by Gemini API. Block reason: {block_reason}")
        raise ValueError("No response candidates returned from Gemini API")

    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason
    finish_reason_name = FINISH_REASONS.get(finish_reason, f"UNKNOWN({finish_reason})")

    # Try to extract text from parts first
    text_content = ""
    if candidate.content and candidate.content.parts:
        text_content = "".join(
            part.text for part in candidate.content.parts
            if hasattr(part, 'text') and part.text
        )

    # Handle different finish reasons
    if finish_reason == 1:  # STOP - normal completion
        if text_content:
            return text_content
        raise ValueError("Response completed but no text content returned")

    elif finish_reason == 2:  # MAX_TOKENS - truncated but may have partial content
        if text_content:
            # Return partial content with a note (still usable for parsing)
            return text_content
        raise ValueError("Response truncated (MAX_TOKENS) with no content")

    elif finish_reason == 3:  # SAFETY
        safety_info = getattr(candidate, 'safety_ratings', 'no details')
        raise ValueError(f"Response blocked by safety filter ({finish_reason_name}): {safety_info}")

    elif finish_reason in (4, 6, 7, 8):  # RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII
        raise ValueError(f"Response blocked: {finish_reason_name}")

    else:  # OTHER or unknown
        if text_content:
            return text_content
        raise ValueError(f"Unexpected finish reason: {finish_reason_name}, no content returned")


def query_model(prompt: str, provider: str, model: str, reasoning_effort: str = "medium") -> str:
    """
    Query the specified model provider.

    Args:
        prompt: The complete prompt
        provider: Either 'openai' or 'google'
        model: Model name to use
        reasoning_effort: Reasoning effort for OpenAI o1/o3 models ("low", "medium", "high")

    Returns:
        Model response text
    """
    if provider == "openai":
        return query_openai(prompt, model, reasoning_effort)
    elif provider == "google":
        return query_google(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Response Parsing
# ============================================================================

def parse_response(response: str, instance_id: str, repo: str, model_used: str) -> TimeEstimation:
    """
    Parse the LLM response and extract the time estimation.

    Args:
        response: Raw LLM response
        instance_id: Task instance ID
        repo: Repository name
        model_used: Model that generated the response

    Returns:
        TimeEstimation object
    """
    # Try to extract JSON from the response
    import re

    # Look for JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[^{}]*"estimated_minutes"[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: try parsing the entire response as JSON
            json_str = response

    try:
        data = json.loads(json_str)

        estimated_minutes = float(data.get("estimated_minutes", -1))

        return TimeEstimation(
            instance_id=instance_id,
            repo=repo,
            estimated_minutes=estimated_minutes,
            justification=data.get("justification", ""),
            time_breakdown=data.get("time_breakdown", {}),
            complexity_indicators=data.get("complexity_indicators", {}),
            model_used=model_used,
            raw_response=response
        )
    except json.JSONDecodeError:
        # Return a partial result if parsing fails
        return TimeEstimation(
            instance_id=instance_id,
            repo=repo,
            estimated_minutes=-1,
            justification=f"Failed to parse response: {response[:500]}...",
            time_breakdown={},
            complexity_indicators={},
            model_used=model_used,
            raw_response=response
        )


# ============================================================================
# Ground Truth Conversion
# ============================================================================

def difficulty_to_minutes_midpoint(difficulty: str) -> Optional[float]:
    """
    Convert SWE-bench difficulty bucket to midpoint minutes for comparison.

    Args:
        difficulty: SWE-bench difficulty string

    Returns:
        Midpoint minutes for the bucket, or None if unknown
    """
    # Map difficulty buckets to their midpoint in minutes
    mapping = {
        "<15 min fix": 7.5,        # Midpoint of 0-15
        "15 min - 1 hour": 37.5,   # Midpoint of 15-60
        "1-4 hours": 150,          # Midpoint of 60-240
        ">4 hours": 360,           # Using 6 hours (360 min) as representative for >4 hours
    }
    return mapping.get(difficulty)


def minutes_to_difficulty_bucket(minutes: float) -> str:
    """
    Convert minute estimate to SWE-bench difficulty bucket.

    Args:
        minutes: Estimated minutes

    Returns:
        SWE-bench difficulty bucket string
    """
    if minutes < 15:
        return "<15 min fix"
    elif minutes < 60:
        return "15 min - 1 hour"
    elif minutes < 240:
        return "1-4 hours"
    else:
        return ">4 hours"


# ============================================================================
# Main Processing
# ============================================================================

def process_single_sample(sample, provider, model, reasoning_effort, include_patch_info):
    """
    Process a single sample and return the result dict.

    This function is designed to be called from a thread pool.
    """
    instance_id = sample['instance_id']

    # Build prompt
    patch_info = None
    if include_patch_info:
        patch_info = extract_patch_summary(sample['patch'])

    prompt = build_estimation_prompt(
        problem_statement=sample['problem_statement'],
        repo=sample['repo'],
        hints_text=sample.get('hints_text', ''),
        patch_info=patch_info
    )

    try:
        # Query model
        response = query_model(prompt, provider, model, reasoning_effort)

        # Parse response
        estimation = parse_response(
            response=response,
            instance_id=instance_id,
            repo=sample['repo'],
            model_used=f"{provider}/{model}"
        )

        # Build result
        result = asdict(estimation)
        result['ground_truth_difficulty'] = sample.get('difficulty', '')
        result['estimated_bucket'] = minutes_to_difficulty_bucket(estimation.estimated_minutes) if estimation.estimated_minutes > 0 else 'PARSE_ERROR'
        result['processed_at'] = datetime.now().isoformat()
        return result

    except Exception as e:
        # Return error result
        return {
            'instance_id': instance_id,
            'repo': sample['repo'],
            'estimated_minutes': -1,
            'justification': str(e),
            'model_used': f"{provider}/{model}",
            'ground_truth_difficulty': sample.get('difficulty', ''),
            'estimated_bucket': 'ERROR',
            'processed_at': datetime.now().isoformat()
        }


def process_dataset(
    provider: str = "openai",
    model: str = "gpt-4o",
    output_file: str = "time_estimations.jsonl",
    max_samples: Optional[int] = None,
    include_patch_info: bool = False,
    start_idx: int = 0,
    reasoning_effort: str = "medium",
    batch_size: int = 10,
    verbose: bool = False,
):
    """
    Process the SWE-bench_Verified dataset and estimate task time.

    Args:
        provider: LLM provider ('openai' or 'google')
        model: Model name to use
        output_file: Path to output JSONL file
        max_samples: Maximum number of samples to process (None for all)
        include_patch_info: Whether to include patch size information in prompt
        start_idx: Starting index for processing (for resumption)
        reasoning_effort: Reasoning effort for OpenAI o1/o3 models ("low", "medium", "high")
        batch_size: Number of concurrent requests (default: 10)
        verbose: Print detailed progress information
    """
    if verbose:
        print("Loading SWE-bench_Verified dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    if verbose:
        print(f"Dataset loaded: {len(dataset)} samples")

    # Determine samples to process
    if max_samples:
        end_idx = min(start_idx + max_samples, len(dataset))
    else:
        end_idx = len(dataset)

    if verbose:
        print(f"Processing samples {start_idx} to {end_idx - 1}")
        print(f"Using provider: {provider}, model: {model}")
        if provider == "openai" and any(model.startswith(p) for p in ("o1", "o3", "gpt-5")):
            print(f"Reasoning effort: {reasoning_effort}")
        print(f"Output file: {output_file}")
        print(f"Batch size (concurrent requests): {batch_size}")
        print("-" * 50)

    # Load existing results if resuming
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get('instance_id'))
                except Exception:
                    pass
        if existing_ids and verbose:
            print(f"Found {len(existing_ids)} existing results, will skip those")

    # Collect samples to process (excluding already processed)
    samples_to_process = []
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        if sample['instance_id'] not in existing_ids:
            samples_to_process.append(sample)

    if not samples_to_process:
        if verbose:
            print("All samples already processed!")
        return

    if verbose:
        print(f"Samples to process: {len(samples_to_process)}")

    # Process samples in parallel batches
    with open(output_file, 'a') as f:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    process_single_sample,
                    sample,
                    provider,
                    model,
                    reasoning_effort,
                    include_patch_info
                ): sample
                for sample in samples_to_process
            }

            # Process results as they complete
            for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample), desc="Processing"):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    if result.get('estimated_minutes', -1) < 0:
                        print(f"\nError processing {sample['instance_id']}: {result.get('justification', 'Unknown error')}")
                except Exception as e:
                    print(f"\nUnexpected error processing {sample['instance_id']}: {e}")
                    error_result = {
                        'instance_id': sample['instance_id'],
                        'repo': sample['repo'],
                        'estimated_minutes': -1,
                        'justification': str(e),
                        'model_used': f"{provider}/{model}",
                        'ground_truth_difficulty': sample.get('difficulty', ''),
                        'estimated_bucket': 'ERROR',
                        'processed_at': datetime.now().isoformat()
                    }
                    f.write(json.dumps(error_result) + '\n')
                    f.flush()

    if verbose:
        print(f"\nProcessing complete. Results saved to {output_file}")


def analyze_results(results_file: str) -> None:
    """
    Analyze and summarize the time estimation results.

    Args:
        results_file: Path to the JSONL results file
    """
    import statistics

    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("TIME ESTIMATION ANALYSIS (v3 - Point Estimates)")
    print(f"{'='*60}")
    print(f"Total samples analyzed: {len(results)}")

    # Valid difficulty levels for bucket comparison
    valid_levels = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]

    # Filter to valid results only (positive minute estimates)
    valid_results = [r for r in results if r.get('estimated_minutes', -1) > 0]
    error_count = len(results) - len(valid_results)

    print(f"Valid estimates: {len(valid_results)}")
    if error_count:
        print(f"Errors/Parse failures: {error_count}")

    if not valid_results:
        print("No valid results to analyze!")
        return

    # ========== Minute Estimate Statistics ==========
    print(f"\n{'Estimated Minutes Statistics':}")
    print("-" * 40)

    minutes_list = [r['estimated_minutes'] for r in valid_results]
    print(f"Mean: {statistics.mean(minutes_list):.1f} minutes")
    print(f"Median: {statistics.median(minutes_list):.1f} minutes")
    print(f"Std Dev: {statistics.stdev(minutes_list):.1f} minutes" if len(minutes_list) > 1 else "Std Dev: N/A")
    print(f"Min: {min(minutes_list):.1f} minutes")
    print(f"Max: {max(minutes_list):.1f} minutes")

    # Percentiles
    sorted_minutes = sorted(minutes_list)
    n = len(sorted_minutes)
    p25_idx = int(n * 0.25)
    p75_idx = int(n * 0.75)
    p90_idx = int(n * 0.90)
    print(f"25th percentile: {sorted_minutes[p25_idx]:.1f} minutes")
    print(f"75th percentile: {sorted_minutes[p75_idx]:.1f} minutes")
    print(f"90th percentile: {sorted_minutes[p90_idx]:.1f} minutes")

    # ========== Distribution by Estimated Bucket ==========
    print(f"\n{'Estimated Bucket Distribution (derived from minutes)':}")
    print("-" * 40)

    estimated_bucket_counts = Counter(r.get('estimated_bucket', 'UNKNOWN') for r in valid_results)
    for level in valid_levels:
        count = estimated_bucket_counts.get(level, 0)
        pct = count / len(valid_results) * 100 if valid_results else 0
        bar = '\u2588' * int(pct / 2)
        print(f"{level:20} | {count:4} ({pct:5.1f}%) {bar}")

    # ========== Ground Truth Distribution ==========
    print(f"\n{'Ground Truth Difficulty Distribution':}")
    print("-" * 40)

    ground_truth_counts = Counter(r.get('ground_truth_difficulty', 'UNKNOWN') for r in valid_results)
    for level in valid_levels:
        count = ground_truth_counts.get(level, 0)
        pct = count / len(valid_results) * 100 if valid_results else 0
        bar = '\u2588' * int(pct / 2)
        print(f"{level:20} | {count:4} ({pct:5.1f}%) {bar}")

    # ========== Bucket Accuracy Analysis ==========
    print(f"\n{'Bucket Accuracy Analysis (estimated bucket vs ground truth)':}")
    print("-" * 40)

    # Filter to results with valid ground truth
    bucket_comparable = [r for r in valid_results if r.get('ground_truth_difficulty') in valid_levels]

    if bucket_comparable:
        # Exact bucket match accuracy
        exact_matches = sum(1 for r in bucket_comparable
                          if r.get('estimated_bucket') == r.get('ground_truth_difficulty'))
        exact_accuracy = exact_matches / len(bucket_comparable) * 100
        print(f"Exact bucket match: {exact_matches}/{len(bucket_comparable)} ({exact_accuracy:.1f}%)")

        # Within-one-bucket accuracy
        level_order = {level: i for i, level in enumerate(valid_levels)}
        within_one = sum(1 for r in bucket_comparable
                        if abs(level_order.get(r.get('estimated_bucket'), -10) -
                              level_order.get(r.get('ground_truth_difficulty'), -10)) <= 1)
        within_one_accuracy = within_one / len(bucket_comparable) * 100
        print(f"Within-one-bucket: {within_one}/{len(bucket_comparable)} ({within_one_accuracy:.1f}%)")

        # Over/under estimation
        over_estimates = sum(1 for r in bucket_comparable
                           if level_order.get(r.get('estimated_bucket'), 0) >
                              level_order.get(r.get('ground_truth_difficulty'), 0))
        under_estimates = sum(1 for r in bucket_comparable
                            if level_order.get(r.get('estimated_bucket'), 0) <
                               level_order.get(r.get('ground_truth_difficulty'), 0))
        print(f"Over-estimates: {over_estimates} ({over_estimates/len(bucket_comparable)*100:.1f}%)")
        print(f"Under-estimates: {under_estimates} ({under_estimates/len(bucket_comparable)*100:.1f}%)")

    # ========== Minutes vs Ground Truth Midpoint ==========
    print(f"\n{'Minutes Comparison with Ground Truth Midpoints':}")
    print("-" * 40)

    comparable_with_midpoint = []
    for r in bucket_comparable:
        gt_midpoint = difficulty_to_minutes_midpoint(r.get('ground_truth_difficulty', ''))
        if gt_midpoint is not None:
            comparable_with_midpoint.append({
                'estimated': r['estimated_minutes'],
                'gt_midpoint': gt_midpoint,
                'gt_bucket': r['ground_truth_difficulty']
            })

    if comparable_with_midpoint:
        # Calculate error metrics
        errors = [c['estimated'] - c['gt_midpoint'] for c in comparable_with_midpoint]
        abs_errors = [abs(e) for e in errors]

        print(f"Mean Absolute Error (MAE): {statistics.mean(abs_errors):.1f} minutes")
        print(f"Mean Error (bias): {statistics.mean(errors):.1f} minutes")
        print(f"Median Absolute Error: {statistics.median(abs_errors):.1f} minutes")

        # Root Mean Square Error
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        print(f"Root Mean Square Error (RMSE): {rmse:.1f} minutes")

        # By ground truth bucket
        print("\nMean estimated minutes by ground truth bucket:")
        for level in valid_levels:
            bucket_estimates = [c['estimated'] for c in comparable_with_midpoint if c['gt_bucket'] == level]
            if bucket_estimates:
                gt_midpoint = difficulty_to_minutes_midpoint(level)
                mean_est = statistics.mean(bucket_estimates)
                print(f"  {level:20}: {mean_est:6.1f} min (GT midpoint: {gt_midpoint:.1f})")

    # ========== Confusion Matrix ==========
    print(f"\n{'Confusion Matrix (Estimated Bucket vs Ground Truth)':}")
    print("-" * 60)

    # Build confusion matrix
    confusion = {est: {gt: 0 for gt in valid_levels} for est in valid_levels}
    for r in bucket_comparable:
        est = r.get('estimated_bucket')
        gt = r.get('ground_truth_difficulty')
        if est in valid_levels and gt in valid_levels:
            confusion[est][gt] += 1

    # Print header
    header_label = "Estimated \\ Ground Truth"
    print(f"{header_label:25}", end="")
    for gt in valid_levels:
        print(f"{gt:>15}", end="")
    print()

    # Print rows
    for est in valid_levels:
        print(f"{est:25}", end="")
        for gt in valid_levels:
            count = confusion[est][gt]
            print(f"{count:>15}", end="")
        print()

    # ========== Repository Breakdown ==========
    print(f"\n{'Repository Breakdown (bucket accuracy)':}")
    print("-" * 40)

    repo_results = {}
    for r in bucket_comparable:
        repo = r.get('repo', 'unknown')
        if repo not in repo_results:
            repo_results[repo] = {'correct': 0, 'total': 0, 'estimates': []}
        repo_results[repo]['total'] += 1
        repo_results[repo]['estimates'].append(r['estimated_minutes'])
        if r.get('estimated_bucket') == r.get('ground_truth_difficulty'):
            repo_results[repo]['correct'] += 1

    for repo, stats in sorted(repo_results.items(), key=lambda x: x[1]['total'], reverse=True):
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        mean_est = statistics.mean(stats['estimates']) if stats['estimates'] else 0
        print(f"{repo}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) | mean est: {mean_est:.0f} min")


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate task completion time for SWE-bench tasks using LLMs (v3 - point estimates in minutes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with OpenAI GPT-4o
  python swebench_human_time_llm_estimate.py --provider openai --model gpt-4o

  # Process with OpenAI o3-mini with high reasoning effort
  python swebench_human_time_llm_estimate.py --provider openai --model o3-mini --reasoning-effort high

  # Process with Google Gemini
  python swebench_human_time_llm_estimate.py --provider google --model gemini-2.5-pro-preview-05-06

  # Process first 10 samples only
  python swebench_human_time_llm_estimate.py --max-samples 10

  # Resume from sample 100
  python swebench_human_time_llm_estimate.py --start-idx 100

  # Analyze existing results
  python swebench_human_time_llm_estimate.py --analyze-only --results-file time_estimations.jsonl

Environment variables:
  OPENAI_API_KEY    - Required for OpenAI models
  GOOGLE_API_KEY    - Required for Google Gemini models
        """
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "google"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use. Defaults: openai=gpt-4o, google=gemini-2.5-flash",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file path (default: time_estimations_v3_{provider}_{model}.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for processing (default: 0)",
    )
    parser.add_argument(
        "--include-patch-info",
        action="store_true",
        help="Include patch size information in the prompt (may bias estimates)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort for OpenAI o1/o3/gpt-5 models (default: medium)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results, don't process new samples",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Results file to analyze (used with --analyze-only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set default model based on provider
    if args.model is None:
        if args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "gemini-2.5-flash"

    # Set default output file
    if args.output_file is None:
        model_safe = args.model.replace("/", "_").replace(":", "_")
        args.output_file = f"time_estimations_v3_{args.provider}_{model_safe}.jsonl"

    if args.analyze_only:
        results_file = args.results_file or args.output_file
        if not os.path.exists(results_file):
            raise SystemExit(f"Error: Results file not found: {results_file}")
        analyze_results(results_file)
    else:
        # Check for API keys
        if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Error: OPENAI_API_KEY environment variable not set")
        if args.provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
            raise SystemExit("Error: GOOGLE_API_KEY environment variable not set")

        process_dataset(
            provider=args.provider,
            model=args.model,
            output_file=args.output_file,
            max_samples=args.max_samples,
            include_patch_info=args.include_patch_info,
            start_idx=args.start_idx,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        # Also run analysis after processing
        if os.path.exists(args.output_file):
            analyze_results(args.output_file)


if __name__ == "__main__":
    main()
