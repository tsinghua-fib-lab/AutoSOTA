import argparse
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"


# ============================================================================
# Constants
# ============================================================================

# CyBench difficulty levels (from metadata schema)
DIFFICULTY_LABELS = {
    "0": "other",
    "1": "easy",
    "2": "intermediate",
    "3": "hard",
    "4": "expert",
    "5": "master",
}

# CTF categories
CTF_CATEGORIES = [
    "crypto", "forensics", "misc", "ppc", "pwn",
    "reverse", "web", "blockchain", "steganography"
]

# First Solve Time (FST) data from the CyBench paper (arxiv.org/abs/2408.08926)
# Times are in minutes, representing how long it took human CTF teams to first solve each task
# Format: normalized task name -> FST in minutes
TASK_FST_MINUTES = {
    # Tasks with Subtasks (from paper Table 2)
    "urgent": 4,
    "flag command": 6,
    "flag_command": 6,
    "it has begun": 7,
    "it_has_begun": 7,
    "dynastic": 9,
    "primary knowledge": 11,
    "primary_knowledge": 11,
    "data siege": 78,  # 01:18
    "data_siege": 78,
    "unbreakable": 123,  # 02:03
    "flecks of gold": 368,  # 06:08
    "flecksofgold": 368,
    "missing bits": 9,
    "slcg": 29,
    "rpgo": 45,
    "walking to the seaside": 133,  # 02:13
    "eval me": 11,
    "eval_me": 11,
    "noisier crc": 85,  # 01:25
    "noisier_crc": 85,
    "noisier-crc": 85,
    "frog waf": 330,  # 05:30
    "frog_waf": 330,
    "frog-waf": 330,
    "back to the past": 32,
    "motp": 52,

    # Tasks without Subtasks (from paper Table 3)
    "loot stash": 2,
    "lootstash": 2,
    "packed away": 6,
    "packedaway": 6,
    "crushing": 20,
    "permuted": 42,
    "labyrinth linguist": 43,
    "labyrinth_linguist": 43,
    "partial tenacity": 65,  # 01:05
    "partial_tenacity": 65,
    "delulu": 75,  # 01:15
    "were pickle phreaks revenge": 120,  # 02:00
    "were_pickle_phreaks_revenge": 120,
    "lock talk": 132,  # 02:12
    "locktalk": 132,
    "skilift": 5,
    "glacier exchange": 9,
    "sop": 51,
    "avatar": 84,  # 01:24
    "shuffled aes": 159,  # 02:39
    "noisy crc": 35,
    "noisy_crc": 35,
    "noisy-crc": 35,
    "network tools": 42,
    "network_tools": 42,
    "network-tools": 42,
    "chunky": 105,  # 01:45
    "just another pickle jail": 244,  # 04:04
    "just-another-pickle-jail": 244,
    "randsubware": 356,  # 05:56
    "failproof": 60,  # 01:00
    "ezmaze": 204,  # 03:24
    "diffecient": 454,  # 07:34
    "robust cbc": 1494,  # 24:54
    "robust_cbc": 1494,
    "robust-cbc": 1494,

    # Additional task name variations for matching
    "boxcutter": None,
    "quickscan": None,
    "maze": None,
    "character": None,
    "stop drop and roll": None,
    "path of survival": None,
    "rids": None,
    "the prom": None,
    "flash-ing logs": None,
    "iced tea": None,
    "blunt": None,
    "tsayaki": None,
    "rot128": None,
    "arranged": None,
    "makeshift": None,
    "lucky faucet": None,
    "recovery": None,
    "russian roulette": None,
    "cubicle riddle": None,
    "were pickle phreaks": None,
    "multidigilingual": None,
    "colored squares": None,
    "elgamal": None,
    "matrix lab 2": None,
    "guardians of the kernel": None,
    "text sender": None,
    "algorithm multitool": None,
}


# ============================================================================
# Data Classes
# ============================================================================

def lookup_fst(task_name: str) -> Optional[float]:
    """
    Look up the First Solve Time (FST) for a task by name.

    Args:
        task_name: The task name (various formats supported)

    Returns:
        FST in minutes, or None if not found
    """
    # Normalize the task name for lookup
    # Remove difficulty brackets like [Easy], [Hard], etc.
    normalized = re.sub(r'\[.*?\]', '', task_name).strip()
    # Convert to lowercase and try various formats
    normalized_lower = normalized.lower()

    # Try exact match first
    if normalized_lower in TASK_FST_MINUTES:
        return TASK_FST_MINUTES[normalized_lower]

    # Try with underscores/hyphens replaced by spaces
    with_spaces = normalized_lower.replace('_', ' ').replace('-', ' ')
    if with_spaces in TASK_FST_MINUTES:
        return TASK_FST_MINUTES[with_spaces]

    # Try removing "very easy", "easy", "medium", "hard", "insane" prefixes/suffixes
    for prefix in ['very easy ', 'easy ', 'medium ', 'hard ', 'insane ', 'expert ', 'master ']:
        if with_spaces.startswith(prefix):
            stripped = with_spaces[len(prefix):]
            if stripped in TASK_FST_MINUTES:
                return TASK_FST_MINUTES[stripped]

    # Try removing suffixes like " very easy", " easy", etc.
    for suffix in [' very easy', ' easy', ' medium', ' hard', ' insane', ' expert', ' master']:
        if with_spaces.endswith(suffix):
            stripped = with_spaces[:-len(suffix)]
            if stripped in TASK_FST_MINUTES:
                return TASK_FST_MINUTES[stripped]

    # Handle numbered prefix format like "22-back-to-the-past" -> "back to the past"
    numbered_match = re.match(r'^\d+[\-_\s]+(.+)$', with_spaces)
    if numbered_match:
        stripped = numbered_match.group(1)
        if stripped in TASK_FST_MINUTES:
            return TASK_FST_MINUTES[stripped]

    # Try exact substring match only for longer names (avoid false matches like "maze" -> "ezmaze")
    if len(normalized_lower) >= 6:
        for key, value in TASK_FST_MINUTES.items():
            if value is not None and len(key) >= 6:
                # Only match if it's a substantial overlap
                if normalized_lower == key or with_spaces == key:
                    return value

    return None


@dataclass
class CyBenchTask:
    """Represents a single CyBench CTF task."""
    task_id: str
    task_name: str
    task_path: str
    categories: list
    competition: str
    competition_year: str
    difficulty: str
    easy_prompt: str
    hard_prompt: str
    target_host: str
    tools: dict
    internet: dict
    stateful_shell: bool
    subtasks: list
    num_subtasks: int
    fst_minutes: Optional[float]  # First Solve Time from paper (ground truth)


@dataclass
class TimeEstimation:
    """Stores the time estimation result for a single task."""
    task_id: str
    task_name: str
    categories: list
    competition: str
    difficulty: str
    fst_minutes: Optional[float]  # Ground truth from paper
    estimated_minutes: float
    justification: str
    time_breakdown: dict
    complexity_indicators: dict
    model_used: str
    raw_response: str


# ============================================================================
# Task Discovery
# ============================================================================

def discover_tasks(benchmark_path: str) -> list[CyBenchTask]:
    """
    Discover all CyBench tasks by finding metadata.json files.

    Args:
        benchmark_path: Path to the CyBench benchmark directory

    Returns:
        List of CyBenchTask objects
    """
    tasks = []
    benchmark_path = Path(benchmark_path)

    # Find all metadata.json files
    for metadata_file in benchmark_path.rglob("metadata/metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract task info from path
            task_dir = metadata_file.parent.parent
            rel_path = task_dir.relative_to(benchmark_path)
            parts = rel_path.parts

            # Determine competition and structure
            if len(parts) >= 2:
                competition = parts[0]
                competition_year = ""
                for part in parts:
                    if any(str(year) in part for year in range(2020, 2030)):
                        competition_year = part
                        break
            else:
                competition = "unknown"
                competition_year = ""

            # Build task ID
            task_id = str(rel_path).replace(" ", "_").replace("[", "").replace("]", "")
            task_name = task_dir.name

            # Look up FST from paper data
            fst = lookup_fst(task_name)

            task = CyBenchTask(
                task_id=task_id,
                task_name=task_name,
                task_path=str(task_dir),
                categories=metadata.get("categories", []),
                competition=competition,
                competition_year=competition_year,
                difficulty=str(metadata.get("difficulty", "0")),
                easy_prompt=metadata.get("easy_prompt", ""),
                hard_prompt=metadata.get("hard_prompt", ""),
                target_host=metadata.get("target_host", ""),
                tools=metadata.get("tools", {}),
                internet=metadata.get("internet", {}),
                stateful_shell=metadata.get("stateful_shell", False),
                subtasks=metadata.get("subtasks", []),
                num_subtasks=len(metadata.get("subtasks", [])),
                fst_minutes=fst,
            )
            tasks.append(task)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {metadata_file}: {e}")
            continue

    return tasks


# ============================================================================
# Prompt Construction
# ============================================================================

def build_estimation_prompt(task: CyBenchTask, use_hard_prompt: bool = False,
                            include_subtasks: bool = True) -> str:
    """
    Build the meta-prompt for time estimation.

    Args:
        task: CyBenchTask object
        use_hard_prompt: Whether to use the harder prompt variant
        include_subtasks: Whether to include subtask information

    Returns:
        Complete prompt string for the LLM
    """
    prompt_text = task.hard_prompt if use_hard_prompt else task.easy_prompt
    categories_str = ", ".join(task.categories) if task.categories else "unknown"
    difficulty_label = DIFFICULTY_LABELS.get(task.difficulty, "unknown")

    # Build tools summary
    tools_list = []
    if task.tools.get("terminal"):
        tools_list.append(f"Terminal: {', '.join(task.tools['terminal'])}")
    if task.tools.get("gui"):
        tools_list.append(f"GUI: {', '.join(task.tools['gui'])}")
    if task.tools.get("languages"):
        tools_list.append(f"Languages: {', '.join(task.tools['languages'])}")
    tools_str = "; ".join(tools_list) if tools_list else "Not specified"

    prompt = f"""# CTF Task Time Estimation

You are an expert cybersecurity analyst and CTF (Capture The Flag) competitor. Your task is to estimate how many minutes it would take an experienced human security researcher to complete the following CTF challenge.

## Challenge Context

**Competition:** {task.competition} ({task.competition_year})
**Category:** {categories_str}
**Labeled Difficulty:** {difficulty_label}
**Task Name:** {task.task_name}
**Tools Required:** {tools_str}
**Internet Required:** {"Yes" if task.internet.get("necessary") else "No"} (Useful: {"Yes" if task.internet.get("useful") else "No"})
**Requires Stateful Shell:** {"Yes" if task.stateful_shell else "No"}
**Number of Subtasks:** {task.num_subtasks}

## Important Assumptions

When estimating time, assume the human:
1. Has solid CTF experience (has competed in several CTFs before)
2. Is familiar with the category (e.g., knows common crypto attacks, web exploits, reverse engineering tools)
3. Has access to standard security tools (Ghidra, Burp Suite, pwntools, etc.)
4. Is working alone without AI assistance
5. May need to research specific vulnerabilities or techniques
6. Needs to understand the challenge before exploiting it

## Challenge Description

<challenge_prompt>
{prompt_text}
</challenge_prompt>
"""

    if include_subtasks and task.subtasks:
        prompt += "\n## Challenge Subtasks/Steps\n\n"
        for i, subtask in enumerate(task.subtasks, 1):
            subtask_desc = subtask.get("subtask", subtask.get("question", ""))
            hints_count = len(subtask.get("hints", []))
            prompt += f"{i}. {subtask_desc}"
            if hints_count > 0:
                prompt += f" ({hints_count} hints available)"
            prompt += "\n"

    if task.target_host:
        prompt += f"\n**Note:** This challenge requires interacting with a server at {task.target_host}\n"

    prompt += """
## Your Task

Analyze the challenge and estimate the total time in minutes. Your response MUST be in the following JSON format:

```json
{
    "estimated_minutes": <number - your point estimate in minutes>,
    "justification": "<A detailed 2-4 sentence explanation of why this time estimate is appropriate>",
    "time_breakdown": {
        "reconnaissance_and_analysis": <minutes for understanding the challenge, analyzing files/code>,
        "research_and_learning": <minutes for looking up techniques, vulnerabilities, tools>,
        "exploitation_development": <minutes for writing exploits, scripts, payloads>,
        "execution_and_flag_capture": <minutes for running exploits, debugging, capturing flag>
    },
    "complexity_indicators": {
        "category_expertise_needed": "<basic/intermediate/advanced/expert>",
        "tool_complexity": "<simple/moderate/complex/specialized>",
        "attack_chain_length": "<single_step/few_steps/multi_step/complex_chain>",
        "research_required": "<none/minimal/moderate/extensive>",
        "debugging_difficulty": "<straightforward/moderate/challenging/very_challenging>"
    }
}
```

## Estimation Guidelines

Consider these factors when making your estimate:

1. **Category Complexity**: Different categories have different baseline times
   - **Crypto**: Often requires mathematical analysis and custom script writing
   - **Web**: May involve enumeration, multiple vulnerability types, chained exploits
   - **Pwn/Binary**: Requires reverse engineering, exploit development, dealing with protections
   - **Reverse**: Time-intensive analysis of binaries or obfuscated code
   - **Forensics**: May require specialized tools and careful artifact analysis
   - **Misc**: Highly variable, often involves creative problem-solving

2. **Challenge Scope**: How many components/steps are involved?

3. **Research Requirements**: Does the challenge require looking up specific CVEs, techniques, or algorithms?

4. **Tool Setup**: Are specialized tools needed that might require setup time?

5. **Difficulty Label**: Use the labeled difficulty as a calibration point, but analyze the actual challenge

## Time Reference Points for CTF Challenges

Use these as calibration points (for an experienced CTF player):
- **5-20 minutes**: Trivial challenges, simple encoding, obvious vulnerabilities
- **20-45 minutes**: Easy challenges requiring basic exploitation or analysis
- **45-120 minutes (0.75-2 hours)**: Intermediate challenges with some complexity
- **120-300 minutes (2-5 hours)**: Hard challenges requiring significant effort
- **300-600 minutes (5-10 hours)**: Expert challenges with complex attack chains
- **600+ minutes (10+ hours)**: Master-level challenges requiring deep expertise

## Important Notes

- Be realistic about all phases: understanding, research, implementation, debugging
- CTF challenges often have rabbit holes that consume time
- Consider that challenges may require iterative approaches
- Labeled difficulty is a hint but may not reflect actual time required
- Provide your estimate as a single number (e.g., 45, 90, 180), not a range

Provide your estimation now:
"""

    return prompt


# ============================================================================
# LLM API Clients
# ============================================================================

def query_openai(prompt: str, model: str = "gpt-4o", reasoning_effort: str = "medium") -> str:
    """Query OpenAI API with the given prompt."""
    from openai import OpenAI

    client = OpenAI()

    is_reasoning_model = any(
        model.startswith(prefix) or model.startswith(f"openai/{prefix}")
        for prefix in ("o1", "o3", "gpt-5")
    )

    system_message = (
        "You are an expert cybersecurity analyst and CTF competitor specializing in "
        "estimating task complexity and completion time for security challenges. "
        "Always respond with valid JSON as specified in the prompt."
    )

    if is_reasoning_model:
        combined_prompt = f"Instructions: {system_message}\n\n{prompt}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_prompt}],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=32000
        )
    else:
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
    """Query Google Gemini API with the given prompt."""
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 32000,
    }

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction="You are an expert cybersecurity analyst and CTF competitor "
                          "specializing in estimating task complexity and completion time. "
                          "Always respond with valid JSON as specified in the prompt."
    )

    response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

    FINISH_REASONS = {
        0: "FINISH_REASON_UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS",
        3: "SAFETY", 4: "RECITATION", 5: "OTHER",
        6: "BLOCKLIST", 7: "PROHIBITED_CONTENT", 8: "SPII",
    }

    if not response.candidates:
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'unknown')
            raise ValueError(f"Prompt blocked by Gemini API. Block reason: {block_reason}")
        raise ValueError("No response candidates returned from Gemini API")

    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason
    finish_reason_name = FINISH_REASONS.get(finish_reason, f"UNKNOWN({finish_reason})")

    text_content = ""
    if candidate.content and candidate.content.parts:
        text_content = "".join(
            part.text for part in candidate.content.parts
            if hasattr(part, 'text') and part.text
        )

    if finish_reason == 1:  # STOP
        if text_content:
            return text_content
        raise ValueError("Response completed but no text content returned")
    elif finish_reason == 2:  # MAX_TOKENS
        if text_content:
            return text_content
        raise ValueError("Response truncated (MAX_TOKENS) with no content")
    elif finish_reason == 3:  # SAFETY
        safety_info = getattr(candidate, 'safety_ratings', 'no details')
        raise ValueError(f"Response blocked by safety filter ({finish_reason_name}): {safety_info}")
    elif finish_reason in (4, 6, 7, 8):
        raise ValueError(f"Response blocked: {finish_reason_name}")
    else:
        if text_content:
            return text_content
        raise ValueError(f"Unexpected finish reason: {finish_reason_name}, no content returned")


def query_model(prompt: str, provider: str, model: str, reasoning_effort: str = "medium") -> str:
    """Query the specified model provider."""
    if provider == "openai":
        return query_openai(prompt, model, reasoning_effort)
    elif provider == "google":
        return query_google(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Response Parsing
# ============================================================================

def parse_response(response: str, task: CyBenchTask, model_used: str) -> TimeEstimation:
    """Parse the LLM response and extract the time estimation."""
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
            json_str = response

    try:
        data = json.loads(json_str)
        estimated_minutes = float(data.get("estimated_minutes", -1))

        return TimeEstimation(
            task_id=task.task_id,
            task_name=task.task_name,
            categories=task.categories,
            competition=task.competition,
            difficulty=task.difficulty,
            fst_minutes=task.fst_minutes,
            estimated_minutes=estimated_minutes,
            justification=data.get("justification", ""),
            time_breakdown=data.get("time_breakdown", {}),
            complexity_indicators=data.get("complexity_indicators", {}),
            model_used=model_used,
            raw_response=response
        )
    except json.JSONDecodeError:
        return TimeEstimation(
            task_id=task.task_id,
            task_name=task.task_name,
            categories=task.categories,
            competition=task.competition,
            difficulty=task.difficulty,
            fst_minutes=task.fst_minutes,
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

def minutes_to_difficulty_bucket(minutes: float) -> str:
    """Convert minute estimate to CyBench difficulty bucket."""
    if minutes < 20:
        return "0"  # other/trivial
    elif minutes < 45:
        return "1"  # easy
    elif minutes < 150:
        return "2"  # intermediate
    elif minutes < 300:
        return "3"  # hard
    elif minutes < 480:
        return "4"  # expert
    else:
        return "5"  # master


# ============================================================================
# Main Processing
# ============================================================================

def process_single_task(task: CyBenchTask, provider: str, model: str,
                        reasoning_effort: str, use_hard_prompt: bool,
                        include_subtasks: bool) -> dict:
    """Process a single task and return the result dict."""
    prompt = build_estimation_prompt(
        task=task,
        use_hard_prompt=use_hard_prompt,
        include_subtasks=include_subtasks
    )

    try:
        response = query_model(prompt, provider, model, reasoning_effort)
        estimation = parse_response(response=response, task=task, model_used=f"{provider}/{model}")

        result = asdict(estimation)
        result['ground_truth_difficulty'] = task.difficulty
        result['ground_truth_label'] = DIFFICULTY_LABELS.get(task.difficulty, 'unknown')
        result['estimated_difficulty'] = minutes_to_difficulty_bucket(estimation.estimated_minutes) if estimation.estimated_minutes > 0 else 'PARSE_ERROR'
        result['estimated_label'] = DIFFICULTY_LABELS.get(result['estimated_difficulty'], 'unknown') if result['estimated_difficulty'] != 'PARSE_ERROR' else 'PARSE_ERROR'
        result['num_subtasks'] = task.num_subtasks
        result['processed_at'] = datetime.now().isoformat()
        return result

    except Exception as e:
        return {
            'task_id': task.task_id,
            'task_name': task.task_name,
            'categories': task.categories,
            'competition': task.competition,
            'difficulty': task.difficulty,
            'fst_minutes': task.fst_minutes,
            'estimated_minutes': -1,
            'justification': str(e),
            'model_used': f"{provider}/{model}",
            'ground_truth_difficulty': task.difficulty,
            'ground_truth_label': DIFFICULTY_LABELS.get(task.difficulty, 'unknown'),
            'estimated_difficulty': 'ERROR',
            'estimated_label': 'ERROR',
            'num_subtasks': task.num_subtasks,
            'processed_at': datetime.now().isoformat()
        }


def process_tasks(
    benchmark_path: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    output_file: str = "cybench_time_estimations.jsonl",
    max_tasks: Optional[int] = None,
    category_filter: Optional[str] = None,
    competition_filter: Optional[str] = None,
    use_hard_prompt: bool = False,
    include_subtasks: bool = True,
    reasoning_effort: str = "medium",
    batch_size: int = 10,
    verbose: bool = False,
) -> None:
    """Process CyBench tasks and estimate completion time."""
    if verbose:
        print("Discovering CyBench tasks...")
    tasks = discover_tasks(benchmark_path)
    if verbose:
        print(f"Found {len(tasks)} tasks")

    # Apply filters
    if category_filter:
        category_filter = category_filter.lower()
        tasks = [t for t in tasks if category_filter in [c.lower() for c in t.categories]]
        if verbose:
            print(f"After category filter '{category_filter}': {len(tasks)} tasks")

    if competition_filter:
        competition_filter = competition_filter.lower()
        tasks = [t for t in tasks if competition_filter in t.competition.lower()]
        if verbose:
            print(f"After competition filter '{competition_filter}': {len(tasks)} tasks")

    if max_tasks:
        tasks = tasks[:max_tasks]
        if verbose:
            print(f"Limited to {len(tasks)} tasks")

    if verbose:
        print(f"Using provider: {provider}, model: {model}")
        if provider == "openai" and any(model.startswith(p) for p in ("o1", "o3", "gpt-5")):
            print(f"Reasoning effort: {reasoning_effort}")
        print(f"Prompt variant: {'hard' if use_hard_prompt else 'easy'}")
        print(f"Include subtasks: {include_subtasks}")
        print(f"Output file: {output_file}")
        print(f"Batch size: {batch_size}")
        print("-" * 50)

    # Load existing results if resuming
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get('task_id'))
                except Exception:
                    pass
        if existing_ids and verbose:
            print(f"Found {len(existing_ids)} existing results, will skip those")

    # Filter out already processed tasks
    tasks_to_process = [t for t in tasks if t.task_id not in existing_ids]

    if not tasks_to_process:
        if verbose:
            print("All tasks already processed!")
        return

    if verbose:
        print(f"Tasks to process: {len(tasks_to_process)}")

    # Process tasks in parallel
    with open(output_file, 'a') as f:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_task = {
                executor.submit(
                    process_single_task, task, provider, model,
                    reasoning_effort, use_hard_prompt, include_subtasks
                ): task
                for task in tasks_to_process
            }

            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing"):
                task = future_to_task[future]
                try:
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    if result.get('estimated_minutes', -1) < 0:
                        print(f"\nError processing {task.task_id}: {result.get('justification', 'Unknown error')}")
                except Exception as e:
                    print(f"\nUnexpected error processing {task.task_id}: {e}")
                    error_result = {
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                        'categories': task.categories,
                        'competition': task.competition,
                        'difficulty': task.difficulty,
                        'estimated_minutes': -1,
                        'justification': str(e),
                        'model_used': f"{provider}/{model}",
                        'ground_truth_difficulty': task.difficulty,
                        'estimated_difficulty': 'ERROR',
                        'num_subtasks': task.num_subtasks,
                        'processed_at': datetime.now().isoformat()
                    }
                    f.write(json.dumps(error_result) + '\n')
                    f.flush()

    if verbose:
        print(f"\nProcessing complete. Results saved to {output_file}")


def analyze_results(results_file: str) -> None:
    """Analyze and summarize the time estimation results."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("CYBENCH TIME ESTIMATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total tasks analyzed: {len(results)}")

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

    # ========== Distribution by Category ==========
    print(f"\n{'Mean Estimated Minutes by Category':}")
    print("-" * 40)

    category_estimates = {}
    for r in valid_results:
        for cat in r.get('categories', ['unknown']):
            if cat not in category_estimates:
                category_estimates[cat] = []
            category_estimates[cat].append(r['estimated_minutes'])

    for cat, estimates in sorted(category_estimates.items(), key=lambda x: -len(x[1])):
        mean_est = statistics.mean(estimates)
        print(f"{cat:15} | {len(estimates):3} tasks | mean: {mean_est:6.1f} min")

    # ========== FST Comparison (Ground Truth) ==========
    print(f"\n{'Comparison with First Solve Time (FST) Ground Truth':}")
    print("-" * 50)

    fst_results = [r for r in valid_results if r.get('fst_minutes') is not None]
    print(f"Tasks with FST ground truth: {len(fst_results)}/{len(valid_results)}")

    if fst_results:
        errors = [r['estimated_minutes'] - r['fst_minutes'] for r in fst_results]
        abs_errors = [abs(e) for e in errors]
        relative_errors = [abs(r['estimated_minutes'] - r['fst_minutes']) / r['fst_minutes'] * 100
                          for r in fst_results if r['fst_minutes'] > 0]

        print(f"Mean Absolute Error (MAE): {statistics.mean(abs_errors):.1f} minutes")
        print(f"Median Absolute Error: {statistics.median(abs_errors):.1f} minutes")
        print(f"Mean Error (bias): {statistics.mean(errors):+.1f} minutes")
        if relative_errors:
            print(f"Mean Relative Error: {statistics.mean(relative_errors):.1f}%")

        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        print(f"Root Mean Square Error (RMSE): {rmse:.1f} minutes")

        if len(fst_results) > 2:
            fst_vals = [r['fst_minutes'] for r in fst_results]
            est_vals = [r['estimated_minutes'] for r in fst_results]
            mean_fst = statistics.mean(fst_vals)
            mean_est = statistics.mean(est_vals)
            numerator = sum((f - mean_fst) * (e - mean_est) for f, e in zip(fst_vals, est_vals))
            denom_fst = sum((f - mean_fst) ** 2 for f in fst_vals) ** 0.5
            denom_est = sum((e - mean_est) ** 2 for e in est_vals) ** 0.5
            if denom_fst > 0 and denom_est > 0:
                correlation = numerator / (denom_fst * denom_est)
                print(f"Pearson Correlation with FST: {correlation:.3f}")

        print("\nPer-task FST vs Estimated (sorted by FST):")
        print(f"{'Task':<35} {'FST':>8} {'Est':>8} {'Error':>8}")
        print("-" * 65)
        for r in sorted(fst_results, key=lambda x: x['fst_minutes']):
            error = r['estimated_minutes'] - r['fst_minutes']
            task_short = r['task_name'][:33] if len(r['task_name']) > 33 else r['task_name']
            print(f"{task_short:<35} {r['fst_minutes']:>7.0f}m {r['estimated_minutes']:>7.0f}m {error:>+7.0f}m")

    # ========== Distribution by Ground Truth Difficulty ==========
    print(f"\n{'Mean Estimated Minutes by Difficulty Label':}")
    print("-" * 40)

    for diff in sorted(DIFFICULTY_LABELS.keys()):
        diff_results = [r for r in valid_results if r.get('ground_truth_difficulty') == diff]
        if diff_results:
            mean_est = statistics.mean([r['estimated_minutes'] for r in diff_results])
            label = DIFFICULTY_LABELS[diff]
            fst_for_diff = [r['fst_minutes'] for r in diff_results if r.get('fst_minutes') is not None]
            fst_str = f" (mean FST: {statistics.mean(fst_for_diff):.0f}m)" if fst_for_diff else ""
            print(f"{diff} ({label:12}) | {len(diff_results):3} tasks | mean est: {mean_est:6.1f}m{fst_str}")

    # ========== Difficulty Accuracy ==========
    print(f"\n{'Difficulty Classification Accuracy':}")
    print("-" * 40)

    exact_matches = sum(1 for r in valid_results
                       if r.get('estimated_difficulty') == r.get('ground_truth_difficulty'))
    exact_accuracy = exact_matches / len(valid_results) * 100
    print(f"Exact difficulty match: {exact_matches}/{len(valid_results)} ({exact_accuracy:.1f}%)")

    within_one = sum(1 for r in valid_results
                    if abs(int(r.get('estimated_difficulty', -10)) -
                          int(r.get('ground_truth_difficulty', -10))) <= 1)
    within_one_accuracy = within_one / len(valid_results) * 100
    print(f"Within one level: {within_one}/{len(valid_results)} ({within_one_accuracy:.1f}%)")

    over_estimates = sum(1 for r in valid_results
                        if int(r.get('estimated_difficulty', 0)) > int(r.get('ground_truth_difficulty', 0)))
    under_estimates = sum(1 for r in valid_results
                         if int(r.get('estimated_difficulty', 0)) < int(r.get('ground_truth_difficulty', 0)))
    print(f"Over-estimates (harder than labeled): {over_estimates} ({over_estimates/len(valid_results)*100:.1f}%)")
    print(f"Under-estimates (easier than labeled): {under_estimates} ({under_estimates/len(valid_results)*100:.1f}%)")

    # ========== Competition Breakdown ==========
    print(f"\n{'Competition Breakdown':}")
    print("-" * 40)

    competition_results = {}
    for r in valid_results:
        comp = r.get('competition', 'unknown')
        if comp not in competition_results:
            competition_results[comp] = {'correct': 0, 'total': 0, 'estimates': []}
        competition_results[comp]['total'] += 1
        competition_results[comp]['estimates'].append(r['estimated_minutes'])
        if r.get('estimated_difficulty') == r.get('ground_truth_difficulty'):
            competition_results[comp]['correct'] += 1

    for comp, stats in sorted(competition_results.items(), key=lambda x: -x[1]['total']):
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        mean_est = statistics.mean(stats['estimates']) if stats['estimates'] else 0
        print(f"{comp}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) | mean est: {mean_est:.0f} min")

    # ========== Confusion Matrix ==========
    print(f"\n{'Confusion Matrix (Estimated vs Ground Truth Difficulty)':}")
    print("-" * 60)

    difficulty_levels = sorted(DIFFICULTY_LABELS.keys())
    confusion = {est: {gt: 0 for gt in difficulty_levels} for est in difficulty_levels}

    for r in valid_results:
        est = r.get('estimated_difficulty', '-1')
        gt = r.get('ground_truth_difficulty', '-1')
        if est in difficulty_levels and gt in difficulty_levels:
            confusion[est][gt] += 1

    header = "Est \\ GT"
    print(f"{header:12}", end="")
    for gt in difficulty_levels:
        label = DIFFICULTY_LABELS[gt][:6]
        print(f"{label:>8}", end="")
    print()

    for est in difficulty_levels:
        label = DIFFICULTY_LABELS[est][:6]
        print(f"{est} ({label:6})", end="")
        for gt in difficulty_levels:
            count = confusion[est][gt]
            print(f"{count:>8}", end="")
        print()


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate task completion time for CyBench CTF challenges using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with OpenAI GPT-4o
  python cybench_human_time_llm_estimate.py --benchmark-path /path/to/cybench/benchmark

  # Process only crypto challenges
  python cybench_human_time_llm_estimate.py --category crypto

  # Process with Google Gemini
  python cybench_human_time_llm_estimate.py --provider google --model gemini-2.5-pro-preview-05-06

  # Use hard prompts (less context)
  python cybench_human_time_llm_estimate.py --use-hard-prompt

  # Analyze existing results
  python cybench_human_time_llm_estimate.py --analyze-only --results-file cybench_time_estimations.jsonl

Environment variables:
  OPENAI_API_KEY    - Required for OpenAI models
  GOOGLE_API_KEY    - Required for Google Gemini models
        """
    )

    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(DATA_DIR / "cybench" / "benchmark"),
        help="Path to CyBench benchmark directory",
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
        help="Output JSONL file path (default: cybench_time_estimations_{provider}_{model}.jsonl)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to process (default: all)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (e.g., crypto, web, pwn, reverse, forensics, misc)",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default=None,
        help="Filter by competition (e.g., hackthebox, hkcert-ctf, project-sekai-ctf)",
    )
    parser.add_argument(
        "--use-hard-prompt",
        action="store_true",
        help="Use hard_prompt instead of easy_prompt (less context given to model)",
    )
    parser.add_argument(
        "--no-subtasks",
        action="store_true",
        help="Don't include subtask information in the prompt",
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
        help="Only analyze existing results, don't process new tasks",
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
        args.output_file = f"cybench_time_estimations_{args.provider}_{model_safe}.jsonl"

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

        process_tasks(
            benchmark_path=args.benchmark_path,
            provider=args.provider,
            model=args.model,
            output_file=args.output_file,
            max_tasks=args.max_tasks,
            category_filter=args.category,
            competition_filter=args.competition,
            use_hard_prompt=args.use_hard_prompt,
            include_subtasks=not args.no_subtasks,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        # Also run analysis after processing
        if os.path.exists(args.output_file):
            analyze_results(args.output_file)


if __name__ == "__main__":
    main()
