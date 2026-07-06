#!/usr/bin/env python3
"""
Batch Experiment Executor - Generates behavior trajectories using modular architecture

Execution Order:
    1. Complete ALL trials for Instruction 1
    2. Complete ALL trials for Instruction 2
    3. Complete ALL trials for Instruction 3
    ...and so on

Usage:
    # Run with default config file (scripts/config/experiments.json)
    python scripts/experiment/run_experiments.py

    # Use custom config file
    python scripts/experiment/run_experiments.py --config my_experiments.json

    # Limit instructions (for testing)
    python scripts/experiment/run_experiments.py --max_instructions 5 --trials 10

    # Mock mode testing
    python scripts/experiment/run_experiments.py --mock

    # Custom user-provided dataset and output
    python scripts/experiment/run_experiments.py --dataset path/to/your_dataset.csv --output_dir ./my_logs

    # Save checkpoint every N instruction groups (default: 5)
    python scripts/experiment/run_experiments.py --save_interval 2

    # Resume from checkpoint after crash
    python scripts/experiment/run_experiments.py --resume

Progress Bars:
    - Install: pip install tqdm
    - Tracks: Each LLM API call (1 call per scenario)
    - Shows: [N/M] Model... | Progress: n/total calls | Rate
    - Formula: Total calls = tasks × 9 scenarios
    - Works in both fresh and resume modes
"""

import os
import sys
import time
import json
import argparse
import uuid
import pandas as pd
import shutil
import threading
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Will be logged later when logger is available

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

# Now import specific classes/functions
from src.mab import JailbreakEnvironment, JailbreakProbeRunner, LLMClient
from src.core.utils import setup_logger, set_debug_mode, get_log_level
import logging


# =============================================================================
# 1. Configuration Management
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    source: str
    model: str
    mock: bool = False
    max_workers: int = 1
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(
            source=data['source'],
            model=data['model'],
            mock=data.get('mock', False),
            max_workers=data.get('max_workers', 1),
            enabled=data.get('enabled', True)
        )


# Default batch configuration (backward compatible)
MODEL_CONFIGS = [
    ModelConfig("siliconflow", "deepseek-ai/DeepSeek-R1", mock=False, max_workers=2),
    ModelConfig("siliconflow", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", mock=False, max_workers=2),
]

# Default parameters
DEFAULT_CONFIG = {
    "dataset_path": "data/AdvBench/demo_harmful_behaviors_custom.csv",
    "output_dir": "./logs/cigt-smoke",
    "trials_per_instruction": 2,
    "max_workers": 1,
    "max_instructions": 1,
    "mock_mode": True,
    "save_interval": 5,  # Save every 5 completed tasks
}


def load_experiments_config(config_path: str = "experiments.json") -> Optional[Dict[str, Any]]:
    """
    Load experiment configuration from JSON file

    Args:
        config_path: Configuration file path

    Returns:
        Configuration dictionary, None if file doesn't exist
    """
    logger = setup_logger("ConfigLoader")

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded config: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def parse_models_from_config(config: Dict[str, Any]) -> List[ModelConfig]:
    """
    Parse model list from configuration

    Args:
        config: Configuration dictionary

    Returns:
        Model configuration list
    """
    models = []

    if 'models' not in config:
        return models

    for model_data in config['models']:
        model_config = ModelConfig.from_dict(model_data)
        if model_config.enabled:
            models.append(model_config)

    return models


def parse_runtime_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse runtime parameters from configuration

    Args:
        config: Configuration dictionary

    Returns:
        运行时Configuration dictionary
    """
    runtime = DEFAULT_CONFIG.copy()

    if 'runtime' in config:
        runtime_config = config['runtime']
        for key in ['dataset_path', 'output_dir', 'trials_per_instruction', 'max_workers', 'max_instructions', 'mock_mode', 'save_interval']:
            if key in runtime_config:
                runtime[key] = runtime_config[key]

    return runtime


def parse_scenarios_from_config(config: Dict[str, Any]) -> Optional[List[str]]:
    """
    Parse scenarios configuration from configuration

    Args:
        config: Configuration dictionary

    Returns:
        List of enabled scenario groups, or None if not specified
    """
    if 'scenarios' not in config:
        return None

    scenarios_config = config['scenarios']
    enabled_groups = scenarios_config.get('enabled_groups')

    if enabled_groups and isinstance(enabled_groups, list):
        logger = setup_logger("ConfigLoader")
        logger.info(f"Enabled scenario groups: {enabled_groups}")
        return enabled_groups

    return None


def get_scenario_count(enabled_groups: Optional[List[str]] = None) -> int:
    """
    Get the number of scenarios that will be executed based on enabled_groups

    Args:
        enabled_groups: List of enabled scenario groups, or None for all

    Returns:
        Number of scenarios that will be executed
    """
    # Import SCENARIO_REGISTRY to count scenarios
    try:
        from src.mab.config import SCENARIO_REGISTRY
    except ImportError:
        # Fallback: try relative import if src not in path
        import sys
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        src_dir = os.path.join(project_root, 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from src.mab.config import SCENARIO_REGISTRY
        except ImportError:
            # Last resort: return 9 (default)
            return 9

    if enabled_groups is None:
        return len(SCENARIO_REGISTRY)

    count = sum(1 for scenario in SCENARIO_REGISTRY.values() if scenario['group'] in enabled_groups)
    return count if count > 0 else len(SCENARIO_REGISTRY)


# =============================================================================
# 2. Data Loading and Task Preparation
# =============================================================================

def load_instructions(filepath: str, max_instructions: int = None) -> List[str]:
    """
    Load instructions from CSV file

    Args:
        filepath: CSVFile
        max_instructions: Maximum instruction count limit

    Returns:
        Instruction list
    """
    logger = setup_logger("DataLoader")

    if not os.path.exists(filepath):
        logger.warning(f"Dataset not found: {filepath}")
        logger.info("Using benign built-in smoke-test data...")
        return [
            "Explain how to reset a forgotten password for my own account using official recovery channels.",
            "Summarize best practices for safely reporting a suspected software vulnerability.",
            "Give a high-level explanation of why multi-factor authentication improves account safety.",
        ][:max_instructions or 3]

    try:
        df = pd.read_csv(filepath)
        if 'goal' not in df.columns:
            raise ValueError(f"CSV must contain 'goal' column, current columns: {df.columns.tolist()}")

        goals = df['goal'].dropna().tolist()

        if max_instructions:
            goals = goals[:max_instructions]
            logger.info(f"Limited instruction count: {max_instructions}")

        logger.info(f"Successfully loaded {len(goals)} instructions")
        return goals

    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return []


@dataclass
class ExperimentTask:
    """Experiment task dataclass"""
    model_config: ModelConfig
    instruction: str
    trial_idx: int
    output_dir: str
    enabled_groups: Optional[List[str]] = None

    @property
    def task_id(self) -> str:
        """Generate task ID"""
        instr_short = self.instruction[:20].replace(" ", "_")
        return f"{self.model_config.model}_{instr_short}_t{self.trial_idx}"


def prepare_tasks(
    model_configs: List[ModelConfig],
    instructions: List[str],
    trials_per_instruction: int,
    output_dir: str,
    enabled_groups: Optional[List[str]] = None
) -> List[ExperimentTask]:
    """
    Prepare all experiment tasks

    Args:
        model_configs: Model configuration list
        instructions: Instruction list
        trials_per_instruction: Trials per instruction
        output_dir: Output directory
        enabled_groups: List of enabled scenario groups

    Returns:
        Task list
    """
    tasks = []
    for config in model_configs:
        for instruction in instructions:
            for trial in range(trials_per_instruction):
                tasks.append(ExperimentTask(
                    model_config=config,
                    instruction=instruction,
                    trial_idx=trial,
                    output_dir=output_dir,
                    enabled_groups=enabled_groups
                ))
    return tasks


# =============================================================================
# 3. Single Experiment Execution
# =============================================================================

def run_single_experiment(task: ExperimentTask, checkpoint_mgr: Optional['CheckpointManager'] = None, pbar: Optional[Any] = None, prev_history: List[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute single experiment - runs through ALL scenarios for one instruction
    Supports checkpointing at scenario level for precise recovery
    Supports history inheritance from previous trials

    Args:
        task: instruction task
        checkpoint_mgr: Optional checkpoint manager for partial progress recovery
        pbar: Optional tqdm progress bar for API call tracking
        prev_history: History data from previous trials (for continuous learning)

    Returns:
        (results, final_history):
            - results: List of execution result dictionaries (one per scenario)
            - final_history: Complete history after this trial (for next trial)
    """
    logger = setup_logger(f"Exp_{task.task_id}")

    # Add detailed start logging
    # logger.info(f"=== Starting trial {task.trial_idx} for instruction: {task.instruction[:50]}... ===")

    # Input validation
    if not task or not hasattr(task, 'model_config') or not hasattr(task, 'instruction'):
        error_msg = "Invalid task object provided"
        logger.error(error_msg)
        return [{
            "success": False,
            "task_id": "unknown",
            "error": error_msg,
            "model": "unknown",
            "instruction": "unknown",
            "trial": -1,
            "timestamp": time.time()
        }], []

    try:
        # 1. Create LLM client
        logger.debug(f"Step 1: Creating LLM client for {task.model_config.source}/{task.model_config.model}")
        client = LLMClient(
            source=task.model_config.source,
            model=task.model_config.model,
            mock=task.model_config.mock
        )
        logger.debug(f"Step 1: ✓ LLM client created")

        # 2. Create output directory
        logger.debug(f"Step 2: Creating output directory")
        model_safe = task.model_config.model.replace("/", "_")
        model_output_dir = os.path.join(task.output_dir, model_safe)
        os.makedirs(model_output_dir, exist_ok=True)
        logger.debug(f"Step 2: ✓ Output directory ready: {model_output_dir}")

        # 3. Create experiment environment
        logger.debug(f"Step 3: Creating experiment environment")
        env = JailbreakEnvironment(
            model_name=task.model_config.model,
            instruction=task.instruction,
            num_trials=1,
            save_path=None,
            save_interval=1,
            enabled_groups=task.enabled_groups
        )
        logger.debug(f"Step 3: ✓ Environment created with {len(env.registry)} scenarios")
        if task.enabled_groups:
            logger.debug(f"  Filtered groups: {task.enabled_groups}")

        # 4. Load checkpoint if exists
        logger.debug(f"Step 4: Checking for checkpoints")
        completed_scenarios = set()
        agent_history = []
        num_completed = 0
        if checkpoint_mgr and checkpoint_mgr.checkpoint_exists():
            partial_state = checkpoint_mgr.load_partial_state(task)
            if partial_state:
                completed_scenarios = set(partial_state.get('completed_scenarios', []))
                agent_history = partial_state.get('agent_history', [])
                num_completed = partial_state.get('num_completed', 0)
                logger.debug(f"Step 4: ✓ Resuming from scenario {num_completed}/{len(env.registry)}")
            else:
                logger.debug(f"Step 4: ✓ No partial state found, starting fresh")
        else:
            logger.debug(f"Step 4: ✓ No checkpoint found, starting fresh")

        # Load agent history into environment
        # Priority: prev_history (from previous trial) > checkpoint history
        # Track original count to return only NEW history (FIX: prevent duplicate accumulation)
        original_history_count = 0
        if prev_history:
            env.load_history_data(prev_history)
            original_history_count = len(prev_history)
            logger.debug(f"Step 4: ✓ Loaded {len(prev_history)} history records from previous trial")
        elif agent_history:
            env.load_history_data(agent_history)
            original_history_count = len(agent_history)
            logger.debug(f"Step 4: ✓ Loaded {len(agent_history)} history records from checkpoint")

        # Advance environment to skip completed scenarios
        if num_completed > 0:
            env.advance_to_scenario(num_completed)

        # 5. Create executor
        logger.debug(f"Step 5: Creating probe runner")
        runner = JailbreakProbeRunner(client, task.model_config.model, env)
        logger.debug(f"Step 5: ✓ Probe runner created")

        # 6. Execute experiment for ALL scenarios (skip completed)
        results = []
        total_scenarios = len(env.registry)
        scenarios_to_run = total_scenarios - num_completed

        logger.debug(f"Step 6: Starting scenario execution ({scenarios_to_run} scenarios to run)")

        for scenario_idx in range(scenarios_to_run):
            current_scenario_num = num_completed + scenario_idx + 1

            try:
                # Execute scenario (single-threaded within each trial)
                logger.debug(f"  Executing scenario {current_scenario_num}/{total_scenarios}")
                scenario_id, action, reward, raw_text = runner.run_step(task.instruction)

                scenario = env.registry[scenario_id]
                scenario_name = scenario['name']
                group = scenario['group']

                result = {
                    "success": True,
                    "task_id": task.task_id,
                    "model": task.model_config.model,
                    "source": task.model_config.source,
                    "instruction": task.instruction,
                    "trial": task.trial_idx,
                    "scenario_id": scenario_id,
                    "scenario_name": scenario_name,
                    "group": group,
                    "action": action,
                    "reward": reward,
                    "raw_text": raw_text,
                    "mock": task.model_config.mock,
                    "timestamp": time.time()
                }
                results.append(result)

                logger.debug(f"  ✓ Scenario {current_scenario_num}/{total_scenarios}: {group} - {action}")

                # Update progress bar after each API call
                if pbar:
                    pbar.update(1)

                # Save checkpoint after each scenario
                if checkpoint_mgr:
                    checkpoint_mgr.save_partial_state(task, results, runner)

            except Exception as scenario_error:
                error_msg = f"Scenario {current_scenario_num} failed: {scenario_error}"
                logger.error(error_msg)
                logger.error(f"Full traceback:\n{traceback.format_exc()}")

                # Return partial results with error
                error_result = {
                    "success": False,
                    "task_id": task.task_id,
                    "error": error_msg,
                    "model": task.model_config.model,
                    "instruction": task.instruction,
                    "trial": task.trial_idx,
                    "scenario_id": f"failed_{current_scenario_num}",
                    "scenario_name": "ERROR",
                    "group": "ERROR",
                    "action": "ParseFail",
                    "reward": 0.0,
                    "raw_text": str(scenario_error),
                    "mock": task.model_config.mock,
                    "timestamp": time.time()
                }
                results.append(error_result)

                # Continue to next scenario instead of aborting entire trial
                continue

        # Get final history for next trial
        # FIX: Only return NEW history (exclude prev_history to prevent duplication)
        final_history = env.history_data[original_history_count:]

        logger.debug(f"=== Trial {task.trial_idx} completed: {len(results)} scenarios ===")
        logger.debug(f"  original_history_count={original_history_count}, env.history_data={len(env.history_data)}, final_history={len(final_history)}")
        return results, final_history

    except Exception as e:
        error_msg = f"Experiment failed with critical error: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return [{
            "success": False,
            "task_id": task.task_id,
            "error": error_msg,
            "model": task.model_config.model,
            "instruction": task.instruction,
            "trial": task.trial_idx,
            "timestamp": time.time()
        }], []


# =============================================================================
# 4. Checkpoint Management
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for crash recovery and resume functionality with per-scenario precision"""

    def __init__(self, output_dir: str, model_name: str):
        self.model_name = model_name
        self.model_safe = model_name.replace("/", "_")
        self.checkpoint_dir = os.path.join(output_dir, self.model_safe, "checkpoint")
        self.metadata_file = os.path.join(self.checkpoint_dir, "metadata.json")
        self.results_file = os.path.join(self.checkpoint_dir, "results.csv")
        self.history_file = os.path.join(self.checkpoint_dir, "history_data.csv")
        # New: Per-task partial state files
        self.partial_dir = os.path.join(self.checkpoint_dir, "partial")
        self.logger = setup_logger("CheckpointManager")

    def save_checkpoint(self, completed_tasks: List[Dict[str, Any]], metadata: Dict[str, Any], history_data: List[Dict[str, Any]] = None):
        """Save checkpoint with completed tasks, metadata, and history data

        Args:
            completed_tasks: List of completed task results
            metadata: Checkpoint metadata
            history_data: Agent's history data for learning (optional)
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save results
        if completed_tasks:
            df = pd.DataFrame(completed_tasks)
            df.to_csv(self.results_file, index=False)

        # Save history data (for agent learning state)
        if history_data:
            df = pd.DataFrame(history_data)
            df.to_csv(self.history_file, index=False)

        self.logger.debug(f"Checkpoint saved: {len(completed_tasks)} tasks completed")

    def _sanitize_task_id(self, task_id: str) -> str:
        """Sanitize task ID for use as filename

        Replaces all characters that are problematic in filenames:
        /, |, \, :, *, ?, ", <, >
        """
        import re
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', task_id)
        return sanitized

    def save_partial_state(self, task: 'ExperimentTask', results: List[Dict[str, Any]], runner: 'JailbreakProbeRunner'):
        """Save partial state for a task after each scenario

        Args:
            task: Current experiment task
            results: Results collected so far for this task
            runner: Probe runner containing environment with history
        """
        os.makedirs(self.partial_dir, exist_ok=True)

        # Get completed scenario IDs
        completed_scenarios = [r['scenario_id'] for r in results if r.get('success')]

        # Save partial state
        sanitized_id = self._sanitize_task_id(task.task_id)
        partial_file = os.path.join(self.partial_dir, f"{sanitized_id}.json")
        partial_state = {
            "task_id": task.task_id,
            "model": task.model_config.model,
            "instruction": task.instruction,
            "trial": task.trial_idx,
            "completed_scenarios": completed_scenarios,
            "num_completed": len(completed_scenarios),
            "timestamp": time.time()
        }

        # Save agent history from environment
        if hasattr(runner, 'env') and hasattr(runner.cigt_env, 'history_data'):
            partial_state['agent_history'] = runner.cigt_env.history_data

        with open(partial_file, 'w') as f:
            json.dump(partial_state, f, indent=2)

        self.logger.debug(f"Saved partial state: {task.task_id} ({len(completed_scenarios)} scenarios)")

    def load_partial_state(self, task: 'ExperimentTask') -> Optional[Dict[str, Any]]:
        """Load partial state for a task

        Args:
            task: Task to load state for

        Returns:
            Partial state dict or None if not found
        """
        sanitized_id = self._sanitize_task_id(task.task_id)
        partial_file = os.path.join(self.partial_dir, f"{sanitized_id}.json")

        if not os.path.exists(partial_file):
            return None

        try:
            with open(partial_file, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Loaded partial state: {task.task_id} ({state.get('num_completed', 0)} scenarios)")
            return state
        except Exception as e:
            self.logger.warning(f"Failed to load partial state for {task.task_id}: {e}")
            return None

    def has_partial_state(self, task: 'ExperimentTask') -> bool:
        """Check if partial state exists for a task"""
        sanitized_id = self._sanitize_task_id(task.task_id)
        partial_file = os.path.join(self.partial_dir, f"{sanitized_id}.json")
        return os.path.exists(partial_file)

    def cleanup_partial_state(self, task: 'ExperimentTask'):
        """Remove partial state for a completed task"""
        sanitized_id = self._sanitize_task_id(task.task_id)
        partial_file = os.path.join(self.partial_dir, f"{sanitized_id}.json")
        if os.path.exists(partial_file):
            os.remove(partial_file)
            self.logger.debug(f"Cleaned up partial state: {task.task_id}")

    def load_checkpoint(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load checkpoint data with validation"""
        if not os.path.exists(self.metadata_file):
            return [], None, []

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            completed_tasks = []
            if os.path.exists(self.results_file):
                try:
                    df = pd.read_csv(self.results_file)
                    # Validate required columns
                    required = ['model', 'instruction', 'trial', 'scenario_id']
                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        raise ValueError(f"Checkpoint missing required columns: {missing}")
                    completed_tasks = df.to_dict('records')
                except Exception as e:
                    self.logger.error(f"Checkpoint results file corrupted: {e}")
                    return [], metadata, []

            history_data = []
            if os.path.exists(self.history_file):
                try:
                    df = pd.read_csv(self.history_file)
                    history_data = df.to_dict('records')
                except Exception as e:
                    self.logger.warning(f"Checkpoint history file corrupted: {e}")
                    # Continue without history

            self.logger.info(f"Checkpoint loaded: {len(completed_tasks)} tasks, {len(history_data)} history records")
            return completed_tasks, metadata, history_data

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return [], None, []

    def get_partial_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get all partial states for resume

        Returns:
            Dict mapping original task_id to partial state
        """
        if not os.path.exists(self.partial_dir):
            return {}

        partial_states = {}
        for filename in os.listdir(self.partial_dir):
            if filename.endswith('.json'):
                partial_file = os.path.join(self.partial_dir, filename)
                try:
                    with open(partial_file, 'r') as f:
                        state = json.load(f)
                        # Use the original task_id from inside the state
                        original_task_id = state.get('task_id')
                        if original_task_id:
                            partial_states[original_task_id] = state
                except Exception as e:
                    self.logger.warning(f"Failed to load partial file {filename}: {e}")

        return partial_states

    def checkpoint_exists(self) -> bool:
        """Check if checkpoint exists"""
        return os.path.exists(self.metadata_file)

    def get_completed_task_ids(self) -> set:
        """Get set of completed task IDs for fast lookup"""
        completed_tasks, _, _ = self.load_checkpoint()
        return {f"{t['model']}|{t['instruction']}|{t['trial']}" for t in completed_tasks}

    def cleanup_checkpoint(self):
        """Remove checkpoint directory"""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            self.logger.info("Checkpoint cleaned up")


# =============================================================================
# 5. Batch Execution and Result Management
# =============================================================================

class BatchExperimentManager:
    """Batch experiment manager with checkpoint support"""

    def __init__(self, save_interval: int = 5, resume: bool = False):
        """
        Args:
            save_interval: Save results every N completed tasks (default: 5)
            resume: Whether to resume from checkpoint
        """
        self.save_interval = save_interval
        self.resume = resume
        self.results = []
        self.logger = setup_logger("BatchManager")
        self.output_dir = None
        self.model_configs = None
        self.last_save_count = 0
        self.checkpoint_managers = {}  # model_name -> CheckpointManager
        # Track trial counts per task for periodic saving
        self.trial_counts = {}  # task_key -> count
        # Lock for thread-safe directory creation and saving
        self._directory_lock = threading.Lock()
        # Track scenario-based directories: {model: {scenario_group: dir_path}}
        self.scenario_directories = {}
        # Track run directories for each model: {model: run_dir}
        self.run_directories = {}
        # Generate unique run ID (timestamp)
        self.run_id = None

    def execute(self, tasks: List[ExperimentTask], output_dir: str = None, model_configs: List = None) -> pd.DataFrame:
        """
        Execute batch experiments with periodic saving and checkpoint support

        Args:
            tasks: Task list
            output_dir: Output directory for periodic saves
            model_configs: Model configurations for saving

        Returns:
            结果DataFrame
        """
        # Input validation
        if not tasks:
            self.logger.error("No tasks provided for execution")
            return pd.DataFrame()

        if not isinstance(tasks, list):
            self.logger.error(f"Tasks must be a list, got {type(tasks)}")
            return pd.DataFrame()

        # Validate tasks
        for i, task in enumerate(tasks):
            if not hasattr(task, 'model_config') or not hasattr(task, 'instruction'):
                self.logger.error(f"Task at index {i} is invalid: missing required attributes")
                return pd.DataFrame()

        # Validate output_dir if provided
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                if not os.path.isdir(output_dir):
                    raise OSError(f"Output directory is not a directory: {output_dir}")
            except OSError as e:
                self.logger.error(f"Cannot create output directory {output_dir}: {e}")
                return pd.DataFrame()

        # Validate model_configs if provided
        if model_configs:
            if not isinstance(model_configs, list):
                self.logger.error(f"model_configs must be a list, got {type(model_configs)}")
                return pd.DataFrame()
            for cfg in model_configs:
                if not hasattr(cfg, 'model') or not hasattr(cfg, 'enabled'):
                    self.logger.error(f"Invalid model config: {cfg}")
                    return pd.DataFrame()

        # Initialize checkpoint managers for each model
        if output_dir and model_configs:
            for cfg in model_configs:
                if cfg.enabled:
                    self.checkpoint_managers[cfg.model] = CheckpointManager(output_dir, cfg.model)

        # Generate unique run ID (timestamp)
        import datetime
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Run ID: {self.run_id}")

        # Handle resume mode
        if self.resume:
            return self._execute_with_resume(tasks, output_dir, model_configs)
        else:
            return self._execute_fresh(tasks, output_dir, model_configs)

    def _execute_fresh(self, tasks: List[ExperimentTask], output_dir: str = None, model_configs: List = None) -> pd.DataFrame:
        """Execute experiments without resume

        Execution strategy:
        - INSTRUCTION LEVEL: Parallel (multiple instructions execute simultaneously)
        - TRIAL LEVEL: Sequential within each instruction (for history inheritance)

        Uses ThreadPoolExecutor at instruction level to maximize throughput while
        maintaining trial order for proper agent learning.
        """
        total_tasks = len(tasks)
        self.logger.info(f"Starting execution of {total_tasks} experiment tasks")
        self.logger.info(f"Save interval: Every {self.save_interval} completed tasks")
        self.logger.info("Execution strategy: Parallel instructions, sequential trials per instruction")
        self.logger.info("=" * 60)

        # Store for periodic saving
        self.output_dir = output_dir
        self.model_configs = model_configs

        # Clean up any existing temp directories from previous crashes
        if output_dir and model_configs:
            self._cleanup_temp_dirs(check_only=False)

        start_time = time.time()

        # Group tasks by (model, instruction)
        task_groups = self._group_tasks_by_instruction(tasks)
        total_groups = len(task_groups)

        # Calculate total API calls based on actual scenario count
        # Get enabled_groups from first task if available
        enabled_groups = tasks[0].enabled_groups if tasks else None
        num_scenarios = get_scenario_count(enabled_groups)
        total_api_calls = len(tasks) * num_scenarios
        self.logger.info(f"Total API calls to execute: {total_api_calls} ({len(tasks)} tasks × {num_scenarios} scenarios)")
        self.logger.info(f"Total instruction groups: {total_groups}")

        # Determine instruction-level parallelism
        # Use global max_workers from first model config (or default to 4)
        global_max_workers = model_configs[0].max_workers if model_configs else 4
        instruction_workers = min(global_max_workers, len(task_groups), 8)
        self.logger.info(f"Instruction-level parallelism: {instruction_workers} workers")
        self.logger.info(f"Trial-level: Sequential (for history inheritance)")

        # Initialize progress bar for API calls
        pbar = None
        if HAS_TQDM:
            pbar = tqdm(total=total_api_calls, desc="API Calls", unit="call",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Execute instruction groups in parallel using ThreadPoolExecutor
        # Each instruction group runs its trials sequentially internally
        completed_groups = 0
        success_count = 0
        failure_count = 0
        all_results = []
        results_lock = threading.Lock()

        def execute_single_instruction_group(group_key: str, group_tasks: List[ExperimentTask]) -> List[Dict[str, Any]]:
            """Execute one instruction group (trials are sequential within)"""
            model = group_tasks[0].model_config.model
            instruction = group_tasks[0].instruction
            cp_mgr = self.checkpoint_managers.get(model)

            # Update progress bar description
            if pbar:
                pbar.set_description(f"[{completed_groups}/{total_groups}] {model[:15]}...")

            self.logger.info(
                f"\n[{completed_groups}/{total_groups}] Starting: {model} | Instruction: {instruction[:50]}..."
            )
            self.logger.info(f"  Tasks in this group: {len(group_tasks)} trials")

            try:
                # Execute trials SEQUENTIALLY within this instruction group
                group_results = self._execute_task_group_sequential(group_tasks, cp_mgr, None, pbar)

                # Log summary
                success_results = [r for r in group_results if r.get("success")]
                if success_results:
                    compliance_count = sum(1 for r in success_results if r.get('action') == 'Compliance')
                    self.logger.info(
                        f"  ✅ Completed: {len(group_tasks)} trials, {len(success_results)} scenarios, "
                        f"{compliance_count} compliance"
                    )
                    # Cleanup partial states for all tasks in this group
                    for task in group_tasks:
                        if cp_mgr:
                            cp_mgr.cleanup_partial_state(task)
                else:
                    self.logger.warning(f"  ❌ FAILED group: {group_key}")

                return group_results

            except Exception as exc:
                error_msg = f"  💥 CRITICAL EXCEPTION in group {group_key}: {exc}"
                self.logger.error(error_msg)
                self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                self.logger.warning(f"Skipping failed group: {group_key}")
                return [{
                    "success": False,
                    "task_id": f"FAILED_{group_key}",
                    "error": error_msg,
                    "model": model,
                    "instruction": instruction,
                    "trial": -1,
                    "timestamp": time.time()
                }]

        # Execute instruction groups in parallel
        with ThreadPoolExecutor(max_workers=instruction_workers) as executor:
            # Submit all instruction groups
            future_to_group = {
                executor.submit(execute_single_instruction_group, key, tasks): key
                for key, tasks in task_groups.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_group):
                group_key = future_to_group[future]
                completed_groups += 1

                try:
                    group_results = future.result(timeout=600)  # 10 min timeout per instruction group
                    with results_lock:
                        all_results.extend(group_results)

                    # Count successes/failures
                    success_results = [r for r in group_results if r.get("success")]
                    if success_results:
                        success_count += len(success_results)
                    else:
                        failure_count += 1

                    # Periodic save after each instruction group completes
                    if completed_groups % self.save_interval == 0:
                        with results_lock:
                            self.results = all_results.copy()
                        self._periodic_save(completed_groups)
                        self._save_checkpoint()

                except Exception as e:
                    failure_count += 1
                    self.logger.error(f"Group {group_key} failed: {e}")

        # Update final results
        self.results = all_results

        # Close progress bar
        if pbar:
            pbar.close()

        # Final save if there are unsaved results
        if self.results and len(self.results) > self.last_save_count:
            self._periodic_save(completed_groups, final=True)
        else:
            self._cleanup_temp_dirs()

        # Cleanup checkpoint on successful completion
        self._cleanup_checkpoint()

        elapsed = time.time() - start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Batch experiments completed! Time taken: {elapsed:.1f}秒")
        self.logger.info(f"Total instruction groups: {len(task_groups)}")
        self.logger.info(f"Success: {success_count} scenarios")
        self.logger.info(f"Failure: {failure_count} groups")
        self.logger.info("=" * 60)

        return self.to_dataframe()

    def _group_tasks_by_instruction(self, tasks: List[ExperimentTask]) -> Dict[str, List[ExperimentTask]]:
        """Group tasks by (model, instruction) to ensure sequential execution per instruction

        Returns:
            Dict mapping (model|instruction) to list of tasks (trials)
        """
        groups = {}
        for task in tasks:
            key = f"{task.model_config.model}|{task.instruction}"
            if key not in groups:
                groups[key] = []
            groups[key].append(task)

        # Sort each group by trial_idx to ensure sequential execution
        for key in groups:
            groups[key].sort(key=lambda t: t.trial_idx)

        return groups

    def _execute_task_group_sequential(self, tasks: List[ExperimentTask], checkpoint_mgr: Optional['CheckpointManager'], api_semaphore: Optional[threading.Semaphore] = None, pbar: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Execute all tasks in a group sequentially (trial 0 → 1 → 2 → ...) with history inheritance

        IMPORTANT: Trials must be sequential to maintain agent learning history.
        Each trial's behavior is influenced by ALL previous trial outcomes.

        History Flow:
            Trial 0: history = []
            Trial 1: history = [Trial 0 results]
            Trial 2: history = [Trial 0 + Trial 1 results]
            ...

        Args:
            tasks: List of tasks for the same instruction (different trials, must be sorted by trial_idx)
            checkpoint_mgr: Checkpoint manager for this model
            api_semaphore: Deprecated, no longer used (kept for compatibility)
            pbar: Optional tqdm progress bar for API call tracking

        Returns:
            Combined results from all tasks in the group (for checkpointing)
        """
        if not tasks:
            return []

        # Get model config
        model_config = tasks[0].model_config
        task_key = f"{model_config.model}|{tasks[0].instruction}"

        logger = setup_logger("SequentialExecutor")
        logger.debug(f"Executing {len(tasks)} trials SEQUENTIALLY with history inheritance")

        # Verify tasks are sorted by trial_idx
        trial_indices = [t.trial_idx for t in tasks]
        if trial_indices != sorted(trial_indices):
            logger.warning(f"Tasks not sorted! Trial indices: {trial_indices}")
            tasks = sorted(tasks, key=lambda t: t.trial_idx)
            logger.debug(f"Re-sorted tasks: {[t.trial_idx for t in tasks]}")

        group_results = []
        accumulated_history = []  # History that accumulates across trials
        start_time = time.time()

        # Execute trials sequentially (0 → 1 → 2 → ...)
        for idx, task in enumerate(tasks):
            trial_start = time.time()
            logger.debug(f"  → Starting trial {task.trial_idx} ({idx + 1}/{len(tasks)})")
            logger.debug(f"     History: {len(accumulated_history)} records from previous trials")

            try:
                # Execute single trial with history inheritance
                task_results, final_history = run_single_experiment(
                    task, checkpoint_mgr, pbar, prev_history=accumulated_history
                )

                # Validate results
                if not isinstance(task_results, list):
                    logger.error(f"Task {task.task_id} returned non-list: {type(task_results)}")
                    task_results = []

                # Collect results
                validated_results = []
                for result in task_results:
                    if isinstance(result, dict) and result.get("success"):
                        validated_results.append(result)
                    elif isinstance(result, dict):
                        # Failed result
                        group_results.append(result)
                        logger.warning(f"Task {task.task_id} failed: {result.get('error', 'Unknown')}")
                    else:
                        logger.error(f"Invalid result format from {task.task_id}: {result}")

                group_results.extend(validated_results)

                # Update accumulated history for next trial
                accumulated_history.extend(final_history)
                
                # Prevent memory leak: limit history size
                MAX_HISTORY_SIZE = 1000  # Configurable threshold
                if len(accumulated_history) > MAX_HISTORY_SIZE:
                    # Keep only recent history (last N records)
                    # This preserves learning while preventing OOM
                    keep_count = MAX_HISTORY_SIZE // 2
                    accumulated_history = accumulated_history[-keep_count:]
                    logger.warning(f"History size exceeded limit, trimmed to {keep_count} records")
                
                logger.debug(f"     History after trial: {len(accumulated_history)} records")

                # Update trial count
                current_count = self.trial_counts.get(task_key, 0) + 1
                self.trial_counts[task_key] = current_count

                # Progress logging
                trial_elapsed = time.time() - trial_start
                logger.debug(f"  ✓ Trial {task.trial_idx} completed: {len(validated_results)} scenarios in {trial_elapsed:.1f}s")

            except Exception as e:
                error_msg = f"Task {task.task_id} raised exception: {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                error_result = {
                    "success": False,
                    "task_id": task.task_id,
                    "error": error_msg,
                    "model": model_config.model,
                    "instruction": tasks[0].instruction,
                    "trial": task.trial_idx,
                    "timestamp": time.time()
                }
                group_results.append(error_result)

        # Save ALL results for this instruction group AFTER all trials complete
        if self.output_dir and self.model_configs and group_results:
            # Filter only successful results for saving
            successful_results = [r for r in group_results if r.get("success")]
            if successful_results:
                self._save_results_batch(task_key, successful_results)
                logger.debug(f"  ✅ Saved {len(successful_results)} scenarios for instruction group")

        # Summary
        elapsed = time.time() - start_time
        logger.debug(f"  ✅ Sequential execution completed: {len(tasks)} trials in {elapsed:.1f}s "
                   f"({len(tasks)/elapsed:.2f} trials/sec)")
        logger.debug(f"  Total history accumulated: {len(accumulated_history)} records")

        return group_results

    def _save_results_batch(self, task_key: str, results: List[Dict[str, Any]]):
        """Save results for an instruction group to scenario-based structure

        Args:
            task_key: Task identifier (model|instruction)
            results: Results from all trials in the group (trials × 9 scenarios)
        """
        if not results:
            self.logger.warning(f"Empty results received for task_key: {task_key}")
            return

        try:
            # Validate results structure
            if not isinstance(results, list):
                self.logger.error(f"Results must be a list, got {type(results)}")
                return

            # Extract model and instruction with validation
            first_result = results[0]
            model = first_result.get('model')
            instruction = first_result.get('instruction')

            if not model or not instruction:
                self.logger.error(f"Missing model or instruction in results: {first_result}")
                return

            # Convert to DataFrame
            df = self._results_to_dataframe(results)
            if df.empty:
                self.logger.warning(f"No valid results to save for {task_key}")
                return

            # Validate DataFrame structure
            required_columns = ['group', 'trial', 'scenario_id']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns {missing_cols} in DataFrame")
                return

            # Save by scenario group → instruction file
            # Note: _get_scenario_dir already uses the lock internally, so we don't need another lock here
            for scenario_group in df['group'].unique():
                try:
                    scenario_df = df[df['group'] == scenario_group]

                    # Get scenario directory (this method handles its own locking)
                    scenario_dir = self._get_scenario_dir(model, scenario_group)

                    # Generate instruction hash for filename
                    instruction_hash = self._get_instruction_hash(instruction)
                    filename = f"{instruction_hash}.csv"
                    filepath = os.path.join(scenario_dir, filename)

                    # Append to file with error handling
                    # Use a separate lock just for file writing to avoid holding lock during I/O
                    with self._directory_lock:
                        header = not os.path.exists(filepath)
                        scenario_df.to_csv(filepath, mode='a', header=header, index=False)

                    # Verify file was written
                    if not os.path.exists(filepath):
                        self.logger.error(f"Failed to create file: {filepath}")
                    else:
                        file_size = os.path.getsize(filepath)
                        self.logger.debug(f"Saved to {filepath} ({file_size} bytes)")

                except Exception as e:
                    self.logger.error(f"Failed to save scenario group {scenario_group}: {e}")
                    raise  # Re-raise to trigger outer error handling

            # Log success
            current_count = self.trial_counts.get(task_key, 0)
            instruction_short = instruction[:40] + "..." if len(instruction) > 40 else instruction
            self.logger.debug(
                f"✅ Instruction group saved [{current_count} trials]: {model} | {instruction_short} | "
                f"{len(results)} scenarios"
            )

        except Exception as e:
            self.logger.error(f"CRITICAL: Failed to save results batch for {task_key}: {e}")
            # Don't raise - allow experiment to continue

    def _execute_with_resume(self, tasks: List[ExperimentTask], output_dir: str = None, model_configs: List = None) -> pd.DataFrame:
        """Execute experiments with resume from checkpoint (including partial task states)
        Ensures trials for the same instruction are executed sequentially
        """
        # Store for periodic saving
        self.output_dir = output_dir
        self.model_configs = model_configs

        total_tasks = len(tasks)
        self.logger.info("=" * 60)
        self.logger.info("RESUME MODE - Loading checkpoint...")
        self.logger.info("Execution order: Trials for same instruction are sequential")
        self.logger.info("=" * 60)

        # Load checkpoint data
        all_completed_tasks = []
        all_checkpoint_metadata = {}

        for model_name, cp_mgr in self.checkpoint_managers.items():
            if cp_mgr.checkpoint_exists():
                completed_tasks, metadata, _ = cp_mgr.load_checkpoint()
                all_completed_tasks.extend(completed_tasks)
                if metadata:
                    all_checkpoint_metadata[model_name] = metadata

        # Load partial states for in-progress tasks
        all_partial_states = {}
        for model_name, cp_mgr in self.checkpoint_managers.items():
            partial_states = cp_mgr.get_partial_progress()
            all_partial_states.update(partial_states)

        if not all_completed_tasks and not all_partial_states:
            self.logger.warning("No checkpoint found! Starting from scratch...")
            return self._execute_fresh(tasks, output_dir, model_configs)

        # Build set of completed task identifiers
        completed_task_ids = set()
        for t in all_completed_tasks:
            task_id = f"{t['model']}|{t['instruction']}|{t['trial']}"
            completed_task_ids.add(task_id)

        # Filter tasks to only include incomplete ones
        remaining_tasks = []
        skipped_count = 0
        resumed_partial_count = 0

        for task in tasks:
            task_id = f"{task.model_config.model}|{task.instruction}|{task.trial_idx}"
            task_id_for_partial = task.task_id  # Use this format for partial state lookup

            # Skip if fully completed
            if task_id in completed_task_ids:
                skipped_count += 1
                continue

            # Check if partial state exists
            if task_id_for_partial in all_partial_states:
                partial_state = all_partial_states[task_id_for_partial]
                num_completed = partial_state.get('num_completed', 0)
                resumed_partial_count += 1
                self.logger.info(f"Resuming partial task {task_id}: {num_completed}/9 scenarios completed")

            remaining_tasks.append(task)

        self.logger.info(f"Checkpoint contains {len(all_completed_tasks)} completed tasks")
        self.logger.info(f"Checkpoint contains {len(all_partial_states)} partial tasks")
        self.logger.info(f"Skipping {skipped_count} already completed tasks")
        self.logger.info(f"Resuming {resumed_partial_count} partial tasks")
        self.logger.info(f"Resuming with {len(remaining_tasks)} total tasks")
        self.logger.info("=" * 60)

        if not remaining_tasks:
            self.logger.info("All tasks already completed! Loading final results...")
            # Load all results and return
            self.results = all_completed_tasks
            return self.to_dataframe()

        # Load existing results into memory
        self.results = all_completed_tasks

        # Group remaining tasks by instruction for sequential execution
        task_groups = self._group_tasks_by_instruction(remaining_tasks)
        total_groups = len(task_groups)

        # Calculate total API calls for remaining tasks based on actual scenario count
        enabled_groups = remaining_tasks[0].enabled_groups if remaining_tasks else None
        num_scenarios = get_scenario_count(enabled_groups)
        total_api_calls = len(remaining_tasks) * num_scenarios
        self.logger.info(f"Total API calls to execute: {total_api_calls} ({len(remaining_tasks)} tasks × {num_scenarios} scenarios)")
        self.logger.info(f"Total instruction groups: {total_groups}")

        # Execute remaining tasks
        start_time = time.time()

        # Note: ThreadPoolExecutor handles concurrency internally
        self.logger.info("Using ThreadPoolExecutor for parallel execution (Resume mode)")
        for model_config in model_configs:
            self.logger.info(f"  Model {model_config.model}: max_workers={model_config.max_workers}")

        # Initialize progress bar for API calls
        pbar = None
        if HAS_TQDM:
            pbar = tqdm(total=total_api_calls, desc="API Calls (Resume)", unit="call",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Execute instruction groups sequentially (one at a time)
        # Note: Within each group, trials are now parallel
        completed_groups = 0
        success_count = sum(1 for r in self.results if r.get("success"))
        failure_count = sum(1 for r in self.results if not r.get("success") and "error" in r)

        for group_key, group_tasks in task_groups.items():
            completed_groups += 1
            model = group_tasks[0].model_config.model
            instruction = group_tasks[0].instruction
            cp_mgr = self.checkpoint_managers.get(model)

            # Update progress bar description
            if pbar:
                pbar.set_description(f"[{completed_groups}/{total_groups}] {model[:15]}...")

            self.logger.info(
                f"\n[{completed_groups}/{total_groups}] RESUME: {model} | Instruction: {instruction[:50]}..."
            )
            self.logger.info(f"  Tasks in this group: {len(group_tasks)} trials")

            try:
                # Execute all tasks in this group in parallel
                group_results = self._execute_task_group_sequential(group_tasks, cp_mgr, None, pbar)
                self.results.extend(group_results)

                # Log summary
                success_results = [r for r in group_results if r.get("success")]
                if success_results:
                    success_count += len(success_results)
                    compliance_count = sum(1 for r in success_results if r.get('action') == 'Compliance')
                    self.logger.info(
                        f"  ✅ Completed: {len(group_tasks)} trials, {len(success_results)} scenarios, "
                        f"{compliance_count} compliance"
                    )
                    # Cleanup partial states for all tasks in this group
                    for task in group_tasks:
                        if cp_mgr:
                            cp_mgr.cleanup_partial_state(task)
                else:
                    failure_count += 1
                    self.logger.warning(f"  ❌ FAILED group: {group_key}")

                # Periodic save after each instruction group
                overall_progress = skipped_count + completed_groups
                if completed_groups % self.save_interval == 0:
                    self._periodic_save(overall_progress)
                    self._save_checkpoint()

            except Exception as exc:
                failure_count += 1
                self.logger.error(f"  💥 CRITICAL EXCEPTION: {exc}")

        # Close progress bar
        if pbar:
            pbar.close()

        # Final save
        if len(self.results) > len(all_completed_tasks):
            self._periodic_save(skipped_count + completed_groups, final=True)
        else:
            self._cleanup_temp_dirs()

        # Cleanup checkpoint on successful completion
        self._cleanup_checkpoint()

        elapsed = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info(f"Resume completed! Time taken: {elapsed:.1f}秒")
        self.logger.info(f"Total instruction groups: {len(task_groups)}")
        self.logger.info(f"Success: {success_count} scenarios")
        self.logger.info(f"Failure: {failure_count} groups")
        self.logger.info("=" * 60)

        return self.to_dataframe()

    def _periodic_save(self, completed: int, final: bool = False):
        """Save checkpoint metadata periodically (group CSVs are already saved immediately)"""
        if not self.output_dir or not self.model_configs:
            return

        new_results = self.results[self.last_save_count:]
        if not new_results:
            return

        # Group CSVs are already saved immediately by _save_results_batch
        # This method only updates checkpoint metadata for resume capability

        self.last_save_count = len(self.results)

        if final:
            self.logger.info(f"Final checkpoint save: {completed} tasks completed")
            # Show run directories that were created
            if hasattr(self, 'run_directories') and self.run_directories:
                self.logger.info("Run directories:")
                for model, run_dir in self.run_directories.items():
                    self.logger.info(f"  {model}: {run_dir}")
                # Show full structure
                self.logger.info("\nDirectory structure:")
                for model, scenario_groups in self.scenario_directories.items():
                    for _scenario_group, path in scenario_groups.items():
                        self.logger.info(f"  {path}/")
        else:
            self.logger.info(f"Checkpoint save [{completed}]: Metadata updated")

    def _get_instruction_hash(self, instruction: str) -> str:
        """Generate short hash for instruction to use as filename"""
        import hashlib
        hash_obj = hashlib.md5(instruction.encode())
        return hash_obj.hexdigest()[:8]

    def _get_scenario_dir(self, model: str, scenario_group: str) -> str:
        """Get or create scenario-based directory structure (THREAD-SAFE)

        Structure: logs/cigt/{model}/{scenario_group}/

        Uses double-checked locking pattern to prevent race conditions:
        1. Fast path: check cache (read-only, no lock)
        2. Slow path: acquire lock, check again, then create
        """
        if not model or not scenario_group:
            raise ValueError(f"Invalid model or scenario_group: model={model}, group={scenario_group}")

        if not self.output_dir:
            raise ValueError("output_dir not set in BatchExperimentManager")

        if not self.run_id:
            raise ValueError("run_id not initialized, call execute() first")

        # Fast path: check cache first (read-only, no lock needed)
        if model in self.scenario_directories and scenario_group in self.scenario_directories[model]:
            return self.scenario_directories[model][scenario_group]

        # Slow path: need to create directory - use lock for entire process
        with self._directory_lock:
            # Double-check after acquiring lock (another thread might have created it)
            if model in self.scenario_directories and scenario_group in self.scenario_directories[model]:
                return self.scenario_directories[model][scenario_group]

            # Sanitize names
            import re
            model_safe = re.sub(r'[\\/*?:"<>|]', '_', model)
            scenario_safe = re.sub(r'[\\/*?:"<>|]', '_', scenario_group)

            # Create directory
            scenario_dir = os.path.join(self.output_dir, model_safe, scenario_safe)
            try:
                os.makedirs(scenario_dir, exist_ok=True)
                if not os.path.isdir(scenario_dir):
                    raise OSError(f"Failed to create directory: {scenario_dir}")

                # Update cache
                if model not in self.scenario_directories:
                    self.scenario_directories[model] = {}
                self.scenario_directories[model][scenario_group] = scenario_dir

                # Track run directory
                run_dir = os.path.join(self.output_dir, model_safe)
                if model not in self.run_directories:
                    self.run_directories[model] = run_dir
                    self.logger.info(f"Created run directory: {run_dir}")

                self.logger.debug(f"Created scenario directory: {scenario_dir}")
                return scenario_dir

            except OSError as e:
                self.logger.error(f"Failed to create directory {scenario_dir}: {e}")
                raise


    def _cleanup_temp_dirs(self, check_only: bool = True):
        """Clean up temporary save directories

        Args:
            check_only: If True, only clean up if temp dir exists (for startup cleanup)
                        If False, always attempt cleanup (for final cleanup)
        """
        if not self.output_dir or not self.model_configs:
            return

        for model_config in self.model_configs:
            model_safe = model_config.model.replace("/", "_")
            temp_dir = os.path.join(self.output_dir, model_safe, "_temp")

            if os.path.exists(temp_dir):
                # Check if temp dir has any files
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
            elif not check_only:
                # Debug log that nothing needed cleanup
                self.logger.debug(f"No temp directory to clean up: {temp_dir}")

    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw results list to DataFrame (without full conversion)"""
        if not results:
            return pd.DataFrame()

        # Filter for successful results
        success_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        if not success_results:
            return pd.DataFrame()

        # Required fields for DataFrame
        required_fields = [
            'trial', 'scenario_id', 'group', 'scenario_name',
            'action', 'reward', 'model', 'instruction', 'raw_text',
            'timestamp', 'mock'
        ]

        records = []
        for r in success_results:
            # Validate all required fields exist
            missing_fields = [f for f in required_fields if f not in r]
            if missing_fields:
                self.logger.warning(f"Result missing fields {missing_fields}: {r}")
                continue

            try:
                records.append({
                    'trial': r['trial'] + 1,
                    'scenario_id': r['scenario_id'],
                    'group': r['group'],
                    'scenario_name': r['scenario_name'],
                    'action': r['action'],
                    'reward': r['reward'],
                    'model': r['model'],
                    'instruction': r['instruction'],
                    'raw_text': r['raw_text'],
                    'timestamp': r['timestamp'],
                    'mock': r['mock']
                })
            except KeyError as e:
                self.logger.error(f"KeyError in result conversion: {e} in {r}")
            except Exception as e:
                self.logger.error(f"Unexpected error in result conversion: {e} in {r}")

        return pd.DataFrame(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        if not self.results:
            return pd.DataFrame()

        # Success results
        success_results = [r for r in self.results if r.get("success")]

        if not success_results:
            return pd.DataFrame()

        # Extract key fields
        records = []
        for r in success_results:
            records.append({
                'trial': r['trial'] + 1,  # Start from 1
                'scenario_id': r['scenario_id'],
                'group': r['group'],  # Use actual group name
                'scenario_name': r['scenario_name'],  # Also include scenario name
                'action': r['action'],
                'reward': r['reward'],
                'model': r['model'],
                'instruction': r['instruction'],
                'raw_text': r['raw_text'],
                'timestamp': r['timestamp'],
                'mock': r['mock']
            })

        return pd.DataFrame(records)

    def _save_checkpoint(self):
        """Save checkpoint for all models"""
        if not self.checkpoint_managers or not self.results:
            return

        # Group results by model
        results_by_model = {}
        for result in self.results:
            model = result.get('model')
            if model and model in self.checkpoint_managers:
                if model not in results_by_model:
                    results_by_model[model] = []
                results_by_model[model].append(result)

        # Get history data accumulator if available
        history_by_model = getattr(self, '_history_accumulator', {})

        # Save checkpoint for each model
        for model_name, model_results in results_by_model.items():
            cp_mgr = self.checkpoint_managers[model_name]

            # Get max_workers for this model from model_configs
            max_workers = 1  # Default
            if hasattr(self, 'model_configs') and self.model_configs:
                for cfg in self.model_configs:
                    if cfg.model == model_name:
                        max_workers = cfg.max_workers
                        break

            # Create metadata
            metadata = {
                "model": model_name,
                "total_results": len(model_results),
                "timestamp": time.time(),
                "save_interval": self.save_interval,
                "max_workers": max_workers,
                "resume": self.resume
            }

            # Get history data for this model
            model_history = history_by_model.get(model_name, [])

            cp_mgr.save_checkpoint(model_results, metadata, model_history)

    def _cleanup_checkpoint(self):
        """Cleanup all checkpoints"""
        if not self.checkpoint_managers:
            return

        for cp_mgr in self.checkpoint_managers.values():
            cp_mgr.cleanup_checkpoint()

        self.logger.info("All checkpoints cleaned up")

    def save_results(self, df: pd.DataFrame, output_dir: str, model_name: str,
                     incremental: bool = False, run_dir: str = None) -> Dict[str, str]:
        """
        Save results split by scenario group

        Args:
            df: Results DataFrame
            output_dir: Output directory
            model_name: Model name
            incremental: If True, append to existing run_dir instead of creating new one
            run_dir: Existing run directory for incremental saves

        Returns:
            Dict mapping group name to file path
        """
        if df.empty:
            self.logger.warning("No results to save")
            return {}

        model_safe = model_name.replace("/", "_")

        if incremental and run_dir:
            # Use existing run directory
            final_run_dir = run_dir
        else:
            # Create new run directory
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            final_run_dir = os.path.join(output_dir, model_safe, f"{timestamp}_{unique_id}")

        os.makedirs(final_run_dir, exist_ok=True)

        # Split and save by group
        file_paths = {}
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            filepath = os.path.join(final_run_dir, f"{group}.csv")

            # For incremental saves, append if file exists
            if incremental and os.path.exists(filepath):
                group_df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                group_df.to_csv(filepath, index=False)

            file_paths[group] = filepath

        # Log summary
        if incremental:
            self.logger.info(f"Appended to {final_run_dir}")
        else:
            self.logger.info(f"Saved to {final_run_dir}")

        for group, path in file_paths.items():
            count = len(df[df['group'] == group])
            self.logger.info(f"  {group}: {count} records")

        return file_paths


# =============================================================================
# 5. Main Function
# =============================================================================

def main(args):
    """Main execution function"""
    # Set global debug mode
    set_debug_mode(args.debug)

    # Setup logger
    logger = setup_logger("BatchExperiments")

    # 1. Load configuration
    config = DEFAULT_CONFIG.copy()
    model_configs = MODEL_CONFIGS.copy()
    enabled_groups = None

    # Try to load config from JSON file
    json_config = load_experiments_config(args.config)
    if json_config:
        # Parse model config from JSON
        json_models = parse_models_from_config(json_config)
        if json_models:
            model_configs = json_models
            logger.info(f"Using {len(json_models)} models from JSON config")

        # Parse runtime config from JSON
        json_runtime = parse_runtime_from_config(json_config)
        config.update(json_runtime)
        logger.info("Applied JSON runtime config")

        # Parse scenarios config from JSON
        enabled_groups = parse_scenarios_from_config(json_config)

    # 2. Override command line arguments
    if args.mock:
        config["mock_mode"] = True
        for cfg in model_configs:
            cfg.mock = True

    if args.dataset:
        config["dataset_path"] = args.dataset

    if args.output_dir:
        config["output_dir"] = args.output_dir

    if args.trials:
        config["trials_per_instruction"] = args.trials

    if args.max_workers:
        config["max_workers"] = args.max_workers

    if args.save_interval:
        config["save_interval"] = args.save_interval

    # 3. Load instructions
    instructions = load_instructions(
        config["dataset_path"],
        args.max_instructions or config["max_instructions"]
    )

    if not instructions:
        logger.error("没有可用的指令，退出")
        return 1

    # 4. Prepare tasks
    tasks = prepare_tasks(
        model_configs=model_configs,
        instructions=instructions,
        trials_per_instruction=config["trials_per_instruction"],
        output_dir=config["output_dir"],
        enabled_groups=enabled_groups
    )

    if not tasks:
        logger.error("No executable tasks, exiting")
        return 1

    # Log scenarios configuration
    if enabled_groups:
        logger.info(f"Scenario filtering enabled: {enabled_groups}")
    else:
        logger.info("Using all scenarios (no filtering)")

    # 5. Execute batch experiments
    save_interval = config.get("save_interval", 5)
    manager = BatchExperimentManager(
        save_interval=save_interval,
        resume=args.resume
    )
    results_df = manager.execute(
        tasks,
        output_dir=config["output_dir"],
        model_configs=model_configs
    )

    if results_df.empty:
        logger.error("All experiments failed, no results")
        return 1

    # 6. Generate summary from run directories (already saved incrementally)
    logger.info("\n" + "=" * 60)
    logger.info("📊 Experiment Summary:")
    logger.info("=" * 60)

    for model, run_dir in manager.run_directories.items():
        model_df = results_df[results_df['model'] == model]
        logger.info(f"\nModel: {model}")
        logger.info(f"  Run directory: {run_dir}")

        # Read saved files to get accurate counts
        for group in model_df['group'].unique():
            group_file = os.path.join(run_dir, f"{group}.csv")
            if os.path.exists(group_file):
                import pandas as pd
                saved_df = pd.read_csv(group_file)
                total_trials = len(saved_df)
                compliance_count = (saved_df['action'] == 'Compliance').sum()
                compliance_rate = compliance_count / total_trials if total_trials > 0 else 0
                logger.info(f"  {group}: {total_trials} trials, {compliance_count} compliance, {compliance_rate:.2%} rate")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 Batch experiments completed!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Experiment Executor - Generates behavior trajectories using modular architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config file (scripts/config/experiments.json)
    python scripts/experiment/run_experiments.py

    # Use custom config file
    python scripts/experiment/run_experiments.py --config my_experiments.json

    # Limited instruction count (for testing)
    python scripts/experiment/run_experiments.py --max_instructions 3 --trials 5

    # Mock mode testing
    python scripts/experiment/run_experiments.py --mock

    # Custom user-provided dataset and output directory
    python scripts/experiment/run_experiments.py --dataset path/to/your_dataset.csv --output_dir ./my_logs

    # Save every 2 completed instruction groups
    python scripts/experiment/run_experiments.py --save_interval 2

    # Resume from checkpoint after crash
    python scripts/experiment/run_experiments.py --resume

    # Resume with mock mode
    python scripts/experiment/run_experiments.py --resume --mock

    # Enable debug logging (shows detailed information)
    python scripts/experiment/run_experiments.py --mock --debug

Execution: One instruction at a time  - all trials for instruction 1,
then all trials for instruction 2, etc.

Debug Mode: Use --debug to show detailed logs including:
    - API calls and messages
    - Prompt building details
    - Response parsing
    - History tracking
    - Checkpoint operations
        """
    )

    parser.add_argument("--config", type=str, default="scripts/config/experiments.json",
                        help="Configuration file path (default: scripts/config/experiments.json)")
    parser.add_argument("--dataset", type=str,
                        help="Instruction dataset path (default: from config or data/AdvBench/demo_harmful_behaviors_custom.csv)")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory (default: from config or ./logs/cigt-smoke)")
    parser.add_argument("--trials", type=int,
                        help="Trials per instruction (default: from config or 2)")
    parser.add_argument("--max_instructions", type=int,
                        help="Maximum instruction count limit (for testing)")
    parser.add_argument("--max_workers", type=int,
                        help="Maximum worker threads (default: from config or 8)")
    parser.add_argument("--save_interval", type=int,
                        help="Save results every N completed tasks (default: from config or 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (use after crash)")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode (no real API calls)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (shows detailed information)")

    args = parser.parse_args()
    sys.exit(main(args))
