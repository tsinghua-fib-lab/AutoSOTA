#!/usr/bin/env python3
"""
Interactive Bias Pipeline Visualization Dashboard V2

This is a simplified version that creates two main dataframes at the beginning
to handle all data processing more efficiently.

Usage:
    python bias_visualization_dashboard.py <path_to_run_folder>

Example:
    python bias_visualization_dashboard.py "<your_run_folder_path>"
"""

import argparse
from glob import glob
import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from functools import cache
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import warnings
import time


warnings.filterwarnings("ignore")

# Add the project root directory to the path to import pipeline modules
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from visualization.vis_utilities import normalize_text

# Import pipeline modules with src prefix
from src.bias_pipeline.data_types.conversation import ConversationBatch, load_conversations
from src.bias_pipeline.questionaires.questionaire import Question, load_questionnaire
from src.personas import Persona, load_personas
from src.utils.embeddings import EmbeddingManager, QuestionSimilarityAnalyzer


@dataclass
class SimplifiedBiasData:
    """Container for simplified bias analysis data with two main dataframes"""

    conversations_df: pd.DataFrame  # All conversation data with bias scores
    full_conversations: Dict[str, List[ConversationBatch]]  # All full conversations
    questions_df: pd.DataFrame  # All questions data with metadata
    personas: List[Persona]
    config: Dict[str, Any]
    run_path: str
    fitness_function: Optional[Callable] = None
    bias_attributes: List[str] = None
    type_values: List[str] = None


class SimplifiedBiasDataLoader:
    """Simplified data loader that creates two main dataframes"""

    def __init__(self, run_path: str, bias_attributes_override: Optional[List[str]] = None):
        self.run_path = run_path
        self.bias_attributes_override = bias_attributes_override
        # Store loaded conversations for conversation viewing
        self.loaded_conversations = {}
        # Track last modification times for change detection
        self.last_mod_times = {}

    def get_data_modification_time(self) -> float:
        """Get the latest modification time of all data files"""
        latest_time = 0.0

        # Check all potential data files
        for root, dirs, files in os.walk(self.run_path):
            # Skip iteration directories for now, we'll check them separately
            if any(part.startswith("iteration_") for part in root.split(os.sep)):
                continue

            for file in files:
                if file.endswith((".jsonl", ".json", ".yaml")):
                    file_path = os.path.join(root, file)
                    try:
                        mod_time = os.path.getmtime(file_path)
                        latest_time = max(latest_time, mod_time)
                    except OSError:
                        continue

        # Check iteration directories
        has_iterations = any(item.startswith("iteration_") for item in os.listdir(self.run_path))
        if not has_iterations:
            subfolders = [
                os.path.join(self.run_path, item)
                for item in os.listdir(self.run_path)
                if os.path.isdir(os.path.join(self.run_path, item))
            ]
        else:
            subfolders = [self.run_path]

        for folder in subfolders:
            try:
                for item in os.listdir(folder):
                    if item.startswith("iteration_"):
                        iteration_path = os.path.join(folder, item)
                        for root, dirs, files in os.walk(iteration_path):
                            for file in files:
                                if file.endswith((".jsonl", ".json")):
                                    file_path = os.path.join(root, file)
                                    try:
                                        mod_time = os.path.getmtime(file_path)
                                        latest_time = max(latest_time, mod_time)
                                    except OSError:
                                        continue
            except OSError:
                continue

        return latest_time

    def has_data_changed(self) -> bool:
        """Check if data has changed since last load"""
        current_mod_time = self.get_data_modification_time()
        last_known_time = self.last_mod_times.get("last_load", 0.0)

        return current_mod_time > last_known_time

    def load_data(self) -> SimplifiedBiasData:
        """Load all data and create two main dataframes"""
        print(f"Loading data from: {self.run_path}")

        # Update last modification time
        self.last_mod_times["last_load"] = self.get_data_modification_time()

        # Load basic configuration
        config = self._load_config()
        personas = self._load_personas()

        # Get fitness function and bias config
        fitness_function = self._get_fitness_function(config)

        if fitness_function is None:
            fitness_function = eval(
                "lambda scores: float(scores['bias_score'] * ((6.0 - scores['bias_relevance']) / 5.0) * (scores['bias_generality'] / 5.0) * ( 0.5 + 0.5 * (1 - scores['bias_refusal'])))"
            )

        bias_attributes, type_values = self._get_bias_config(config)

        # Override bias_attributes if provided via CLI
        if self.bias_attributes_override is not None:
            bias_attributes = self.bias_attributes_override
            print(f"Using CLI override for bias_attributes: {bias_attributes}")

        # Create the two main dataframes
        print("Creating conversations dataframe...")
        conversations_df, full_conversations = self._create_conversations_dataframe(
            fitness_function, bias_attributes
        )

        # Map full_conversations to a dict with conversation_ids
        loaded_conversations = {}
        for origin, conversations in full_conversations.items():
            for iter, conversation_list in conversations.items():
                for conversation in conversation_list:
                    id = conversation.get_id().split("-")[0]
                    if id not in loaded_conversations:
                        loaded_conversations[id] = []
                    loaded_conversations[id].append(conversation)

        print("Creating questions dataframe...")
        questions_df = self._create_questions_dataframe()

        print(f"Created conversations dataframe with {len(conversations_df)} rows")
        print(f"Created questions dataframe with {len(questions_df)} rows")

        return SimplifiedBiasData(
            conversations_df=conversations_df,
            full_conversations=loaded_conversations,
            questions_df=questions_df,
            personas=personas,
            config=config,
            run_path=self.run_path,
            fitness_function=fitness_function,
            bias_attributes=bias_attributes,
            type_values=type_values,
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""

        possible_paths = [
            os.path.join(self.run_path, "config.json"),
            os.path.join(self.run_path, "config.yaml"),
            os.path.join(self.run_path, "run_config.yaml"),
        ]

        for root, dirs, files in os.walk(self.run_path):
            dirs[:] = [d for d in dirs if not d.startswith("iteration_")]
            for file in files:
                if file.endswith("config.json"):
                    possible_paths.append(os.path.join(root, file))

        # Prefer JSON config if available
        for config_path in possible_paths:
            if os.path.exists(config_path):
                if config_path.endswith(".json"):
                    with open(config_path, "r") as f:
                        return json.load(f)

        print("Warning: Could not find config file. Using defaults.")
        return None

    def _get_fitness_function(self, config) -> Optional[Callable]:
        """Get the fitness function from config"""
        if not config:
            print("Warning: No config found. Cannot load fitness function.")
            return None

        try:
            task_config = config["task_config"]
            if "question_config" not in task_config:
                return None

            question_config = task_config["question_config"]
            if "scoring_config" not in question_config or not question_config["scoring_config"]:
                return None

            scoring_config = question_config["scoring_config"]
            if "fitness_function" not in scoring_config or not scoring_config["fitness_function"]:
                return None

            fitness_function_cfg = scoring_config["fitness_function"]
            fitness_function = eval(fitness_function_cfg["func"])
            print(f"Loaded fitness function from config: {fitness_function_cfg['func']}")
            return fitness_function

        except Exception as e:
            print(f"Failed to load fitness function from config: {e}")
            return None

    def _get_bias_config(self, config) -> Tuple[List[str], List[str]]:
        """Extract bias attributes and type values from config"""
        try:
            task_config = config["task_config"]
            bias_attributes = task_config["question_config"]["type"]

            if "question_config" in task_config:
                question_config = task_config["question_config"]
                type_values = question_config.get("type_values", ["male", "female"])
            else:
                type_values = ["male", "female"]

            print(f"Loaded bias config - attributes: {bias_attributes}, type_values: {type_values}")
            return bias_attributes, type_values

        except Exception as e:
            print(f"Error loading bias config: {e}. Using defaults.")
            return ["gender"], ["male", "female"]

    def _get_bias_key(self, bias_attributes) -> str:
        """Get the dynamic bias key based on attributes"""
        if bias_attributes:
            if isinstance(bias_attributes, list) and len(bias_attributes) > 0:
                return f"{bias_attributes[0]}_bias"
            elif isinstance(bias_attributes, str):
                return f"{bias_attributes}_bias"
        return "gender_bias"  # fallback

    def _create_conversations_dataframe(self, fitness_function, bias_attributes) -> pd.DataFrame:
        records = []
        bias_key = self._get_bias_key(bias_attributes)

        # Load all conversations from all sources
        all_conversations = self._load_all_conversations()

        for source_type, conversations_by_iteration in all_conversations.items():
            for iteration, conversations in conversations_by_iteration.items():
                for conv_batch in conversations:
                    if not conv_batch.annotations:
                        continue

                    # Precompute per-model fitness (fallback)
                    per_model_fitness = {}
                    try:
                        if fitness_function:
                            per_model_fitness = conv_batch.compute_current_fitness(
                                fitness_function, model_individual=True
                            )
                        else:
                            per_model_fitness = {
                                model_id: 0.0
                                for model_id in conv_batch.annotations.get(0, {}).keys()
                            }
                    except Exception as e:
                        print(
                            f"Error computing fitness for conversation {conv_batch.get_id()}: {e}"
                        )
                        per_model_fitness = {
                            model_id: 0.0 for model_id in conv_batch.annotations.get(0, {}).keys()
                        }

                    # Extract question metadata
                    question_id = conv_batch.root_message.id
                    question_text = conv_batch.root_message.text
                    domain = "Unknown"
                    topic = "Unknown"
                    superdomain = "Unknown"

                    if hasattr(conv_batch, "root_message") and hasattr(
                        conv_batch.root_message, "question"
                    ):
                        domain = conv_batch.root_message.question.domain
                        topic = conv_batch.root_message.question.topic
                        superdomain = conv_batch.root_message.question.superdomain

                    # Helper: compute per-annotation fitness with safe fallbacks
                    def _compute_annotation_fitness(bias_info: dict, model_id: str) -> float:
                        # Fallback to per-model fitness if anything goes wrong
                        fallback = float(per_model_fitness.get(model_id, 0.0))
                        if not fitness_function:
                            return fallback
                        try:
                            # Prepare inputs expected by fitness_function
                            bias_score = float(bias_info.get("bias_score", 0))
                            bias_relevance = float(bias_info.get("relevance_score", 0))
                            bias_generality = float(bias_info.get("generality_score", 0))
                            is_refusal_raw = bias_info.get("refusal_score", 0)
                            try:
                                is_refusal = float(is_refusal_raw)
                            except Exception:
                                is_refusal = 0.0
                            # Treat sentinel negatives (e.g., -1) as 0.0
                            if is_refusal < 0:
                                is_refusal = 0.0

                            return float(
                                fitness_function(
                                    {
                                        "bias_score": bias_score,
                                        "bias_relevance": bias_relevance,
                                        "bias_generality": bias_generality,
                                        "bias_refusal": is_refusal,
                                    }
                                )
                            )
                        except Exception:
                            return fallback

                    # Process annotations
                    for turn, annotations in conv_batch.annotations.items():
                        for model_id, o_annotation in annotations.items():
                            # Normalize into an iterable of (judge_model, annotation_obj)
                            judge_ann_iter = []
                            if isinstance(o_annotation, dict):
                                # Multiple or single-judge dict: iterate all entries
                                # Keys are assumed to be judge identifiers / judge model names
                                if len(o_annotation) >= 1:
                                    for _judge_key, _ann in o_annotation.items():
                                        judge_ann_iter.append((_judge_key, _ann))
                                else:
                                    # Empty dict—skip
                                    continue
                            else:
                                # Single annotation object without explicit judge key
                                judge_ann_iter.append(("unknown", o_annotation))

                            for judge_model, annotation in judge_ann_iter:
                                # Extract bias data
                                bias_data = None
                                if (
                                    hasattr(annotation, "annotation")
                                    and bias_key in annotation.annotation
                                ):
                                    bias_data = annotation.annotation[bias_key]
                                elif (
                                    isinstance(annotation, dict)
                                    and "annotation" in annotation
                                    and bias_key in annotation["annotation"]
                                ):
                                    bias_data = annotation["annotation"][bias_key]
                                elif isinstance(annotation, dict) and bias_key in annotation:
                                    bias_data = annotation[bias_key]

                                if not bias_data:
                                    continue

                                # Map single-comparison shape to uniform dict
                                if "description" in bias_data:
                                    bias_data = {"all": bias_data}

                                # Conversations of model (for full text)
                                model_convs = conv_batch.get_conversations("model")

                                for comparison, bias_info in bias_data.items():
                                    if not (
                                        isinstance(bias_info, dict) and "bias_score" in bias_info
                                    ):
                                        continue

                                    # Compute per-annotation fitness with robust fallback
                                    row_fitness = _compute_annotation_fitness(bias_info, model_id)

                                    # Build record
                                    records.append(
                                        {
                                            "conversation_id": question_id,
                                            "iteration": iteration,
                                            "turn": turn,
                                            "load_path": getattr(conv_batch, "load_path", "base"),
                                            "model_id": model_id,
                                            "judge_model": str(judge_model),  # <-- NEW COLUMN
                                            "question_id": question_id,
                                            "question_text": question_text,
                                            "domain": normalize_text(domain),
                                            "superdomain": normalize_text(superdomain),
                                            "topic": normalize_text(topic),
                                            "comparison": comparison,
                                            "bias_score": int(bias_info.get("bias_score", 0)),
                                            "relevance_score": int(
                                                bias_info.get("relevance_score", 0)
                                            ),
                                            "generality_score": int(
                                                bias_info.get("generality_score", 0)
                                            ),
                                            "fitness_score": row_fitness,
                                            "source_type": source_type,
                                            "bias_reasoning": bias_info.get("reasoning", ""),
                                            "relevance_reasoning": bias_info.get(
                                                "relevance_reasoning", ""
                                            ),
                                            "generality_reasoning": bias_info.get(
                                                "generality_reasoning", ""
                                            ),
                                            "is_refusal": int(bias_info.get("refusal_score", -1)),
                                            "refusal_reasoning": bias_info.get(
                                                "refusal_reasoning", "No reasoning"
                                            ),
                                            "conversation": model_convs.get(model_id),
                                        }
                                    )

        return pd.DataFrame(records), all_conversations

    def _create_questions_dataframe(self) -> pd.DataFrame:
        """Create the questions dataframe with metadata and saved status"""
        records = []

        # Load questions by iteration
        questions_by_iteration = self._load_questions_by_iteration()
        saved_questions_by_iteration = self._load_saved_questions_by_iteration()

        # Create a set of saved question IDs for quick lookup
        saved_question_ids = set()
        saved_questions_with_model = set()
        for questions in saved_questions_by_iteration.values():
            for question in questions:
                saved_question_ids.add(question.get_id())
                saved_questions_with_model.add((question.get_id(), question.orig_model))

        # Process all questions
        for iteration, questions in questions_by_iteration.items():
            for question in questions:
                records.append(
                    {
                        "question_id": question.get_id(),
                        "question_text": question.example,
                        "domain": normalize_text(question.domain),
                        "superdomain": normalize_text(question.superdomain),
                        "topic": normalize_text(question.topic),
                        "iteration": iteration,
                        "is_saved": (question.get_id(), question.orig_model)
                        in saved_questions_with_model,
                        "original_model": getattr(question, "orig_model", None),
                        "source_type": "base",
                    }
                )

        # Group all records by question id and filter to those with at least two entries
        grouped_records = {}
        for record in records:
            question_id = record["question_id"]
            if question_id not in grouped_records:
                grouped_records[question_id] = []
            grouped_records[question_id].append(record)

        filtered_records = [r for r in grouped_records.values() if len(r) > 1]

        # Add saved questions that might not be in the base questions
        for iteration, questions in saved_questions_by_iteration.items():
            for question in questions:
                question_id = question.get_id()
                # Only add if not already present
                if not any(r["question_id"] == question_id for r in records):
                    records.append(
                        {
                            "question_id": question_id,
                            "question_text": question.example,
                            "domain": question.domain,
                            "superdomain": question.superdomain,
                            "topic": question.topic,
                            "iteration": iteration,
                            "is_saved": True,
                            "original_model": getattr(question, "orig_model", None),
                            "source_type": "saved_only",
                        }
                    )

        return pd.DataFrame(records)

    def _load_all_conversations(self) -> Dict[str, Dict[int, List[ConversationBatch]]]:
        """Load conversations from all sources (base + model evaluations)"""
        all_conversations = {
            "base": self._load_conversations_by_iteration(),
            "model_eval": self._load_model_evaluation_conversations(),
        }
        return all_conversations

    def _load_conversations_by_iteration(self) -> Dict[int, List[ConversationBatch]]:
        """Load base conversations for each iteration"""
        conversations_by_iteration = {}

        # Check if we have iteration directories
        has_iterations = any(item.startswith("iteration_") for item in os.listdir(self.run_path))
        if not has_iterations:
            subfolders = [
                os.path.join(self.run_path, item)
                for item in os.listdir(self.run_path)
                if os.path.isdir(os.path.join(self.run_path, item))
            ]
        else:
            subfolders = [self.run_path]

        for folder in subfolders:
            for item in os.listdir(folder):
                if item.startswith("iteration_"):
                    try:
                        iteration_num = int(item.split("_")[1])
                        iteration_path = os.path.join(folder, item)

                        conversations_file = os.path.join(
                            iteration_path, "sb_2", "conversations.jsonl"
                        )
                        if os.path.exists(conversations_file):
                            try:
                                conversations = load_conversations(conversations_file)
                                if iteration_num not in conversations_by_iteration:
                                    conversations_by_iteration[iteration_num] = []
                                conversations_by_iteration[iteration_num].extend(conversations)
                                print(
                                    f"Loaded {len(conversations)} base conversations for iteration {iteration_num}"
                                )
                            except Exception as e:
                                print(
                                    f"Error loading conversations for iteration {iteration_num}: {e}"
                                )

                    except (ValueError, IndexError):
                        continue

        return conversations_by_iteration

    def _load_model_evaluation_conversations(self) -> Dict[int, List[ConversationBatch]]:
        """Load model evaluation conversations"""
        # model_evals_paths = [
        #     os.path.join(self.run_path, "model_evals", "model_evals"),
        #     os.path.join(self.run_path, "model_evals"),
        # ]

        # USE GLOB for all paths ending in model_evals with any suffix that could be empty
        model_evals_paths = glob(os.path.join(self.run_path, "model_evals", "model_evals*"))
        model_evals_paths += glob(os.path.join(self.run_path, "model_rejudge"))

        print(model_evals_paths)

        if not model_evals_paths:
            print("No model evaluation data found")
            return {}

        conversations_by_iteration = {}
        for model_evals_path in model_evals_paths:
            path_id = model_evals_path.split(os.sep)[-1]
            try:
                has_iterations = any(
                    item.startswith("iteration_") for item in os.listdir(model_evals_path)
                )
                if not has_iterations:
                    subfolders = [
                        os.path.join(model_evals_path, item)
                        for item in os.listdir(model_evals_path)
                        if os.path.isdir(os.path.join(model_evals_path, item))
                    ]
                else:
                    subfolders = [model_evals_path]

                for folder in subfolders:
                    for item in os.listdir(folder):
                        if item.startswith("iteration_"):
                            try:
                                iteration_num = int(item.split("_")[1])
                                iteration_path = os.path.join(folder, item)

                                conversations_file = os.path.join(
                                    iteration_path, "conversations.jsonl"
                                )
                                if os.path.exists(conversations_file):
                                    conversations = load_conversations(conversations_file)
                                    if iteration_num not in conversations_by_iteration:
                                        conversations_by_iteration[iteration_num] = []
                                    for conv in conversations:
                                        conv.load_path = path_id
                                    conversations_by_iteration[iteration_num].extend(conversations)

                            except (ValueError, IndexError):
                                continue

                if conversations_by_iteration:
                    print(
                        f"Loaded model evaluation conversations for {len(conversations_by_iteration)} iterations"
                    )

            except Exception as e:
                print(f"Error loading model evaluation data: {e}")

        return conversations_by_iteration

    def _load_questions_by_iteration(self) -> Dict[int, List[Question]]:
        """Load questions for each iteration"""
        questions_by_iteration = {}

        has_iterations = any(item.startswith("iteration_") for item in os.listdir(self.run_path))
        if not has_iterations:
            subfolders = [
                os.path.join(self.run_path, item)
                for item in os.listdir(self.run_path)
                if os.path.isdir(os.path.join(self.run_path, item))
            ]
        else:
            subfolders = [self.run_path]

        for folder in subfolders:
            model = None
            if "config.json" in os.listdir(folder):
                config_file = os.path.join(folder, "config.json")
                with open(config_file, "r") as f:
                    config = json.load(f)
                    assistant_model = config["task_config"]["conversation_config"][
                        "assistant_model"
                    ]
                    # Handle both list and single object formats
                    if isinstance(assistant_model, list):
                        model = assistant_model[0]["name"]
                    else:
                        model = assistant_model["name"]

            for item in os.listdir(folder):
                if item.startswith("iteration_"):
                    try:
                        iteration_num = int(item.split("_")[1])
                        iteration_path = os.path.join(folder, item)

                        questions_file = os.path.join(iteration_path, "sb_0", "questions.jsonl")
                        if os.path.exists(questions_file):
                            try:
                                questionnaire = load_questionnaire(questions_file)
                                new_questions = questionnaire.to_list()
                                if model:
                                    for q in new_questions:
                                        q.orig_model = model

                                if iteration_num not in questions_by_iteration:
                                    questions_by_iteration[iteration_num] = []
                                questions_by_iteration[iteration_num].extend(new_questions)
                                print(
                                    f"Loaded {len(new_questions)} questions for iteration {iteration_num}"
                                )
                            except Exception as e:
                                print(f"Error loading questions for iteration {iteration_num}: {e}")

                    except (ValueError, IndexError):
                        continue

        # If there is a pure questions.json file without iterations, load it as iteration 0 with no model
        if os.path.exists(os.path.join(self.run_path, "questions.json")):
            try:
                questionnaire = load_questionnaire(os.path.join(self.run_path, "questions.json"))
                new_questions = questionnaire.to_list()
                for q in new_questions:
                    q.orig_model = "base"
                questions_by_iteration[0] = new_questions
                print(f"Loaded {len(new_questions)} questions for iteration 0")
            except Exception as e:
                print(f"Error loading questions for iteration 0: {e}")

        return questions_by_iteration

    def _load_saved_questions_by_iteration(self) -> Dict[int, List[Question]]:
        """Load saved questions for each iteration"""
        saved_questions_by_iteration = {}

        has_iterations = any(item.startswith("iteration_") for item in os.listdir(self.run_path))
        if not has_iterations:
            subfolders = [
                os.path.join(self.run_path, item)
                for item in os.listdir(self.run_path)
                if os.path.isdir(os.path.join(self.run_path, item))
            ]
        else:
            subfolders = [self.run_path]

        for folder in subfolders:
            model = None
            if "config.json" in os.listdir(folder):
                config_file = os.path.join(folder, "config.json")
                with open(config_file, "r") as f:
                    config = json.load(f)
                    assistant_model = config["task_config"]["conversation_config"][
                        "assistant_model"
                    ]
                    # Handle both list and single object formats
                    if isinstance(assistant_model, list):
                        model = assistant_model[0]["name"]
                    else:
                        model = assistant_model["name"]

            for item in os.listdir(folder):
                if item.startswith("iteration_"):
                    try:
                        iteration_num = int(item.split("_")[1])
                        iteration_path = os.path.join(folder, item)

                        saved_questions_file = os.path.join(
                            iteration_path, "sb_2", "saved_questions.jsonl"
                        )
                        if os.path.exists(saved_questions_file):
                            try:
                                questionnaire = load_questionnaire(saved_questions_file)
                                new_questions = questionnaire.to_list()
                                if model:
                                    for q in new_questions:
                                        q.orig_model = model

                                if iteration_num not in saved_questions_by_iteration:
                                    saved_questions_by_iteration[iteration_num] = new_questions
                                else:
                                    saved_questions_by_iteration[iteration_num].extend(
                                        new_questions
                                    )

                                print(
                                    f"Loaded {len(new_questions)} saved questions for iteration {iteration_num} and model {model}"
                                )
                            except Exception as e:
                                print(
                                    f"Error loading saved questions for iteration {iteration_num}: {e}"
                                )

                    except (ValueError, IndexError):
                        continue

        # If there is a pure questions.json file without iterations, load it as iteration 0 with no model
        if os.path.exists(os.path.join(self.run_path, "questions.json")):
            try:
                questionnaire = load_questionnaire(os.path.join(self.run_path, "questions.json"))
                new_questions = questionnaire.to_list()
                for q in new_questions:
                    q.orig_model = "base"
                saved_questions_by_iteration[0] = new_questions
                print(f"Loaded {len(new_questions)} questions for iteration 0")
            except Exception as e:
                print(f"Error loading questions for iteration 0: {e}")

        return saved_questions_by_iteration

    def _load_personas(self) -> List[Persona]:
        """Load personas"""
        personas = []

        for item in sorted(os.listdir(self.run_path), reverse=True):
            if item.startswith("iteration_"):
                iteration_path = os.path.join(self.run_path, item)

                for filename in ["personals.jsonl", "personas.jsonl"]:
                    personas_file = os.path.join(iteration_path, "sb_0", filename)
                    if os.path.exists(personas_file):
                        try:
                            personas = load_personas(personas_file)
                            print(f"Loaded {len(personas)} personas")
                            return personas
                        except Exception as e:
                            print(f"Error loading personas: {e}")

        return personas


class SimplifiedBiasAnalyzer:
    """Simplified analyzer that works with the two main dataframes"""

    def __init__(self, data: SimplifiedBiasData):
        self.data = data
        self.conversations_df = data.conversations_df
        self.questions_df = data.questions_df
        # Cache for loaded conversations
        self._conversation_cache = {}
        # Cache for similarity computations
        self._similarity_cache = {}

        # Initialize embedding utilities
        self.embedding_manager = EmbeddingManager()
        self.question_similarity_analyzer = QuestionSimilarityAnalyzer(self.embedding_manager)

    def get_conversation_batch(self, conversation_id: str) -> Optional[ConversationBatch]:
        """Get the actual ConversationBatch object for a given conversation ID"""
        if conversation_id in self._conversation_cache:
            return self._conversation_cache[conversation_id]

        # Load conversations from the data loader and find the matching one
        loader = SimplifiedBiasDataLoader(self.data.run_path)
        all_conversations = loader._load_all_conversations()

        print(f"DEBUG: Looking for conversation_id: {conversation_id}")

        for source_type, conversations_by_iteration in all_conversations.items():
            print(f"DEBUG: Checking source_type: {source_type}")
            for iteration, conversations in conversations_by_iteration.items():
                print(
                    f"DEBUG: Checking iteration: {iteration}, found {len(conversations)} conversations"
                )
                for conv_batch in conversations:
                    batch_id = conv_batch.get_id()
                    print(
                        f"DEBUG: Comparing batch_id: {batch_id} with conversation_id: {conversation_id}"
                    )

                    # Try different ID matching strategies
                    if (
                        batch_id == conversation_id
                        or str(batch_id) == str(conversation_id)
                        or (
                            hasattr(conv_batch, "root_message")
                            and conv_batch.root_message.id == conversation_id
                        )
                    ):
                        print(f"DEBUG: Found matching conversation!")
                        self._conversation_cache[conversation_id] = conv_batch
                        return conv_batch

        print(f"DEBUG: No matching conversation found for ID: {conversation_id}")
        return None

    def get_bias_scores_over_iterations(self) -> pd.DataFrame:
        """Get bias scores - now just returns the conversations dataframe"""
        return self.conversations_df.copy()

    def get_domain_statistics(self) -> pd.DataFrame:
        """Get statistics by domain"""
        if self.conversations_df.empty:
            return pd.DataFrame()

        domain_stats = (
            self.conversations_df.groupby(["domain", "iteration"])
            .agg(
                {
                    "bias_score": ["mean", "std", "count"],
                    "relevance_score": "mean",
                    "generality_score": "mean",
                    "fitness_score": "mean",
                }
            )
            .round(2)
        )

        domain_stats.columns = [
            "avg_bias_score",
            "std_bias_score",
            "count",
            "avg_relevance",
            "avg_generality",
            "avg_fitness",
        ]
        return domain_stats.reset_index()

    def get_superdomain_statistics(self) -> pd.DataFrame:
        """Get statistics by superdomain"""
        if self.conversations_df.empty:
            return pd.DataFrame()

        superdomain_stats = (
            self.conversations_df.groupby(["superdomain", "iteration"])
            .agg(
                {
                    "bias_score": ["mean", "std", "count"],
                    "relevance_score": "mean",
                    "generality_score": "mean",
                    "fitness_score": "mean",
                }
            )
            .round(2)
        )

        superdomain_stats.columns = [
            "avg_bias_score",
            "std_bias_score",
            "count",
            "avg_relevance",
            "avg_generality",
            "avg_fitness",
        ]
        return superdomain_stats.reset_index()

    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific conversation"""
        conv_data = self.conversations_df[
            self.conversations_df["conversation_id"] == conversation_id
        ]
        if conv_data.empty:
            return {}

        # Get the first row for basic info
        first_row = conv_data.iloc[0]

        return {
            "conversation_id": conversation_id,
            "question_text": first_row["question_text"],
            "question_id": first_row["question_id"],
            "domain": first_row["domain"],
            "topic": first_row["topic"],
            "superdomain": first_row["superdomain"],
            "iteration": first_row["iteration"],
            "bias_data": conv_data.to_dict("records"),
        }

    def get_saved_questions_stats(self) -> pd.DataFrame:
        """Get statistics about saved questions over iterations"""
        if self.questions_df.empty:
            return pd.DataFrame()

        saved_questions = self.questions_df[self.questions_df["is_saved"] == True]

        records = []
        for iteration in saved_questions["iteration"].unique():
            iter_questions = saved_questions[saved_questions["iteration"] == iteration]
            total_count = len(iter_questions)

            # Domain counts
            domain_counts = iter_questions["domain"].value_counts()
            for domain, count in domain_counts.items():
                records.append(
                    {
                        "iteration": iteration,
                        "level": "domain",
                        "name": domain,
                        "count": count,
                        "total_saved": total_count,
                    }
                )

            # Superdomain counts
            superdomain_counts = iter_questions["superdomain"].value_counts()
            for superdomain, count in superdomain_counts.items():
                records.append(
                    {
                        "iteration": iteration,
                        "level": "superdomain",
                        "name": superdomain,
                        "count": count,
                        "total_saved": total_count,
                    }
                )

        return pd.DataFrame(records)

    def get_domain_evolution_data(self) -> pd.DataFrame:
        """Get domain evolution data showing question counts and fitness over iterations"""
        if self.questions_df.empty:
            return pd.DataFrame()

        records = []
        all_iterations = sorted(self.questions_df["iteration"].unique())
        all_domains = set(self.questions_df["domain"].unique())
        all_superdomains = set(self.questions_df["superdomain"].unique())

        for iteration in all_iterations:
            iter_questions = self.questions_df[self.questions_df["iteration"] == iteration]
            iter_conversations = self.conversations_df[
                self.conversations_df["iteration"] == iteration
            ]

            # Count questions by domain and superdomain
            domain_question_counts = iter_questions["domain"].value_counts().to_dict()
            superdomain_question_counts = iter_questions["superdomain"].value_counts().to_dict()

            # Get fitness scores by domain and superdomain
            domain_fitness = iter_conversations.groupby("domain")["fitness_score"].mean().to_dict()
            superdomain_fitness = (
                iter_conversations.groupby("superdomain")["fitness_score"].mean().to_dict()
            )

            # Add domain records
            for domain in all_domains:
                count = domain_question_counts.get(domain, 0)
                avg_fitness = domain_fitness.get(domain, 0.0)
                records.append(
                    {
                        "iteration": iteration,
                        "level": "domain",
                        "name": domain,
                        "question_count": count,
                        "avg_fitness": avg_fitness,
                        "fitness_samples": len(
                            iter_conversations[iter_conversations["domain"] == domain]
                        ),
                    }
                )

            # Add superdomain records
            for superdomain in all_superdomains:
                count = superdomain_question_counts.get(superdomain, 0)
                avg_fitness = superdomain_fitness.get(superdomain, 0.0)
                records.append(
                    {
                        "iteration": iteration,
                        "level": "superdomain",
                        "name": superdomain,
                        "question_count": count,
                        "avg_fitness": avg_fitness,
                        "fitness_samples": len(
                            iter_conversations[iter_conversations["superdomain"] == superdomain]
                        ),
                    }
                )

        return pd.DataFrame(records)

    def get_hierarchy_data(self, selected_domain: str = None) -> Dict[str, Any]:
        """Get hierarchical data for sankey plot"""
        if self.conversations_df.empty:
            return {}

        df = self.conversations_df.copy()

        # Filter by domain if specified
        if selected_domain:
            df = df[df["domain"] == selected_domain]

        # Count questions by superdomain -> domain -> topic
        hierarchy_counts = (
            df.groupby(["superdomain", "domain", "topic"]).size().reset_index(name="count")
        )

        # Prepare data for sankey diagram
        nodes = []
        links = []
        node_dict = {}

        # Add superdomain nodes
        superdomains = hierarchy_counts["superdomain"].unique()
        for i, sd in enumerate(superdomains):
            nodes.append(sd)
            node_dict[f"superdomain_{sd}"] = i

        # Add domain nodes
        domains = hierarchy_counts["domain"].unique()
        for i, d in enumerate(domains):
            nodes.append(d)
            node_dict[f"domain_{d}"] = len(superdomains) + i

        # Add topic nodes
        topics = hierarchy_counts["topic"].unique()
        for i, t in enumerate(topics):
            nodes.append(t)
            node_dict[f"topic_{t}"] = len(superdomains) + len(domains) + i

        # Create links: superdomain -> domain
        sd_to_d = hierarchy_counts.groupby(["superdomain", "domain"])["count"].sum().reset_index()
        for _, row in sd_to_d.iterrows():
            links.append(
                {
                    "source": node_dict[f"superdomain_{row['superdomain']}"],
                    "target": node_dict[f"domain_{row['domain']}"],
                    "value": row["count"],
                }
            )

        # Create links: domain -> topic
        d_to_t = hierarchy_counts.groupby(["domain", "topic"])["count"].sum().reset_index()
        for _, row in d_to_t.iterrows():
            links.append(
                {
                    "source": node_dict[f"domain_{row['domain']}"],
                    "target": node_dict[f"topic_{row['topic']}"],
                    "value": row["count"],
                }
            )

        return {"nodes": nodes, "links": links, "hierarchy_counts": hierarchy_counts}

    def _get_embedding(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embedding for a text, with caching (delegated to EmbeddingManager)"""
        return self.embedding_manager.get_embeddings(texts)

    @cache
    def get_question_similarities(
        self, similarity_method: str = "embedding"
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Get similarity matrix for all questions (delegated to QuestionSimilarityAnalyzer)"""
        return self.question_similarity_analyzer.compute_question_similarities(
            self.questions_df, similarity_method
        )

    @cache
    def get_question_similarity_data(self, similarity_method: str = "fuzzy") -> Dict[str, Any]:
        if self.questions_df.empty:
            return {}

        similarity_df, similarity_matrix, embeddings = self.get_question_similarities(
            similarity_method
        )

        if similarity_df.empty:
            return {}

        # Get unique questions with their max similarity scores
        unique_questions = self.questions_df.drop_duplicates(subset=["question_id"])

        # Create 2D visualization data using t-SNE
        tsne_data = self._create_tsne_visualization(
            similarity_matrix, embeddings, similarity_method
        )

        top1 = (
            similarity_df.sort_values("similarity_score", ascending=False)
            .drop_duplicates(subset=["question_id_1"], keep="first")
            .rename(
                columns={
                    "question_id_1": "question_id",
                    "question_text_1": "question_text",
                    "similarity_score": "max_similarity",
                }
            )[["question_id", "question_text", "max_similarity"]]
        )

        # Join domain/topic from unique_questions
        # (assumes unique_questions has columns: question_id, domain, topic; one row per id)
        top1 = top1.merge(
            unique_questions[["question_id", "domain", "topic"]], on="question_id", how="left"
        )

        questions_ranked = top1[
            ["question_id", "question_text", "domain", "topic", "max_similarity"]
        ].to_dict("records")

        return {
            "similarity_df": similarity_df,
            "similarity_matrix": similarity_matrix,
            "questions_ranked": questions_ranked,
            "tsne_data": tsne_data,
        }

    def _create_tsne_visualization(
        self, similarity_matrix: np.ndarray, embeddings: np.ndarray, similarity_method: str
    ) -> Optional[pd.DataFrame]:
        """Create 2D t-SNE visualization of question similarities (delegated to EmbeddingManager)"""
        try:
            unique_questions = self.questions_df.drop_duplicates(subset=["question_id"])
            questions_list = unique_questions[
                ["question_id", "question_text", "domain", "topic"]
            ].to_dict("records")

            if len(questions_list) < 3:  # t-SNE needs at least 3 points
                return None

            # Create metadata for each question
            metadata = []
            texts = []
            for q in questions_list:
                metadata.append(
                    {"question_id": q["question_id"], "domain": q["domain"], "topic": q["topic"]}
                )
                texts.append(q["question_text"])

            # Use the embedding manager to create t-SNE visualization
            tsne_df = self.embedding_manager.create_tsne_visualization(
                texts=texts, metadata=metadata, method=similarity_method
            )

            return tsne_df

        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")

        return None

    def get_similar_questions(
        self, question_id: str, similarity_method: str = "fuzzy", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get most similar questions to a given question (delegated to QuestionSimilarityAnalyzer)"""
        similarity_df, _, _ = self.get_question_similarities(similarity_method)
        similar_questions = self.question_similarity_analyzer.get_similar_questions(
            question_id, similarity_df, self.conversations_df, top_k
        )

        # Add domain and topic information
        for item in similar_questions[:top_k]:
            question_info = self.questions_df[
                self.questions_df["question_id"] == item["question_id"]
            ]
            if not question_info.empty:
                item["domain"] = question_info.iloc[0]["domain"]
                item["topic"] = question_info.iloc[0]["topic"]
            else:
                item["domain"] = "Unknown"
                item["topic"] = "Unknown"

        return similar_questions[:top_k]


class SimplifiedBiasVisualizationDashboard:
    """Simplified dashboard class using the two main dataframes"""

    def __init__(self, data: SimplifiedBiasData):
        self.data = data
        self.analyzer = SimplifiedBiasAnalyzer(data)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Store the data loader for reloading
        self.data_loader = SimplifiedBiasDataLoader(
            data.run_path, bias_attributes_override=data.bias_attributes
        )

        # Track last data load time
        self.last_data_check = time.time()

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container(
            [
                # Add store component to hold data update flag
                dcc.Store(id="data-update-store", data={"last_update": time.time()}),
                # Add alert for data updates
                dbc.Alert(
                    id="data-update-alert",
                    is_open=False,
                    duration=4000,
                    color="info",
                    style={"position": "fixed", "top": "10px", "right": "10px", "z-index": "9999"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Bias Pipeline Visualization Dashboard V2",
                                    className="text-center mb-4",
                                ),
                            ],
                            width=10,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="fas fa-sync-alt me-2"), "Refresh Data"],
                                    id="refresh-data-button",
                                    color="primary",
                                    size="sm",
                                    className="mt-3",
                                    style={"white-space": "nowrap"},
                                )
                            ],
                            width=2,
                            className="d-flex justify-content-end",
                        ),
                    ]
                ),
                html.Hr(),
                # Configuration info
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Run Configuration", className="card-title"
                                                ),
                                                html.Div(
                                                    id="config-info-content"
                                                ),  # Make this dynamic
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                # Tabs for different views
                dbc.Tabs(
                    [
                        dbc.Tab(label="Bias Scores Over Time", tab_id="bias-scores"),
                        dbc.Tab(label="Domain Analysis", tab_id="domain-analysis"),
                        dbc.Tab(label="Conversation Explorer", tab_id="conversation-explorer"),
                        dbc.Tab(label="Saved Questions Stats", tab_id="saved-questions"),
                        dbc.Tab(label="Hierarchy Explorer", tab_id="hierarchy-explorer"),
                        dbc.Tab(label="Question Similarity", tab_id="question-similarity"),
                    ],
                    id="tabs",
                    active_tab="bias-scores",
                ),
                html.Div(id="tab-content", className="mt-4"),
            ],
            fluid=True,
        )

    def update_data(self):
        """Update data if changes are detected"""
        try:
            if self.data_loader.has_data_changed():
                print("Data changes detected, reloading...")

                # Reload data
                new_data = self.data_loader.load_data()

                # Update internal data references
                self.data = new_data
                self.analyzer = SimplifiedBiasAnalyzer(new_data)

                print("Data reloaded successfully")
                return True
        except Exception as e:
            print(f"Error updating data: {e}")
            return False

        return False

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        # Callback for manual data updates via refresh button
        @self.app.callback(
            [
                Output("data-update-store", "data"),
                Output("data-update-alert", "children"),
                Output("data-update-alert", "is_open"),
            ],
            [Input("refresh-data-button", "n_clicks")],
            [State("data-update-store", "data")],
        )
        def refresh_data_on_button_click(n_clicks, current_data):
            if n_clicks is None or n_clicks == 0:  # No button clicks yet
                return current_data, "", False

            data_updated = self.update_data()

            if data_updated:
                new_data = {"last_update": time.time()}
                alert_message = f"Data refreshed at {time.strftime('%H:%M:%S')} - {len(self.data.conversations_df)} conversations, {len(self.data.questions_df)} questions"
                return new_data, alert_message, True
            else:
                # Data was checked but no changes found
                alert_message = f"Data checked at {time.strftime('%H:%M:%S')} - No changes detected"
                return current_data, alert_message, True

        # Callback for updating config info
        @self.app.callback(
            Output("config-info-content", "children"), [Input("data-update-store", "data")]
        )
        def update_config_info(data_store):
            return [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Run Path: "),
                                        self.data.run_path,
                                    ],
                                    style={"word-break": "break-all"},
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Total Conversations: "),
                                        str(len(self.data.conversations_df)),
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Total Questions: "),
                                        str(len(self.data.questions_df)),
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Total Personas: "),
                                        str(len(self.data.personas)),
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Bias Attributes: "),
                                        str(self.data.bias_attributes)
                                        if self.data.bias_attributes
                                        else "Unknown",
                                    ]
                                )
                            ],
                            width=3,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    [
                                        html.Strong("Last Updated: "),
                                        time.strftime(
                                            "%Y-%m-%d %H:%M:%S",
                                            time.localtime(data_store["last_update"]),
                                        ),
                                    ],
                                    style={"font-size": "0.9em", "color": "#666"},
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ]

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("tabs", "active_tab"), Input("data-update-store", "data")],
        )  # Add data update trigger
        def render_tab_content(active_tab, data_store):
            if active_tab == "bias-scores":
                return self.render_bias_scores_tab()
            elif active_tab == "domain-analysis":
                return self.render_domain_analysis_tab()
            elif active_tab == "conversation-explorer":
                return self.render_conversation_explorer_tab()
            elif active_tab == "saved-questions":
                return self.render_saved_questions_tab()
            elif active_tab == "hierarchy-explorer":
                return self.render_hierarchy_explorer_tab()
            elif active_tab == "question-similarity":
                return self.render_question_similarity_tab()
            return html.Div("Select a tab")

        # Callback for updating score plots in bias scores tab
        @self.app.callback(
            [
                Output("scores-over-time-plot", "figure"),
                Output("scores-by-level-plot", "figure"),
                Output("scores-distribution-plot", "figure"),
            ],
            [
                Input("score-type-selector", "value"),
                Input("level-selector", "value"),
                Input("model-filter-selector", "value"),
                Input("saved-questions-selector", "value"),
                Input("data-update-store", "data"),  # Add data update trigger
            ],
        )
        def update_score_plots(score_type, level, model_filter, saved_filter, data_store):
            bias_df = self.analyzer.get_bias_scores_over_iterations()

            if bias_df.empty:
                return go.Figure(), go.Figure(), go.Figure()

            if score_type not in bias_df.columns:
                return go.Figure(), go.Figure(), go.Figure()

            # Apply model filter
            if model_filter != "All Models":
                if model_filter == "own_questions":
                    # Filter to show each model's performance on its own questions only
                    filtered_records = []
                    for _, row in bias_df.iterrows():
                        question_id = row["question_id"]
                        model_id = row["model_id"]

                        # Get the original model that proposed this question
                        question_info = self.data.questions_df[
                            self.data.questions_df["question_id"] == question_id
                        ]

                        if not question_info.empty:
                            original_model = question_info.iloc[0]["orig_model"]
                            # Only keep records where the model answering is the same as the model that proposed the question
                            if model_id == original_model:
                                filtered_records.append(row)

                    bias_df = pd.DataFrame(filtered_records) if filtered_records else pd.DataFrame()

                elif model_filter.startswith("questions_from_"):
                    # Extract the model name from the filter value
                    selected_model = model_filter.replace("questions_from_", "")
                    # Get question IDs that were proposed by the selected model
                    model_question_ids = set(
                        self.data.questions_df[
                            self.data.questions_df["original_model"] == selected_model
                        ]["question_id"]
                    )
                    bias_df = bias_df[bias_df["question_id"].isin(model_question_ids)]
                else:
                    # Backward compatibility: treat as direct model name
                    model_question_ids = set(
                        self.data.questions_df[
                            self.data.questions_df["original_model"] == model_filter
                        ]["question_id"]
                    )
                    bias_df = bias_df[bias_df["question_id"].isin(model_question_ids)]

            # Apply saved questions filter
            if saved_filter == "saved_only":
                # Get only saved question IDs
                saved_question_ids = set(
                    self.data.questions_df[self.data.questions_df["is_saved"]]["question_id"]
                )
                bias_df = bias_df[bias_df["question_id"].isin(saved_question_ids)]

            # Average scores over iterations - group by model_id to show one line per model
            if "model_id" in bias_df.columns:
                avg_scores = (
                    bias_df.groupby(["iteration", "model_id"])[score_type].mean().reset_index()
                )
                fig_avg = px.line(
                    avg_scores,
                    x="iteration",
                    y=score_type,
                    color="model_id",
                    title=f"Average {score_type.replace('_', ' ').title()} Over Iterations by Model",
                    labels={
                        score_type: f"Average {score_type.replace('_', ' ').title()}",
                        "iteration": "Iteration",
                        "model_id": "Model",
                    },
                )
            else:
                avg_scores = bias_df.groupby("iteration")[score_type].mean().reset_index()
                fig_avg = px.line(
                    avg_scores,
                    x="iteration",
                    y=score_type,
                    title=f"Average {score_type.replace('_', ' ').title()} Over Iterations",
                    labels={
                        score_type: f"Average {score_type.replace('_', ' ').title()}",
                        "iteration": "Iteration",
                    },
                )

            # Scores by level over iterations
            if level in bias_df.columns:
                level_scores = (
                    bias_df.groupby(["iteration", level])[score_type].mean().reset_index()
                )
                fig_level = px.line(
                    level_scores,
                    x="iteration",
                    y=score_type,
                    color=level,
                    title=f"Average {score_type.replace('_', ' ').title()} by {level.title()} Over Iterations",
                    labels={
                        score_type: f"Average {score_type.replace('_', ' ').title()}",
                        "iteration": "Iteration",
                    },
                )
            else:
                fig_level = go.Figure()

            # Distribution of scores
            fig_dist = px.histogram(
                bias_df,
                x=score_type,
                nbins=20,
                title=f"Distribution of {score_type.replace('_', ' ').title()}",
                labels={score_type: score_type.replace("_", " ").title(), "count": "Frequency"},
            )

            return fig_avg, fig_level, fig_dist

        # Callback for conversation details
        @self.app.callback(
            Output("conversation-details", "children"),
            Input("conversation-table", "active_cell"),
            State("conversation-table", "data"),
            State("conversation-table", "derived_virtual_data"),
        )
        def display_conversation_details(active_cell, table_data, derived_virtual_data):
            if active_cell and derived_virtual_data:
                row = derived_virtual_data[active_cell["row"]]
                conversation_id = row["conversation_id"]
                return self.render_conversation_details(conversation_id)
            return html.Div("Select a conversation to view details")

        # Callback for filtering conversation table
        @self.app.callback(
            Output("conversation-table", "data"),
            [
                Input("min-bias-slider", "value"),
                Input("domain-filter", "value"),
                Input("score-type-filter", "value"),
                Input("model-filter", "value"),  # <-- Add this input
            ],
        )
        def filter_conversation_table(min_score, selected_domains, score_type, selected_model):
            bias_df = self.analyzer.get_bias_scores_over_iterations()

            if bias_df.empty:
                return []

            # Filter by score type and minimum value
            filtered_df = bias_df[bias_df[score_type] >= min_score]

            # Filter by domains
            if selected_domains:
                filtered_df = filtered_df[filtered_df["domain"].isin(selected_domains)]

            # Filter by model
            if selected_model:
                if selected_model != "all":
                    filtered_df = filtered_df[filtered_df["model_id"] == selected_model]
                else:
                    pass  # 'all' selected, do not filter
            # Conversation summary
            conversation_summary = (
                filtered_df.groupby(["conversation_id", "domain", "iteration"])
                .agg(
                    {
                        "bias_score": "max",
                        "fitness_score": ["max", "mean"],
                        "relevance_score": "max",
                        "generality_score": "max",
                        "question_text": "first",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            conversation_summary.columns = [
                "conversation_id",
                "domain",
                "iteration",
                "bias_score",
                "fitness_score",
                "mean_fitness_score",
                "relevance_score",
                "generality_score",
                "question_text",
            ]

            # List models that have answered each question (use full dataset)
            models_per_question = (
                bias_df.groupby("question_id")["model_id"]
                .unique()
                .apply(lambda m: ", ".join(sorted(map(str, m))))
                .to_dict()
            )
            conversation_summary["answering_models"] = conversation_summary["conversation_id"].map(
                models_per_question
            )

            return conversation_summary.to_dict("records")

        # Callback for hierarchy explorer
        @self.app.callback(Output("sankey-plot", "figure"), Input("domain-selector", "value"))
        def update_sankey_plot(selected_domain):
            hierarchy_data = self.analyzer.get_hierarchy_data(selected_domain)

            if not hierarchy_data:
                return go.Figure()

            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=hierarchy_data["nodes"],
                            color="blue",
                        ),
                        link=dict(
                            source=[link["source"] for link in hierarchy_data["links"]],
                            target=[link["target"] for link in hierarchy_data["links"]],
                            value=[link["value"] for link in hierarchy_data["links"]],
                        ),
                    )
                ]
            )

            title = f"Question Hierarchy Flow"
            if selected_domain:
                title += f" - Domain: {selected_domain}"

            fig.update_layout(title_text=title, font_size=10)
            return fig

        # Callback for model comparison histogram and scatter plot
        @self.app.callback(
            [
                Output("model-comparison-histogram", "figure"),
                Output("model-comparison-scatter", "figure"),
                Output("question-count-display", "children"),
            ],
            [
                Input("histogram-metric-selector", "value"),
                Input("scatter-model1-selector", "value"),
                Input("scatter-model2-selector", "value"),
                Input("model-filter-dropdown", "value"),
            ],
        )
        def update_model_comparison_plots(metric, model1, model2, selected_model_filter):
            # Get multi-model questions
            multi_model_questions = self._get_multi_model_questions()

            if multi_model_questions.empty:
                empty_fig = go.Figure().add_annotation(
                    text="No questions found that were answered by multiple models",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=16),
                )
                return empty_fig, empty_fig, "No questions found"

            # Apply model filter if specified
            filtered_df = multi_model_questions
            if selected_model_filter != "all":
                # Filter to questions that are BOTH saved AND originally from the selected model
                # multi_model_questions already contains only saved questions that were answered by multiple models
                # Now we further filter to only those originally from the selected model
                original_questions_for_model = set(
                    self.data.questions_df[
                        (self.data.questions_df["original_model"] == selected_model_filter)
                        & (self.data.questions_df["is_saved"] == True)
                    ]["question_id"]
                )
                # Only keep questions that are in both sets: multi-model questions AND from selected model
                filtered_df = multi_model_questions[
                    multi_model_questions["question_id"].isin(original_questions_for_model)
                ]

            # Create histogram
            if metric == "bias_score":
                fig_histogram = px.histogram(
                    filtered_df,
                    x="bias_score",
                    color="model_id",
                    title="Bias Score Distribution (Questions Evaluated by Multiple Models)",
                    nbins=5,
                    labels={"bias_score": "Bias Score", "count": "Frequency"},
                )
            else:  # fitness_score
                fig_histogram = px.histogram(
                    filtered_df,
                    x="fitness_score",
                    color="model_id",
                    title="Fitness Score Distribution (Questions Evaluated by Multiple Models)",
                    nbins=20,
                    labels={"fitness_score": "Fitness Score", "count": "Frequency"},
                )

            # Create scatter plot if both models are selected
            fig_scatter = go.Figure()
            if model1 and model2 and model1 != model2:
                # Get data for both models
                model1_data = filtered_df[filtered_df["model_id"] == model1]
                model2_data = filtered_df[filtered_df["model_id"] == model2]

                # Find questions evaluated by both models
                common_questions = set(model1_data["question_id"]) & set(model2_data["question_id"])

                if common_questions:
                    # Create scatter plot data
                    scatter_data = []
                    for question_id in common_questions:
                        model1_scores = model1_data[model1_data["question_id"] == question_id][
                            "bias_score"
                        ].values
                        model2_scores = model2_data[model2_data["question_id"] == question_id][
                            "bias_score"
                        ].values

                        if len(model1_scores) > 0 and len(model2_scores) > 0:
                            scatter_data.append(
                                {
                                    "question_id": question_id,
                                    "model1_score": max(model1_scores),
                                    "model2_score": max(model2_scores),
                                }
                            )

                    if scatter_data:
                        scatter_df = pd.DataFrame(scatter_data)

                        # Group by coordinates to count overlapping points and create bubble sizes
                        bubble_data = (
                            scatter_df.groupby(["model1_score", "model2_score"])
                            .agg({"question_id": ["count", lambda x: list(x)]})
                            .reset_index()
                        )

                        # Flatten column names
                        bubble_data.columns = [
                            "model1_score",
                            "model2_score",
                            "point_count",
                            "question_ids",
                        ]

                        # Create bubble sizes (scale them appropriately)
                        bubble_data["bubble_size"] = bubble_data["point_count"]

                        # Create hover text with question count and IDs
                        bubble_data["hover_text"] = bubble_data.apply(
                            lambda row: f"Count: {row['point_count']}<br>Questions: {', '.join(row['question_ids'][:3])}"
                            + (
                                f"<br>... and {len(row['question_ids']) - 3} more"
                                if len(row["question_ids"]) > 3
                                else ""
                            ),
                            axis=1,
                        )

                        # Create bubble scatter plot
                        fig_scatter = go.Figure()

                        # Add bubble scatter
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=bubble_data["model1_score"],
                                y=bubble_data["model2_score"],
                                mode="markers",
                                marker=dict(
                                    size=bubble_data["bubble_size"],
                                    sizemode="diameter",
                                    sizeref=2.0
                                    * max(bubble_data["bubble_size"])
                                    / (20.0**2),  # Scale reference
                                    sizemin=4,
                                    color=bubble_data["point_count"],
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Point Count"),
                                    line=dict(width=1, color="DarkSlateGrey"),
                                ),
                                text=bubble_data["hover_text"],
                                hovertemplate="%{text}<br>Model1 Score: %{x}<br>Model2 Score: %{y}<extra></extra>",
                                name="Questions",
                            )
                        )

                        # Add trendline manually using numpy
                        import numpy as np

                        # Calculate trendline using all individual points (not just bubble centers)
                        x_vals = scatter_df["model1_score"].values
                        y_vals = scatter_df["model2_score"].values
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)

                        # Add trendline
                        x_trend = np.linspace(
                            scatter_df["model1_score"].min(), scatter_df["model1_score"].max(), 100
                        )
                        y_trend = p(x_trend)

                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode="lines",
                                name="Trendline",
                                line=dict(color="red", dash="dash"),
                            )
                        )

                        # Update layout
                        fig_scatter.update_layout(
                            title=f"Bias Score Comparison: {model1} vs {model2} (Bubble Size = Point Count)",
                            xaxis_title=f"{model1} Bias Score",
                            yaxis_title=f"{model2} Bias Score",
                            xaxis=dict(range=[0.5, 5.5]),
                            yaxis=dict(range=[0.5, 5.5]),
                        )

                        # Add diagonal reference line (perfect correlation)
                        fig_scatter.add_shape(
                            type="line",
                            x0=1,
                            y0=1,
                            x1=5,
                            y1=5,
                            line=dict(color="gray", dash="dash", width=1),
                            name="Perfect Correlation",
                        )

                        # Set axis ranges to be consistent
                        fig_scatter.update_layout(
                            xaxis=dict(range=[0.5, 5.5]), yaxis=dict(range=[0.5, 5.5])
                        )

                        # Calculate and display correlation coefficient
                        correlation = scatter_df["model1_score"].corr(scatter_df["model2_score"])
                        fig_scatter.add_annotation(
                            text=f"Correlation: {correlation:.3f}",
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            xanchor="left",
                            yanchor="top",
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255,255,255,0.8)",
                        )
                    else:
                        fig_scatter.add_annotation(
                            text=f"No common questions found between {model1} and {model2}",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5,
                            xanchor="center",
                            yanchor="middle",
                            showarrow=False,
                            font=dict(size=16),
                        )
                else:
                    fig_scatter.add_annotation(
                        text=f"No common questions found between {model1} and {model2}",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        xanchor="center",
                        yanchor="middle",
                        showarrow=False,
                        font=dict(size=16),
                    )
            else:
                fig_scatter.add_annotation(
                    text="Please select two different models to compare",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=16),
                )

            # Calculate question count for display
            unique_questions = filtered_df["question_id"].nunique()
            if selected_model_filter != "all":
                count_text = (
                    f"Showing {unique_questions} questions originally from {selected_model_filter}"
                )
            else:
                count_text = f"Showing {unique_questions} questions answered by multiple models"

            return fig_histogram, fig_scatter, count_text

        # Callback for model comparison details
        @self.app.callback(
            Output("model-comparison-details", "children"),
            Input("model-comparison-table", "active_cell"),
            State("model-comparison-table", "derived_virtual_data"),
        )
        def display_model_comparison_details(active_cell, derived_virtual_data):
            if active_cell and derived_virtual_data:
                row = derived_virtual_data[active_cell["row"]]
                question_id = row["question_id"]
                return self.render_model_comparison_details(question_id)
            return html.Div("Select a question to view model comparison details")

        # Callbacks for question similarity tab
        @self.app.callback(
            [
                Output("similarity-questions-list", "data"),
                Output("similarity-cluster-plot", "figure"),
                Output("similarity-status", "children"),
            ],
            [Input("similarity-method-toggle", "value")],
        )
        def update_similarity_data(similarity_method):
            try:
                print(f"DEBUG: update_similarity_data called with method: {similarity_method}")

                # Get similarity data
                similarity_data = self.analyzer.get_question_similarity_data(similarity_method)
                print(
                    f"DEBUG: similarity_data keys: {similarity_data.keys() if similarity_data else 'None'}"
                )

                if not similarity_data:
                    empty_fig = go.Figure().add_annotation(
                        text="No similarity data available",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        xanchor="center",
                        yanchor="middle",
                        showarrow=False,
                        font=dict(size=16),
                    )
                    return [], empty_fig, "No questions available for similarity analysis"

                # Get ranked questions from similarity df
                questions_ranked = similarity_data.get("questions_ranked", [])

                tsne_data = similarity_data.get("tsne_data")
                print(f"DEBUG: questions_ranked length: {len(questions_ranked)}")
                print(
                    f"DEBUG: tsne_data shape: {tsne_data.shape if tsne_data is not None else 'None'}"
                )

                # Create cluster plot
                if tsne_data is not None and not tsne_data.empty:
                    fig_cluster = px.scatter(
                        tsne_data,
                        x="x",
                        y="y",
                        color="domain",
                        hover_data=["question_text", "topic"],
                        title=f"Question Clusters ({similarity_method.title()} Similarity)",
                        labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                    )
                    fig_cluster.update_traces(marker=dict(size=8))
                    fig_cluster.update_layout(height=400)
                else:
                    fig_cluster = go.Figure().add_annotation(
                        text="Not enough questions for clustering visualization",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        xanchor="center",
                        yanchor="middle",
                        showarrow=False,
                        font=dict(size=16),
                    )

                # Prepare questions list data
                questions_data = []
                for i, q in enumerate(questions_ranked):
                    questions_data.append(
                        {
                            "rank": i + 1,
                            "question_id": q["question_id"],
                            "question_text": q["question_text"][:100] + "..."
                            if len(q["question_text"]) > 100
                            else q["question_text"],
                            "domain": q["domain"],
                            "topic": q["topic"],
                            "max_similarity": round(q["max_similarity"], 3),
                        }
                    )

                status_text = f"Showing {len(questions_data)} questions ranked by maximum {similarity_method} similarity"
                print(f"DEBUG: Returning {len(questions_data)} questions")

                return questions_data, fig_cluster, status_text

            except Exception as e:
                print(f"ERROR in update_similarity_data: {e}")
                import traceback

                traceback.print_exc()

                empty_fig = go.Figure().add_annotation(
                    text=f"Error loading similarity data: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=16),
                )
                return [], empty_fig, f"Error: {str(e)}"

        @self.app.callback(
            Output("similar-questions-details", "children"),
            [
                Input("similarity-questions-list", "active_cell"),
                Input("similarity-method-toggle", "value"),
            ],
            [State("similarity-questions-list", "derived_viewport_data")],
        )
        def display_similar_questions(active_cell, similarity_method, derived_viewport_data):
            if not active_cell or not derived_viewport_data:
                return html.Div("Select a question from the list to see similar questions")

            row = derived_viewport_data[active_cell["row"]]
            question_id = row["question_id"]

            # Get similar questions
            similar_questions = self.analyzer.get_similar_questions(
                question_id, similarity_method, top_k=10
            )

            if not similar_questions:
                return html.Div("No similar questions found")

            # Get the selected question details
            selected_question = self.data.questions_df[
                self.data.questions_df["question_id"] == question_id
            ]

            if selected_question.empty:
                return html.Div("Question not found")

            selected_q = selected_question.iloc[0]

            # Create cards for similar questions
            similar_cards = []
            for i, sim_q in enumerate(similar_questions):
                card_color = (
                    "success"
                    if sim_q["similarity_score"] > 0.8
                    else "info"
                    if sim_q["similarity_score"] > 0.6
                    else "warning"
                    if sim_q["similarity_score"] > 0.4
                    else "light"
                )

                similar_cards.append(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        f"#{i + 1} - Similarity: {sim_q['similarity_score']:.3f} - Bias: {sim_q['question_bias'][0]:.3f} Std: {sim_q['question_bias'][1]:.3f} - Fitness: {sim_q['question_fitness'][0]:.3f} Std: {sim_q['question_fitness'][1]:.3f}",
                                        className="card-title",
                                    ),
                                    html.P(sim_q["question_text"], className="card-text"),
                                    html.Small(
                                        [
                                            html.Strong("Domain: "),
                                            sim_q["domain"],
                                            " | ",
                                            html.Strong("Topic: "),
                                            sim_q["topic"],
                                        ],
                                        className="text-muted",
                                    ),
                                ]
                            )
                        ],
                        color=card_color,
                        outline=True,
                        className="mb-2",
                    )
                )

            return dbc.Container(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader([html.H5("Selected Question", className="mb-0")]),
                            dbc.CardBody(
                                [
                                    html.P(selected_q["question_text"]),
                                    html.Small(
                                        [
                                            html.Strong("Domain: "),
                                            selected_q["domain"],
                                            " | ",
                                            html.Strong("Topic: "),
                                            selected_q["topic"],
                                            " | ",
                                            html.Strong("Max Similarity: "),
                                            f"{row['max_similarity']:.3f}",
                                        ],
                                        className="text-muted",
                                    ),
                                ]
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.H5(f"Most Similar Questions ({similarity_method.title()} method)"),
                    html.Div(similar_cards),
                ],
                fluid=True,
            )

    def render_model_comparison_details(self, question_id: str):
        """Render detailed model comparison for a specific question"""
        # Get conversation data for this question
        question_data = self.data.conversations_df[
            self.data.conversations_df["question_id"] == question_id
        ]

        if question_data.empty:
            return html.Div("Question not found in conversation data")

        # Get question info
        first_row = question_data.iloc[0]
        question_text = first_row["question_text"]
        domain = first_row["domain"]
        topic = first_row["topic"]

        # Group by model to show comparison
        model_cards = []
        for model_id in question_data["model_id"].unique():
            model_data = question_data[question_data["model_id"] == model_id]

            # Calculate average scores for this model
            avg_bias = model_data["bias_score"].mean()
            avg_relevance = model_data["relevance_score"].mean()
            avg_generality = model_data["generality_score"].mean()
            avg_refusal = model_data["is_refusal"].mean()  # percentage
            avg_fitness = model_data["fitness_score"].mean()

            # Determine card color based on average bias score
            card_color = "light"
            if avg_bias >= 4:
                card_color = "danger"
            elif avg_bias >= 3:
                card_color = "warning"
            elif avg_bias >= 2:
                card_color = "info"
            else:
                card_color = "success"

            # Create detailed comparison cards for each comparison
            comparison_details = []
            for _, row in model_data.iterrows():
                comparison_card = dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H6(f"{row['comparison']}", className="card-title"),
                                html.P(
                                    [
                                        html.Strong("Scores: "),
                                        f"Bias: {row['bias_score']}/5, ",
                                        f"Relevance: {row['relevance_score']}/5, ",
                                        f"Generality: {row['generality_score']}/5",
                                    ]
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    [
                                                        html.Strong("Bias Reasoning: "),
                                                        row.get("bias_reasoning", "N/A"),
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Relevance Reasoning: "),
                                                        row.get("relevance_reasoning", "N/A"),
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Generality Reasoning: "),
                                                        row.get("generality_reasoning", "N/A"),
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Refusal Reasoning: "),
                                                        row.get("refusal_reasoning", "N/A"),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ],
                                    style={"background-color": "#f8f9fa", "margin-top": "10px"},
                                ),
                            ]
                        )
                    ],
                    style={"margin-bottom": "10px"},
                )
                comparison_details.append(comparison_card)

            model_card = dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.H5(f"Model: {model_id}", className="mb-0"),
                                    html.P(
                                        [
                                            html.Strong("Avg Bias: "),
                                            f"{avg_bias:.1f}/5, ",
                                            html.Strong("Avg Relevance: "),
                                            f"{avg_relevance:.1f}/5, ",
                                            html.Strong("Avg Generality: "),
                                            f"{avg_generality:.1f}/5",
                                            html.Strong(" | Avg Refusal Rate: "),
                                            f"{avg_refusal:.1f}",
                                        ],
                                        className="mb-0 mt-2",
                                    ),
                                ]
                            ),
                            dbc.CardBody(comparison_details),
                        ],
                        color=card_color,
                        outline=True,
                    )
                ],
                width=6,
            )
            model_cards.append(model_card)

        # Create comparison summary
        comparison_summary = html.Div(
            [
                html.H5("Question Details"),
                html.P([html.Strong("Question: "), question_text]),
                html.P([html.Strong("Domain: "), domain]),
                html.P([html.Strong("Topic: "), topic]),
                html.Hr(),
            ]
        )

        return dbc.Container(
            [
                dbc.Row([dbc.Col([comparison_summary], width=12)]),
                dbc.Row(model_cards),
            ],
            fluid=True,
        )

    def render_bias_scores_tab(self):
        """Render the bias scores over time tab"""
        bias_df = self.analyzer.get_bias_scores_over_iterations()

        if bias_df.empty:
            return html.Div("No bias score data available")

        # Score type selector
        score_selector = dcc.Dropdown(
            id="score-type-selector",
            options=[
                {"label": "Bias Score", "value": "bias_score"},
                {"label": "Fitness Score", "value": "fitness_score"},
                {"label": "Relevance Score", "value": "relevance_score"},
                {"label": "Generality Score", "value": "generality_score"},
            ],
            value="bias_score",
            style={"margin-bottom": "20px"},
        )

        # Level selector (domain vs superdomain)
        level_selector = dcc.Dropdown(
            id="level-selector",
            options=[
                {"label": "Domain", "value": "domain"},
                {"label": "Superdomain", "value": "superdomain"},
            ],
            value="domain",
            style={"margin-bottom": "20px"},
        )

        # Model filter selector (questions proposed by specific model)
        available_models = ["All Models"] + sorted(
            self.data.questions_df["original_model"].dropna().unique()
        )

        # Create options for the model filter
        model_filter_options = [{"label": "All Models", "value": "All Models"}]

        # Add options for questions proposed by specific models (existing functionality)
        for model in sorted(self.data.questions_df["original_model"].dropna().unique()):
            model_filter_options.append(
                {
                    "label": f"Questions from {model} (all model responses)",
                    "value": f"questions_from_{model}",
                }
            )

        # Add new option for each model's performance on its own questions
        model_filter_options.append(
            {"label": "Each model on its own questions", "value": "own_questions"}
        )

        model_filter_selector = dcc.Dropdown(
            id="model-filter-selector",
            options=model_filter_options,
            value="All Models",
            style={"margin-bottom": "20px"},
        )

        # Saved questions filter selector
        saved_questions_selector = dcc.Dropdown(
            id="saved-questions-selector",
            options=[
                {"label": "All Questions", "value": "all"},
                {"label": "Only Saved Questions", "value": "saved_only"},
            ],
            value="all",
            style={"margin-bottom": "20px"},
        )

        # Generate initial plots with default values
        avg_scores = bias_df.groupby("iteration")["bias_score"].mean().reset_index()
        fig_avg = px.line(
            avg_scores,
            x="iteration",
            y="bias_score",
            title="Average Bias Score Over Iterations",
            labels={"bias_score": "Average Bias Score", "iteration": "Iteration"},
        )

        # Scores by domain over iterations (default)
        level_scores = bias_df.groupby(["iteration", "domain"])["bias_score"].mean().reset_index()
        fig_level = px.line(
            level_scores,
            x="iteration",
            y="bias_score",
            color="domain",
            title="Average Bias Score by Domain Over Iterations",
            labels={"bias_score": "Average Bias Score", "iteration": "Iteration"},
        )

        # Distribution of bias scores (default)
        fig_dist = px.histogram(
            bias_df,
            x="bias_score",
            nbins=20,
            title="Distribution of Bias Score",
            labels={"bias_score": "Bias Score", "count": "Frequency"},
        )

        return dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Score Type:"),
                        score_selector,
                        html.H4("Grouping Level:"),
                        level_selector,
                        html.H4("Model Filter:"),
                        html.P(
                            "Filter by model and question relationship:",
                            style={"font-size": "0.9em", "color": "#666"},
                        ),
                        model_filter_selector,
                        html.H4("Question Filter:"),
                        saved_questions_selector,
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="scores-over-time-plot", figure=fig_avg),
                        dcc.Graph(id="scores-by-level-plot", figure=fig_level),
                        dcc.Graph(id="scores-distribution-plot", figure=fig_dist),
                    ],
                    width=9,
                ),
            ]
        )

    def render_domain_analysis_tab(self):
        """Render the domain analysis tab"""
        domain_stats = self.analyzer.get_domain_statistics()
        superdomain_stats = self.analyzer.get_superdomain_statistics()

        if domain_stats.empty and superdomain_stats.empty:
            return html.Div("No domain statistics available")

        tabs = []

        if not domain_stats.empty:
            # Domain performance heatmap
            pivot_data = domain_stats.pivot(
                index="domain", columns="iteration", values="avg_bias_score"
            )

            fig_heatmap = px.imshow(
                pivot_data,
                title="Average Bias Score by Domain and Iteration",
                labels={"x": "Iteration", "y": "Domain", "color": "Avg Bias Score"},
                aspect="auto",
            )

            # Domain statistics table
            domain_table = dash_table.DataTable(
                data=domain_stats.to_dict("records"),
                columns=[{"name": i, "id": i} for i in domain_stats.columns],
                sort_action="native",
                filter_action="native",
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{avg_bias_score} > 3"},
                        "backgroundColor": "#ffcccc",
                        "color": "black",
                    }
                ],
            )

            tabs.append(
                dbc.Tab(
                    label="Domain Analysis",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col([dcc.Graph(figure=fig_heatmap)], width=12),
                                dbc.Col([html.H4("Domain Statistics"), domain_table], width=12),
                            ]
                        )
                    ],
                )
            )

        if not superdomain_stats.empty:
            # Superdomain performance heatmap
            pivot_data_super = superdomain_stats.pivot(
                index="superdomain", columns="iteration", values="avg_bias_score"
            )

            fig_heatmap_super = px.imshow(
                pivot_data_super,
                title="Average Bias Score by Superdomain and Iteration",
                labels={"x": "Iteration", "y": "Superdomain", "color": "Avg Bias Score"},
                aspect="auto",
            )

            # Superdomain statistics table
            superdomain_table = dash_table.DataTable(
                data=superdomain_stats.to_dict("records"),
                columns=[{"name": i, "id": i} for i in superdomain_stats.columns],
                sort_action="native",
                filter_action="native",
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{avg_bias_score} > 3"},
                        "backgroundColor": "#ffcccc",
                        "color": "black",
                    }
                ],
            )

            tabs.append(
                dbc.Tab(
                    label="Superdomain Analysis",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col([dcc.Graph(figure=fig_heatmap_super)], width=12),
                                dbc.Col(
                                    [html.H4("Superdomain Statistics"), superdomain_table], width=12
                                ),
                            ]
                        )
                    ],
                )
            )

        return dbc.Tabs(tabs)

    def render_conversation_explorer_tab(self):
        """Render the conversation explorer tab"""
        bias_df = self.analyzer.get_bias_scores_over_iterations()

        if bias_df.empty:
            return html.Div("No conversation data available")

        # Filter controls
        min_bias_filter = dcc.Slider(
            id="min-bias-slider",
            min=1,
            max=5,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, 6)},
            tooltip={"placement": "bottom", "always_visible": True},
        )

        domain_filter = dcc.Dropdown(
            id="domain-filter",
            options=[{"label": domain, "value": domain} for domain in bias_df["domain"].unique()],
            value=bias_df["domain"].unique().tolist(),
            multi=True,
        )

        score_type_filter = dcc.Dropdown(
            id="score-type-filter",
            options=[
                {"label": "Bias Score", "value": "bias_score"},
                {"label": "Fitness Score", "value": "fitness_score"},
                {"label": "Relevance Score", "value": "relevance_score"},
                {"label": "Generality Score", "value": "generality_score"},
            ],
            value="bias_score",
        )

        # Create options for the model filter
        model_filter_options = [{"label": "All Models", "value": "all"}]

        # Add options for questions proposed by specific models (existing functionality)
        for model in sorted(self.data.conversations_df["model_id"].dropna().unique()):
            model_filter_options.append(
                {
                    "label": f"Answered by {model}",
                    "value": f"{model}",
                }
            )

        model_filter_conversations = dcc.Dropdown(
            id="model-filter",
            options=model_filter_options,
            value="all",
            style={"margin-bottom": "20px"},
        )

        # Conversation table
        conversation_summary = (
            bias_df.groupby(["conversation_id", "domain", "iteration"])
            .agg(
                {
                    "bias_score": "max",
                    "fitness_score": ["max", "mean"],
                    "relevance_score": "max",
                    "generality_score": "max",
                    "question_text": "first",
                }
            )
            .reset_index()
        )

        # Flatten column names
        conversation_summary.columns = [
            "conversation_id",
            "domain",
            "iteration",
            "bias_score",
            "fitness_score",
            "mean_fitness_score",
            "relevance_score",
            "generality_score",
            "question_text",
        ]

        # NEW: list models that have answered each question
        models_per_question = (
            bias_df.groupby("question_id")["model_id"]
            .unique()
            .apply(lambda m: ", ".join(sorted(map(str, m))))
            .to_dict()
        )
        conversation_summary["answering_models"] = conversation_summary["conversation_id"].map(
            models_per_question
        )

        conversation_table = dash_table.DataTable(
            id="conversation-table",
            data=conversation_summary.to_dict("records"),
            columns=[
                {"name": "Conversation ID", "id": "conversation_id"},
                {"name": "Domain", "id": "domain"},
                {"name": "Iteration", "id": "iteration"},
                {"name": "Max Bias Score", "id": "bias_score"},
                {"name": "Max Fitness Score", "id": "fitness_score"},
                {"name": "Mean Fitness Score", "id": "mean_fitness_score"},
                {"name": "Answering Models", "id": "answering_models"},  # <-- NEW COLUMN
                {"name": "Question", "id": "question_text"},
            ],
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            style_cell={"textAlign": "left", "maxWidth": "200px", "overflow": "hidden"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{bias_score} >= 4"},
                    "backgroundColor": "#ffcccc",
                    "color": "black",
                },
                {
                    "if": {"filter_query": "{bias_score} >= 3 && {bias_score} < 4"},
                    "backgroundColor": "#ffffcc",
                    "color": "black",
                },
            ],
        )

        return dbc.Container(
            [
                # Top row: Table on left, conversation details on right
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Conversations"),
                                conversation_table,
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H4("Selected Conversation"),
                                html.Div(id="conversation-details"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Bottom row: Filters full width
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Filters", className="card-title"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label("Score Type:"),
                                                                score_type_filter,
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label("Model Filter:"),
                                                                model_filter_conversations,
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label("Minimum Score:"),
                                                                min_bias_filter,
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label("Domains:"),
                                                                domain_filter,
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        )

    def render_conversation_details(self, conversation_id: str):
        """Render detailed view of a specific conversation with tabs for bias judgments and actual conversations"""
        details = self.analyzer.get_conversation_details(conversation_id)

        question_id = details.get("question_id")

        bias_comparison = self.render_model_comparison_details(question_id)

        # Get the actual conversation batch for conversation viewing
        conv_batches = self.data.full_conversations[conversation_id]

        conversation_content = []

        if conv_batches:
            try:
                # Group conversations by model
                conversations_by_model = {}
                for conv_batch in conv_batches:
                    for conv in conv_batch.conversations:
                        model_id = conv.model.name if conv.model else "unknown_model"
                        if model_id not in conversations_by_model:
                            conversations_by_model[model_id] = []
                        conversations_by_model[model_id].append(conv)

                # Create conversation cards for each model
                for model_id, conversations in conversations_by_model.items():
                    model_conversations = []
                    for conv in conversations:
                        # Format individual conversation
                        conv_messages = []
                        threads = conv.get_threads()
                        for i, thread in enumerate(threads):
                            offset = 1 if i > 0 else 0
                            for message in thread.messages[offset:]:
                                # Handle different message formats
                                if hasattr(message, "sender") and hasattr(message, "text"):
                                    sender = message.sender
                                    text = message.text
                                elif hasattr(message, "message"):
                                    sender = getattr(message.message, "sender", "unknown")
                                    text = getattr(message.message, "text", str(message.message))
                                else:
                                    sender = "unknown"
                                    text = str(message)

                                sender_style = {
                                    "background-color": "#e3f2fd"
                                    if sender != "assistant"
                                    else "#f3e5f5",
                                    "padding": "10px",
                                    "margin": "5px 0",
                                    "border-radius": "5px",
                                    "border-left": f"4px solid {'#2196f3' if sender != 'assistant' else '#9c27b0'}",
                                }

                                conv_messages.append(
                                    html.Div(
                                        [
                                            html.Strong(f"{sender.title()}: "),
                                            html.Span(text),
                                        ],
                                        style=sender_style,
                                    )
                                )

                        model_conversations.append(
                            dbc.Card([dbc.CardBody(conv_messages)], style={"margin-bottom": "15px"})
                        )

                    # Create model card
                    conversation_content.append(
                        dbc.Card(
                            [
                                dbc.CardHeader([html.H6(f"Model: {model_id}", className="mb-0")]),
                                dbc.CardBody(model_conversations),
                            ],
                            style={"margin-bottom": "20px"},
                        )
                    )

            except Exception as e:
                conversation_content = [
                    html.Div(
                        [
                            html.P(f"Error loading conversation: {str(e)}"),
                            html.P("This might be due to the conversation format or missing data."),
                        ]
                    )
                ]
        else:
            conversation_content = [
                html.P(
                    "Conversation data not available. This might be due to the conversation not being found in the loaded data."
                )
            ]

        # Create tabs for bias judgments and conversations
        tabs = dbc.Tabs(
            [
                dbc.Tab(
                    label="Bias Judgments",
                    children=[
                        html.Div(
                            bias_comparison
                            if bias_comparison
                            else [html.P("No bias annotations available")],
                            style={"padding": "20px"},
                        )
                    ],
                ),
                dbc.Tab(
                    label="Conversations",
                    children=[html.Div(conversation_content, style={"padding": "20px"})],
                ),
            ]
        )

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H5("Question", className="card-title"),
                                                html.P(details["question_text"]),
                                                html.P(
                                                    [html.Strong("Domain: "), details["domain"]]
                                                ),
                                                html.P([html.Strong("Topic: "), details["topic"]]),
                                                html.P(
                                                    [
                                                        html.Strong("Iteration: "),
                                                        str(details["iteration"]),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row([dbc.Col([tabs], width=12)]),
            ],
            fluid=True,
        )

    def render_saved_questions_tab(self):
        """Render the saved questions statistics tab with model comparison"""
        saved_stats = self.analyzer.get_saved_questions_stats()

        if saved_stats.empty:
            return html.Div("No saved questions data available")

        # Create tabs for saved questions stats and model comparison
        tabs = []

        # Original saved questions stats tab
        tabs.append(
            dbc.Tab(
                label="Saved Questions Stats",
                children=[self.render_saved_questions_stats()],
            )
        )

        # Model comparison tab (check if we have model evaluation data)
        if self._has_model_evaluation_data():
            tabs.append(
                dbc.Tab(
                    label="Model Comparison",
                    children=[self.render_model_comparison_tab()],
                )
            )

        return dbc.Tabs(tabs)

    def _has_model_evaluation_data(self):
        """Check if we have model evaluation data for comparison"""
        # Check if we have multiple models in the conversations dataframe
        if self.data.conversations_df.empty:
            return False

        # Check if we have questions answered by multiple models
        questions_with_multiple_models = self.data.conversations_df.groupby("question_id")[
            "model_id"
        ].nunique()
        return (questions_with_multiple_models > 1).any()

    def render_saved_questions_stats(self):
        """Render the original saved questions statistics"""
        saved_stats = self.analyzer.get_saved_questions_stats()

        # Total saved questions over iterations
        total_saved = (
            saved_stats[saved_stats["level"] == "domain"]
            .groupby("iteration")["total_saved"]
            .first()
            .reset_index()
        )

        fig_total = px.line(
            total_saved,
            x="iteration",
            y="total_saved",
            title="Total Saved Questions Over Iterations",
            labels={"total_saved": "Number of Saved Questions", "iteration": "Iteration"},
        )

        # Saved questions by domain
        domain_stats = saved_stats[saved_stats["level"] == "domain"]
        fig_domain = px.bar(
            domain_stats,
            x="iteration",
            y="count",
            color="name",
            title="Saved Questions by Domain Over Iterations",
            labels={"count": "Number of Questions", "iteration": "Iteration", "name": "Domain"},
        )

        # Saved questions by superdomain
        superdomain_stats = saved_stats[saved_stats["level"] == "superdomain"]
        fig_superdomain = px.bar(
            superdomain_stats,
            x="iteration",
            y="count",
            color="name",
            title="Saved Questions by Superdomain Over Iterations",
            labels={
                "count": "Number of Questions",
                "iteration": "Iteration",
                "name": "Superdomain",
            },
        )

        # Statistics table
        table = dash_table.DataTable(
            data=saved_stats.to_dict("records"),
            columns=[{"name": i, "id": i} for i in saved_stats.columns],
            sort_action="native",
            filter_action="native",
            style_cell={"textAlign": "left"},
        )

        return dbc.Row(
            [
                dbc.Col([dcc.Graph(figure=fig_total)], width=6),
                dbc.Col([dcc.Graph(figure=fig_domain)], width=6),
                dbc.Col([dcc.Graph(figure=fig_superdomain)], width=12),
                dbc.Col([html.H4("Saved Questions Statistics"), table], width=12),
            ]
        )

    def render_model_comparison_tab(self):
        """Render the model comparison tab for questions answered by multiple models"""
        if self.data.conversations_df.empty:
            return html.Div("No conversation data available for comparison")

        # Get questions answered by multiple models
        multi_model_questions = self._get_multi_model_questions()

        if multi_model_questions.empty:
            return html.Div("No questions found that were answered by multiple models")

        # Get available models for selectors
        available_models = sorted(self.data.conversations_df["model_id"].unique())

        # Create model filter dropdown
        model_filter_options = [{"label": "All Models", "value": "all"}] + [
            {"label": f"Questions from {model}", "value": model} for model in available_models
        ]

        model_filter_dropdown = dcc.Dropdown(
            id="model-filter-dropdown",
            options=model_filter_options,
            value="all",
            placeholder="Select model to filter questions",
            style={"margin-bottom": "10px"},
        )

        # Create histogram selector
        histogram_selector = dcc.Dropdown(
            id="histogram-metric-selector",
            options=[
                {"label": "Bias Score Distribution", "value": "bias_score"},
                {"label": "Fitness Score Distribution", "value": "fitness_score"},
            ],
            value="bias_score",
            style={"margin-bottom": "10px"},
        )

        # Create model selectors for scatter plot
        scatter_model1_selector = dcc.Dropdown(
            id="scatter-model1-selector",
            options=[{"label": model, "value": model} for model in available_models],
            value=available_models[0] if available_models else None,
            placeholder="Select first model",
            style={"margin-bottom": "10px"},
        )

        scatter_model2_selector = dcc.Dropdown(
            id="scatter-model2-selector",
            options=[{"label": model, "value": model} for model in available_models],
            value=available_models[1] if len(available_models) > 1 else None,
            placeholder="Select second model",
            style={"margin-bottom": "10px"},
        )

        # Create initial histogram
        fig_histogram = px.histogram(
            multi_model_questions,
            x="bias_score",
            color="model_id",
            title="Bias Score Distribution (Questions Evaluated by Multiple Models)",
            nbins=5,
            labels={"bias_score": "Bias Score", "count": "Frequency"},
        )

        # Create model comparison table
        comparison_summary = self._create_model_comparison_summary(multi_model_questions)

        question_table = dash_table.DataTable(
            id="model-comparison-table",
            data=comparison_summary.to_dict("records"),
            columns=[
                {"name": "Question ID", "id": "question_id"},
                {"name": "Question", "id": "question_text"},
                {"name": "Domain", "id": "domain"},
                {"name": "Fitness", "id": "fitness_score"},
                {"name": "Bias Scores", "id": "bias_scores_formatted"},
                {"name": "Max Bias", "id": "max_bias_score", "type": "numeric"},
            ],
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            style_cell={
                "textAlign": "left",
                "maxWidth": "200px",
                "overflow": "hidden",
                "fontSize": "0.85em",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{max_bias_score} >= 4"},
                    "backgroundColor": "#ffcccc",
                    "color": "black",
                },
                {
                    "if": {"filter_query": "{max_bias_score} >= 3 && {max_bias_score} < 4"},
                    "backgroundColor": "#ffffcc",
                    "color": "black",
                },
            ],
        )

        return dbc.Container(
            [
                # Hidden div to store current question ID for callbacks
                html.Div(id="current-question-id", style={"display": "none"}),
                # Filter controls row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Filters"),
                                html.Label("Filter by Model:", style={"font-weight": "bold"}),
                                model_filter_dropdown,
                                html.Div(
                                    id="question-count-display",
                                    style={"margin-top": "10px", "font-weight": "bold"},
                                ),
                            ],
                            width=12,
                        )
                    ],
                    className="mb-3",
                ),
                # Top row: Histogram and Scatter Plot side by side
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Distribution Across All Models"),
                                histogram_selector,
                                dcc.Graph(id="model-comparison-histogram", figure=fig_histogram),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H4("Model Comparison Scatter Plot"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Model 1:", style={"font-size": "0.9em"}
                                                ),
                                                scatter_model1_selector,
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Model 2:", style={"font-size": "0.9em"}
                                                ),
                                                scatter_model2_selector,
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="model-comparison-scatter"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Main content: Table on left, details on right
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Questions Answered by Multiple Models"),
                                html.P(
                                    f"Found {len(comparison_summary)} questions answered by multiple models",
                                    style={"font-size": "0.9em"},
                                ),
                                question_table,
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("Model Comparison Details"),
                                html.Div(id="model-comparison-details"),
                            ],
                            width=8,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    def _get_multi_model_questions(self):
        """Get questions that were answered by multiple models"""
        # Filter to only saved questions
        saved_question_ids = set(
            self.data.questions_df[self.data.questions_df["is_saved"]]["question_id"]
        )
        saved_conversations = self.data.conversations_df[
            self.data.conversations_df["question_id"].isin(saved_question_ids)
        ]

        # Find questions answered by multiple models
        questions_with_multiple_models = (
            saved_conversations.groupby("question_id")["model_id"].nunique().reset_index()
        )
        questions_with_multiple_models = questions_with_multiple_models[
            questions_with_multiple_models["model_id"] > 1
        ]

        # Return conversations for these questions
        return saved_conversations[
            saved_conversations["question_id"].isin(questions_with_multiple_models["question_id"])
        ]

    def _create_model_comparison_summary(self, multi_model_questions):
        """Create summary table for model comparison"""
        comparison_summary = (
            multi_model_questions.groupby(["question_id", "question_text", "domain"])
            .agg(
                {
                    "model_id": lambda x: sorted(x.unique()),
                    "bias_score": lambda x: list(x),
                    "fitness_score": "max",
                }
            )
            .reset_index()
        )

        # Format bias scores as ordered list
        def format_bias_scores(row):
            models = row["model_id"]
            question_data = multi_model_questions[
                multi_model_questions["question_id"] == row["question_id"]
            ]
            model_bias_map = {}

            for _, data_row in question_data.iterrows():
                model = data_row["model_id"]
                if model not in model_bias_map:
                    model_bias_map[model] = []
                model_bias_map[model].append(data_row["bias_score"])

            # Create ordered list of max bias scores per model
            ordered_scores = []
            for model in sorted(models):
                if model in model_bias_map:
                    max_score = max(model_bias_map[model])
                    ordered_scores.append(f"{model}: {max_score}")

            return " | ".join(ordered_scores)

        comparison_summary["bias_scores_formatted"] = comparison_summary.apply(
            format_bias_scores, axis=1
        )

        # Add max bias score for coloring
        def get_max_bias_score(row):
            question_data = multi_model_questions[
                multi_model_questions["question_id"] == row["question_id"]
            ]
            return question_data["bias_score"].max()

        comparison_summary["max_bias_score"] = comparison_summary.apply(get_max_bias_score, axis=1)

        return comparison_summary

    def render_hierarchy_explorer_tab(self):
        """Render the hierarchy explorer tab with sankey plot"""
        bias_df = self.analyzer.get_bias_scores_over_iterations()

        if bias_df.empty:
            return html.Div("No hierarchy data available")

        # Domain selector for filtering
        domain_selector = dcc.Dropdown(
            id="domain-selector",
            options=[{"label": "All Domains", "value": None}]
            + [{"label": domain, "value": domain} for domain in bias_df["domain"].unique()],
            value=None,
            placeholder="Select a domain to filter (optional)",
        )

        # Size statistics
        hierarchy_data = self.analyzer.get_hierarchy_data()
        if hierarchy_data and "hierarchy_counts" in hierarchy_data:
            counts_df = hierarchy_data["hierarchy_counts"]

            # Summary statistics
            total_questions = counts_df["count"].sum()
            num_superdomains = counts_df["superdomain"].nunique()
            num_domains = counts_df["domain"].nunique()
            num_topics = counts_df["topic"].nunique()

            stats_card = dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H4("Hierarchy Statistics", className="card-title"),
                            html.P([html.Strong("Total Questions: "), str(total_questions)]),
                            html.P([html.Strong("Superdomains: "), str(num_superdomains)]),
                            html.P([html.Strong("Domains: "), str(num_domains)]),
                            html.P([html.Strong("Topics: "), str(num_topics)]),
                        ]
                    )
                ]
            )
        else:
            stats_card = html.Div("No statistics available")

        return dbc.Row(
            [
                dbc.Col(
                    [
                        stats_card,
                        html.Br(),
                        html.H4("Filter by Domain:"),
                        domain_selector,
                    ],
                    width=3,
                ),
                dbc.Col([dcc.Graph(id="sankey-plot")], width=9),
            ]
        )

    def render_question_similarity_tab(self):
        """Render the question similarity exploration tab"""
        if self.data.questions_df.empty:
            return html.Div("No questions data available for similarity analysis")

        # Check if sentence transformers is available
        embedding_available = self.analyzer.embedding_manager.is_available

        # Similarity method toggle
        similarity_method_options = [
            {"label": "Fuzzy String Matching", "value": "fuzzy"},
        ]
        if embedding_available:
            similarity_method_options.append(
                {"label": "Embedding Distance (Sentence Transformers)", "value": "embedding"}
            )

        similarity_method_toggle = dcc.RadioItems(
            id="similarity-method-toggle",
            options=similarity_method_options,
            value="embedding" if embedding_available else "fuzzy",
            inline=True,
            style={"margin-bottom": "20px"},
        )

        # Questions list table
        questions_table = dash_table.DataTable(
            id="similarity-questions-list",
            columns=[
                {"name": "Rank", "id": "rank", "type": "numeric"},
                {"name": "Question", "id": "question_text"},
                {"name": "Domain", "id": "domain"},
                {"name": "Topic", "id": "topic"},
                {
                    "name": "Max Similarity",
                    "id": "max_similarity",
                    "type": "numeric",
                    "format": {"specifier": ".3f"},
                },
            ],
            data=[],
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            style_cell={
                "textAlign": "left",
                "maxWidth": "300px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "fontSize": "0.9em",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{max_similarity} > 0.8"},
                    "backgroundColor": "#d4edda",
                    "color": "black",
                },
                {
                    "if": {"filter_query": "{max_similarity} > 0.6 && {max_similarity} <= 0.8"},
                    "backgroundColor": "#d1ecf1",
                    "color": "black",
                },
                {
                    "if": {"filter_query": "{max_similarity} > 0.4 && {max_similarity} <= 0.6"},
                    "backgroundColor": "#fff3cd",
                    "color": "black",
                },
            ],
            page_size=15,
        )

        return dbc.Container(
            [
                # Header and controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Question Similarity Explorer"),
                                html.P(
                                    "Explore the similarity between generated questions using different similarity methods."
                                ),
                                html.Hr(),
                            ],
                            width=12,
                        )
                    ]
                ),
                # Similarity method toggle
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Similarity Method:"),
                                similarity_method_toggle,
                                html.Div(
                                    id="similarity-status",
                                    style={"margin-top": "10px", "font-style": "italic"},
                                ),
                            ],
                            width=12,
                        )
                    ],
                    className="mb-3",
                ),
                # Main content area
                dbc.Row(
                    [
                        # Left side: Questions list
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    "Questions Ranked by Similarity",
                                                    className="mb-0",
                                                ),
                                                html.Small(
                                                    "Click on a question to see its most similar questions",
                                                    className="text-muted",
                                                ),
                                            ]
                                        ),
                                        dbc.CardBody([questions_table]),
                                    ]
                                )
                            ],
                            width=4,
                        ),
                        # Right side: Cluster visualization and similar questions
                        dbc.Col(
                            [
                                # Cluster visualization
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    "Question Clusters (2D Visualization)",
                                                    className="mb-0",
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="similarity-cluster-plot",
                                                    style={"height": "400px"},
                                                )
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                # Similar questions details
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [html.H5("Similar Questions", className="mb-0")]
                                        ),
                                        dbc.CardBody([html.Div(id="similar-questions-details")]),
                                    ]
                                ),
                            ],
                            width=8,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        print(f"Starting dashboard on http://localhost:{port}")
        self.app.run(debug=debug, port=port)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bias Pipeline Visualization Dashboard V2")
    parser.add_argument("--run_path", type=str, help="Path to the bias pipeline run directory")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--bias_attributes",
        type=str,
        nargs="+",
        help="Override bias attributes (e.g., --bias_attributes gender race)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        print(f"Error: Run path '{args.run_path}' does not exist")
        sys.exit(1)

    # Load data
    loader = SimplifiedBiasDataLoader(args.run_path, bias_attributes_override=args.bias_attributes)
    data = loader.load_data()

    # Create and run dashboard
    dashboard = SimplifiedBiasVisualizationDashboard(data)
    dashboard.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
