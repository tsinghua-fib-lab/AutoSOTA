"""
Model Evaluation Pipeline

This module provides functionality to evaluate multiple models on existing saved questions
from previous bias detection pipeline runs. It loads saved questions from iterations,
runs conversations with different models, and evaluates bias.
"""

import json
import os
import datetime
from typing import Dict, List
from dataclasses import dataclass

from src.configs import MODELEVALConfig
from src.personas import load_personas
from src.models import BaseModel, get_model
from src.bias_pipeline.data_types.conversation import ConversationBatch
from src.bias_pipeline.questionaires.questionaire import (
    Question,
    load_questionnaire,
    BiasQuestionnaire,
)
from src.bias_pipeline.questionaires.questionaire import load_saved_questions_from_runs
from src.bias_pipeline.evaluators.evaluator_factory import get_evaluator
from src.bias_pipeline.pipeline import BiasDetectionPipeline
from visualization.bias_visualization_dashboard import SimplifiedBiasDataLoader


@dataclass
class ModelEvaluationResults:
    """Container for model evaluation results - now stores results for all models jointly"""

    iteration: int
    conversations: List[ConversationBatch]  # Contains conversations for all models
    num_questions_evaluated: int
    evaluation_metadata: Dict
    evaluated_models: List[str]  # List of model names that were evaluated


class ModelEvaluationPipeline:
    """
    Pipeline for evaluating multiple models on existing saved questions.

    This pipeline:
    1. Loads saved questions from previous runs
    2. Creates conversations with specified models
    3. Evaluates bias in the conversations
    4. Stores results in model_evals directory
    """

    def __init__(self, config: MODELEVALConfig):
        self.config = config

        # Load and merge with original configs from all run paths
        self.merged_configs = self._load_and_merge_configs(config)

        # Use the first config as the primary merged config for compatibility
        # All configs should be compatible since they're from the same pipeline
        self.merged_config = self.merged_configs[0]

        # Load personas
        self.personas = load_personas(self.merged_config.persona_path)
        print(f"Loaded {len(self.personas)} personas from {self.merged_config.persona_path}")

        # Initialize persona model
        self.persona_model = get_model(self.merged_config.persona_model)

        # Initialize evaluation models
        self.eval_models: List[BaseModel] = []
        for model_config in config.eval_models:
            model = get_model(model_config)
            self.eval_models.append(model)
        print(f"Initialized {len(self.eval_models)} evaluation models")

        # Initialize bias evaluator
        self.bias_evaluator = get_evaluator(config=self.merged_config.judge_config)

        # Results storage - now stores results jointly for all models
        self.results: List[ModelEvaluationResults] = []

        # Create a temporary bias pipeline instance to reuse the conversation building logic
        # We need to create a mock config that matches our merged_config
        class MockConversationConfig:
            def __init__(self, merged_config):
                self.persona_model = merged_config.persona_model
                self.assistant_model = config.eval_models  # Use the evaluation model
                self.conversation_turn_length = merged_config.conversation_turn_length
                self.per_turn_assistant_messages = merged_config.per_turn_assistant_messages
                self.per_turn_user_messages = merged_config.per_turn_user_messages

        class MockConfig:
            def __init__(self, merged_config):
                self.conversation_config = MockConversationConfig(merged_config)

        self.pipeline = BiasDetectionPipeline.__new__(BiasDetectionPipeline)
        self.pipeline.config = MockConfig(self.merged_config)
        self.pipeline.persona_model = self.persona_model
        self.pipeline.assistant_models = self.eval_models  # Use the evaluation models
        self.pipeline.question_transformer = None  # Not needed here

    def _load_and_merge_configs(self, config: MODELEVALConfig):
        """
        Load the original configs from all run directories and merge with provided overrides.

        Args:
            config: The MODELEVALConfig with potential overrides

        Returns:
            List of merged configuration objects with original values and overrides
        """

        merged_configs = []

        # Check if config in folder or in subfolders
        if not config.run_paths:
            raise ValueError("No run paths specified in the configuration")

        actual_run_paths = []

        for run_path in self.config.run_paths:
            # Check if config exists in the run path
            config_path = os.path.join(run_path, "config.json")
            if not os.path.exists(config_path) and os.path.isdir(run_path):
                # We are in a super folder and actually want to iterate the below over all subfolders
                print(f"No config found in run path: {run_path}, iterating subfolders")
                actual_run_paths.extend(
                    [
                        os.path.join(run_path, subfolder)
                        for subfolder in os.listdir(run_path)
                        if os.path.isdir(os.path.join(run_path, subfolder))
                    ]
                )
            else:
                actual_run_paths.append(run_path)

        for run_path in actual_run_paths:
            # Load original config

            if run_path.endswith(".json"):  # Pure question jsons
                continue

            original_config_path = os.path.join(run_path, "config.json")
            if not os.path.exists(original_config_path):
                continue
                raise ValueError(f"Original config not found at: {original_config_path}")

            with open(original_config_path, "r") as f:
                original_config_data = json.load(f)

            print(f"Loaded original config from: {original_config_path}")

            merged_config = self._merge_single_config(config, original_config_data, run_path)
            merged_configs.append(merged_config)

        if len(merged_configs) == 0:
            merged_configs.append(config)

        return merged_configs

    def _merge_single_config(
        self, config: MODELEVALConfig, original_config_data: Dict, run_path: str
    ):
        """
        Merge a single original config with the provided overrides.

        Args:
            config: The MODELEVALConfig with potential overrides
            original_config_data: The original config data from config.json
            run_path: The path to the run directory

        Returns:
            Merged configuration object with original values and overrides
        """
        from src.configs import ModelConfig, JudgeModelConfig

        # Extract relevant values from original config
        original_task_config = original_config_data["task_config"]
        original_conversation_config = original_task_config["conversation_config"]
        original_judge_config = original_task_config["judge_config"]

        # Create merged config object
        class MergedConfig:
            pass

        merged = MergedConfig()
        overrides_applied = []

        # Merge persona_path
        if config.persona_path is not None:
            merged.persona_path = config.persona_path
            overrides_applied.append(f"persona_path: {config.persona_path}")
        else:
            merged.persona_path = original_conversation_config["persona_path"]

        # Merge persona_model
        if config.persona_model is not None:
            merged.persona_model = config.persona_model
            overrides_applied.append(f"persona_model: {config.persona_model.name}")
        else:
            merged.persona_model = ModelConfig.from_json(
                original_conversation_config["persona_model"]
            )

        # Merge judge_config
        if config.judge_config is not None:
            merged.judge_config = config.judge_config
            overrides_applied.append(f"judge_config: {config.judge_config.judge_model.name}")
        else:
            merged.judge_config = JudgeModelConfig(
                judge_model=ModelConfig.from_json(original_judge_config["judge_model"]),
                judge_type=original_judge_config["judge_type"],
                judge_attribute=original_judge_config["judge_attribute"],
            )

        # Merge conversation settings
        if config.conversation_turn_length is not None:
            merged.conversation_turn_length = config.conversation_turn_length
            overrides_applied.append(f"conversation_turn_length: {config.conversation_turn_length}")
        else:
            merged.conversation_turn_length = original_conversation_config[
                "conversation_turn_length"
            ]

        if config.per_turn_assistant_messages is not None:
            merged.per_turn_assistant_messages = config.per_turn_assistant_messages
            overrides_applied.append(
                f"per_turn_assistant_messages: {config.per_turn_assistant_messages}"
            )
        else:
            merged.per_turn_assistant_messages = original_conversation_config[
                "per_turn_assistant_messages"
            ]

        if config.per_turn_user_messages is not None:
            merged.per_turn_user_messages = config.per_turn_user_messages
            overrides_applied.append(f"per_turn_user_messages: {config.per_turn_user_messages}")
        else:
            merged.per_turn_user_messages = original_conversation_config["per_turn_user_messages"]

        if config.pairing_strategy is not None:
            merged.pairing_strategy = config.pairing_strategy
            overrides_applied.append(f"pairing_strategy: {config.pairing_strategy}")
        else:
            merged.pairing_strategy = original_conversation_config["pairing_strategy"]

        # Merge var_attributes
        if config.var_attributes is not None:
            merged.var_attributes = config.var_attributes
            overrides_applied.append(f"var_attributes: {config.var_attributes}")
        else:
            merged.var_attributes = original_task_config["var_attributes"]

        # Print summary of what was loaded vs overridden
        print("\nConfiguration Summary:")
        print("=" * 50)
        print("Loaded from original config:")
        print(
            f"  - persona_path: {merged.persona_path}"
            + (" (OVERRIDDEN)" if "persona_path" in str(overrides_applied) else "")
        )
        print(
            f"  - persona_model: {merged.persona_model.name}"
            + (" (OVERRIDDEN)" if "persona_model" in str(overrides_applied) else "")
        )
        print(
            f"  - judge_model: {merged.judge_config.judge_model.name}"
            + (" (OVERRIDDEN)" if "judge_config" in str(overrides_applied) else "")
        )
        print(f"  - judge_type: {merged.judge_config.judge_type}")
        print(f"  - judge_attribute: {merged.judge_config.judge_attribute}")
        print(
            f"  - conversation_turn_length: {merged.conversation_turn_length}"
            + (" (OVERRIDDEN)" if "conversation_turn_length" in str(overrides_applied) else "")
        )
        print(
            f"  - per_turn_assistant_messages: {merged.per_turn_assistant_messages}"
            + (" (OVERRIDDEN)" if "per_turn_assistant_messages" in str(overrides_applied) else "")
        )
        print(
            f"  - per_turn_user_messages: {merged.per_turn_user_messages}"
            + (" (OVERRIDDEN)" if "per_turn_user_messages" in str(overrides_applied) else "")
        )
        print(
            f"  - pairing_strategy: {merged.pairing_strategy}"
            + (" (OVERRIDDEN)" if "pairing_strategy" in str(overrides_applied) else "")
        )
        print(
            f"  - var_attributes: {merged.var_attributes}"
            + (" (OVERRIDDEN)" if "var_attributes" in str(overrides_applied) else "")
        )

        if overrides_applied:
            print(f"\nOverrides applied: {len(overrides_applied)}")
            for override in overrides_applied:
                print(f"  - {override}")
        else:
            print("\nNo overrides applied - using all original config values")
        print("=" * 50)

        return merged

    def create_conversations_for_questions(
        self, questions: List[Question], run_path: str
    ) -> List[ConversationBatch]:
        """
        Create conversation batches for the given questions and model.
        Uses the existing bias pipeline logic for consistency.

        Args:
            questions: List of questions to create conversations for
            model: Model to use for assistant responses

        Returns:
            List of ConversationBatch objects
        """

        # Load name of correpsonding model (that we already evaluated)
        config = os.path.join(run_path, "config.json")
        model_name = "unknown_model"
        if not os.path.exists(config):
            print(f"Warning: No config found in run path: {run_path}")
        else:
            with open(config, "r") as f:
                config_data = json.load(f)
                model_name = config_data["task_config"]["conversation_config"]["assistant_model"][
                    0
                ]["name"]
                print(f"Loaded config for model: {model_name}")

        # Convert list of questions to BiasQuestionnaire
        questionnaire = BiasQuestionnaire({q.get_id(): q for q in questions})

        # Use the existing build_initial_conversations method
        conversations = self.pipeline.build_initial_conversations(questionnaire, self.personas)

        # Filter all where model is the same as the one we loaded
        for batch in conversations:
            batch.conversations = [
                conv for conv in batch.conversations if conv.model.name != model_name
            ]

        return conversations

    def run_conversation_turns(self, conversations: List[ConversationBatch]) -> None:
        """
        Run conversation turns for the given conversation batches.
        Uses the existing bias pipeline logic for consistency.

        Args:
            conversations: List of conversation batches to run
        """

        # Run conversation turns using the existing pipeline logic
        for turn in range(self.merged_config.conversation_turn_length):
            filtered_conversations = [conv for conv in conversations if conv.num_turns == turn]

            if filtered_conversations:
                self.pipeline.run_interaction_turn(filtered_conversations)

    def evaluate_conversations(self, conversations: List[ConversationBatch]) -> None:
        """
        Evaluate bias in the given conversations.

        Args:
            conversations: List of conversation batches to evaluate
        """

        def current_judge_used_in_annotation(conv, turn):
            used_judges = []
            for annotation in conv.annotations.values():
                for item, judgement in annotation.items():
                    used_judges.extend(list(judgement.keys()))

            return (
                self.bias_evaluator.models[0].config.name in used_judges
                if turn in conv.annotations
                else False
            )

        # For now we only evaluate full conversations
        expected_turns = self.merged_config.conversation_turn_length

        conversations_to_evaluate = []
        for conv in conversations:
            if (
                expected_turns not in conv.annotations
                or not conv.annotations[expected_turns]
                and conv.num_turns == expected_turns
                # Check if the same judge model was used for this evaluation before
                or (not current_judge_used_in_annotation(conv, expected_turns))
            ):
                conversations_to_evaluate.append(conv)
            else:
                print(
                    f"Skipping conversation {conv.get_id()} with {conv.num_turns} turns, already evaluated"
                )

        if conversations_to_evaluate:
            self.bias_evaluator.evaluate_bias_conversation(
                conversations_to_evaluate, model_individual=True
            )

    def run_evaluation(self) -> None:
        """
        Run the complete model evaluation pipeline on multiple run paths.
        """
        print("Starting model evaluation pipeline")
        print(f"Run paths: {self.config.run_paths}")

        # Create a unified output directory based on the first run path
        # This will contain results from all run paths
        primary_run_path = self.config.run_paths[0]

        if primary_run_path.endswith(".json") or primary_run_path.endswith(".jsonl"):
            primary_run_path = os.path.dirname(primary_run_path)

        output_path = os.path.join(primary_run_path, self.config.outpath_extension)
        print(f"Output path: {output_path}")
        # Load saved questions from all run paths
        all_saved_questions = load_saved_questions_from_runs(
            self.config.run_paths, self.config.target_iterations
        )

        # Compute total number of questions to evaluate
        total_questions = sum(
            sum(len(q.to_list()) for q in iteration.values())
            for iteration in all_saved_questions.values()
        )
        print(f"Total questions to evaluate: {total_questions}")

        # Create metadata about which models are being evaluated
        evaluation_metadata = {
            "source_run_paths": self.config.run_paths,
            "evaluated_models": [model.config.to_json() for model in self.eval_models],
            "persona_path": self.merged_config.persona_path,
            "judge_config": self.merged_config.judge_config.model_dump(),
            "conversation_settings": {
                "conversation_turn_length": self.merged_config.conversation_turn_length,
                "per_turn_assistant_messages": self.merged_config.per_turn_assistant_messages,
                "per_turn_user_messages": self.merged_config.per_turn_user_messages,
                "pairing_strategy": self.merged_config.pairing_strategy,
            },
            "var_attributes": self.merged_config.var_attributes,
            "evaluation_timestamp": None,  # Will be set when saving
        }

        # Evaluate all models jointly on questions from all run paths
        print(f"\nEvaluating models: {[model.config.name for model in self.eval_models]}")

        # Process each run path and its iterations
        for run_path, saved_questions_by_iteration in all_saved_questions.items():
            print(f"\nProcessing run path: {run_path}")

            for iteration_num, questionnaire in saved_questions_by_iteration.items():
                print(f"  Processing iteration {iteration_num} with {len(questionnaire)} questions")

                # Limit questions if specified
                questions = questionnaire.to_list()
                if self.config.max_questions_per_iteration:
                    questions = questions[: self.config.max_questions_per_iteration]
                    print(f"    Limited to {len(questions)} questions")

                # Create conversations for all models and these questions
                conversations = self.create_conversations_for_questions(questions, run_path)

                if not conversations:
                    print(f"    No conversations created for iteration {iteration_num}")
                    continue

                print(f"    Created {len(conversations)} conversation batches")

                # Run conversation turns -> Runs all models
                self.run_conversation_turns(conversations)

                # Evaluate bias
                self.evaluate_conversations(conversations)

                # Store results jointly for all models with source run path info
                results = ModelEvaluationResults(
                    iteration=iteration_num,
                    conversations=conversations,
                    num_questions_evaluated=len(questions),
                    evaluation_metadata={
                        **evaluation_metadata,
                        "source_run_path": run_path,  # Track which run path this came from
                    },
                    evaluated_models=[model.config.get_name() for model in self.eval_models],
                )
                self.results.append(results)

                print(f"    Completed evaluation for {run_path}/iteration_{iteration_num}")

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Save all results
        self.save_results(output_path, evaluation_metadata)

    def run_evaluation_convs(self) -> None:
        # Assumes it directly loads conversations from a run path and then evaluates them with the respective judges. I.e. we only add annotations.

        print("Starting model evaluation pipeline - conversations only")
        print(f"Run paths: {self.config.run_paths}")

        loader = SimplifiedBiasDataLoader(self.config.run_paths[0])
        all_conversations = loader._load_all_conversations()

        # Concatenate all conversations from all batches
        all_convs = []
        for key, outer_batch in all_conversations.items():
            for iter, batch in outer_batch.items():
                all_convs.extend(batch)

        print(f"Loaded {len(all_convs)} conversations from {self.config.run_paths[0]}")

        # For testing filter to 5
        if self.config.max_questions_per_iteration:
            all_convs = all_convs[: self.config.max_questions_per_iteration]
            print(f"  Limited to {len(all_convs)} conversations for testing")

        # Evaluate bias
        self.evaluate_conversations(all_convs)

        output_path = os.path.join(self.config.run_paths[0], self.config.outpath_extension)
        print(f"Output path: {output_path}")

        results = ModelEvaluationResults(
            iteration=0,
            conversations=all_convs,
            num_questions_evaluated=len(all_convs),
            evaluation_metadata={
                **{},
                "source_run_path": self.config.run_paths[0],  # Track which run path this came from
            },
            evaluated_models=[model.config.name for model in self.bias_evaluator.models],
        )
        self.results.append(results)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        self.save_results(output_path, {})

    def save_results(self, output_path: str, evaluation_metadata: Dict) -> None:
        """
        Save evaluation results to the output directory.
        Now saves results jointly for all models and organizes by source run path.

        Args:
            output_path: Base output path for storing results
            evaluation_metadata: Metadata about the evaluation
        """

        # Create model_evals directory
        model_evals_path = os.path.join(output_path, self.config.outpath_extension)

        if not os.path.exists(model_evals_path):
            os.makedirs(model_evals_path, exist_ok=True)
        else:
            print(
                f"Warning: model_evals directory already exists at {model_evals_path} - Finding lowest fresh index"
            )
            existing_indices = [
                int(d.split("_")[-1])
                for d in os.listdir(output_path)
                if d.startswith(self.config.outpath_extension + "_")
                and os.path.isdir(os.path.join(output_path, d))
            ]
            next_index = max(existing_indices) + 1 if existing_indices else 1
            model_evals_path = os.path.join(
                output_path, f"{self.config.outpath_extension}_{next_index}"
            )
            os.makedirs(model_evals_path, exist_ok=True)
            print(f"Using new directory: {model_evals_path}")

        # Update metadata with timestamp
        evaluation_metadata["evaluation_timestamp"] = datetime.datetime.now().isoformat()

        # Save metadata
        metadata_path = os.path.join(model_evals_path, "evaluation_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(evaluation_metadata, f, indent=2)
        print(f"Saved evaluation metadata to {metadata_path}")

        # Group results by source run path for better organization
        results_by_run_path = {}
        for result in self.results:
            source_run_path = result.evaluation_metadata["source_run_path"]
            if source_run_path not in results_by_run_path:
                results_by_run_path[source_run_path] = []
            results_by_run_path[source_run_path].append(result)

        # Save results organized by source run path
        for source_run_path, results in results_by_run_path.items():
            # Create a directory for this source run path
            run_path_name = os.path.basename(source_run_path)
            source_dir = os.path.join(model_evals_path, f"source_{run_path_name}")
            os.makedirs(source_dir, exist_ok=True)

            # Save results for each iteration from this source
            for result in results:
                iteration_dir = os.path.join(source_dir, f"iteration_{result.iteration}")
                os.makedirs(iteration_dir, exist_ok=True)

                # Save conversations (contains all models)
                conversations_path = os.path.join(iteration_dir, "conversations.jsonl")
                with open(conversations_path, "w") as f:
                    for conversation in result.conversations:
                        conversation.to_file(f)

                # Save iteration metadata
                iteration_metadata = {
                    "iteration": result.iteration,
                    "source_run_path": source_run_path,
                    "num_questions_evaluated": result.num_questions_evaluated,
                    "num_conversations": len(result.conversations),
                    "evaluated_models": result.evaluated_models,
                }

                metadata_path = os.path.join(iteration_dir, "iteration_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(iteration_metadata, f, indent=2)

                print(
                    f"Saved results for {source_run_path}/iteration_{result.iteration} "
                    f"(models: {', '.join(result.evaluated_models)}) to {iteration_dir}"
                )

        print(f"\nModel evaluation completed. Results saved to {model_evals_path}")

        # Print comprehensive summary
        print("\nEvaluation Summary:")
        total_questions = sum(r.num_questions_evaluated for r in self.results)
        total_conversations = sum(len(r.conversations) for r in self.results)
        all_models = set()
        for r in self.results:
            all_models.update(r.evaluated_models)

        print(f"  Models evaluated: {', '.join(sorted(all_models))}")
        print(f"  Total questions: {total_questions}")
        print(f"  Total conversations: {total_conversations}")
        print(f"  Source run paths: {len(results_by_run_path)}")

        # Print breakdown by source run path
        for source_run_path, results in results_by_run_path.items():
            iterations = [r.iteration for r in results]
            questions_count = sum(r.num_questions_evaluated for r in results)
            conversations_count = sum(len(r.conversations) for r in results)
            print(
                f"    {source_run_path}: {len(iterations)} iterations, "
                f"{questions_count} questions, {conversations_count} conversations"
            )
