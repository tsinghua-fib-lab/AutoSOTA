import os
import json
from typing import List, Dict, Tuple, Iterator
from src.configs import Config
from src.bias_pipeline.questionaires.questionaire import (
    load_questionnaire,
    Question,
    BiasQuestionnaire,
)
from src.bias_pipeline.question_transfomer.question_transfomer import get_question_transformer
from src.bias_pipeline.data_types.data_types import Message
from src.models import run_parallel


def transform_questions(config: Config):
    """
    Transform questions using the specified question transformer.

    This function loads questions from a file, applies the configured question transformer,
    and saves the transformed questions with all necessary information preserved.

    Args:
        config (Config): Configuration object containing transformation settings
    """

    print(f"Starting question transformation with config: {config.task_config}")

    # Load the questions to transform
    question_path = config.task_config.question_path
    if not os.path.exists(question_path):
        raise FileNotFoundError(f"Question file not found: {question_path}")

    print(f"Loading questions from: {question_path}")
    questionnaire = load_questionnaire(question_path)
    questions = questionnaire.to_list()

    # Apply max_questions limit if specified
    if config.task_config.max_questions and len(questions) > config.task_config.max_questions:
        questions = questions[: config.task_config.max_questions]
        print(f"Limited to {config.task_config.max_questions} questions")

    print(f"Loaded {len(questions)} questions for transformation")

    # Initialize the question transformer
    transformer_config = config.task_config.question_transformer_config
    question_transformer = get_question_transformer(transformer_config)

    print(
        f"Initialized {transformer_config.type} transformer for attribute: {transformer_config.attribute}"
    )

    # Transform questions in batch
    print("Starting transformation...")
    transformed_results = list(question_transformer.transform_batch(questions))

    # Process the results
    transformed_questions = []
    transformation_info = []

    for i, (original_question, transformed_messages) in enumerate(transformed_results):
        if not transformed_messages:
            print(f"Warning: No transformation result for question {i}")
            continue

        # For each transformed message, create a new question
        for j, message in enumerate(transformed_messages):
            # Create a new question with the transformed text
            transformed_question = Question(
                superdomain=original_question.superdomain,
                domain=original_question.domain,
                topic=original_question.topic,
                example=message.text,
                score=original_question.score,  # Preserve original score if any
                orig_model=getattr(original_question, "orig_model", None),
            )

            transformed_questions.append(transformed_question)

            # Store transformation metadata
            transformation_info.append(
                {
                    "original_id": original_question.get_id(),
                    "original_text": original_question.example,
                    "transformed_id": transformed_question.get_id(),
                    "transformed_text": message.text,
                    "transformation_index": j,
                    "transformer_type": transformer_config.type,
                    "transformer_attribute": transformer_config.attribute,
                    "expected_options": transformer_config.expected_options,
                }
            )

    print(
        f"Successfully transformed {len(questions)} questions into {len(transformed_questions)} variants"
    )

    # Create output questionnaire
    question_dict = {q.get_id(): q for q in transformed_questions}
    transformed_questionnaire = BiasQuestionnaire(question_dict)

    # Prepare output directory
    out_path = config.get_out_path()
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save transformed questions
    questions_output_path = os.path.join(out_path, "transformed_questions.jsonl")
    with open(questions_output_path, "w") as f:
        for question in transformed_questions:
            question.to_file(f)

    print(f"Saved {len(transformed_questions)} transformed questions to: {questions_output_path}")

    # Save transformation metadata
    metadata_output_path = os.path.join(out_path, "transformation_metadata.json")
    with open(metadata_output_path, "w") as f:
        json.dump(
            {
                "transformation_config": {
                    "transformer_type": transformer_config.type,
                    "attribute": transformer_config.attribute,
                    "expected_options": transformer_config.expected_options,
                    "model": transformer_config.model.model_dump(),
                },
                "source_file": question_path,
                "original_question_count": len(questions),
                "transformed_question_count": len(transformed_questions),
                "transformations": transformation_info,
            },
            f,
            indent=2,
        )

    print(f"Saved transformation metadata to: {metadata_output_path}")

    # Save as BiasQuestionnaire JSON format as well for compatibility
    questionnaire_output_path = os.path.join(out_path, "transformed_questions.json")
    with open(questionnaire_output_path, "w") as f:
        json.dump(transformed_questionnaire.to_json(), f, indent=2)

    print(f"Saved questionnaire format to: {questionnaire_output_path}")

    # Print summary statistics
    print("\n=== Transformation Summary ===")
    print(f"Original questions: {len(questions)}")
    print(f"Transformed questions: {len(transformed_questions)}")
    print(f"Transformation ratio: {len(transformed_questions) / len(questions):.2f}")
    print(f"Transformer type: {transformer_config.type}")
    print(f"Target attribute: {transformer_config.attribute}")
    print(f"Expected options: {transformer_config.expected_options}")
    print(f"Output directory: {out_path}")

    # Show some examples
    if transformation_info:
        print("\n=== Example Transformations ===")
        for i, info in enumerate(transformation_info[:3]):  # Show first 3 examples
            print(f"\nExample {i + 1}:")
            print(f"  Original: {info['original_text'][:100]}...")
            print(f"  Transformed: {info['transformed_text'][:100]}...")

    print("\nTransformation completed successfully!")


def _validate_transformation_config(config: Config):
    """
    Validate the transformation configuration.

    Args:
        config (Config): Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not hasattr(config.task_config, "question_path"):
        raise ValueError("question_path is required in task_config")

    if not hasattr(config.task_config, "question_transformer_config"):
        raise ValueError("question_transformer_config is required in task_config")

    transformer_config = config.task_config.question_transformer_config

    required_fields = [
        "type",
        "attribute",
        "expected_options",
        "model",
        "system_prompt_path",
        "user_prompt_path",
    ]
    for field in required_fields:
        if not hasattr(transformer_config, field):
            raise ValueError(f"{field} is required in question_transformer_config")

    # Validate paths exist
    if not os.path.exists(config.task_config.question_path):
        raise ValueError(f"Question file not found: {config.task_config.question_path}")

    if not os.path.exists(transformer_config.system_prompt_path):
        raise ValueError(f"System prompt file not found: {transformer_config.system_prompt_path}")

    if not os.path.exists(transformer_config.user_prompt_path):
        raise ValueError(f"User prompt file not found: {transformer_config.user_prompt_path}")
