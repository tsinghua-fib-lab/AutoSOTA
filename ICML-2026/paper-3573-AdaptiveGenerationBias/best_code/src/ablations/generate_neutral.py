import os
import json
from typing import List, Dict, Tuple
from src.configs import Config
from src.ablations.neutral_question_generator import NeutralQuestionGenerator
from src.bias_pipeline.questionaires.questionaire import (
    load_saved_questions_from_runs,
    Question,
    BiasQuestionnaire,
)
from src.models import run_parallel, get_model


def generate_neutral_questions(
    config: Config,
):
    """Generate neutral questions using different modes based on configuration"""

    model_config = config.task_config.model

    # Initialize generator
    model = get_model(model_config)
    generator = NeutralQuestionGenerator(model_config, config.task_config.type_values)

    # Load the examples
    if config.task_config.examples_path and os.path.exists(config.task_config.examples_path):
        with open(config.task_config.examples_path, "r") as f:
            examples = json.load(f)
    else:
        examples = []

    # Determine generation mode based on config
    generation_mode = getattr(config.task_config, "generation_mode", "from_runs")

    if generation_mode == "from_runs":
        # Original mode: load questions from existing runs
        prompts, topic_data = _generate_from_runs(config, generator, examples)
    elif generation_mode == "from_topics":
        # New mode: load topics from topic mapping files
        prompts, topic_data = _generate_from_topics(config, generator, examples)
    elif generation_mode == "creative":
        # New mode: generate completely new questions with creative topics
        prompts, topic_data = _generate_creative(config, generator, examples)
    else:
        raise ValueError(f"Unknown generation_mode: {generation_mode}")

    if not prompts:
        print("No prompts generated. Check your configuration.")
        return

    print(f"Generated {len(prompts)} prompts")

    # Run parallel prompting
    def run_parallel_func(prompt_data: Tuple) -> str:
        topic, prompt = prompt_data
        return generator._parse_generated_question(
            model.predict_string(prompt, model_config.system_prompt)
        )

    tuple_prompts = list(zip(topic_data, prompts))
    res = run_parallel(
        run_parallel_func, tuple_prompts, max_workers=config.task_config.model.max_workers
    )
    new_questions = _process_results_from_runs(res)

    if not new_questions:
        print("No valid questions were generated.")
        return

    question_dict = dict([(q.get_id(), q) for q in new_questions])

    # New Questionnaire
    new_questionnaire = BiasQuestionnaire(question_dict)

    out_path = config.get_out_path()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save
    with open(os.path.join(out_path, "questions.json"), "w") as f:
        json.dump(new_questionnaire.to_json(), f, indent=2)

    # test load the questions
    with open(os.path.join(out_path, "questions.json"), "r") as f:
        test_questions = BiasQuestionnaire.from_json(json.load(f))

    print(f"Generated and saved {len(test_questions)} questions to {out_path}")


def _generate_from_runs(
    config: Config, generator: NeutralQuestionGenerator, examples: List[Dict]
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Generate prompts from existing run data"""
    run_paths = config.task_config.run_paths

    questions: Dict[str, Dict[int, BiasQuestionnaire]] = {}
    for path in run_paths:
        if not os.path.exists(path):
            print(f"Run path {path} does not exist. Please check the path.")
            continue

        questions.update(load_saved_questions_from_runs([path]))

    # Group all questions
    generated_questions: List[Question] = []
    for path, questionnaires in questions.items():
        for questionnaire in questionnaires.values():
            generated_questions.extend(questionnaire.to_list())

    print(f"Loaded {len(generated_questions)} questions from runs")

    if not generated_questions:
        return [], []

    # Extract Superdomains and domains and topics
    tuples = list(set([(q.superdomain, q.domain, q.topic) for q in generated_questions]))
    print(f"Found {len(tuples)} unique topic tuples")

    # Generate Prompts
    prompts = generator.generate_prompts(tuples, examples)

    return prompts, tuples


def _generate_from_topics(
    config: Config, generator: NeutralQuestionGenerator, examples: List[Dict]
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Generate prompts from topic mapping files"""
    topics_path = getattr(config.task_config, "topics_path", None)
    if not topics_path or not os.path.exists(topics_path):
        raise ValueError(f"topics_path not specified or file does not exist: {topics_path}")

    with open(topics_path, "r") as f:
        topics_data = json.load(f)

    num_questions_per_topic = getattr(config.task_config, "num_questions_per_topic", 1)

    print(f"Loaded {len(topics_data)} domain categories from {topics_path}")

    prompts, topic_info = generator.generate_prompts_from_topics(
        topics_data, examples, num_questions_per_topic
    )

    return prompts, topic_info


def _generate_creative(
    config: Config, generator: NeutralQuestionGenerator, examples: List[Dict]
) -> Tuple[List[str], None]:
    """Generate prompts for creative question generation"""
    num_questions = getattr(config.task_config, "num_questions", 100)

    print(f"Generating {num_questions} creative questions")

    prompts = generator.generate_prompts_creative(num_questions, examples)

    return prompts, None


def _process_results_from_runs(res: List[Tuple]) -> List[Question]:
    """Process results from the from_runs generation mode"""
    new_questions: List[Question] = []

    for topic, new_question in res:
        if new_question:
            new_questions.append(
                Question(
                    example=new_question,
                    superdomain=topic[0][0],
                    domain=topic[0][1],
                    topic=topic[0][2],
                )
            )

    return new_questions


def _process_results_other_modes(res: List[str]) -> List[Question]:
    """Process results from topic mapping or creative generation modes"""
    new_questions: List[Question] = []

    for new_question in res:
        if new_question:
            # For creative and topic-based generation, we'll use placeholder values
            # since the actual superdomain/domain/topic are embedded in the question
            new_questions.append(
                Question(
                    example=new_question,
                    superdomain="Generated",
                    domain="Generated",
                    topic="Generated",
                )
            )

    return new_questions
