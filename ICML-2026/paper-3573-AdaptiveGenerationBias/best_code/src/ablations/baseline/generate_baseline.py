import os
import json
import random
from typing import List, Dict, Tuple
from src.configs import Config
from src.ablations.baseline.baseline_question_generator import BaselineQuestionGenerator
from src.bias_pipeline.questionaires.questionaire import (
    load_saved_questions_from_runs,
    Question,
    BiasQuestionnaire,
)
from src.models import run_parallel, get_model
from src.prompts.prompt_loader import get_prompt_loader


def generate_baseline_questions(
    config: Config,
):
    """Generate baseline biased questions using different modes based on configuration"""

    model_config = config.task_config.model

    # Initialize generator
    attribute = getattr(config.task_config, "attribute", "gender")
    generator = BaselineQuestionGenerator(model_config, config.task_config.type_values, attribute)

    # Load the examples
    if config.task_config.examples_path and os.path.exists(config.task_config.examples_path):
        with open(config.task_config.examples_path, "r") as f:
            examples = json.load(f)
    else:
        examples = []

    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_template(
        "ablations/baseline/baseline_system.j2",
        attribute=attribute,
        type_values=config.task_config.type_values,
        type_examples=config.task_config.type_examples,
    )

    # Determine generation mode based on config
    generation_mode = getattr(config.task_config, "generation_mode", "from_runs")
    generation_format = getattr(config.task_config, "generation_format", "from_question")

    if generation_mode == "from_runs":
        # load questions from existing runs
        prompts, topic_data = _generate_from_runs(config, generator, examples)
    elif generation_mode == "from_topics":
        # load topics from topic mapping files
        prompts, topic_data = _generate_from_topics(config, generator, examples, generation_format)
    elif generation_mode == "creative":
        # generate completely new questions with creative topics
        prompts, topic_data = _generate_creative(config, generator, examples)
    else:
        raise ValueError(f"Unknown generation_mode: {generation_mode}")

    if not prompts:
        print("No prompts generated. Check your configuration.")
        return

    num_questions_to_select = min(config.task_config.num_questions, len(prompts))

    # Joint shuffeling of prompts and topic_data
    combined = (
        list(zip(prompts, topic_data)) if topic_data else list(zip(prompts, [None] * len(prompts)))
    )
    random.shuffle(combined)
    prompts, topic_data = zip(*combined)

    prompts = prompts[:num_questions_to_select]
    topic_data = topic_data[:num_questions_to_select] if topic_data else None

    print(f"Generated {len(prompts)} prompts using format: {generation_format}")

    # Get model with appropriate response format for the generation format
    format_model = generator._get_model_with_format(generation_format)

    # Run parallel prompting with structured response format
    def run_parallel_func(prompt_data) -> Dict[str, str]:
        topic_data, prompt = prompt_data
        response = format_model.predict_string(prompt, system_prompt)
        return generator._parse_generated_question(response)

    if topic_data:
        tuple_prompts = list(zip(topic_data, prompts))
    else:
        tuple_prompts = list(zip([None] * len(prompts), prompts))
    res = run_parallel(
        run_parallel_func, tuple_prompts, max_workers=config.task_config.model.max_workers
    )

    new_questions = _process_results_from_runs(res, generation_format)

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

    print(f"Generated and saved {len(test_questions)} baseline questions to {out_path}")


def _generate_from_runs(
    config: Config, generator: BaselineQuestionGenerator, examples: List[Dict]
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
    config: Config,
    generator: BaselineQuestionGenerator,
    examples: List[Dict],
    generation_format: str,
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
        topics_data, examples, num_questions_per_topic, generation_format
    )

    return prompts, topic_info


def _generate_creative(
    config: Config, generator: BaselineQuestionGenerator, examples: List[Dict]
) -> Tuple[List[str], None]:
    """Generate prompts for creative question generation"""
    num_questions = getattr(config.task_config, "num_questions", 100)

    print(f"Generating {num_questions} creative questions")

    prompts = generator.generate_prompts_creative(num_questions, examples)

    return prompts, None


def _process_results_from_runs(res: List[Tuple], generation_format: str) -> List[Question]:
    """Process results from the from_runs generation mode"""
    new_questions: List[Question] = []

    for meta_info, new_question_data in res:
        if (
            new_question_data
            and isinstance(new_question_data, dict)
            and "question" in new_question_data
        ):
            if not meta_info[0]:
                # No meta info, use defaults
                superdomain = new_question_data.get("superdomain", "Generated")
                domain = new_question_data.get("domain", "Generated")
                topic = new_question_data.get("topic", "Generated")
            elif len(meta_info[0]) == 3:
                superdomain, domain, topic = meta_info[0]
            elif len(meta_info[0]) == 2:
                superdomain, domain = meta_info[0]
                topic = new_question_data.get("topic", "Generated")
            elif len(meta_info[0]) == 1:
                superdomain = meta_info[0][0]
                domain = new_question_data.get("domain", "Generated")
                topic = new_question_data.get("topic", "Generated")

            new_questions.append(
                Question(
                    example=new_question_data["question"],
                    superdomain=superdomain,
                    domain=domain,
                    topic=topic,
                )
            )

    return new_questions
