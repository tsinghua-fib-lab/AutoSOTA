import os
import json
from typing import List, Dict, Tuple
from src.configs import Config
from src.ablations.transfer.bias_transfer_generator import BiasTransferGenerator
from src.bias_pipeline.questionaires.questionaire import (
    load_saved_questions_from_runs,
    Question,
    BiasQuestionnaire,
)
from src.models import run_parallel, get_model
import re

# Match {{ ... }} with any content inside
pattern = re.compile(r"\{\{\s*.*?\s*\}\}")


def generate_transfer_questions(
    config: Config,
):
    """Transfer questions from one bias type to another"""

    model_config = config.task_config.model
    run_paths = config.task_config.run_paths
    source_bias = config.task_config.source_bias
    target_bias = config.task_config.target_bias

    # Initialize generator
    model = get_model(model_config)
    generator = BiasTransferGenerator(model_config, type_values=config.task_config.type_values)

    # Test loading questions
    questions: Dict[str, Dict[int, BiasQuestionnaire]] = {}
    for path in run_paths:
        if not os.path.exists(path):
            print(f"Run path {path} does not exist. Please check the path.")
            continue

        print(f"Testing with run path: {path}")
        questions.update(load_saved_questions_from_runs([path]))

    # Group all questions
    loaded_questions: List[Question] = []
    for path, questionnaires in questions.items():
        for questionnaire in questionnaires.values():
            loaded_questions.extend(questionnaire.to_list())

    print(f"Loaded {len(loaded_questions)} questions")

    if loaded_questions:
        # Create question contexts with bias information
        question_contexts = []
        for q in loaded_questions:
            question_contexts.append(
                (q.example, q.superdomain, q.domain, q.topic, source_bias, target_bias)
            )

        print(
            f"Prepared {len(question_contexts)} questions for transfer from {source_bias} to {target_bias}"
        )

        # Generate Prompts
        prompts = generator.generate_prompts(question_contexts)

        context_prompts = list(zip(question_contexts, prompts))

        # Run parallel prompting
        def run_parallel_func(context_prompt: Tuple[Tuple[str, ...], str]) -> List[str]:
            return generator._parse_generated_question(
                model.predict_string(context_prompt[1], model_config.system_prompt)
            )

        if config.task_config.llm_replace:
            res = run_parallel(run_parallel_func, context_prompts, max_workers=16)
        else:
            # Take all {{ }} and replace with the new type values using regex

            def replace_double_mustache(text: str) -> str:
                return pattern.sub("{{" + "/".join(config.task_config.type_values) + "}}", text)

            res = []
            for context, prompt in context_prompts:
                prompt = replace_double_mustache(question_contexts[0][0])
                res.append(((context), prompt))

        new_questions: List[Question] = []

        for context, new_question in res:
            if new_question:
                new_questions.append(
                    Question(
                        example=new_question,
                        superdomain=context[0][1],
                        domain=context[0][2],
                        topic=context[0][3],
                    )
                )

        question_dict = dict([(q.get_id(), q) for q in new_questions])

        # New Questionaire
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

        print(f"Loaded {len(test_questions)} transferred questions from {out_path}")
        print(f"Successfully transferred questions from {source_bias} bias to {target_bias} bias")
