import random
from typing import Iterator, List, Dict, Any, TYPE_CHECKING, Tuple
from abc import ABC, abstractmethod
import json
import os


from src.bias_pipeline.data_types.data_types import Message
from src.bias_pipeline.questionaires.questionaire import Question

from src.configs import QuestionTransformerConfig

from src.models import get_model
from src.models import run_parallel
from src.prompts.prompt_loader import get_prompt_loader


class QuestionTransformer(ABC):
    def __init__(self, config: QuestionTransformerConfig):
        self.config = config
        self.model = get_model(config.model)
        self.prompt_loader = get_prompt_loader()

    @abstractmethod
    def transform(self, question: Question) -> List[Message]:
        """Transforms the input question according to the specific strategy."""
        pass

    def transform_batch(
        self, questions: List[Question]
    ) -> Iterator[Tuple[Question, List[Message]]]:
        """Transforms a batch of questions."""

        convs = run_parallel(self.transform, questions, self.config.model.max_workers)
        return convs


class ImplicitQuestionTransfomer(QuestionTransformer):
    def __init__(self, config: QuestionTransformerConfig):
        super().__init__(config)

    def transform(self, question: Question) -> List[Message]:
        prompt = self.config.prompt_template.format(question=question.text)
        response = self.model.generate(prompt)
        messages = "".join(response)
        return messages


class NameQuestionTransformer(QuestionTransformer):
    def __init__(self, config: QuestionTransformerConfig):
        super().__init__(config)
        self.name_samples: Dict[str, List[str] | Dict[str, List[str]]] = self._load_names()

    def _load_names(self) -> Dict[str, Any]:
        """Load names from the appropriate JSON file based on the attribute."""
        path = f"src/prompts/examples/names/{self.config.attribute}_names.json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Names file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def transform(self, question: Question) -> List[Message]:
        # Sample replacement options based on the attribute
        replacement_options = []
        options = self.name_samples.keys()

        if self.config.attribute == "gender":
            male_option = random.choice(self.name_samples["male"])
            female_option = random.choice(self.name_samples["female"])
            replacement_options = [male_option, female_option]
        elif self.config.attribute == "race" or self.config.attribute == "religion":
            # Flip coin if female or male names should be used
            gender_key = random.choice(["male", "female"])
            for key in self.config.expected_options:
                replacement_options.append(random.choice(self.name_samples[key][gender_key]))

        filled_user_prompt = self.prompt_loader.get_template(
            self.config.user_prompt_path,
            original_text=question.example,
            replacement_options=replacement_options,
        )
        filled_system_prompt = self.prompt_loader.get_template(
            self.config.system_prompt_path,
            type_values=self.config.expected_options,
        )

        response = self.model.predict_string(filled_user_prompt, system_prompt=filled_system_prompt)

        return_message = Message(
            text=response,
            sender="user",
        )

        return [return_message]


class ExtendedContextQuestionTransformer(QuestionTransformer):
    def __init__(self, config: QuestionTransformerConfig, context_sentences: List[str]):
        super().__init__(config)
        self.context_sentences = context_sentences

    def transform(self, question: Question) -> List[Message]:
        context = "\n".join(self.context_sentences)
        prompt = self.config.prompt_template.format(context=context, question=question.text)
        response = self.model.generate(prompt)
        messages = "".join(response)
        return messages


def get_question_transformer(
    config: QuestionTransformerConfig,
) -> QuestionTransformer:
    if config.type == "implicit":
        return ImplicitQuestionTransfomer(config)
    elif config.type in ["name", "names"]:
        return NameQuestionTransformer(config)
    elif config.type == "extended_context":
        return ExtendedContextQuestionTransformer(config)
    else:
        raise ValueError(f"Unknown transformer type: {config.type}")
