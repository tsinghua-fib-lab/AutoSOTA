import random
from abc import ABC, abstractmethod


class PromptFormatter(ABC):
    """Abstract class for formatting prompts."""

    @abstractmethod
    def format(self, task_description: str, prompt: str, *args, **kwargs) -> str:
        """Format a prompt string with given arguments."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the prompt formatter."""
        pass


class ZeroShot(PromptFormatter):
    """Zero-shot prompt formatter."""

    def format(self, task_description: str, prompt: str, *args, **kwargs) -> str:
        """Return the prompt as is."""
        return task_description + "\n\n" + prompt

    @property
    def name(self) -> str:
        """Return the name of the prompt formatter."""
        return "zero-shot"


class FewShot(PromptFormatter):
    """Few-shot prompt formatter."""

    def __init__(self, shots):
        self.shots = shots

    def format(
        self, task_description, prompt, questions, answers, *args, **kwargs
    ) -> str:
        """Prepend a few-shot prompt to the given prompt."""
        if self.shots >= len(questions):
            raise AssertionError(
                "Number of shots should be less than the number of questions."
            )
        QAs = random.sample(list(zip(questions, answers)), self.shots)
        few_header = "\n\n".join([f"{qa[0]}{qa[1]}" for qa in QAs])
        return f"{task_description}\n\n{few_header}\n\n{prompt}"

    @property
    def name(self) -> str:
        """Return the name of the prompt formatter."""
        return "few-shot"


class ChainOfThought(PromptFormatter):
    def format(self, task_description, prompt, cot, *args, **kwargs) -> str:
        """Prepend a chain-of-thought prompt to the given prompt."""
        return f"{task_description}\n\n{cot}\n\n{prompt}"

    @property
    def name(self) -> str:
        """Return the name of the prompt formatter."""
        return "chain-of-thought"
