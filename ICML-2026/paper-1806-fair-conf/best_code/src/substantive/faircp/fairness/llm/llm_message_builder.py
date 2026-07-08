from abc import ABC, abstractmethod
from typing import List
from openai.types.chat import ChatCompletionMessageParam


class LlmMessageBuilder(ABC):
    def __init__(self, cfg: dict, label_map: dict[int, str]):
        self.label_map = label_map

        self.user_prompt_template: str = cfg["llm_user_prompt"]
        self.user_prompt_control_template: str = cfg["llm_user_prompt_control"]
        self.user_prompt_cvg: str | None = cfg.get("llm_user_prompt_cvg")
        self.user_prompt_no_cvg: str = cfg.get("llm_user_prompt_no_cvg")

        system_instruct_template: str = cfg["llm_system_instruct"]
        self.system_instruct: str = system_instruct_template.format(
            all_options=", ".join(self.label_map.values())
        )

    @abstractmethod
    def construct_messages(
        self,
        prompt: str,
        predictions: List[int],
        coverage: str,
    ) -> List[ChatCompletionMessageParam]:
        """
        Returns a list of messages suitable for openai.chat.completions.create.
        Can include text, images, or audio depending on subclass implementation.
        """
        pass

    @abstractmethod
    def construct_control_messages(
        self,
        prompt: str,
    ) -> List[ChatCompletionMessageParam]:
        """
        Returns a list of messages suitable for openai.chat.completions.create.
        Can include text, images, or audio depending on subclass implementation.
        """
        pass

    def format_question(self, predictions: list[int], coverage: str) -> str:
        options = ",".join(
            f"{self.label_map.get(i, f'Option {i}')}" for i in predictions
        )
        if self.user_prompt_cvg is None:
            coverage_info = ""
        else:
            coverage_info = (
                self.user_prompt_cvg.format(coverage=coverage)
                if coverage != "-1"
                else self.user_prompt_no_cvg
            )
        return self.user_prompt_template.format(
            options=options.strip(), coverage_info=coverage_info
        )


class VisionLlmMessageBuilder(LlmMessageBuilder):
    """Base class for image-based LLM message builders."""

    def __init__(self, cfg: dict, label_map: dict[int, str]):
        super().__init__(cfg, label_map)

    def construct_messages(
        self, prompt: str, predictions: List[int], coverage: str
    ) -> List[ChatCompletionMessageParam]:
        user_prompt = self.format_question(predictions, coverage)
        return [
            {"role": "system", "content": self.system_instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(prompt)},
                    },
                ],
            },
        ]

    def construct_control_messages(
        self, prompt: str
    ) -> List[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": self.system_instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt_control_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(prompt)},
                    },
                ],
            },
        ]

    def _encode_image(self, prompt: str) -> str:
        """Encode image to base64 data URI. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _encode_image method")
