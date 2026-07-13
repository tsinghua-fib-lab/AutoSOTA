from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from typing import List, Tuple, Union, Dict
from copy import deepcopy
import pandas as pd

from rlie.datasets.tasks import BaseTask

from rlie.utils.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("Prompt")

class BasePrompt(ABC):
    """
    This class gives us a way to conviniently generate prompts.
    """

    def __init__(self, task: BaseTask):
        self.task = task

    def _is_multi_content(self, obj) -> bool:
        """
        Check if the object is a multi content object

        Parameters:
            obj: Object to be checked

        Returns:
            bool: True if the object is a multi content object
        """
        return isinstance(obj, dict) and "multi_content" in obj

    def _fill_multi_content(
        self,
        substitute_dicts: Union[List, Tuple[Dict, Union[List, Tuple]]],
        multi_content,
    ) -> str:
        """
        Fill multi content with the given substitute_dicts

        Parameters:
            substitute_dicts: List of substitute dictionaries or
                Tuple of sub dict for prefix and suffix and `substitute_dicts` for multi content
            multi_content: Multi content to be filled.
                An `multi_content` object can be a string or
                a dictionary with keys `multi_content`, `prefix` (optional) and `suffix` (optional).
                If the length of `substitute_dicts` for key `multi_content` is 0, then it will return empty string.

        Returns:
            res: Filled multi content
        """
        res = ""
        if isinstance(multi_content, str):
            for idx, substitute_dict in enumerate(substitute_dicts):
                res += self._substitute_obj(
                    {"idx": idx + 1, **substitute_dict}, multi_content
                )
        elif isinstance(multi_content, dict):
            if len(substitute_dicts[1]) == 0:
                return ""

            if "prefix" in multi_content:
                res += self._substitute_obj(
                    substitute_dicts[0], multi_content["prefix"]
                )
            # Must have multi content
            res += self._fill_multi_content(
                substitute_dicts[1], multi_content["multi_content"]
            )
            if "suffix" in multi_content:
                res += self._substitute_obj(
                    substitute_dicts[0], multi_content["suffix"]
                )
        return res

    def _get_substitute_key(self, template_str: Union[str, List, Dict]):
        keys = set()
        if isinstance(template_str, str):
            template = Template(template_str)
            for group in template.pattern.findall(template_str):
                for it in group:
                    if it:
                        keys.add(it)
        elif isinstance(template_str, list):
            for item in template_str:
                keys.update(self._get_substitute_key(item))
        elif isinstance(template_str, dict):
            for value in template_str.values():
                keys.update(self._get_substitute_key(value))
        else:
            raise ValueError(f"Invalid template type {type(template_str)}")
        return list(keys)

    def _get_substitute_dict(
        self, data_dict: pd.DataFrame, example_idx
    ) -> Dict[str, str]:
        substitute_dict = data_dict.loc[example_idx].to_dict()
        return substitute_dict

    def _substitute_obj(
        self, substitute_dict: Dict[str, str], obj: Union[str, List, Dict]
    ):
        if isinstance(obj, str):
            return Template(obj).substitute(substitute_dict)
        elif isinstance(obj, list):
            return [self._substitute_obj(substitute_dict, o) for o in obj]
        elif isinstance(obj, dict):
            return {k: self._substitute_obj(substitute_dict, v) for k, v in obj.items()}

    def _information_prompt(
        self, substitute_dict: Dict[str, str], info_key: str
    ) -> Dict[str, str]:
        prompt = self._convert_to_messages(info_key)
        return self._substitute_obj(substitute_dict, prompt)

    def _get_prompt_template(self, key: str) -> Union[str, List[Dict[str, str]], Dict]:
        return deepcopy(self.task.prompt_template[key])

    def _convert_to_messages(self, key: str) -> List[Dict]:
        messages = [
            {"role": kk, "content": vv}
            for kk, vv in self.task.prompt_template[key].items()
        ]
        return messages

    def _fill_multi_in_sub_dict(
        self,
        init_dict: Dict[str, str],
        multi_sub_dicts: Dict[str, List[Dict[str, str]]],
        prompt_key: str,
    ):
        substitute_dict = init_dict
        keys = self._get_substitute_key(self._get_prompt_template(prompt_key))
        keys = [key for key in keys if key not in substitute_dict]

        for key in keys:
            if self._is_multi_content(self._get_prompt_template(key)):
                substitute_dict[key] = self._fill_multi_content(
                    ({}, multi_sub_dicts[key]),
                    self._get_prompt_template(key),
                )
            else:
                substitute_dict[key] = self._get_prompt_template(key)

        return substitute_dict

    def batched_generation(self, train_data, num_hypotheses):
        """
        Generate hypotheses that is useful for predicting the color of the shoes given the appearance of the person.
        """

        substitute_dict = {"num_hypotheses": num_hypotheses}

        multi_sub_dicts = {"observations": []}
        for example_idx in range(len(train_data)):
            multi_sub_dicts["observations"].append(
                self._get_substitute_dict(train_data, example_idx)
            )
        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "batched_generation"
        )

        prompt = self._information_prompt(substitute_dict, "batched_generation")

        return prompt

    def batched_generation_refine(
        self,
        train_data,
        num_hypotheses,
        hypotheses_text: str,
    ):
        substitute_dict = {
            "num_hypotheses": num_hypotheses,
            "hypotheses_text": hypotheses_text,
        }
        multi_sub_dicts = {"observations": []}
        for example_idx in range(len(train_data)):
            multi_sub_dicts["observations"].append(
                self._get_substitute_dict(train_data, example_idx)
            )
        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict,
            multi_sub_dicts,
            "batched_generation_refine",
        )
        prompt = self._information_prompt(
            substitute_dict,
            "batched_generation_refine",
        )
        return prompt

    def inference(self, hypotheses_dict, test_data, test_idx):
        """
        Create inference prompt.
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        prompt = self._information_prompt(substitute_dict, "inference")

        return prompt

    def multiple_hypotheses_inference(
        self,
        hypotheses: List[str],
        test_data: pd.DataFrame,
        test_idx: int,
    ) -> List[Dict[str, str]]:
        """Create inference prompt that includes a bank of hypotheses."""

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        formatted = []
        for idx, hypothesis in enumerate(hypotheses, start=1):
            formatted.append(f"{idx}. {hypothesis}")
        substitute_dict["hypotheses"] = "\n".join(formatted)

        prompt = self._information_prompt(
            substitute_dict,
            "multiple_hypotheses_inference",
        )

        return prompt

    def multiple_hypotheses_inference_with_linear_regression_and_label(
        self,
        weighted_hypotheses: List[Tuple[str, float]],
        bias: float,
        model_prediction: str,
        test_data: pd.DataFrame,
        test_idx: int,
    ) -> List[Dict[str, str]]:
        """Create inference prompt that exposes a model-provided label and weights."""

        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        formatted = []
        for idx, (hypothesis, weight) in enumerate(weighted_hypotheses, start=1):
            formatted.append(f"{idx}. {hypothesis} (weight = {weight:.4f})")
        substitute_dict["weighted_hypotheses"] = "\n".join(formatted)
        substitute_dict["bias"] = f"{bias:.4f}"
        substitute_dict["model_prediction"] = model_prediction

        prompt = self._information_prompt(
            substitute_dict,
            "multiple_hypotheses_inference_with_linear_regression_and_label",
        )

        return prompt

    def multiple_hypotheses_inference_with_linear_regression(
        self,
        weighted_hypotheses: List[Tuple[str, float]],
        bias: float,
        test_data: pd.DataFrame,
        test_idx: int,
    ) -> List[Dict[str, str]]:
        """Create inference prompt that exposes linear-regression weights without suggested label."""

        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        formatted = []
        for idx, (hypothesis, weight) in enumerate(weighted_hypotheses, start=1):
            formatted.append(f"{idx}. {hypothesis} (weight = {weight:.4f})")
        substitute_dict["weighted_hypotheses"] = "\n".join(formatted)
        substitute_dict["bias"] = f"{bias:.4f}"

        prompt = self._information_prompt(
            substitute_dict,
            "multiple_hypotheses_inference_with_linear_regression",
        )

        return prompt

    def one_step_adaptive_inference(
        self, hypotheses_dict, train_data, test_data, test_idx
    ):
        """
        One step adaptive inference prompt
        """

        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        multi_sub_dicts = {"adaptive_info_prompt": []}
        hyp_idx = 0
        for _, hypothesis_class in hypotheses_dict.items():
            hypothesis_related_examples = hypothesis_class.correct_examples
            hyp_substitute_dict = {
                "hypothesis_text": hypothesis_class.hypothesis,
                "idx": hyp_idx + 1,
            }
            observations_dict = {"observations": []}
            for example_info in hypothesis_related_examples:
                observations_dict["observations"].append(
                    self._get_substitute_dict(train_data, example_info[0])
                )
            
            hyp_substitute_dict = self._fill_multi_in_sub_dict(
                hyp_substitute_dict, observations_dict, "adaptive_info_prompt"
            )
            multi_sub_dicts["adaptive_info_prompt"].append(
                hyp_substitute_dict
            )
            hyp_idx += 1

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "adaptive_inference"
        )

        prompt = self._information_prompt(substitute_dict, "adaptive_inference")
        logger.debug(f"System prompt: {prompt[0]['content']}")
        logger.debug(f"User prompt: {prompt[1]['content']}")

        return prompt

    def adaptive_selection(self, hypotheses_dict, train_data, test_data, test_idx):
        """
        Hypothesis selection prompt for two step adaptive inference
        """

        substitute_dict = self._get_substitute_dict(test_data, test_idx)

        multi_sub_dicts = {"adaptive_info_prompt": []}
        hyp_idx = 0
        for _, hypothesis_class in hypotheses_dict.items():
            hypothesis_related_examples = hypothesis_class.correct_examples
            hyp_substitute_dict = {
                "hypothesis_text": hypothesis_class.hypothesis,
                "idx": hyp_idx + 1,
            }
            observations_dict = {"observations": []}
            for example_info in hypothesis_related_examples:
                observations_dict["observations"].append(
                    self._get_substitute_dict(train_data, example_info[0])
                )
            
            hyp_substitute_dict = self._fill_multi_in_sub_dict(
                hyp_substitute_dict, observations_dict, "adaptive_info_prompt"
            )
            multi_sub_dicts["adaptive_info_prompt"].append(
                hyp_substitute_dict
            )
            hyp_idx += 1

        substitute_dict = self._fill_multi_in_sub_dict(
            substitute_dict, multi_sub_dicts, "adaptive_selection"
        )

        prompt = self._information_prompt(substitute_dict, "adaptive_selection")
        logger.debug(f"System prompt: {prompt[0]['content']}")
        logger.debug(f"User prompt: {prompt[1]['content']}")
        return prompt

    def is_relevant(self, hypotheses_dict, test_data, test_idx):
        """
        Check if a hypothesis is relevant to a specific example
        """

        hypothesis = list(hypotheses_dict.keys())[0]

        substitute_dict = self._get_substitute_dict(test_data, test_idx)
        substitute_dict["hypothesis"] = hypothesis

        prompt = self._information_prompt(substitute_dict, "is_relevant")

        return prompt
