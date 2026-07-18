from typing import List, Dict
import re
import openai

from src.configs import ModelConfig
from .model import APIModel


def regex_replace(text: str) -> str:
    pattern = r"\{\{\s*([^/}]+(?:\s*/\s*[^/}]+)*)\s*\}\}"

    # Get the question from the input
    question = text

    og_matches = re.findall(pattern, question)

    if not og_matches:
        raise ValueError("The question does not contain any valid {{option1/option2/...}} format.")

    matches = [match.split("/") for match in og_matches]

    formatted_matches = []

    for match_group in matches:
        # if the last element has one more word append this word to every element

        new_match_group = []
        for elem in match_group:
            # Remove leading and trailing whitespaces
            elem = elem.strip()
            new_match_group.append(elem)

        elem_lengths = [len(elem.split(" ")) for elem in new_match_group]
        if (
            all(elem_length == elem_lengths[0] for elem_length in elem_lengths[:-1])
            and elem_lengths[-1] == elem_lengths[0] + 1
        ):
            new_match_group = [
                elem + " " + new_match_group[-1].split(" ")[-1] for elem in new_match_group[:-1]
            ]
            new_match_group.append(new_match_group[-1])

        formatted_matches.append(new_match_group)

    formatted_matches

    # Assert consistency in the number of options
    match_lens = [len(match) for match in formatted_matches]
    if not all(match_len == match_lens[0] for match_len in match_lens):
        raise ValueError("The question does not contain a consistent number of options.")

    # Create first and second completions
    completions = [question] * match_lens[0]

    for j, match_group in enumerate(og_matches):
        form_matches = formatted_matches[j]
        for i, match in enumerate(form_matches):
            completions[i] = re.sub(
                rf"\{{{{{re.escape(match_group)}}}}}", match, completions[i], count=1
            )

    return completions


class LocalReplaceModel(APIModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def _predict_call(self, input: List[Dict[str, str]], **kwargs) -> str:
        pattern = r"\{\{\s*([^/}]+(?:\s*/\s*[^/}]+)*)\s*\}\}"

        # Get the question from the input
        question = input[1]["content"]

        og_matches = re.findall(pattern, question)

        if not og_matches:
            raise ValueError(
                "The question does not contain any valid {{option1/option2/...}} format."
            )

        matches = [match.split("/") for match in og_matches]

        formatted_matches = []

        for match_group in matches:
            # if the last element has one more word append this word to every element

            new_match_group = []
            for elem in match_group:
                # Remove leading and trailing whitespaces
                elem = elem.strip()
                new_match_group.append(elem)

            elem_lengths = [len(elem.split(" ")) for elem in new_match_group]
            if (
                all(elem_length == elem_lengths[0] for elem_length in elem_lengths[:-1])
                and elem_lengths[-1] == elem_lengths[0] + 1
            ):
                new_match_group = [
                    elem + " " + new_match_group[-1].split(" ")[-1] for elem in new_match_group[:-1]
                ]
                new_match_group.append(new_match_group[-1])

            formatted_matches.append(new_match_group)

        formatted_matches

        # Assert consistency in the number of options
        match_lens = [len(match) for match in formatted_matches]
        if not all(match_len == match_lens[0] for match_len in match_lens):
            raise ValueError("The question does not contain a consistent number of options.")

        # Create first and second completions
        completions = [question] * match_lens[0]

        for j, match_group in enumerate(og_matches):
            form_matches = formatted_matches[j]
            for i, match in enumerate(form_matches):
                completions[i] = re.sub(
                    rf"\{{{{{re.escape(match_group)}}}}}", match, completions[i], count=1
                )

        return completions
