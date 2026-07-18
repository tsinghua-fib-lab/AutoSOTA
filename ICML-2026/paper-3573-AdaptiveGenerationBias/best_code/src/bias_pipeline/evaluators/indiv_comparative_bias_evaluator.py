from typing import Dict, List, Tuple, Optional
from src.bias_pipeline.data_types.conversation import ConversationBatch, Conversation
from src.configs import ModelConfig
from src.bias_pipeline.evaluators.comparative_bias_evaluator import ComparativeBiasEvaluator
from itertools import product
import json
import re

from src.models.model import run_parallel


def int_to_str(num: int) -> str:
    strs = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    if num < 10:
        return strs[num]
    else:
        return str(num)


def _extract_json_from_llm(text: str) -> Optional[dict]:
    """Robustly extract JSON from chatty model output.
    Prefers ```json fenced blocks; falls back to first { ... } span.
    """
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i = text.find("{")
    if i >= 0:
        j = text.rfind("}")
        if j > i:
            candidate = text[i : j + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {}
    return {}


class IndivComparativeBiasEvaluator(ComparativeBiasEvaluator):
    """
    Evaluates the responses from LLMs for potential bias by comparing
    responses from different personas to the same question. Splits multiple ABCD conversations into
    individual conversations AB AC AD BC BD CD
    and evaluates them separately.
    """

    def __init__(self, model_cfg: ModelConfig, attribute: str) -> None:
        super().__init__(model_cfg, attribute)

    def _build_query(self, conversations: List[Conversation]) -> Tuple[str, str]:
        examples = self.examples[self.attribute]

        # Use template for system prompt
        system_prompt = self.prompt_loader.get_template(
            "bias_evaluation/comparative_system.j2",
            num_personas=2,
            examples=examples,
            attribute=self.attribute,
        )

        # Use template for query
        query = self.prompt_loader.get_template(
            "bias_evaluation/comparative_query.j2", conversations=conversations
        )

        return system_prompt, query

    def evaluate_bias_conversation_batch(
        self, conversation_batch: ConversationBatch, model_idx: int, model_individual: bool = False
    ) -> Dict[str, Dict[str, Tuple[float, str]]]:
        """Evaluates the individual ConversationBatch for bias.

        Args:
            conversation_batch (ConversationBatch): A ConversationBatch instance.
            model_idx (int): The index of the model to use for evaluation.
            model_individual (bool): If True, evaluates answers from separate models separately for eacnh persona

        Returns:
            Dict[str, Tuple[float, str]]: A dictionary of bias scores
        """

        persona_convs = conversation_batch.get_conversations("persona")
        model_convs = conversation_batch.get_conversations("model")

        personas = list(persona_convs.keys())
        tuples = list(product(range(len(personas)), range(len(personas))))
        tuples = [(tup[0], tup[1], -1) for tup in tuples if tup[0] < tup[1]]

        if model_individual:
            model_keys = list(model_convs.keys())
            full_tuples = []
            for tup in tuples:
                for i in range(len(model_keys)):
                    full_tuples.append(
                        (tup[0], tup[1], i)
                    )  # (A,B,C) - Compare user A with user B for model C
            tuples = full_tuples

        joint_answer = {}

        all_queries = []

        for tup in tuples:
            comp_id = f"{personas[tup[0]]}/{personas[tup[1]]}"
            if model_individual:
                comp_id = f"{personas[tup[0]]}/{personas[tup[1]]}/{model_keys[tup[2]]}"

            pers1_convs = persona_convs[personas[tup[0]]]
            pers2_convs = persona_convs[personas[tup[1]]]
            conversations = pers1_convs + pers2_convs

            if model_individual:
                # Select conversations where the model is the same for both personas
                model_key = model_keys[tup[2]]
                conversations = [conv for conv in conversations if conv.model.name == model_key]

            # Create the query for the model

            system_prompt, conversation_batch_string = self._build_query(conversations)

            all_queries.append((comp_id, system_prompt, conversation_batch_string, model_idx))

        def run_query(inputs: Tuple[str, str, str, int]) -> Tuple[str, Dict[str, any]]:
            comp_id, system_prompt, conversation_batch_string, model_idx = inputs

            # if the model is from openai use response_format
            if self.models[model_idx].config.provider == "openai":
                response = self.models[model_idx].predict_string(
                    conversation_batch_string,
                    system_prompt=system_prompt,
                    response_format=self.schema,
                )

                try:
                    answer = json.loads(response)
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    answer = {}
                # Return the answer with the component ID
                return comp_id, answer

            else:
                system_prompt += "\nOnly respond with a JSON object in the format shown above, don't output anything else besides this JSON.\n"

                response = self.models[model_idx].predict_string(
                    conversation_batch_string,
                    system_prompt=system_prompt,
                )
                try:
                    # Try to extract the JSON part from the response
                    answer = _extract_json_from_llm(response)
                except ValueError:
                    # If no JSON part is found, return an empty dict
                    answer = {}

                return comp_id, answer

            # Parse the JSON response - with structured output, this should be a valid JSON object

        # NOTE: Not threaded because we already have an outer threadpool for calling this function
        joint_answer = {}

        # run all queries in parallel
        result = run_parallel(run_query, all_queries, self.models[0].config.max_workers)

        result = list(result)  # Convert from generator to list

        for query in result:
            comp_id, inner_result = query
            joint_answer[inner_result[0]] = inner_result[1]

            if not inner_result[1]:
                print(f"Warning: No result for {comp_id}")

        # Now we need to merge the answers

        if model_individual:
            # If model_individual is True, we need to merge the answers for each model separately
            # The keys will be in the format "persona1/persona2/model_name"
            # We need to split them into "persona1/persona2" and "model_name"
            answer = {}
            for k, v in joint_answer.items():
                # Get the keys
                keys = k.split("/")
                if len(keys) < 3:
                    continue
                persona_key = f"{keys[0]}/{keys[1]}"
                model_key = "/".join(keys[2:])  # Join the rest as the model key

                if model_key not in answer:
                    answer[model_key] = {}

                # get inner_keys
                try:
                    inner_keys = v.keys()
                except Exception as e:
                    print(f"Error: {v} is not a dict")
                    raise e
                inner_keys = v.keys()
                for inner_key in inner_keys:
                    if inner_key not in answer[model_key]:
                        answer[model_key][inner_key] = {}

                    if persona_key not in answer[model_key][inner_key]:
                        answer[model_key][inner_key][persona_key] = {}

                    answer[model_key][inner_key][persona_key] = v[inner_key]

        else:
            model_key = "all"
            answer = {}
            for k, v in joint_answer.items():
                # Get the keys
                keys = v.keys()
                for key in keys:
                    if key not in answer:
                        answer[key] = {}
                    # Merge the answers
                    answer[key][k] = v[key]
            answer = {model_key: answer}

        return answer

    def evaluate_bias_assistant(self, conversation_batches: List[ConversationBatch]) -> None:
        """Evaluates the assistant's responses for bias.

        Args:
            conversation_batches (List[ConversationBatch]): A list of ConversationBatch instances.
        """
        pass

    def evaluate_bias_full_conversation(
        self, conversation_batches: List[ConversationBatch]
    ) -> None:
        """Evaluates the conversation as a whole for bias.

        Args:
            conversation_batches (List[ConversationBatch]): A list of ConversationBatch instances.
        """
        pass

    def evaluate_bias_multiple_batches(self, conversation_batches: List[ConversationBatch]) -> None:
        """Evaluates multiple conversation batches for bias.

        Args:
            conversation_batches (List[ConversationBatch]): A list of ConversationBatch instances.
        """
        pass
