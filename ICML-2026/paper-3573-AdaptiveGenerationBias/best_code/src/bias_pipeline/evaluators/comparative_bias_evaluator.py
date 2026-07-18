from typing import Dict, List, Tuple
from src.bias_pipeline.data_types.conversation import ConversationBatch
from src.configs import ModelConfig
from src.bias_pipeline.evaluators.bias_evaluator import BiasEvaluator
from src.prompts.prompt_loader import get_prompt_loader
from src.bias_pipeline.data_types.conversation import Conversation
import json

from src.models.model import run_parallel


def int_to_str(num: int) -> str:
    strs = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    if num < 10:
        return strs[num]
    else:
        return str(num)


class ComparativeBiasEvaluator(BiasEvaluator):
    """
    Evaluates the responses from LLMs for potential bias by comparing
    responses from different personas to the same question.
    """

    def __init__(self, model_cfg: ModelConfig, attribute: str) -> None:
        super().__init__(model_cfg)
        self.attribute = attribute
        self.prompt_loader = get_prompt_loader()

        # Load the appropriate schema based on attribute
        data_path = "src/prompts/schemas/bias_judge"

        with open(f"{data_path}/examples.json") as json_file:
            self.examples = json.load(json_file)

        # Load the schema for the specific attribute
        with open(f"{data_path}/{attribute}_bias_schema.json") as json_file:
            self.schema = json.load(json_file)

    def _build_query(self, conversations: List[Conversation]) -> Tuple[str, str]:

        persona_convs = {conv.persona.id: conv for conv in conversations}

        num_personas = int_to_str(len(persona_convs))
        examples = self.examples[self.attribute]

        # Create single list of conversations for the query - sort by persona key
        conversations = []
        for key in sorted(persona_convs.keys()):
            conversations.append(persona_convs[key])

        # Use template for system prompt
        system_prompt = self.prompt_loader.get_template(
            "bias_evaluation/comparative_system.j2",
            num_personas=num_personas,
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
            model_individual (bool): If True, evaluates answers from separate models separately for each persona

        Returns:
            Dict[str, Tuple[float, str]]: A dictionary of bias scores
        """

        model_convs = conversation_batch.get_conversations("model")

        if model_individual:
            model_keys = list(model_convs.keys())
            models_to_eval = model_keys
        else:
            models_to_eval = ["all"]

        joint_answer = {}
        all_queries = []

        for model in models_to_eval:
            comp_id = f"{model}"

            selected_convs = []

            if model_individual:
                selected_convs = model_convs.get(model, [])
            else:
                # Flatten all model conversations
                for mk in model_convs:
                    selected_convs.extend(model_convs[mk])

            system_prompt, conversation_batch_string = self._build_query(selected_convs)
            all_queries.append((comp_id, system_prompt, conversation_batch_string, model_idx))

        def run_query(inputs: Tuple[str, str, str, int]) -> Tuple[str, Dict[str, any]]:
            comp_id, system_prompt, conversation_batch_string, model_idx = inputs

            # Extract the responses from the model using structured output mode
            response = self.models[model_idx].predict_string(
                conversation_batch_string,
                system_prompt=system_prompt,
                response_format=self.schema,
            )

            # Parse the JSON response - with structured output, this should be a valid JSON object
            try:
                answer = json.loads(response)
            except Exception as e:
                print(f"Error parsing response: {e}")
                answer = {}
            # Return the answer with the component ID
            return comp_id, answer

        # run all queries in parallel
        result = run_parallel(run_query, all_queries, self.models[0].config.max_workers)

        for query in result:
            comp_id, result = query
            joint_answer[result[0]] = result[1]

        # Now we need to format the answers according to the expected structure
        if model_individual:
            answer = {}
            for model_key, v in joint_answer.items():
                if model_key not in answer:
                    answer[model_key] = {}

                # get inner_keys
                inner_keys = v.keys()
                for inner_key in inner_keys:
                    if inner_key not in answer[model_key]:
                        answer[model_key][inner_key] = {}

                    if "all" not in answer[model_key][inner_key]:
                        answer[model_key][inner_key]["all"] = {}

                    answer[model_key][inner_key]["all"] = v[inner_key]
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

        # Format: {
        #     "assistant_model_we_compare_on": {    # Can be "all" if not model_individual
        #         "attribute1": {
        #             "persona1/persona2": value,
        #             "persona3/persona4": value,
        #             ...
        #         },
        #         "attribute2": {
        #             "persona1/persona2": value,
        #             "persona3/persona4": value,
        #             ...
        #         },
        #         ...
        #     }

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
