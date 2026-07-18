from typing import Any, Dict, Optional, List, Tuple
from src.bias_pipeline.data_types.conversation import ConversationBatch
from src.bias_pipeline.data_types.data_types import Annotation
from src.models import BaseModel, get_model, run_parallel
from src.configs import ModelConfig
from abc import ABC, abstractmethod
from itertools import product


class BiasEvaluator(ABC):
    """
    Evaluates the responses from LLMs for potential bias by comparing
    responses from different personas to the same question.
    """

    def __init__(self, model_cfgs: ModelConfig | List[ModelConfig]) -> None:
        if isinstance(model_cfgs, list):
            self.models = []
            self.model_cfgs = model_cfgs
            for model_cfg in model_cfgs:
                self.models.append(get_model(model_cfg))
        else:
            self.model_cfgs = [model_cfgs]
            self.models = [get_model(model_cfgs)]

    @abstractmethod
    def evaluate_bias_conversation_batch(
        self, conversation_batch: ConversationBatch, model_idx: int, model_individual: bool = False
    ) -> Dict[str, Tuple[float, str]]:
        """Evaluates the individual ConversationBatch for bias."""
        pass

    def evaluate_bias_conversation(
        self, conversation_batches: List[ConversationBatch], model_individual: bool = False
    ) -> List[Dict[str, Tuple[float, str]]]:
        """Evaluates the individual ConversationBatches for bias within each respectively.

        Args:
            conversation_batches (List[ConversationBatch]): A list of ConversationBatch instances.
            model_individual (bool): If True, evaluates answers from separate models separately. E.g. one for the GPT-4 model and one for the GPT-3.5 model.

        Returns:

        """

        def func_bias(inputs: Tuple[int, ConversationBatch]) -> Dict[str, Tuple[float, str]]:
            model_idx, conversation_batch = inputs

            attribute_dict = self.evaluate_bias_conversation_batch(
                conversation_batch, model_idx, model_individual
            )

            return attribute_dict

        # Score each conversation with each model
        tuples = list(product(range(len(self.models)), conversation_batches))

        # func_bias(tuples[0])  # Warm up the model

        result = run_parallel(func_bias, tuples, self.models[0].config.max_workers)

        # Store annotations
        for tup, bias_result in result:
            model_idx, conversation_batch = tup

            if model_individual:
                for model_name, bias_res in bias_result.items():
                    annotation = Annotation(
                        model_id=self.model_cfgs[model_idx].name,
                        annotation_subset=model_name,  # Use the model name as the subset
                        annotation=bias_res,
                    )

                    conversation_batch.annotate_curr_state(annotation, subset_key=model_name)
            else:
                assert "all" in bias_result, (
                    "Bias result must contain 'all' key for non-individual models"
                )
                combined_model_names = list(
                    set(
                        conversation.model.name for conversation in conversation_batch.conversations
                    )
                )
                # sort the model names to ensure consistent ordering
                combined_model_names = "-".join(sorted(combined_model_names))

                annotation = Annotation(
                    model_id=self.model_cfgs[model_idx].name,
                    annotation_subset=combined_model_names,  # Use the combined model names
                    annotation=bias_result["all"],
                )

                conversation_batch.annotate_curr_state(annotation, subset_key=combined_model_names)

        return result

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
