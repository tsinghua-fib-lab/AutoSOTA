from typing import Any, Dict, List

from discoverllm.core.tasks.abstract_criteria import abstract_criteria
from discoverllm.core.tasks.hierarchize_criteria import hierarchize_criteria
from discoverllm.pipeline.base import LLMPipeline


def format_alternative(alt: Any) -> str:
    # if alt type is a dict, we should join "{key}: {value}" because it got processed incorreclty
    if isinstance(alt, dict):
        return ", ".join([f"{k}: {v}" for k, v in alt.items()])
    return alt


class Abstractor(LLMPipeline):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        verbose: bool,
        max_num_abstractions: int
    ):
        super().__init__(model_name, temperature, max_tokens, verbose)
        self.max_num_abstractions = max_num_abstractions
        self.verbose = verbose

    def __call__(self, criteria: List[Dict[str, Any]], artifact_type: str, artifact_topic: str) -> List[str]:
        """
        Loop to repetitively sample and abstract criteria until no more criteria can be abstracted.
        """
        criteria_objs = [
            {
                "criterion": criteria[i]['description'],
                "abstractions": [{
                    "description": criteria[i]['description'],
                    "checklist": [
                        { "item": item, "satisfied": 0, "aware": 0, "reason": "" } for item in criteria[i]['checklist']
                    ],
                    "lateral_alternatives": []
                }],
            } for i in range(len(criteria))
        ]

        num_abstractions_per_criteria = [self.max_num_abstractions for _ in range(len(criteria))]

        results = []
        results = abstract_criteria(
            criteria,
            artifact_type,
            artifact_topic,
            num_abstractions_per_criteria,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )

        for criterion_id, result in enumerate(results):
            for abs_idx, abstraction in enumerate(result["abstractions"]):
                criteria_objs[criterion_id]["abstractions"].insert(0, {
                    "description": abstraction["criterion"].strip(),
                    "checklist": [
                        {"item": item, "satisfied": 0, "aware": 0, "reason": "" } for item in abstraction["checklist"]
                    ],
                    "lateral_alternatives": []
                })

                if "is_final" in abstraction and abstraction["is_final"]:
                    break

        criteria_objs = hierarchize_criteria(
            criteria_objs,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )

        return criteria_objs

