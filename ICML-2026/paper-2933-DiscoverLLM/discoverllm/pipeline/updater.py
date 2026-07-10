import copy
import random
from typing import Any, Dict, List

from discoverllm.core.tasks.evaluate_message import evaluate_message
from discoverllm.pipeline.base import LLMPipeline
from discoverllm.utils import filter_hierarchy, is_skip_node, update_hierarchy


class Updater(LLMPipeline):
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        update_probability: float,
        verbose: bool
    ):
        super().__init__(model_name, temperature, max_tokens, verbose)
        self.update_probability = update_probability

    def update_mental_models(self, criterion_obj: Dict[str, Any], hierarchy_evaluation: List[Dict[str, Any]], evaluation_type: str) -> Dict[str, Any]:
        node_id_to_eval_result = {}
        for ev in hierarchy_evaluation:
            node_id_to_eval_result[ev["node_id"]] = ev

        # Nodes to skip: nodes that were already fully satisfied/probed before
        nodes_to_skip = [node["id"] for node in filter_hierarchy(criterion_obj["hierarchy"], is_skip_node)]

        def _is_node_update(node: Dict[str, Any], parent_node: Dict[str, Any]) -> bool:
            is_update = (
                node["id"] not in nodes_to_skip # not skipped
                and node["id"] in node_id_to_eval_result # was evaluated
                and (
                    not parent_node # is a root node
                    or parent_node["id"] in nodes_to_skip # parent was skipped
                    or (
                        # parent was evaluated and satisfied/probed (cascade has not stopped)
                        parent_node["id"] in node_id_to_eval_result
                        and node_id_to_eval_result[parent_node["id"]]["is_satisfied_or_probed"]
                    )
                )
            )
            if not is_update and node["id"] in node_id_to_eval_result:
                del node_id_to_eval_result[node["id"]] # Prevent incorrect cascades
            return is_update

        # Nodes that should be updated with new evaluation results
        nodes_to_update = [
            node["id"]
            for node in filter_hierarchy(
                criterion_obj["hierarchy"],
                lambda node, parent_node: _is_node_update(node, parent_node),
            )
        ]

        def _update_node(node: Dict[str, Any]) -> Dict[str, Any]:
            node["update"] = None
            if node["id"] in nodes_to_skip:
                node["reason"] = ""
                node["update"] = None
                node["update_thresholds"] = []
                return node
            # Nodes to update with the evaluation results
            if node["id"] in nodes_to_update:
                evaluation = node_id_to_eval_result[node["id"]]
                if evaluation["is_satisfied_or_probed"]:
                    node["update_thresholds"] = []
                    if evaluation_type == "probing" and node["aware"] < 1:
                        node["aware"] = 1
                        node["update"] = {"type": "awareness", "status": 1}
                        node["reason"] = evaluation["reasoning"]
                    if evaluation_type == "satisfaction" and node["satisfied"] == 0:
                        node["satisfied"] = 1
                        node["aware"] = 1
                        node["update"] = {"type": "satisfaction", "status": 1}
                        node["reason"] = evaluation["reasoning"]
                else:
                    if "update_thresholds" not in node:
                        node["update_thresholds"] = [random.random() for _ in range(2)]
                    if len(node["update_thresholds"]) > 0 and "near_miss" in evaluation:
                        update_score = self.update_probability * len(evaluation.get("near_miss"))
                        update_threshold = node["update_thresholds"][0]
                        if update_score >= update_threshold:
                            # Update threshold was passed, so increase awareness
                            print(f"Update threshold for node {node['id']} passed: {update_threshold} < {update_score}")
                            node["aware"] += 0.5 if node["aware"] < 1 else 0
                            # Remove the used update threshold
                            node["update_thresholds"].pop(0)
                        else:
                            # Update threshold was not passed, so we will decrease it based on the score
                            node["update_thresholds"][0] = update_threshold - update_score
                            print(f"Update threshold for node {node['id']} decreased: {update_threshold} -> {node['update_thresholds'][0]}")
                    if evaluation_type == "satisfaction":
                        if node["satisfied"] == 1:
                            node["update"] = {"type": "satisfaction", "status": 0}
                        node["satisfied"] = 0
                        node["reason"] = evaluation["reasoning"]
                return node
            else:
                node["satisfied"] = 0
                node["reason"] = ""
                node["update"] = None
                return node

        criterion_obj = copy.deepcopy(criterion_obj)
        criterion_obj["hierarchy"] = update_hierarchy(criterion_obj["hierarchy"], _update_node)

        return criterion_obj

    def __call__(
        self,
        chat_history: List[Dict[str, str]],
        criteria_objs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update the criteria objects based on chat history and assessments/evaluations on the criteria.
        """
        evaluations, evaluation_type = evaluate_message(
            chat_history=chat_history,
            criteria_objs=criteria_objs,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )

        for criterion_idx, evaluation_results in enumerate(evaluations):
            try:
                criterion_obj = criteria_objs[criterion_idx]
            except IndexError:
                print(f"⚠️  Warning: Criterion {criterion_idx} not found in criteria_objs")
                continue
            criteria_objs[criterion_idx] = self.update_mental_models(
                criterion_obj,
                evaluation_results["evaluations"],
                evaluation_type
            )

        return criteria_objs
