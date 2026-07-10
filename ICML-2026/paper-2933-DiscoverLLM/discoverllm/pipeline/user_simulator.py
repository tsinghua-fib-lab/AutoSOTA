import copy
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Self

from discoverllm.core.tasks.create_criteria import create_criteria
from discoverllm.core.tasks.simulate_initial_request import simulate_initial_request
from discoverllm.core.tasks.simulate_user_response import simulate_user_response
from discoverllm.pipeline._criteria_tree import (
    _count_total_root_nodes,
    _extract_root_nodes_at_positions,
    _extract_root_nodes_up_to_position,
    _find_first_unsatisfied_root_position,
    _get_root_node_at_position,
    _is_root_node_tree_satisfied,
    _merge_root_node_back,
)
from discoverllm.pipeline.abstractor import Abstractor
from discoverllm.pipeline.base import LLMPipeline
from discoverllm.pipeline.updater import Updater


class UserSimulator(LLMPipeline):
    def __init__(
        self,
        artifact_type: str | None = None,
        artifact: str | None = None,
        criteria_objs: List[Dict[str, Any]] | None = None,
        initial_request: str | None = None,
        initial_chat_history: Optional[List[Dict[str, str]]] = None,
        initial_criteria_history: Optional[List[List[Dict[str, Any]]]] = None,
        model_name: str = "gpt-5-2025-08-07",
        seed_model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        update_probability: float = 0.25,
        max_num_abstractions: int = 5,
        max_num_criteria: int = 1,
        verbose: bool = False,
        single_dimension_focus: bool = True,
    ):
        super().__init__(model_name, temperature, max_tokens, verbose)
        self.seed_model_name = seed_model_name or model_name
        self.single_dimension_focus = single_dimension_focus
        self.updater = Updater(model_name, temperature, max_tokens, update_probability, verbose)
        self.abstractor = Abstractor(self.seed_model_name, temperature, max_tokens, verbose, max_num_abstractions)

        # Determine initialization mode (mutually exclusive)
        has_artifact = artifact is not None and artifact_type is not None
        has_criteria_init = criteria_objs is not None and initial_request is not None
        has_histories = initial_chat_history is not None and initial_criteria_history is not None and len(initial_chat_history) > 0 and len(initial_criteria_history) > 0
        modes_true = sum([1 if has_artifact else 0, 1 if has_criteria_init else 0, 1 if has_histories else 0])
        if modes_true != 1:
            raise ValueError("Provide exactly one of: (artifact_type & artifact), (criteria_objs & initial_request), or (initial_chat_history & initial_criteria_history)")

        if has_artifact:
            # Case (1): Build from artifact
            self.criteria, artifact_topic = create_criteria(
                artifact,
                artifact_type,
                max_num_criteria,
                self.seed_model_name,
                temperature,
                max_tokens,
                verbose
            )
            self.criteria_objs = self.abstractor(self.criteria, artifact_type, artifact_topic)
            self.initial_request, self.criteria_objs = simulate_initial_request(
                artifact_type,
                artifact_topic,
                self.criteria_objs,
                model_name=self.seed_model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose
            )
            self.chat_history = [{"role": "user", "content": self.initial_request}]
            self.criteria_history = [copy.deepcopy(self.criteria_objs)]
        elif has_criteria_init:
            # Case (2): Use provided criteria and initial request
            self.criteria_objs = copy.deepcopy(criteria_objs)
            self.criteria = [c.get("criterion", "") for c in self.criteria_objs]
            self.initial_request = initial_request
            self.chat_history = [{"role": "user", "content": self.initial_request}]
            self.criteria_history = [copy.deepcopy(self.criteria_objs)]
        else:
            # Case (3): Use provided histories
            self.chat_history = copy.deepcopy(initial_chat_history)
            self.criteria_history = copy.deepcopy(initial_criteria_history)
            # Derive state
            self.criteria_objs = copy.deepcopy(self.criteria_history[-1])
            self.criteria = [c.get("criterion", "") for c in self.criteria_objs]
            # Initial request from first user message
            self.initial_request = self.chat_history[0].get("content", "")

        # Track current root node position (flat index across all criteria)
        self.total_root_nodes = _count_total_root_nodes(self.criteria_objs)
        # When restoring from history, find the first unsatisfied root node
        if has_histories:
            self.current_root_position = _find_first_unsatisfied_root_position(self.criteria_objs)
        else:
            self.current_root_position = 0

    def clone(self) -> Self:
        """Create a fresh UserSimulator with the same state as the source without sharing references."""
        cloned = self.__class__(
            initial_chat_history=copy.deepcopy(self.chat_history),
            initial_criteria_history=copy.deepcopy(self.criteria_history),
            model_name=self.model_name,
            seed_model_name=self.seed_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            update_probability=self.updater.update_probability if hasattr(self, "updater") else 0.25,
            verbose=self.verbose,
            single_dimension_focus=self.single_dimension_focus,
        )
        # Copy the current root position state
        cloned.current_root_position = self.current_root_position
        return cloned

    def get_history(self) -> List[Dict[str, Any]]:
        history = []
        for i, criteria_objs in enumerate(self.criteria_history):
            history.append({ "criteria_objs": criteria_objs })
            history.append(self.chat_history[i*2])
            if i*2+1 < len(self.chat_history):
                history.append(self.chat_history[i*2+1])
        return history

    def is_all_satisfied(self) -> bool:
        """Check if all root nodes across all criteria are satisfied."""
        if not self.single_dimension_focus:
            return _find_first_unsatisfied_root_position(self.criteria_objs) >= self.total_root_nodes
        return self.current_root_position >= self.total_root_nodes

    def get_current_root_node_info(self) -> Tuple[int, int, Dict[str, Any]]:
        """Get the current root node being worked on.

        Returns:
            Tuple of (criterion_idx, root_idx, root_node)

        Raises:
            IndexError if all root nodes are satisfied.
        """
        return _get_root_node_at_position(self.criteria_objs, self.current_root_position)

    def _get_roots_for_response(self) -> List[Dict[str, Any]]:
        """
        Build a criteria_objs list containing root nodes for response generation.
        When single_dimension_focus is True: includes roots 0 up to current position.
        When False: includes all root nodes.
        """
        if not self.single_dimension_focus:
            return _extract_root_nodes_up_to_position(
                self.criteria_objs,
                self.total_root_nodes - 1,
            )
        return _extract_root_nodes_up_to_position(
            self.criteria_objs,
            self.current_root_position
        )

    def _get_roots_for_update(self) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        """
        Build a criteria_objs list for evaluation.
        When single_dimension_focus is True: current root + next root (look-ahead).
        When False: all root nodes simultaneously.

        Returns:
            - criteria_objs with relevant roots
            - position mapping list of (criterion_idx, root_idx) for each root
        """
        if not self.single_dimension_focus:
            positions = list(range(self.total_root_nodes))
        else:
            positions = [self.current_root_position]
            if self.current_root_position + 1 < self.total_root_nodes:
                positions.append(self.current_root_position + 1)

        return _extract_root_nodes_at_positions(self.criteria_objs, positions)

    def _advance_to_next_unsatisfied_root(self) -> None:
        """Advance to the next unsatisfied root node, if any.
        No-op when single_dimension_focus is False (no sequential ordering).
        """
        if not self.single_dimension_focus:
            return
        while self.current_root_position < self.total_root_nodes:
            criterion_idx, root_idx, root_node = _get_root_node_at_position(
                self.criteria_objs,
                self.current_root_position
            )
            if _is_root_node_tree_satisfied(root_node):
                if self.verbose:
                    print(f"Root node at position {self.current_root_position} (criterion {criterion_idx}, root {root_idx}) is satisfied. Advancing...")
                self.current_root_position += 1
            else:
                break

    def update_criteria_only(self, assistant_response: str) -> None:
        """Update criteria for the current root node based on the assistant response.

        Evaluates current root node + next root node (if any) together, but only merges
        the current root node's updates. If the current root becomes fully satisfied,
        also merges the next root node's updates. Then advances to the next unsatisfied root.
        """
        self.chat_history.append({"role": "assistant", "content": assistant_response})

        if self.is_all_satisfied():
            # All root nodes are satisfied, nothing to update
            self.criteria_history.append(copy.deepcopy(self.criteria_objs))
            return

        # Get current + next root nodes for evaluation
        roots_for_update, position_mapping = self._get_roots_for_update()

        if not roots_for_update or not position_mapping:
            self.criteria_history.append(copy.deepcopy(self.criteria_objs))
            return

        # Run update on roots
        updated_roots = self.updater(self.chat_history, roots_for_update)

        # Get the updated hierarchy (list of root nodes)
        updated_hierarchy = updated_roots[0].get("hierarchy", [])

        if not self.single_dimension_focus:
            # Merge all updated roots back
            for i, (crit_idx, root_idx) in enumerate(position_mapping):
                if i < len(updated_hierarchy):
                    self.criteria_objs = _merge_root_node_back(
                        self.criteria_objs,
                        updated_hierarchy[i],
                        crit_idx,
                        root_idx
                    )
        else:
            # Always merge the current (first) root node
            current_criterion_idx, current_root_idx = position_mapping[0]
            if len(updated_hierarchy) > 0:
                self.criteria_objs = _merge_root_node_back(
                    self.criteria_objs,
                    updated_hierarchy[0],
                    current_criterion_idx,
                    current_root_idx
                )

            # Check if current root is now satisfied
            current_root_satisfied = False
            if len(updated_hierarchy) > 0:
                current_root_satisfied = _is_root_node_tree_satisfied(updated_hierarchy[0])

            # If current is satisfied and there's a next root, also merge the next root
            if current_root_satisfied and len(position_mapping) > 1 and len(updated_hierarchy) > 1:
                next_criterion_idx, next_root_idx = position_mapping[1]
                self.criteria_objs = _merge_root_node_back(
                    self.criteria_objs,
                    updated_hierarchy[1],
                    next_criterion_idx,
                    next_root_idx
                )
                if self.verbose:
                    print(f"Current root satisfied, also merged next root at position {self.current_root_position + 1}")

        self.criteria_history.append(copy.deepcopy(self.criteria_objs))

        # Check if we should advance to the next root node
        self._advance_to_next_unsatisfied_root()

    def generate_next_user_response(self) -> str:
        """Generate and append the next user response based on completed + current root nodes."""
        if self.is_all_satisfied():
            # All criteria satisfied - generate a final satisfied response
            user_response = simulate_user_response(
                chat_history=self.chat_history,
                criteria_objs=self.criteria_objs,  # Use full criteria to show satisfaction
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose
            )
        else:
            # Include all roots from 0 up to current (completed roots + current root)
            roots_for_response = self._get_roots_for_response()
            user_response = simulate_user_response(
                chat_history=self.chat_history,
                criteria_objs=roots_for_response,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose
            )
        self.chat_history.append({"role": "user", "content": user_response})
        return user_response

    def __call__(self, assistant_response: str):
        """Backward-compatible combined step: update criteria then generate user response."""
        self.update_criteria_only(assistant_response)
        return self.generate_next_user_response()

