import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio
import traceback

from AgentTailor.ATNetwork.Node import Node
from AgentTailor.agents.agent_registry import AgentRegistry

class Actor(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.

    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 optimized_spatial: bool = False,
                 initial_spatial_probability: float = 0.5,
                 fixed_spatial_masks: List[List[int]] = None,
                 optimized_temporal: bool = False,
                 initial_temporal_probability: float = 0.5,
                 fixed_temporal_masks: List[List[int]] = None,
                 node_kwargs: List[Dict] = None,
                 ):

        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i != j else 0 for j in range(len(agent_names))] for i in
                                   range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks) == len(agent_names) * len(
            agent_names), "The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks) == len(agent_names) * len(
            agent_names), "The fixed_temporal_masks doesn't match the number of agents"

        self.id: str = shortuuid.ShortUUID().random(length=4)
        # Assign a 4-character shortuuid as the node identifier
        self.domain: str = domain
        # Domain of the problems being handled
        self.llm_name: str = llm_name
        # Unified LLM backend used by all agents
        self.agent_names: List[str] = agent_names
        # Names of individual agents (used to instantiate nodes)
        self.optimized_spatial = optimized_spatial
        # Whether spatial optimization is enabled
        self.optimized_temporal = optimized_temporal
        # Whether temporal optimization is enabled
        self.decision_node: Node = AgentRegistry.get(
            decision_method, **{"domain": self.domain, "llm_name": self.llm_name}
        )
        # Final decision node
        self.nodes: Dict[str, Node] = {}
        # Mapping from node id to Node instance
        self.potential_spatial_edges: List[List[str, str]] = []
        self.potential_temporal_edges: List[List[str, str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.init_nodes()  # add nodes to the self.nodes
        self.init_potential_edges()  # add potential edges to the self.potential_spatial/temporal_edges

        init_spatial_logit = (
            torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability)))
            if optimized_spatial
            else 10.0
        )
        # Initialize spatial logits
        self.spatial_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
            requires_grad=optimized_spatial,
        )  # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(
            initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
            requires_grad=optimized_temporal)  # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks, requires_grad=False)  # fixed edge masks

        # LOEO/causal3: if True, mask=1 edges skip Bernoulli sampling (always connect; still acyclic)
        self.loeo_deterministic_edges: bool = False

    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors:
                    matrix[i, j] = 1
        return matrix
    # Return spatial adjacency matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors:
                    matrix[i, j] = 1
        return matrix
    # Return temporal adjacency matrix (connectivity only)

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    # Number of edges

    @property
    def num_nodes(self):
        return len(self.nodes)
    # Number of nodes

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        # Look up node by id and raise if not found

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self, last_node_id: str = None):
        for node_id in self.nodes.keys():
            if last_node_id is None:
                self.nodes[node_id].add_successor(self.decision_node)
            elif last_node_id == node_id:
                self.nodes[node_id].add_successor(self.decision_node)


    def construct_spatial_connection(self, temperature: float = 1.0,
                                     threshold: float = None, ):  # temperature must >= records.md.0
        self.clear_spatial_connection()
        _dev = self.spatial_logits.device
        log_probs = [torch.tensor(0.0, device=_dev, requires_grad=self.optimized_spatial)]

        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits,
                                                               self.spatial_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial == False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue
            if getattr(self, "loeo_deterministic_edges", False):
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(
                        torch.tensor(0.0, device=edge_logit.device, requires_grad=self.optimized_spatial)
                    )
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(
                        1 if edge_prob > threshold else 0,
                        device=edge_logit.device,
                        dtype=edge_prob.dtype,
                    )
                if torch.rand(1, device=edge_prob.device) < edge_prob:
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0,
                                     threshold: float = None, ):  # temperature must >= records.md.0
        self.clear_temporal_connection()
        _dev_t = self.temporal_logits.device
        log_probs = [torch.tensor(0.0, device=_dev_t, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits,
                                                               self.temporal_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            
            # Temporal edges can include self-loops (a node can refer to its own previous output)
            is_self_loop = (out_node == in_node)
            
            if edge_mask == 1.0 and self.optimized_temporal == False:
                # Both self-loops and non-loops can be added
                if is_self_loop or not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                continue

            if getattr(self, "loeo_deterministic_edges", False):
                if is_self_loop or not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                log_probs.append(
                    torch.tensor(0.0, device=edge_logit.device, requires_grad=self.optimized_temporal)
                )
                continue

            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(
                    1 if edge_prob > threshold else 0,
                    device=edge_logit.device,
                    dtype=edge_prob.dtype,
                )
            if torch.rand(1, device=edge_prob.device) < edge_prob:
                # Both self-loops and non-loops can be added
                if is_self_loop or not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    def run(self, inputs: Any,
            num_rounds: int = 3,
            max_tries: int = 3,
            max_time: int = 600,
            aggregate_mode: str = "all connected") -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            # D.A.G
            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs)  # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {repr(e)}")
                        traceback.print_exc()
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()
        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, log_probs
    """
    async def arun(self, input: Dict[str, str],
                   num_rounds: int = 3,
                   max_tries: int = 3,
                   max_time: int = 600,
                   aggregate_mode: str = "all connected", ) -> List[Any]:
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),
                                               timeout=max_time)  # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += records.md
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= records.md
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs

    """

    async def arun(self, input: Dict[str, str],
                   num_rounds: int = 3,
                   max_tries: int = 3,
                   max_time: int = 600,
                   aggregate_mode: str = "all connected",
                   edge_sample_threshold: Optional[float] = None):
        log_probs = 0
        edge_records = []  # Store edge information

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(threshold=edge_sample_threshold)
            log_probs += self.construct_temporal_connection(round, threshold=edge_sample_threshold)
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            # Important fix: clear edge_records at the start of each round to record only edges for this round
            # This avoids accumulating edges across rounds and marking all of them as selected
            # If Training3 needs all rounds' edges, they should be accumulated there instead
            current_round_edge_records = []

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        # Execute current node
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input), timeout=max_time)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {repr(e)}")
                        traceback.print_exc()
                    tries += 1

                current_node = self.nodes[current_node_id]
                for successor in current_node.spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    edge_info = {
                        "out_node_id": current_node_id,
                        "in_node_id": successor.id,
                        "out_output": getattr(current_node, "outputs", None),
                        "type":"spatial",
                        "round": round  # Record which round this edge belongs to
                    }
                    current_round_edge_records.append(edge_info)
                for successor in current_node.temporal_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    edge_info = {
                            "out_node_id": current_node_id,
                            "in_node_id": successor.id,
                            "out_output": getattr(current_node, "outputs", None),
                            "type": "temporal",
                            "round": round  # Record which round this edge belongs to
                    }
                    current_round_edge_records.append(edge_info)

                    # Topology updates
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            # Record edges from all rounds (including the round field for later distinction)
            edge_records.extend(current_round_edge_records)

            self.update_memory()

        # Connect the decision node
        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)

        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        # ✅ Modify return value to include edge records
        return final_answers, log_probs, edge_records

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()
            # Let each node update itself

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def get_node_description(self, node_id: str, include_output: bool = True, max_output_len: Optional[int] = None, include_prompt: bool = True, max_prompt_len: Optional[int] = None) -> str:
        """
        Get a structured description of the specified node.
        
        Args:
            node_id: Node ID (used only for lookup; not included in the description).
            include_output: Whether to include the node's most recent output.
            max_output_len: Maximum length of the output text (characters).
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text (characters).
            
        Returns:
            Structured node description string (without node ID).
        """
        if node_id not in self.nodes:
            return f"Unknown Node"
        
        node = self.nodes[node_id]
        return node.get_node_description(
            include_output=include_output, 
            max_output_len=max_output_len,
            include_prompt=include_prompt,
            max_prompt_len=max_prompt_len
        )
    
    def get_edge_node_info_with_history(self, out_node_id: str, in_node_id: str,
                                        include_output: bool = True, max_output_len: Optional[int] = None,
                                        include_prompt: bool = True, max_prompt_len: Optional[int] = None,
                                        max_history_len: Optional[int] = None) -> Dict[str, Dict[str, str]]:
        """
        Get node information and history for an edge (for the EPN's new input format).
        
        Args:
            out_node_id: Source node ID.
            in_node_id: Target node ID.
            include_output: Whether to include the most recent node output.
            max_output_len: Maximum length of the output text.
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text.
            max_history_len: Maximum length of the history text.
            
        Returns:
            Dict with the following keys:
            - out_node: {"description": ..., "history": ...}
            - in_node: {"description": ..., "history": ...}
        """
        if out_node_id not in self.nodes or in_node_id not in self.nodes:
            empty_info = {"description": "", "history": ""}
            return {"out_node": empty_info, "in_node": empty_info}
        
        out_node = self.nodes[out_node_id]
        in_node = self.nodes[in_node_id]
        
        # Get node parts (role / prompt / output)
        out_parts = out_node.get_node_info_parts(
            include_output=False,
            max_output_len=max_output_len,
            include_prompt=include_prompt,
            max_prompt_len=max_prompt_len,
        )
        in_parts = in_node.get_node_info_parts(
            include_output=False,
            max_output_len=max_output_len,
            include_prompt=include_prompt,
            max_prompt_len=max_prompt_len,
        )
        
        # Node description: role + optional constraint prompt for Critic/EPN disambiguation
        def _build_description(parts: Dict[str, str]) -> str:
            role_text = parts.get("role", "") or ""
            prompt_text = parts.get("prompt", "") or ""
            if prompt_text:
                if role_text:
                    return f"{role_text} || {prompt_text}"
                return prompt_text
            return role_text
        
        out_description = _build_description(out_parts)
        in_description = _build_description(in_parts)
        
        # Get node history
        out_history = out_node.get_node_history(max_history_len=max_history_len)
        in_history = in_node.get_node_history(max_history_len=max_history_len)
        
        return {
            "out_node": {
                "description": out_description,
                "history": out_history
            },
            "in_node": {
                "description": in_description,
                "history": in_history
            }
        }
    
    def get_edge_node_descriptions(self, out_node_id: str, in_node_id: str, 
                                   include_output: bool = True, max_output_len: Optional[int] = None,
                                   include_prompt: bool = True, max_prompt_len: Optional[int] = None) -> tuple:
        """
        Get structured descriptions of the two nodes of an edge.
        
        Args:
            out_node_id: Source node ID (used only for lookup; not included in the description).
            in_node_id: Target node ID (used only for lookup; not included in the description).
            include_output: Whether to include the nodes' most recent outputs.
            max_output_len: Maximum length of the output text (characters).
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text (characters).
            
        Returns:
            (out_node_description, in_node_description) tuple (without node IDs).
        """
        out_desc = self.get_node_description(out_node_id, include_output, max_output_len, include_prompt, max_prompt_len)
        in_desc = self.get_node_description(in_node_id, include_output, max_output_len, include_prompt, max_prompt_len)
        return out_desc, in_desc
    
    def get_edge_node_info_parts(self, out_node_id: str, in_node_id: str,
                                 include_output: bool = True, max_output_len: Optional[int] = None,
                                 include_prompt: bool = True, max_prompt_len: Optional[int] = None) -> Dict[str, Dict[str, str]]:
        """
        Get separated information parts for the two nodes of an edge (without prefixes), for differentiated encoding.
        
        Args:
            out_node_id: Source node ID.
            in_node_id: Target node ID.
            include_output: Whether to include nodes' most recent outputs.
            max_output_len: Maximum length of the output text (characters).
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text (characters).
            
        Returns:
            Dict with the following keys:
            - out_node: Information parts of the source node (role, prompt, output).
            - in_node: Information parts of the target node (role, prompt, output).
        """
        if out_node_id not in self.nodes or in_node_id not in self.nodes:
            # If the node does not exist, return empty info
            empty_info = {"role": "", "prompt": "", "output": ""}
            return {"out_node": empty_info, "in_node": empty_info}
        
        out_node = self.nodes[out_node_id]
        in_node = self.nodes[in_node_id]
        
        out_parts = out_node.get_node_info_parts(include_output, max_output_len, include_prompt, max_prompt_len)
        in_parts = in_node.get_node_info_parts(include_output, max_output_len, include_prompt, max_prompt_len)
        
        return {
            "out_node": out_parts,
            "in_node": in_parts
        }
    
    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            # Count edges
            num_masks = (self.spatial_masks == 0).sum()
            # Count pruned edges
            prune_num_edges = torch.round(num_edges * pruning_rate) if torch.round(num_edges * pruning_rate) > 0 else 1
            # Compute how many edges should be pruned this time; otherwise prune only one
            _edge_logits = self.spatial_logits.clone()
            # Deep-copy a snapshot of spatio-temporal connections
            min_edge_logit = _edge_logits.min()
            # Find the "least useful" edges
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            # Assign a very low value to already pruned edges
            sorted_edges_idx = torch.argsort(_edge_logits)
            # Sort edges
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            # Top-k pruning
            self.spatial_masks[prune_idx] = 0

        # If temporal optimization is enabled, do the same for temporal edges
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges * pruning_rate) if torch.round(num_edges * pruning_rate) > 0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

    # New methods
    def apply_pruning(self, k_spatial: int, k_temporal: int):
        """
        Apply Top-K pruning to the current graph logits and update the masks.
        Args:
            k_spatial: Number of spatial edges to keep.
            k_temporal: Number of temporal edges to keep.
        Returns:
            spatial_mask, temporal_mask (0/1 tensors).
        """
        # --- Spatial edges ---
        spatial_probs = torch.sigmoid(self.spatial_logits.detach())
        # Set masked edges to -inf to ensure they are never selected
        masked_spatial = spatial_probs.clone()
        masked_spatial[self.spatial_masks == 0] = -1.0
        # Top-K selection
        topk_vals, topk_idx = torch.topk(masked_spatial, k_spatial)
        new_spatial_mask = torch.zeros_like(self.spatial_masks)
        new_spatial_mask[topk_idx] = 1

        # --- Temporal edges ---
        temporal_probs = torch.sigmoid(self.temporal_logits.detach())
        masked_temporal = temporal_probs.clone()
        masked_temporal[self.temporal_masks == 0] = -1.0
        topk_vals_t, topk_idx_t = torch.topk(masked_temporal, k_temporal)
        new_temporal_mask = torch.zeros_like(self.temporal_masks)
        new_temporal_mask[topk_idx_t] = 1

        # Update masks inside the Graph
        self.spatial_masks.data = new_spatial_mask
        self.temporal_masks.data = new_temporal_mask

        return new_spatial_mask, new_temporal_mask

