import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}
        self.conversation_history : List[Dict] = [] # chat history of the whole conversation        

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        
    
    def update_memory(self):
        # Update internal memory; this is the main hook for state updates
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def get_spatial_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}
        
        return temporal_info
    
    def get_node_description(self, include_output: bool = True, max_output_len: Optional[int] = None, include_prompt: bool = True, max_prompt_len: Optional[int] = None) -> str:
        """
        Generate a structured description of the node, used as Critic input (backward compatible).
        Note: To reduce unnecessary differences for the Critic, this does not include agent type or domain.
        
        Args:
            include_output: Whether to include the most recent node output.
            max_output_len: Maximum length of the output text (characters). None or <= 0 means no limit.
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text (characters). None or <= 0 means no limit.
            
        Returns:
            Structured node description string (excluding node ID, agent type, and domain).
        """
        desc_parts = []
        
        # 1. Role (if present) - keep role since it may contain useful semantics
        if self.role:
            desc_parts.append(f"Role: {self.role}")
        
        # 2. Prompt/Constraint information (if present and needed)
        if include_prompt and hasattr(self, 'constraint') and self.constraint:
            prompt_text = str(self.constraint)
            if max_prompt_len is not None and max_prompt_len > 0 and len(prompt_text) > max_prompt_len:
                prompt_text = prompt_text[:max_prompt_len] + "..."
            desc_parts.append(f"Prompt: {prompt_text}")
        
        # 3. Most recent output (if present and needed)
        if include_output and hasattr(self, 'outputs') and len(self.outputs) > 0:
            recent_output = str(self.outputs[-1])
            if max_output_len is not None and max_output_len > 0 and len(recent_output) > max_output_len:
                recent_output = recent_output[:max_output_len] + "..."
            desc_parts.append(f"Output: {recent_output}")
        
        # Note:
        # - Do not include node ID (self.id) since it's a random UUID and should not affect encoding
        # - Do not include agent type (self.agent_name) to reduce differences for the Critic
        # - Do not include domain (self.domain) to reduce differences for the Critic
        
        # Join all parts with " | "
        description = " | ".join(desc_parts) if desc_parts else "Empty Node"
        
        return description
    
    def get_node_info_parts(self, include_output: bool = True, max_output_len: Optional[int] = None, include_prompt: bool = True, max_prompt_len: Optional[int] = None) -> Dict[str, str]:
        """
        Get separate node information parts (without prefixes) for differentiated encoding.
        Prefixes such as "Role:", "Prompt:" etc. are removed to reduce similarity between nodes.
        
        Args:
            include_output: Whether to include the most recent node output.
            max_output_len: Maximum length of the output text (characters). None or <= 0 means no limit.
            include_prompt: Whether to include prompt/constraint information.
            max_prompt_len: Maximum length of the prompt text (characters). None or <= 0 means no limit.
            
        Returns:
            Dict with the following keys:
            - role: Role text (plain, without prefix).
            - prompt: Prompt/constraint text (plain, without prefix).
            - output: Most recent output (plain, without prefix).
        """
        result = {
            "role": "",
            "prompt": "",
            "output": ""
        }
        
        # 1. Role (plain text, no prefix)
        if self.role:
            result["role"] = str(self.role).strip()
        
        # 2. Prompt/Constraint info (plain text, no prefix)
        if include_prompt and hasattr(self, 'constraint') and self.constraint:
            prompt_text = str(self.constraint).strip()
            if max_prompt_len is not None and max_prompt_len > 0 and len(prompt_text) > max_prompt_len:
                prompt_text = prompt_text[:max_prompt_len] + "..."
            result["prompt"] = prompt_text
        
        # 3. Most recent output (plain text, no prefix)
        if include_output and hasattr(self, 'outputs') and len(self.outputs) > 0:
            recent_output = str(self.outputs[-1]).strip()
            if max_output_len is not None and max_output_len > 0 and len(recent_output) > max_output_len:
                recent_output = recent_output[:max_output_len] + "..."
            result["output"] = recent_output
        
        return result
    
    def get_node_history(self, max_history_len: Optional[int] = None) -> str:
        """
        Get the node's history string (from last_memory).
        
        Args:
            max_history_len: Maximum length of the history text (characters). None or <= 0 means no limit.
            
        Returns:
            Node history string containing the previous round's input and output.
        """
        # Disable history: if max length <= 0, return empty (avoid noise to Encoder/EPN)
        if max_history_len is not None and max_history_len <= 0:
            return ""
        if not hasattr(self, 'last_memory') or not self.last_memory:
            return ""
        
        history_parts = []
        
        # Get input from the previous round
        if 'inputs' in self.last_memory and self.last_memory['inputs']:
            last_input = str(self.last_memory['inputs'])
            if max_history_len is not None and max_history_len > 0 and len(last_input) > max_history_len:
                last_input = last_input[:max_history_len] + "..."
            history_parts.append(f"Previous input: {last_input}")
        
        # Get output from the previous round
        if 'outputs' in self.last_memory and self.last_memory['outputs']:
            last_outputs = self.last_memory['outputs']
            if isinstance(last_outputs, list) and len(last_outputs) > 0:
                last_output = str(last_outputs[-1])
            else:
                last_output = str(last_outputs)
            if max_history_len is not None and max_history_len > 0 and len(last_output) > max_history_len:
                last_output = last_output[:max_history_len] + "..."
            history_parts.append(f"Previous output: {last_output}")
        
        return " | ".join(history_parts) if history_parts else ""
    
    def execute(self, input:Any, **kwargs):
        self.outputs = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs


    async def async_execute(self, input:Any, **kwargs):
        self.outputs = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs
               
    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
