"""
LangGraph Workflow Definition

Builds the transfer function analysis graph with:
- Classify node
- Netlist build node
- Netlist solve node

Supports configuration-based flow control:
- use_provided_netlist: Skip LLM generation when netlist is provided
"""

from typing import Dict, Any
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import TransferFunctionState
from ..nodes import (
    classify_node,
    build_netlist_node,
    solve_netlist_node,
)


SFG_UNSUPPORTED_MESSAGE = "Please wait for the upcoming open-source release."


def unsupported_sfg_node(state: TransferFunctionState) -> Dict[str, Any]:
    """End SFG cases early while the SFG implementation is not public."""
    print(SFG_UNSUPPORTED_MESSAGE)
    existing_metrics = state.get("metrics", {}) or {}
    return {
        "answer": SFG_UNSUPPORTED_MESSAGE,
        "simplified_answer": SFG_UNSUPPORTED_MESSAGE,
        "success": False,
        "error": SFG_UNSUPPORTED_MESSAGE,
        "ir": None,
        "ir_code": "",
        "solve_steps": [{
            "thought": SFG_UNSUPPORTED_MESSAGE,
            "action": "unsupported_sfg",
            "observation": SFG_UNSUPPORTED_MESSAGE,
        }],
        "metrics": {
            **existing_metrics,
            "unsupported_sfg": {
                "duration_seconds": 0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "llm_calls": 0,
            },
        },
    }


class TransferFunctionGraph:
    """
    Transfer Function Analysis Graph
    
    Workflow:
    1. Classify: Determine IR type and question type
    2. If SFG: print pending-open-source message and stop
    3. If Netlist: build netlist IR and solve with tools
    """
    
    def __init__(self, llm, max_retries: int = 3, use_memory: bool = False):
        """
        Initialize the graph.
        
        Args:
            llm: Language model with vision capability
            max_retries: Reserved for compatibility with existing callers
            use_memory: Whether to use checkpointing
        """
        self.llm = llm
        self.max_retries = max_retries
        self.graph = self._build_graph()
        
        if use_memory:
            memory = MemorySaver()
            self.app = self.graph.compile(checkpointer=memory)
        else:
            self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph with state schema
        workflow = StateGraph(TransferFunctionState)
        
        # === Add Nodes ===
        
        # Classify node - wraps with LLM
        workflow.add_node(
            "classify",
            partial(classify_node, llm=self.llm)
        )
        
        # SFG is intentionally unavailable in this open-source release.
        workflow.add_node("unsupported_sfg", unsupported_sfg_node)
        
        # Netlist Nodes (Symbolic Analysis - Lcapy)
        workflow.add_node("build_netlist", partial(build_netlist_node, llm=self.llm))
        workflow.add_node("solve_netlist", partial(solve_netlist_node, llm=self.llm))
        
        # === Add Edges ===
        
        # Entry point
        workflow.set_entry_point("classify")
        
        # Classify -> Build (Dispatch based on IR type)
        workflow.add_conditional_edges(
            "classify",
            self._route_by_ir_type,
            {
                "sfg_pipeline": "unsupported_sfg",
                "netlist_pipeline": "build_netlist",
            }
        )
        
        # === SFG Placeholder ===
        workflow.add_edge("unsupported_sfg", END)
        
        # === Netlist Pipeline (Symbolic Analysis) ===
        workflow.add_edge("build_netlist", "solve_netlist")
        workflow.add_edge("solve_netlist", END)
        
        return workflow
    
    def _route_by_ir_type(self, state: TransferFunctionState) -> str:
        """Route to SFG or Netlist pipeline"""
        ir_type = state.get("ir_type")
        if ir_type == "sfg":
            return "sfg_pipeline"
        return "netlist_pipeline"  # default
    
    def invoke(self, image_path: str, question: str, 
               config: Dict[str, Any] = None,
               provided_netlist: str = None) -> Dict[str, Any]:
        """
        Run the workflow.
        
        Args:
            image_path: Path to the image
            question: Question to answer
            config: Optional LangGraph config
            provided_netlist: Pre-provided netlist from input data (skip LLM generation if present)
        
        Returns:
            Final state with answer
        """
        initial_state = {
            "image_path": image_path,
            "question": question,
            "provided_netlist": provided_netlist,
            "ir_type": None,
            "analysis_type": None,
            "input_source": None,
            "output_node": None,
            "detected_components": None,  # Detected component types for dynamic prompt
            "ir": None,
            "ir_code": "",
            "solve_steps": [],
            "answer": None,
            "simplified_answer": None,
            "success": False,
            "error": None,
        }
        
        result = self.app.invoke(initial_state, config=config)
        return result
    
    def get_graph_image(self) -> bytes:
        """Get graph visualization as PNG bytes"""
        try:
            return self.app.get_graph().draw_mermaid_png()
        except Exception:
            return None
    
    def get_graph_mermaid(self) -> str:
        """Get graph as Mermaid diagram"""
        mermaid = self.app.get_graph().draw_mermaid()
        hidden_fragments = ("unsupported_sfg", "sfg_pipeline")
        return "\n".join(
            line for line in mermaid.splitlines()
            if not any(fragment in line for fragment in hidden_fragments)
        )


def create_graph(llm, max_retries: int = 3, use_memory: bool = False) -> TransferFunctionGraph:
    """
    Create a transfer function analysis graph.
    
    Args:
        llm: Language model with vision capability
        max_retries: Reserved for compatibility with existing callers
        use_memory: Whether to use checkpointing
    
    Returns:
        TransferFunctionGraph instance
    """
    return TransferFunctionGraph(llm, max_retries, use_memory)
