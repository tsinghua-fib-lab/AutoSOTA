"""
State Definition for TransferFunctionAgent

Defines the global state that flows through the LangGraph workflow.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from operator import add


class SolveStep(TypedDict):
    """Single step in the solving process"""
    thought: str
    action: str
    observation: str


class TransferFunctionState(TypedDict):
    """
    Global state for the transfer function analysis workflow.
    
    This state is passed between nodes and updated throughout the graph execution.
    """
    
    # === Input ===
    image_path: str                    # Path to the circuit/diagram image
    question: str                      # The question to answer
    provided_netlist: Optional[str]    # Pre-provided netlist from input data (skip LLM generation if present)
    
    # === Classification Result ===
    ir_type: Literal["sfg", "netlist", None]  # Type of IR to generate
    analysis_type: Optional[str]  # Analysis type: transfer_function, ac, dc, tran, stability, noise, cmrr, psrr, etc.
    input_source: Optional[str]        # Input source (V1, Vin, etc.)
    output_node: Optional[str]         # Output node or element
    detected_components: Optional[List[str]] # Detected component types for dynamic prompt

    # === IR (Intermediate Representation) ===
    ir: Optional[Dict[str, Any]]       # Parsed IR structure
    ir_code: str                       # Raw netlist code
    
    # === Solving ===
    solve_steps: Annotated[List[SolveStep], add]  # Solving steps (accumulated)
    
    # === Output ===
    answer: Optional[str]              # Final answer
    simplified_answer: Optional[str]   # Simplified form of answer
    success: bool                      # Whether solving succeeded
    error: Optional[str]               # Error message if failed
    
    # === Metrics (Time and Token Tracking) ===
    metrics: Optional[Dict[str, Any]]  # Time and token consumption for each stage


def create_initial_state(
    image_path: str, 
    question: str,
    max_retries: int = 3,
    provided_netlist: Optional[str] = None,
) -> TransferFunctionState:
    """Create initial state for the workflow.
    
    Args:
        image_path: Path to the circuit/diagram image
        question: The question to answer
        max_retries: Reserved for compatibility with existing callers
        provided_netlist: Pre-provided netlist from input data (skip LLM generation if present)
    """
    return TransferFunctionState(
        # Input
        image_path=image_path,
        question=question,
        provided_netlist=provided_netlist,
        
        # Classification (to be filled)
        ir_type=None,
        analysis_type=None,
        input_source=None,
        output_node=None,
        detected_components=None,
        
        # IR (to be filled)
        ir=None,
        ir_code="",
        
        # Solving
        solve_steps=[],
        
        # Output
        answer=None,
        simplified_answer=None,
        success=False,
        error=None,
    )
