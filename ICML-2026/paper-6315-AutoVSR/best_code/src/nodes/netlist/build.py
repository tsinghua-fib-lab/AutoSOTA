"""
Netlist Build Node

Generate Netlist-IR from circuit schematic images using Lcapy format.

Reference: netlist_rules_modular.py

Supports dynamic prompt generation based on detected component types.
"""

import re
import base64
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

# Get global logger
logger = logging.getLogger("TransferFunctionAgent")

# ============================================================
# Configuration Loading
# ============================================================

def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

from ...ir import NetlistIR
from ...utils.response_parser import extract_text_content
from ...utils.metrics import format_duration, format_tokens


# ============================================================
# Load Modular Rules
# ============================================================

def _load_modular_netlist_rules(detected_components: Optional[List[str]] = None) -> str:
    """Load modular netlist rules based on detected component types.
    
    Args:
        detected_components: List of detected component types
            Example: ["resistor", "capacitor", "voltage_source", "opamp"]
    
    Returns:
        Built prompt string (only contains rules for detected component types)
    """
    from ...prompts.netlist_rules_modular import build_dynamic_prompt, get_full_prompt

    if detected_components:
        return build_dynamic_prompt(detected_components)
    return get_full_prompt()

def _compact_braced_expressions(netlist: str) -> str:
    """
    Compact whitespace inside {...} so expressions remain a single token.
    
    Why: our IR parsing/tokenization (and some fixers) split lines by whitespace.
    If an expression contains spaces inside braces, e.g. `{1000 / (1 + s/(2*pi*f))}`,
    it may get truncated into `{1000` and cause missing '}' downstream.
    
    Lcapy does not require spaces inside expressions, so removing them is safe.
    """
    import re
    
    def _repl(match: re.Match) -> str:
        inner = match.group(1)
        inner_compact = "".join(inner.split())
        return "{" + inner_compact + "}"
    
    # Non-greedy capture inside braces, single-line
    return re.sub(r"\{([^}]*)\}", _repl, netlist)

# ============================================================
# Terminal Output Utilities
# ============================================================

class TerminalColors:
    """Terminal output colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_build_header(is_retry: bool = False):
    """Print Build node title"""
    if is_retry:
        print(f"\n{TerminalColors.YELLOW}{'=' * 60}")
        print(f"🔄 BUILD NETLIST (RETRY)")
        print(f"{'=' * 60}{TerminalColors.END}")
    else:
        print(f"\n{TerminalColors.CYAN}{'=' * 60}")
        print(f"📝 BUILD NETLIST")
        print(f"{'=' * 60}{TerminalColors.END}")


def print_generated_ir(netlist: str, input_info: Dict[str, Any] = None, output_info: Dict[str, Any] = None, io_mappings: List[Dict[str, Any]] = None):
    """Print generated IR and Input/Output information (supports multiple inputs and outputs)"""
    print(f"\n{TerminalColors.GREEN}{TerminalColors.BOLD}📋 Generated Netlist IR:{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}{'─' * 40}{TerminalColors.END}")
    
    # Print netlist (with TYPE comments)
    try:
        ir = NetlistIR.from_netlist(netlist)
        netlist_with_types = ir.add_type_comments_to_netlist()
        for line in netlist_with_types.strip().split('\n'):
            print(f"   {TerminalColors.GREEN}{line}{TerminalColors.END}")
    except Exception:
        # Fallback: Print original netlist (no comments)
        for line in netlist.strip().split('\n'):
            print(f"   {TerminalColors.GREEN}{line}{TerminalColors.END}")
    
    # Print I/O mapping information (use new format first)
    if io_mappings:
        print(f"\n{TerminalColors.BLUE}{TerminalColors.BOLD}🔌 I/O Mappings ({len(io_mappings)} pair{'s' if len(io_mappings) > 1 else ''}):{TerminalColors.END}")
        print(f"{TerminalColors.BLUE}{'─' * 40}{TerminalColors.END}")
        
        for i, mapping in enumerate(io_mappings, 1):
            input_src = mapping.get('input_source', 'unknown')
            input_type = mapping.get('input_type', 'voltage')
            output_nodes = mapping.get('output_nodes')
            output_type = mapping.get('output_type', 'voltage')
            
            # Format output nodes
            if output_nodes is None:
                output_str = "* (all nodes)"
            elif isinstance(output_nodes, list):
                output_str = f"V({output_nodes[0]}) - V({output_nodes[1]})" if len(output_nodes) > 1 else f"V({output_nodes[0]})"
            else:
                output_str = f"V({output_nodes})"
            
            type_icon = "⚡" if input_type == 'voltage' else "💧"
            print(f"   {TerminalColors.BLUE}{type_icon} {input_src} → {output_str}{TerminalColors.END}")
    
    # Backward compatibility: if no io_mappings, use old format
    elif input_info or output_info:
        print(f"\n{TerminalColors.BLUE}{TerminalColors.BOLD}🔌 I/O Configuration:{TerminalColors.END}")
        print(f"{TerminalColors.BLUE}{'─' * 40}{TerminalColors.END}")
        
        if input_info:
            input_type = input_info.get('type', 'unknown')
            input_node = input_info.get('node', 'unknown')
            input_name = input_info.get('name', '')
            if input_type == 'voltage':
                print(f"   {TerminalColors.BLUE}📥 Input: Voltage at node {input_node} ({input_name}){TerminalColors.END}")
            elif input_type == 'current':
                print(f"   {TerminalColors.BLUE}📥 Input: Current into node {input_node} ({input_name}){TerminalColors.END}")
            else:
                print(f"   {TerminalColors.BLUE}📥 Input: Node {input_node} ({input_name}){TerminalColors.END}")
        
        if output_info:
            output_node = output_info.get('node', 'unknown')
            output_type = output_info.get('type', 'voltage')
            print(f"   {TerminalColors.BLUE}📤 Output: {output_type.capitalize()} at node {output_node}{TerminalColors.END}")
    
    print()


# System prompt - rules from netlist_rules_modular.py
NETLIST_BUILD_SYSTEM_PROMPT = """You are an expert in electronic circuit analysis.

{rules}
"""


def _extract_token_usage(response) -> Dict[str, int]:
    """Extract token usage information from LLM response"""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if 'usage_metadata' in metadata:
            um = metadata['usage_metadata']
            usage["input_tokens"] = um.get('input_tokens', 0)
            usage["output_tokens"] = um.get('output_tokens', 0)
            usage["total_tokens"] = um.get('total_tokens', 0)
        elif 'token_usage' in metadata:
            tu = metadata['token_usage']
            usage["input_tokens"] = tu.get('prompt_tokens', 0)
            usage["output_tokens"] = tu.get('completion_tokens', 0)
            usage["total_tokens"] = tu.get('total_tokens', 0)
    
    if usage["total_tokens"] == 0 and hasattr(response, 'usage_metadata'):
        um = response.usage_metadata
        if um:
            if isinstance(um, dict):
                usage["input_tokens"] = um.get('input_tokens', 0)
                usage["output_tokens"] = um.get('output_tokens', 0)
                usage["total_tokens"] = um.get('total_tokens', 0)
            else:
                usage["input_tokens"] = getattr(um, 'input_tokens', 0)
                usage["output_tokens"] = getattr(um, 'output_tokens', 0)
                usage["total_tokens"] = getattr(um, 'total_tokens', 0)
    
    return usage


def build_netlist_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """Generate Netlist-IR from circuit image
    
    Generate Lcapy compatible netlist based on the modular netlist rule specification
    
    Supports two modes:
    1. If use_provided_netlist=True and provided_netlist exists, use it directly
    2. Otherwise use LLM generation (supports dynamic prompt to select rules based on component types)
    """
    start_time = time.time()
    
    # Read configuration
    config = _load_config()
    netlist_config = config.get("ir", {}).get("netlist", {})
    use_provided_netlist = netlist_config.get("use_provided_netlist", False)  # Default False, use LLM generation first
    
    image_path = state["image_path"]
    question = state.get("question", "")
    provided_netlist = state.get("provided_netlist")  # Pre-provided netlist
    detected_components = state.get("detected_components")  # Detected component types

    # Analysis information from classify stage
    analysis_type = state.get("analysis_type", "transfer_function")
    input_source = state.get("input_source")
    output_node = state.get("output_node")
    constraints = state.get("constraints")  # Constraints extracted from diagram/question
    
    # ========================================
    # If the switch is open and there is a pre-provided netlist, use it directly
    # ========================================
    if use_provided_netlist and provided_netlist:
        print(f"\n{TerminalColors.CYAN}{'=' * 60}")
        print(f"📋 BUILD NETLIST (Using Provided Netlist)")
        print(f"{'=' * 60}{TerminalColors.END}")
        print(f"   {TerminalColors.GREEN}✅ Skipping LLM generation - using netlist from input data{TerminalColors.END}")
        logger.info("✅ Skipping LLM generation - using netlist from input data")
        
        # Directly use the pre-provided netlist
        netlist = provided_netlist
        content = f"```netlist\n{netlist}\n```"  # For subsequent I/O extraction
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    else:
        # Terminal output - Build title
        print_build_header(is_retry=False)
        
        # ========================================
        # Select prompt rules from the modular rule library
        # ========================================
        if detected_components:
            rules = _load_modular_netlist_rules(detected_components)
            print(f"   {TerminalColors.CYAN}🔧 Using dynamic prompt for: {detected_components}{TerminalColors.END}")
        else:
            rules = _load_modular_netlist_rules()
            print(f"   {TerminalColors.YELLOW}⚠️ No detected components, using full modular prompt{TerminalColors.END}")
        
        # Build prompt with rules
        base_prompt = NETLIST_BUILD_SYSTEM_PROMPT.format(rules=rules)
        
        system_prompt = base_prompt
        
        messages = [SystemMessage(content=system_prompt)]
        
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            media_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            
            # Build prompt
            context_parts = []
            
            if question:
                context_parts.append(f"## QUESTION:\n{question}\n")
            
            
            if input_source:
                context_parts.append(f"\nInput: {input_source}")
            if output_node:
                context_parts.append(f"Output: {output_node}")
            if analysis_type:
                context_parts.append(f"Analysis: {analysis_type}")
                # Distinguish between transfer function and AC steady-state analysis
                if analysis_type in ["transfer_function", "s-domain", "impedance", "frequency"]:
                    context_parts.append("⚠️ For transfer function H(s), input sources MUST use s-domain: `Vname Np Nm s Symbol` (NOT step/dc)")
                elif analysis_type == "ac":
                    context_parts.append("⚠️ For AC steady-state analysis with sin(omega*t) sources, use TIME-DOMAIN expressions: `Vname Np Nm {A*sin(omega*t)}` or `Vname Np Nm {A*cos(omega*t)}`")
                    context_parts.append("⚠️ For reactance values (XL, XC), use: `Lname Np Nm {XL/omega}` or `Cname Np Nm {1/(XC*omega)}`")
            if constraints:
                context_parts.append(f"Component values from problem: {constraints}")
            
            messages.append(HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                {"type": "text", "text": "\n".join(context_parts)}
            ]))
        
        # Call LLM
        response = llm.invoke(messages)
        content = extract_text_content(response.content)
        
        # Extract token usage
        token_usage = _extract_token_usage(response)
        
        # Display full LLM response
        print(f"\n{TerminalColors.MAGENTA}📤 LLM Build Response:{TerminalColors.END}")
        print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
        print(f"   {content}")
        print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
        
        # Parse response - Extract ```netlist ... ``` code block
        netlist = _extract_netlist(content)
        
        # Fallback: Try JSON format (compatible with old format)
        if not netlist:
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    netlist = data.get("netlist", "").replace("\\n", "\n")
            except (json.JSONDecodeError, AttributeError):
                pass
    
    # ========================================
    # Early extraction check
    # ========================================
    if not netlist or not netlist.strip():
        print(f"   {TerminalColors.RED}❌ Failed to extract netlist from LLM response!{TerminalColors.END}")
        return {
            "ir": None,
            "ir_code": "",
            "success": False,
            "error": "LLM output format error: failed to extract netlist",
        }
    
    # Compact whitespace inside {...} before any further processing/parsing.
    # This prevents opamp/VCVS gains like "{1000 / (1 + ...)}" from being split/truncated.
    # netlist = _compact_braced_expressions(netlist)

    # # Auto-fix misused E/F prefixes for independent sources
    # netlist = _fix_source_prefixes(netlist)
    
    # # Auto-fix "s number" format to "dc number" for DC sources
    # netlist = _fix_s_domain_format(netlist)
    
    # # Auto-fix Z prefix to R prefix (Z1 → RZ1)
    # netlist = _fix_impedance_prefix(netlist)
    
    # # Auto-fix invalid source types (square → step, etc.)
    # netlist = _fix_invalid_source_types(netlist)
    
    # # Auto-fix netlist sources to use s-domain
    # netlist = _fix_netlist_sources(netlist)
    
    # # Auto-fix wire components (W has no value)
    # netlist = _fix_wire_components(netlist)
    
    # # Auto-fix engineering notation (1m → 0.001, 1k → 1000)
    # netlist = _fix_engineering_notation(netlist)
    
    # ========================================
    # Extract Input/Output Information (supports multiple inputs and outputs)
    # ========================================
    
    # Step 1: Try to extract from LLM response (new format supports multiple I/O mappings)
    extracted_input, extracted_output, io_mappings = _extract_io_info(content)
    
    # # Step 1.5: Check if extracted input is a 0V sensing source (invalid as input)
    # # 0V sensing sources are for CCCS/CCVS current measurement, NOT circuit excitation
    # if extracted_input and extracted_input.get('name'):
    #     input_name = extracted_input.get('name')
    #     # if _is_zero_sense_source(input_name, netlist):
    #     #     print(f"   {TerminalColors.YELLOW}⚠️ '{input_name}' is a 0V sensing source, cannot be INPUT_SOURCE. Auto-detecting...{TerminalColors.END}")
    #     #     extracted_input = None
    #     #     io_mappings = None  # Clear invalid mappings, will rebuild with fallback
    
    # Step 2: Fallback - auto-detect from netlist if not found
    # if not extracted_input or not extracted_input.get('name') or not extracted_output:
    #     detected_input, detected_output = _detect_io_from_netlist(netlist, question)
        
    #     if not extracted_input or not extracted_input.get('name'):
    #         extracted_input = detected_input
    #     if not extracted_output:
    #         extracted_output = detected_output
        
    #     # If no io_mappings, build one from fallback
    #     if not io_mappings and extracted_input:
    #         io_mappings = [{
    #             'input_source': extracted_input.get('name'),
    #             'input_type': extracted_input.get('type', 'voltage'),
    #             'output_nodes': extracted_output.get('node') if extracted_output else None,
    #             'output_type': 'voltage' if extracted_output else 'all',
    #         }]
    
    # Create IR (with error handling)
    try:
        ir = NetlistIR.from_netlist(netlist)
    except Exception as e:
        print(f"   {TerminalColors.RED}❌ Failed to parse netlist: {e}{TerminalColors.END}")
        return {
            "ir": None,
            "ir_code": netlist,
            "success": False,
            "error": f"Netlist parse error: {e}",
        }
    
    # Check if IR has valid components
    if not ir.components:
        print(f"   {TerminalColors.RED}❌ Netlist has no valid components!{TerminalColors.END}")
        return {
            "ir": None,
            "ir_code": netlist,
            "success": False,
            "error": "Netlist has no valid components. Each line should follow format: ComponentID Node1 Node2 [Value]",
        }
    
    # Terminal output - generated IR and I/O information
    print_generated_ir(netlist, extracted_input, extracted_output, io_mappings)
    
    # Build return dictionary, containing constraints and I/O information
    ir_dict = ir.to_dict()
    if constraints:
        ir_dict["constraints"] = constraints
    
    # Add I/O information to ir_dict, for solve stage use
    # Keep single input/output for backward compatibility
    if extracted_input:
        ir_dict["input_source"] = extracted_input.get('name')
        ir_dict["input_node"] = extracted_input.get('node')
        ir_dict["input_positive_node"] = extracted_input.get('positive_node', extracted_input.get('node'))
        ir_dict["input_negative_node"] = extracted_input.get('negative_node', '0')
    
    if extracted_output:
        ir_dict["output_node"] = extracted_output.get('node')
    
    # Add multiple input/output mappings
    if io_mappings:
        ir_dict["io_mappings"] = io_mappings
    
    result = {
        "ir_code": netlist,
        "ir": ir_dict,
    }
    
    # Add Input/Output information (backward compatibility)
    if extracted_input:
        result["input_info"] = extracted_input
        result["input_node"] = extracted_input.get('node')
        result["input_positive_node"] = extracted_input.get('positive_node', extracted_input.get('node'))
        result["input_negative_node"] = extracted_input.get('negative_node', '0')
    
    if extracted_output:
        result["output_info"] = extracted_output
        result["output_node"] = extracted_output.get('node')
    
    # Add io_mappings to result
    if io_mappings:
        result["io_mappings"] = io_mappings
    
    # Calculate duration and output
    duration = time.time() - start_time
    print(f"\n   {TerminalColors.MAGENTA}⏱️  Build Time: {format_duration(duration)} | "
          f"Tokens: {format_tokens(token_usage['total_tokens'])} "
          f"(in: {format_tokens(token_usage['input_tokens'])}, out: {format_tokens(token_usage['output_tokens'])}){TerminalColors.END}")
    
    # Update metrics
    existing_metrics = state.get("metrics", {})
    
    # Accumulate build stage metrics (may have multiple retries)
    if "build" not in existing_metrics:
        existing_metrics["build"] = {
            "duration_seconds": 0,
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "llm_calls": 0,
        }
    
    existing_metrics["build"]["duration_seconds"] += round(duration, 2)
    existing_metrics["build"]["tokens"]["input_tokens"] += token_usage["input_tokens"]
    existing_metrics["build"]["tokens"]["output_tokens"] += token_usage["output_tokens"]
    existing_metrics["build"]["tokens"]["total_tokens"] += token_usage["total_tokens"]
    existing_metrics["build"]["llm_calls"] += 1
    
    result["metrics"] = existing_metrics
    
    return result


def _extract_io_info(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Extract Input and Output information from LLM response.
    
    Supports new multi-I/O format:
    ```io
    V1 -> 2
    V2 -> 3
    Vin -> 2,3  (differential output)
    Vs -> *     (all outputs, for state-space)
    ```
    
    Also supports legacy format:
    ```io
    INPUT_SOURCE: V1
    OUTPUT_NODE: 3
    ```
    
    Returns:
        Tuple of (input_info, output_info, io_mappings)
        - input_info: first/primary input (for backward compatibility)
        - output_info: first/primary output (for backward compatibility)
        - io_mappings: list of all input-output mappings
    """
    input_info = None
    output_info = None
    io_mappings = []
    
    # Pattern 1: ```io ... ``` code block
    io_block_pattern = r'```io\s*([\s\S]*?)```'
    io_match = re.search(io_block_pattern, text, re.IGNORECASE)
    
    if io_match:
        io_content = io_match.group(1)
        
        # NEW FORMAT: <source> -> <output_node(s)>
        # Matches: V1 -> 2, Vin -> 2,3, Vs -> *
        arrow_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\s*->\s*(\*|[\d,\s]+)'
        arrow_matches = re.findall(arrow_pattern, io_content)
        
        if arrow_matches:
            for source_name, output_str in arrow_matches:
                output_str = output_str.strip()
                
                # Parse output node(s)
                if output_str == '*':
                    # All outputs (state-space analysis)
                    output_nodes = None  # None means all nodes
                    output_type = 'all'
                elif ',' in output_str:
                    # Differential output: 2,3 means V(2) - V(3)
                    nodes = [n.strip() for n in output_str.split(',')]
                    output_nodes = nodes
                    output_type = 'differential'
                else:
                    # Single node
                    output_nodes = output_str
                    output_type = 'voltage'
                
                mapping = {
                    'input_source': source_name,
                    'input_type': 'voltage' if source_name.upper().startswith('V') else 'current',
                    'output_nodes': output_nodes,
                    'output_type': output_type,
                }
                io_mappings.append(mapping)
            
            # Set primary input/output for backward compatibility
            if io_mappings:
                first = io_mappings[0]
                input_info = {
                    'name': first['input_source'],
                    'type': first['input_type'],
                }
                if first['output_nodes'] is not None:
                    if isinstance(first['output_nodes'], list):
                        output_info = {
                            'node': first['output_nodes'][0],
                            'node_negative': first['output_nodes'][1] if len(first['output_nodes']) > 1 else '0',
                            'type': 'differential',
                        }
                    else:
                        output_info = {
                            'node': first['output_nodes'],
                            'type': 'voltage',
                        }
        
        # LEGACY FORMAT: INPUT_SOURCE / OUTPUT_NODE
        if not io_mappings:
            # Parse INPUT_SOURCE (old format)
            input_source_pattern = r'INPUT_SOURCE[=:\s]+([a-zA-Z][a-zA-Z0-9_]*)'
            input_match = re.search(input_source_pattern, io_content, re.IGNORECASE)
            if input_match:
                source_name = input_match.group(1)
                input_info = {
                    'name': source_name,
                    'type': 'voltage' if source_name.upper().startswith('V') else 'current',
                }
            
            # Parse OUTPUT_NODE (old format)
            output_node_pattern = r'OUTPUT_NODE[=:\s]+(\d+)'
            output_match = re.search(output_node_pattern, io_content, re.IGNORECASE)
            if output_match:
                node = output_match.group(1)
                output_info = {
                    'node': node,
                    'type': 'voltage',
                }
            
            # Convert legacy to mapping format
            if input_info:
                mapping = {
                    'input_source': input_info.get('name'),
                    'input_type': input_info.get('type', 'voltage'),
                    'output_nodes': output_info.get('node') if output_info else None,
                    'output_type': 'voltage' if output_info else 'all',
                }
                io_mappings.append(mapping)
        
        # Compatible with older format INPUT: <node> (description)
        if not input_info:
            old_input_pattern = r'INPUT:\s*(\d+)\s*(?:\(([^)]*)\))?'
            old_match = re.search(old_input_pattern, io_content, re.IGNORECASE)
            if old_match:
                node = old_match.group(1)
                desc = old_match.group(2) if old_match.group(2) else ""
                source_name = None
                if desc:
                    source_match = re.search(r'\b([VIEGHF][a-zA-Z0-9_]*)\b', desc, re.IGNORECASE)
                    if source_match:
                        source_name = source_match.group(1)
                input_info = {
                    'node': node,
                    'name': source_name,
                    'type': 'voltage',
                    'positive_node': node,
                    'negative_node': '0',
                }
        
        # Compatible with older format OUTPUT: <node>
        if not output_info:
            old_output_pattern = r'OUTPUT:\s*(\d+)'
            old_match = re.search(old_output_pattern, io_content, re.IGNORECASE)
            if old_match:
                output_info = {
                    'node': old_match.group(1),
                    'type': 'voltage',
                }
    
    # Pattern 2: Inline format (fallback)
    if not input_info:
        inline_source = re.search(r'input[_\s]*source[:\s]+([a-zA-Z][a-zA-Z0-9_]*)', text, re.IGNORECASE)
        if inline_source:
            source_name = inline_source.group(1)
            input_info = {
                'name': source_name,
                'type': 'voltage' if source_name.upper().startswith('V') else 'current',
            }
    
    if not output_info:
        inline_output = re.search(r'output[_\s]*node[:\s]+(\d+)', text, re.IGNORECASE)
        if inline_output:
            output_info = {
                'node': inline_output.group(1),
                'type': 'voltage',
            }
    
    return input_info, output_info, io_mappings if io_mappings else None


def _is_zero_sense_source(source_name: str, netlist: str) -> bool:
    """Check if the specified source is a 0V sensing source (cannot be used as INPUT_SOURCE)
    
    0V sensing source is used for CCCS/CCVS current measurement, not a real excitation source.
    
    Args:
        source_name: Source name (e.g. "Vsense", "V1")
        netlist: Complete netlist string
    
    Returns:
        True if it is a 0V sensing source
    """
    if not source_name or not netlist:
        return False
    
    # Name contains "sense" is usually a sensing source
    if 'sense' in source_name.lower():
        # Further verify if it is a 0V sensing source
        pass  # Continue checking value
    
    # Find this source in the netlist
    for line in netlist.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        comp_name = parts[0]
        if comp_name.upper() != source_name.upper():
            continue
        
        # Found this source, check if it is a 0V sensing source
        # Format: Vsense n1 n2 0 / Vsense n1 n2 dc 0 / Vsense n1 n2 {0}
        rest = ' '.join(parts[3:]).lower() if len(parts) > 3 else ''
        
        # Check if it is a 0V sensing source
        zero_patterns = ['0', 'dc 0', '{0}', 'dc0']
        if rest in zero_patterns:
            return True
        # Check if it ends with 0 (e.g. "dc 0")
        if rest.split()[-1] == '0' if rest else False:
            return True
        # Name contains sense and value is empty or 0
        if 'sense' in source_name.lower() and (not rest or rest == '0'):
            return True
    
    return False


def _detect_io_from_netlist(netlist: str, question: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Auto-detect Input and Output from netlist content.
    
    Heuristics:
    - Input: First V source (V1) or I source
    - Output: Port (P) definitions, or infer from question (V2/V1, Vo/Vi, etc.)
    
    Returns:
        Tuple of (input_info, output_info) dictionaries
    """
    input_info = None
    output_info = None
    
    lines = netlist.strip().split('\n')
    
    # Collect all voltage/current sources and ports
    sources = []
    ports = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        comp_name = parts[0].upper()
        node1 = parts[1]
        node2 = parts[2]
        
        # Voltage sources (skip 0V sensing sources)
        if comp_name.startswith('V'):
            # Check if it's a 0V sensing source
            rest = ' '.join(parts[3:]).lower() if len(parts) > 3 else ''
            is_zero_sense = (
                'sense' in parts[0].lower() or
                rest in ['0', 'dc 0', '{0}', 'dc0'] or
                (rest.split()[-1] == '0' if rest else False)
            )
            if not is_zero_sense:
                sources.append({
                    'name': parts[0],
                    'type': 'voltage',
                    'positive_node': node1,
                    'negative_node': node2
                })
        # Current sources
        elif comp_name.startswith('I'):
            sources.append({
                'name': parts[0],
                'type': 'current',
                'positive_node': node1,
                'negative_node': node2
            })
        # Ports
        elif comp_name.startswith('P'):
            ports.append({
                'name': parts[0],
                'positive_node': node1,
                'negative_node': node2
            })
    
    # Determine input: first voltage source
    if sources:
        src = sources[0]
        input_info = {
            'node': src['positive_node'],
            'type': src['type'],
            'name': src['name'],
            'positive_node': src['positive_node'],
            'negative_node': src['negative_node']
        }
    
    # Determine output from question
    # Look for patterns like V2/V1, Vo/Vi, V(3)/V(1), etc.
    if question:
        # Pattern: V2/V1, V_2/V_1, Vo/Vi
        ratio_patterns = [
            r'[Vv]_?(\d+)\s*/\s*[Vv]_?(\d+)',  # V2/V1 or V_2/V_1
            r'[Vv]_?([a-zA-Z]+)\s*/\s*[Vv]_?([a-zA-Z]+)',  # Vo/Vi or V_o/V_i
            r'[Vv]\((\d+)\)\s*/\s*[Vv]\((\d+)\)',  # V(2)/V(1)
        ]
        
        for pattern in ratio_patterns:
            match = re.search(pattern, question)
            if match:
                output_node = match.group(1)
                input_node = match.group(2)
                
                # Map letter labels to potential nodes
                label_map = {'o': None, 'out': None, '2': '2', '3': '3', 
                             'i': None, 'in': None, '1': '1'}
                
                # If output is numeric, use it directly
                if output_node.isdigit():
                    output_info = {
                        'node': output_node,
                        'type': 'voltage',
                        'name': f"V({output_node})"
                    }
                break
        
        # Pattern: "output voltage" or "Vout" mentioned with node
        if not output_info:
            out_patterns = [
                r'[Vv]_?(?:out|o)\s*(?:at|is)?\s*(?:node)?\s*(\d+)',
                r'output\s+(?:voltage\s+)?(?:at\s+)?(?:node\s+)?(\d+)',
                r'V\s*\(\s*(\d+)\s*\)',  # V(3)
            ]
            for pattern in out_patterns:
                match = re.search(pattern, question)
                if match:
                    output_info = {
                        'node': match.group(1),
                        'type': 'voltage',
                        'name': f"V({match.group(1)})"
                    }
                    break
    
    # Fallback: use ports if defined
    if not output_info and len(ports) >= 2:
        # Assume P1 is input, P2 is output
        output_info = {
            'node': ports[1]['positive_node'],
            'type': 'voltage',
            'name': f"V({ports[1]['positive_node']})"
        }
    elif not output_info and len(ports) == 1 and input_info:
        # Single port could be output
        output_info = {
            'node': ports[0]['positive_node'],
            'type': 'voltage',
            'name': f"V({ports[0]['positive_node']})"
        }
    
    return input_info, output_info


def _extract_netlist(text: str) -> str:
    """Extract netlist from response text"""
    patterns = [
        r'```(?:netlist|spice|lcapy)?\s*([\s\S]*?)```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if _looks_like_netlist(content):
                return content
    
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if _is_netlist_line(line):
            lines.append(line)
    # If no suspected netlist lines are found, return an empty string to let the upper layer try the JSON fallback.
    return '\n'.join(lines) if lines else ""


def _looks_like_netlist(text: str) -> bool:
    """Check if text looks like a netlist"""
    for line in text.split('\n'):
        if _is_netlist_line(line.strip()):
            return True
    return False


def _is_netlist_line(line: str) -> bool:
    """Check if line is a netlist component line"""
    if not line or line.startswith('#') or line.startswith('*') or line.startswith('.'):
        return False
    # Match component prefixes: R, L, C, V, I, E, G, F, H, W, O, P, NR (No resistor)
    pattern = r'^(NR|[RLCVIEGFHWOP])\w*\s+\w+\s+\w+'
    return bool(re.match(pattern, line, re.IGNORECASE))


def _fix_impedance_prefix(netlist: str) -> str:
    """Fix Z prefix (impedance) to R prefix (resistor).
    
    Common mistake: Using Z1, Z2, ZL etc. for impedance, but Lcapy only supports R/L/C.
    Mapping: Zname → RZname (keep original name as value)
    """
    lines = netlist.split('\n')
    fixed_lines = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        if not line:
            fixed_lines.append(original_line)
            continue
        
        # Pattern: Z1 1 2 [value] → RZ1 1 2 [value or Z1]
        match = re.match(r'^(Z\w*)\s+(\w+)\s+(\w+)(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, rest = match.groups()
            rest = rest.strip()
            new_id = f"R{comp_id}"  # Z1 → RZ1
            
            if rest:
                fixed_line = f"{new_id} {node1} {node2} {rest}"
            else:
                # Use original Z name as symbolic value
                fixed_line = f"{new_id} {node1} {node2} {comp_id}"
            
            fixed_lines.append(fixed_line)
            continue
        
        fixed_lines.append(original_line)
    
    return '\n'.join(fixed_lines)


def _fix_invalid_source_types(netlist: str) -> str:
    """Fix invalid source types that are not supported by Lcapy.
    
    Common mistake: Using 'square', 'pulse', 'sine' etc. instead of valid Lcapy types.
    Mapping: square/pulse → step, sine → (remove, use s-domain)
    Valid Lcapy types: dc, ac, s, step
    """
    # Invalid source types and their mappings
    invalid_to_valid = {
        'square': 'step',
        'pulse': 'step', 
        'rect': 'step',
        'rectangular': 'step',
        'sine': '',      # Remove, will become s-domain
        'sin': '',
        'triangle': '',
        'tri': '',
        'ramp': '',
        'sawtooth': '',
    }
    
    lines = netlist.split('\n')
    fixed_lines = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        if not line:
            fixed_lines.append(original_line)
            continue
        
        # Pattern: V1 1 0 <invalid_type> [value] → V1 1 0 <valid_type> [value]
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+(\w+)(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, source_type, rest = match.groups()
            source_type_lower = source_type.lower()
            
            if source_type_lower in invalid_to_valid:
                valid_type = invalid_to_valid[source_type_lower]
                rest = rest.strip()
                
                if valid_type:
                    # Map to valid type (e.g., square → step)
                    if rest:
                        fixed_line = f"{comp_id} {node1} {node2} {valid_type} {rest}"
                    else:
                        fixed_line = f"{comp_id} {node1} {node2} {valid_type}"
                else:
                    # Remove invalid type, keep rest as value (s-domain)
                    if rest:
                        fixed_line = f"{comp_id} {node1} {node2} {rest}"
                    else:
                        fixed_line = f"{comp_id} {node1} {node2}"
                
                fixed_lines.append(fixed_line)
                continue
        
        fixed_lines.append(original_line)
    
    return '\n'.join(fixed_lines)


def _fix_source_prefixes(netlist: str) -> str:
    """Fix misused E/F prefixes for independent sources.
    
    Common mistake: Using E1 (VCVS) instead of V1 (independent voltage source)
    E with only 2 nodes + value should be V (independent source)
    E with 4 nodes + value is VCVS (correct)
    """
    lines = netlist.split('\n')
    fixed_lines = []
    v_counter = 1
    i_counter = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append(line)
            continue
        
        parts = line.split()
        if len(parts) < 3:
            fixed_lines.append(line)
            continue
        
        comp_name = parts[0]
        
        # E with only 3-4 parts (E1 n1 n2 value) is likely meant to be V
        # Real VCVS needs 5+ parts: E1 n1 n2 n3 n4 gain
        if comp_name.upper().startswith('E') and len(parts) <= 5:
            # Check if parts[3] and parts[4] look like control nodes
            if len(parts) == 5:
                # Could be "E1 n1 n2 s value" (wrong) or "E1 n1 n2 nc1 nc2" (incomplete VCVS)
                try:
                    # If parts[3] is 's' or 'dc' or 'ac', it's meant to be independent source
                    if parts[3].lower() in ['s', 'dc', 'ac', 'step']:
                        new_name = f"V{v_counter}"
                        v_counter += 1
                        parts[0] = new_name
                        fixed_lines.append(' '.join(parts))
                        continue
                except:
                    pass
            elif len(parts) == 4:
                # E1 n1 n2 value -> V1 n1 n2 value
                new_name = f"V{v_counter}"
                v_counter += 1
                parts[0] = new_name
                fixed_lines.append(' '.join(parts))
                continue
        
        # Similar fix for F (CCCS) being used as I (current source)
        if comp_name.upper().startswith('F') and len(parts) <= 4:
            # F with only 4 parts is likely meant to be I
            # Real CCCS needs: F1 n1 n2 Vcontrol gain
            if len(parts) == 4:
                new_name = f"I{i_counter}"
                i_counter += 1
                parts[0] = new_name
                fixed_lines.append(' '.join(parts))
                continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def _fix_s_domain_format(netlist: str) -> str:
    """Fix 's number' format to 'dc number' for DC sources.
    
    Common mistake: V1 1 0 s 7 (wrong - s expects symbol, not number)
    Should be: V1 1 0 dc 7 or V1 1 0 7
    """
    lines = netlist.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append(line)
            continue
        
        # Pattern: V1 1 0 s 7 → V1 1 0 dc 7
        # When 's' is followed by a pure number, change 's' to 'dc'
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+s\s+(\d+\.?\d*)(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, value, rest = match.groups()
            fixed_line = f"{comp_id} {node1} {node2} dc {value}"
            if rest.strip():
                fixed_line += f" {rest.strip()}"
            fixed_lines.append(fixed_line)
            continue
        
        # Pattern: V1 1 0 s {7} → V1 1 0 dc 7
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+s\s+\{(\d+\.?\d*)\}(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, value, rest = match.groups()
            fixed_line = f"{comp_id} {node1} {node2} dc {value}"
            if rest.strip():
                fixed_line += f" {rest.strip()}"
            fixed_lines.append(fixed_line)
            continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def _fix_netlist_sources(netlist: str) -> str:
    """Auto-fix netlist source definitions to use s-domain for transfer function analysis"""
    lines = netlist.split('\n')
    fixed_lines = []
    
    def _is_zero_literal(s: str) -> bool:
        """Return True if token represents numeric zero (e.g. '0', '0.0', '{0}')."""
        try:
            return float(str(s).strip().strip("{}")) == 0.0
        except Exception:
            return False
    
    def _is_time_domain_expr(s: str) -> bool:
        """Return True if expression is time-domain (contains sin/cos/omega/t)."""
        s_lower = s.lower()
        # Check for time-domain indicators: sin, cos, omega, t
        return any(indicator in s_lower for indicator in ['sin', 'cos', 'omega', 't', 'exp'])
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append(line)
            continue
        
        # Pattern: V1 1 0 {expression} - check if it's time-domain or symbolic
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+\{([^}]+)\}(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, expr, rest = match.groups()
            # If expression contains time-domain indicators, keep it as is
            if _is_time_domain_expr(expr):
                # Keep time-domain expression in braces
                fixed_line = f"{comp_id} {node1} {node2} {{{expr}}}"
                if rest.strip():
                    fixed_line += f" {rest.strip()}"
                fixed_lines.append(fixed_line)
                continue
            # If it's a simple symbol (no operators), convert to s-domain
            elif re.match(r'^\w+$', expr.strip()):
                symbol = re.sub(r'\([st]\)', '', expr)
                if comp_id.lower().startswith('vsense') or _is_zero_literal(symbol):
                    fixed_line = f"{comp_id} {node1} {node2} 0"
                else:
                    fixed_line = f"{comp_id} {node1} {node2} s {symbol}"
                if rest.strip():
                    fixed_line += f" {rest.strip()}"
                fixed_lines.append(fixed_line)
                continue
            else:
                # Keep complex expressions as is
                fixed_line = f"{comp_id} {node1} {node2} {{{expr}}}"
                if rest.strip():
                    fixed_line += f" {rest.strip()}"
                fixed_lines.append(fixed_line)
                continue
        
        # Pattern: V1 1 0 ac V1 → V1 1 0 s V1
        # Also handles SPICE-style: V1 1 0 ac 220 25 → V1 1 0 s 220 (drop phase, Lcapy only accepts single value)
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+ac\s+(\S+)(.*)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, symbol, rest = match.groups()
            # ⚠️ Lcapy s-domain source only accepts ONE value, not "amplitude phase"
            # If rest contains additional values (like phase), ignore them
            if comp_id.lower().startswith('vsense') or _is_zero_literal(symbol):
                fixed_line = f"{comp_id} {node1} {node2} 0"
            else:
                # Only use the first value (amplitude), discard any trailing values (phase, etc.)
                fixed_line = f"{comp_id} {node1} {node2} s {symbol}"
            # Don't append rest - it may contain phase or other SPICE-specific values not supported by Lcapy
            fixed_lines.append(fixed_line)
            continue
        
        # Pattern: V1 1 0 Vin → V1 1 0 s Vin (when no domain specifier, add s)
        match = re.match(r'^([VI]\w*)\s+(\w+)\s+(\w+)\s+(\w+)$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2, value = match.groups()
            # If value is not a domain specifier (dc/ac/s/step), add s
            if value.lower() not in ['dc', 'ac', 's', 'step']:
                if comp_id.lower().startswith('vsense') or _is_zero_literal(value):
                    fixed_lines.append(f"{comp_id} {node1} {node2} 0")
                else:
                    fixed_lines.append(f"{comp_id} {node1} {node2} s {value}")
                continue
        
        # Keep line as is
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def _fix_wire_components(netlist: str) -> str:
    """Remove values from wire components (W has no value)"""
    lines = netlist.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append(line)
            continue
        
        # Pattern: W1 1 2 value → W1 1 2 (remove extra values)
        match = re.match(r'^(W\w*)\s+(\w+)\s+(\w+)\s+.+$', line, re.IGNORECASE)
        if match:
            comp_id, node1, node2 = match.groups()
            fixed_lines.append(f"{comp_id} {node1} {node2}")
            continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)
    """Check for invalid element values containing omega, j, or s expressions.
    
    Lcapy element values must be constants or symbolic names, NOT expressions
    containing omega (angular frequency), j (imaginary unit), or s (Laplace variable).
    
    Returns:
        List of error messages for invalid elements
    """
    errors = []
    
    # Patterns that indicate invalid values
    invalid_patterns = [
        # omega patterns
        (r'\bomega\b', 'omega (angular frequency)'),
        (r'\bω\b', 'ω (angular frequency)'),
        
        # j (imaginary unit) patterns - but not component names starting with j
        (r'[{(]\s*j\s*[*\s]', 'j (imaginary unit)'),
        (r'[*\s]\s*j\s*[})]', 'j (imaginary unit)'),
        (r'\bj\s*\*', 'j (imaginary unit)'),
        (r'\*\s*j\b', 'j (imaginary unit)'),
        
        # s expressions in element values (not source type 's')
        # Match s with operators: s+, s-, s*, s/, s**, (s, s)
        (r'\{[^}]*\bs\s*[\*\+\-\/\*]', 's (Laplace variable in expression)'),
        (r'\{[^}]*[\*\+\-\/]\s*s\b', 's (Laplace variable in expression)'),
        (r'\{[^}]*\bs\*\*', 's (Laplace variable with exponent)'),
        (r'\{[^}]*\(\s*s\s*[\*\+\-\/]', 's (Laplace variable in parentheses)'),
    ]
    
    lines = netlist.strip().split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#') or line.startswith('*'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        comp_name = parts[0]
        prefix = comp_name[0].upper()
        
        # Skip voltage/current sources (they can have 's' as domain specifier)
        if prefix in ['V', 'I']:
            # But check if there's an expression with s in the value part
            # Valid: V1 1 0 s Vin
            # Invalid: V1 1 0 {s*something}
            if len(parts) >= 4:
                value_part = ' '.join(parts[3:])
                # Check for s expressions in braces
                if re.search(r'\{[^}]*\bs\s*[\*\+\-\/\*\(]', value_part):
                    errors.append(f"Line {line_num}: {comp_name} has invalid s-expression in value: {value_part}")
            continue
        
        # For passive elements and controlled sources, check the value parts
        # Get the value portion (everything after node specifications)
        if prefix in ['R', 'L', 'C', 'G']:
            # 2-terminal: Rname Np Nm value
            value_start = 3
        elif prefix in ['E', 'H', 'F']:
            # E: 4-terminal VCVS or 2-terminal with opamp keyword
            # Check if 'opamp' keyword is present
            if 'opamp' in line.lower():
                # opamp format: E1 Nout 0 opamp Nplus Nminus [Ad]
                # Skip opamp lines for now (Ad check is separate)
                continue
            # 4-terminal: Ename Np Nm Ncp Ncm gain
            value_start = 5
        else:
            # Other components
            value_start = 3
        
        if len(parts) > value_start:
            value_part = ' '.join(parts[value_start:])
            
            for pattern, desc in invalid_patterns:
                if re.search(pattern, value_part, re.IGNORECASE):
                    errors.append(
                        f"Line {line_num}: {comp_name} has invalid value containing {desc}. "
                        f"Element values must be constants or symbolic names, not expressions. "
                        f"Value: {value_part}"
                    )
                    break
    
    return errors

def _fix_engineering_notation(netlist: str) -> str:
    """Convert engineering notation to numeric values.
    
    Lcapy may not support shorthand like 1m, 1k, 1u, etc.
    Convert to numeric: 1m → 0.001, 1k → 1000, etc.
    """
    # Engineering notation mapping
    notation_map = {
        'f': 1e-15,  # femto
        'p': 1e-12,  # pico
        'n': 1e-9,   # nano
        'u': 1e-6,   # micro
        'm': 1e-3,   # milli
        'k': 1e3,    # kilo
        'K': 1e3,    # kilo (uppercase)
        'M': 1e6,    # mega
        'G': 1e9,    # giga
        'T': 1e12,   # tera
    }
    
    lines = netlist.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Split line into parts
        parts = line.split()
        if len(parts) >= 4:
            # Check the value part (usually last or second-to-last)
            new_parts = []
            for i, part in enumerate(parts):
                # Skip component name and node numbers
                if i < 3:
                    new_parts.append(part)
                    continue
                
                # Try to convert engineering notation
                # Pattern: number followed by single letter (e.g., 1m, 10k, 4.7u)
                eng_match = re.match(r'^(\d+\.?\d*)([fpnumkKMGT])$', part)
                if eng_match:
                    value, suffix = eng_match.groups()
                    multiplier = notation_map.get(suffix, 1)
                    numeric_value = float(value) * multiplier
                    # Format nicely
                    if numeric_value >= 1:
                        new_parts.append(str(int(numeric_value)) if numeric_value == int(numeric_value) else str(numeric_value))
                    else:
                        new_parts.append(f"{numeric_value:.10g}")
                else:
                    new_parts.append(part)
            
            fixed_lines.append(' '.join(new_parts))
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)
