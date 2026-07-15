"""
Classify Node

Two-stage classification:
1. IR Type Classification: Determines which pipeline to use (sfg/netlist)
2. Component Detection: For Netlist prompt selection

Component Detection:
- Netlist: Detects resistor, capacitor, inductor, voltage_source, current_source, 
           controlled sources (vcvs, vccs, cccs, ccvs), opamp, etc.

This separation ensures:
- SFG cases can be detected and skipped while awaiting the later open-source release
- Netlist dynamic prompt selection in build nodes can use detected components
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils.response_parser import extract_text_content
from src.utils.metrics import format_duration, format_tokens

def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


# ============================================================
# Step 1: IR Type Classification Prompt
# ============================================================
CLASSIFY_IR_TYPE_PROMPT = """Classify the diagram type and extract analysis information.

## ir_type (choose one):

### "sfg" - Signal Flow Graph
Block diagrams with transfer function blocks G(s), H(s) and summing junctions.
This path is not available in the current open-source release.

### "netlist" - Circuit Analysis (Lcapy)
Linear circuits with R, L, C, sources, and controlled sources.
Use for: transfer function, voltage gain, impedance, frequency response.

## Decision Rule:
- Block diagram / signal flow → sfg
- Circuit schematic → netlist
- If unclear → netlist

## analysis_type:
Choose based on the requested final quantity, NOT just the presence of "s-domain".

- "transfer_function":
  Use ONLY when the question asks for a ratio/gain/transfer function such as:
  H(s), Vout/Vin, Vo/Vi, voltage gain, current gain, transfer function,
  transimpedance, transadmittance, frequency response.

- "transient_response":
  Use when the question asks for an absolute s-domain node voltage or branch current,
  not a ratio, such as:
  Vn1(s), Vn2(s), node voltage, nodal equation, voltage source current iv1,
  inductor current il1, branch current in s-domain.

- "dc":
  Use when the question asks for DC operating point, DC voltage/current,
  steady-state with dc sources only.

- "ac":
  Use when the question asks for sinusoidal AC steady-state response,
  magnitude/phase at frequency, or response vs omega.

Important disambiguation:
- "Derive the nodal equation for node 3 in the s-domain" => transient_response
- "What is iv1(s)?" => transient_response
- "Find H(s)=Vout/Vin" => transfer_function
- "Find voltage gain Vout/Vin" => transfer_function

## constraints:
Extract component values from the diagram/question:
- R, L, C values (e.g., "R1=10k, C1=1pF")
- Source values (e.g., "Vin=1V")
- Controlled source parameters (e.g., "gm=1mS")

## Output:
{"ir_type": "...", "analysis_type": "...", "constraints": "..."}

Output ONLY JSON."""


# ============================================================
# Step 2a: Component Detection Prompt (for Netlist)
# ============================================================
DETECT_NETLIST_COMPONENTS_PROMPT = """Identify ALL component types present in this circuit schematic.

Look carefully at the image and list ONLY the component types you see:

## Component Types:
- "resistor": R symbols, zigzag lines, rectangular boxes with R label
- "capacitor": C symbols, parallel plates, or labeled C values
- "inductor": L symbols, coils, or labeled L values
- "voltage_source": V symbols, circles with +/-, battery symbols
- "current_source": I symbols, circles with arrows
- "vcvs": Diamond voltage source controlled by voltage (E type, labeled with voltage ratio)
- "vccs": Diamond current source controlled by voltage (G/gm type, transconductance)
- "cccs": Diamond current source controlled by current (F type, current ratio)
- "ccvs": Diamond current source controlled by current (H type, transresistance)
- "opamp": Triangle with + and - inputs (operational amplifier symbol)
- "conductance": G symbols as 2-terminal elements

## Output Format:
{"detected_components": ["resistor", "capacitor", ...]}

List ONLY the types that appear in the image. Output ONLY JSON."""


class TerminalColors:
    """Terminal output colors"""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    END = '\033[0m'


def _extract_token_usage(response) -> Dict[str, int]:
    """Extract token usage information from LLM response"""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # try response_metadata (LangChain standard)
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
    
    # try usage_metadata attribute
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



def _encode_image(image_path: str) -> tuple:
    """Read and encode image"""
    import base64
    
    image_data = None
    media_type = "image/png"
    
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            # png jpg can be read correctly, so no need to set media_type based on file extension
            image_data = base64.b64encode(f.read()).decode("utf-8")
        if image_path.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
    
    return image_data, media_type


def classify_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """Two-stage classification:
    
    Stage 1: IR Type Classification
        - Always runs
        - Determines: sfg / netlist
        
    Stage 2: Component Detection (conditional)
        - Runs for Netlist unless a provided netlist is used
        - For Netlist: skipped if using provided netlist
        - Used for: dynamic prompt selection in build nodes
    """
    # read config
    config = _load_config()
    
    # Netlist config
    netlist_config = config.get("ir", {}).get("netlist", {})
    use_provided_netlist = netlist_config.get("use_provided_netlist", False)
    
    provided_netlist = state.get("provided_netlist")
    image_path = state["image_path"]
    question = state["question"]
    
    # encode image (both stages may use)
    image_data, media_type = _encode_image(image_path)
    
    # initialize metrics
    total_duration = 0
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    llm_calls = 0
    
    # ============================================================
    # Stage 1: IR Type Classification
    # ============================================================
    print(f"\n{TerminalColors.CYAN}{'=' * 60}")
    print(f"📋 CLASSIFY - Stage 1: IR Type")
    print(f"{'=' * 60}{TerminalColors.END}")
    
    stage1_start = time.time()
    
    # build messages
    messages = [SystemMessage(content=CLASSIFY_IR_TYPE_PROMPT)]
    if image_data:
        messages.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
            {"type": "text", "text": f"Question: {question}"}
        ]))
    else:
        messages.append(HumanMessage(content=f"Question: {question}"))
    
    # call LLM
    response = llm.invoke(messages)
    content = extract_text_content(response.content)
    
    # extract token usage
    stage1_tokens = _extract_token_usage(response)
    stage1_duration = time.time() - stage1_start
    
    # show response
    print(f"\n{TerminalColors.MAGENTA}📤 LLM Response:{TerminalColors.END}")
    print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
    print(f"   {content}")
    print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
    
    # parse result
    ir_result = _parse_ir_type(content)
    ir_type = ir_result.get("ir_type")
    analysis_type = ir_result.get("analysis_type")
    constraints = ir_result.get("constraints")
    
    # show result
    print(f"\n   {TerminalColors.GREEN}📊 IR Type: {ir_type}{TerminalColors.END}")
    print(f"   {TerminalColors.BLUE}📈 Analysis Type: {analysis_type}{TerminalColors.END}")
    if constraints:
        print(f"   {TerminalColors.YELLOW}📐 Constraints: {constraints}{TerminalColors.END}")
    print(f"   {TerminalColors.MAGENTA}⏱️  Time: {format_duration(stage1_duration)} | "
          f"Tokens: {format_tokens(stage1_tokens['total_tokens'])}{TerminalColors.END}")
    
    # accumulate statistics
    total_duration += stage1_duration
    total_tokens["input_tokens"] += stage1_tokens["input_tokens"]
    total_tokens["output_tokens"] += stage1_tokens["output_tokens"]
    total_tokens["total_tokens"] += stage1_tokens["total_tokens"]
    llm_calls += 1
    
    # initialize component detection result
    detected_components = None
    
    # ============================================================
    # Stage 2: Component Detection (conditional)
    # ============================================================
    # determine whether to perform component detection based on ir_type:
    # - Netlist: run unless using provided netlist
    
    need_netlist_detection = (
        ir_type == "netlist" and 
        (not use_provided_netlist or not provided_netlist)
    )
    
    need_component_detection = need_netlist_detection
    
    if need_component_detection:
        detection_prompt = DETECT_NETLIST_COMPONENTS_PROMPT
        parse_func = _parse_netlist_components
        detection_hint = "Identify all component types in this circuit."
        pipeline_name = "Netlist"
        
        print(f"\n{TerminalColors.CYAN}{'=' * 60}")
        print(f"🔧 CLASSIFY - Stage 2: Component Detection ({pipeline_name})")
        print(f"{'=' * 60}{TerminalColors.END}")
        
        stage2_start = time.time()
        
        # build messages (only image, no question)
        messages = [SystemMessage(content=detection_prompt)]
        if image_data:
            messages.append(HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                {"type": "text", "text": detection_hint}
            ]))
        else:
            # if no image, skip component detection
            print(f"   {TerminalColors.YELLOW}⚠️ No image available, skipping component detection{TerminalColors.END}")
            detected_components = []
        
        if image_data:
            # call LLM
            response = llm.invoke(messages)
            content = extract_text_content(response.content)
            
            # extract token usage
            stage2_tokens = _extract_token_usage(response)
            stage2_duration = time.time() - stage2_start
            
            # Display LLM response
            print(f"\n{TerminalColors.MAGENTA}📤 LLM Response:{TerminalColors.END}")
            print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
            print(f"   {content}")
            print(f"{TerminalColors.MAGENTA}{'─' * 40}{TerminalColors.END}")
            
            # parse components, validate, normalize, deduplicate (using corresponding parse function)
            detected_components = parse_func(content)
            
            # show result
            print(f"\n   {TerminalColors.CYAN}🔧 Detected: {detected_components}{TerminalColors.END}")
            print(f"   {TerminalColors.MAGENTA}⏱️  Time: {format_duration(stage2_duration)} | "
                  f"Tokens: {format_tokens(stage2_tokens['total_tokens'])}{TerminalColors.END}")
            
            # accumulate statistics
            total_duration += stage2_duration
            total_tokens["input_tokens"] += stage2_tokens["input_tokens"]
            total_tokens["output_tokens"] += stage2_tokens["output_tokens"]
            total_tokens["total_tokens"] += stage2_tokens["total_tokens"]
            llm_calls += 1
    else:
        # explain why component detection is skipped
        if ir_type == "netlist":
            if use_provided_netlist and provided_netlist:
                print(f"\n   {TerminalColors.YELLOW}ℹ️  Component detection skipped (using provided netlist){TerminalColors.END}")
        elif ir_type == "sfg":
            print(f"\n   {TerminalColors.YELLOW}ℹ️  SFG support is pending a later open-source release{TerminalColors.END}")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n   {TerminalColors.MAGENTA}{'─' * 40}")
    print(f"   📊 Total: {format_duration(total_duration)} | "
          f"Tokens: {format_tokens(total_tokens['total_tokens'])} | "
          f"LLM Calls: {llm_calls}{TerminalColors.END}")
    print()
    
    # build metrics
    stage_metrics = {
        "classify": {
            "duration_seconds": round(total_duration, 2),
            "tokens": total_tokens,
            "llm_calls": llm_calls,
        }
    }
    
    # merge existing metrics
    existing_metrics = state.get("metrics", {})
    if existing_metrics:
        stage_metrics = {**existing_metrics, **stage_metrics}
    
    return {
        "ir_type": ir_type,
        "analysis_type": analysis_type,
        "constraints": constraints,
        "detected_components": detected_components,
        "metrics": stage_metrics,
    }


def _extract_first_json_object(content: str) -> Optional[str]:
    """Extract the first valid JSON object from an LLM response."""
    if not content:
        return None

    decoder = json.JSONDecoder()

    # Prefer explicit JSON code fences; models often put the intended object there
    # before continuing with explanatory text.
    for match in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', content, re.IGNORECASE):
        candidate = match.group(1).strip()
        if not candidate.startswith("{"):
            continue
        try:
            _, end = decoder.raw_decode(candidate)
            return candidate[:end]
        except json.JSONDecodeError:
            continue

    # Fall back to scanning for the first decodable object. This avoids greedy
    # regex captures that accidentally include LaTeX braces after the JSON.
    for match in re.finditer(r'\{', content):
        candidate = content[match.start():]
        try:
            _, end = decoder.raw_decode(candidate)
            return candidate[:end]
        except json.JSONDecodeError:
            continue

    return None


def _parse_ir_type(content: str) -> Dict[str, Any]:
    """Parse IR type classification result.
    
    Raises error if JSON parsing fails.
    """
    json_text = _extract_first_json_object(content)
    if not json_text:
        raise ValueError(f"LLM classification failed: No JSON found in response.\nResponse: {content[:500]}")
    
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM classification failed: Invalid JSON format.\nError: {e}\nResponse: {content[:500]}")
    
    ir_type = result.get("ir_type")
    if ir_type not in ["sfg", "netlist"]:
        raise ValueError(f"LLM classification failed: Invalid ir_type '{ir_type}'.\nMust be one of: sfg, netlist.\nResponse: {content[:500]}")
    
    return {
        "ir_type": ir_type,
        "analysis_type": result.get("analysis_type", "transfer_function"),
        "constraints": result.get("constraints"),
    }


def _parse_netlist_components(content: str) -> List[str]:
    """Parse netlist component detection result.
    
    Returns normalized list of circuit component types.
    """
    json_text = _extract_first_json_object(content)
    if not json_text:
        return []
    
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        return []
    
    detected = result.get("detected_components", [])
    if not isinstance(detected, list):
        return []
    
    # normalize component names
    valid_components = {
        "resistor", "capacitor", "inductor", "conductance",
        "voltage_source", "current_source",
        "vcvs", "vccs", "cccs", "ccvs", "controlled_source",
        "opamp", "amplifier",
        "open", "port"
    }
    
    normalized = []
    for comp in detected:
        comp_lower = comp.lower().strip()
        if comp_lower in valid_components:
            normalized.append(comp_lower)
        # handle common aliases
        elif comp_lower in ["r", "res"]:
            normalized.append("resistor")
        elif comp_lower in ["c", "cap"]:
            normalized.append("capacitor")
        elif comp_lower in ["l", "ind", "coil"]:
            normalized.append("inductor")
        elif comp_lower in ["v", "vs", "vsource", "voltage"]:
            normalized.append("voltage_source")
        elif comp_lower in ["i", "is", "isource", "current"]:
            normalized.append("current_source")
        elif comp_lower in ["op-amp", "op_amp", "operational_amplifier"]:
            normalized.append("opamp")
        elif comp_lower in ["e", "e_source"]:
            normalized.append("vcvs")
        elif comp_lower in ["g", "gm", "g_source", "transconductance"]:
            normalized.append("vccs")
        elif comp_lower in ["f", "f_source"]:
            normalized.append("cccs")
        elif comp_lower in ["h", "h_source"]:
            normalized.append("ccvs")
    
    return list(set(normalized))  # deduplicate
