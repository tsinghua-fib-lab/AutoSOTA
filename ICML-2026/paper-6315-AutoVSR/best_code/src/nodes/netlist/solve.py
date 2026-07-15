"""
Solve Netlist Agent - Symbolic Circuit Analysis (Plan-and-Execute)

Focuses on:
- Transfer Function Analysis
- Impedance Analysis
- State-Space Analysis
- AC/DC Symbolic Analysis
"""

from typing import Any, Dict, List, Optional, TypedDict
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from ...tools import create_netlist_tools
from ...utils.response_parser import extract_text_content


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


def print_separator(title: str = "", char: str = "=", length: int = 60):
    if title:
        side_len = (length - len(title) - 2) // 2
        print(f"\n{TerminalColors.CYAN}{char * side_len} {title} {char * side_len}{TerminalColors.END}")
    else:
        print(f"{TerminalColors.CYAN}{char * length}{TerminalColors.END}")


def is_valid_final_answer(answer: str) -> bool:
    """
    Verify if final_answer is a valid answer, not transitional text.
    
    Invalid final answer examples:
    - "Proceeding to step 2."
    - "I will now calculate..."
    - "Next, I need to..."
    """
    if not answer or len(answer.strip()) < 2:
        return False
    
    answer_lower = answer.lower().strip()
    
    # Keywords for transitional text (indicating no real answer yet)
    invalid_patterns = [
        "proceeding to",
        "proceed to", 
        "moving to",
        "next step",
        "i will",
        "i need to",
        "let me",
        "now i",
        "going to",
        "continue to",
        "step 2",
        "step 3",
        "step 4",
        "step 5",
    ]
    
    for pattern in invalid_patterns:
        if pattern in answer_lower:
            print(f"{TerminalColors.YELLOW}⚠️ Ignoring invalid FINAL ANSWER: '{answer[:50]}...' (contains transitional text){TerminalColors.END}")
            return False
    
    return True


def print_step_header(step_num: int):
    print(f"\n{TerminalColors.BOLD}{TerminalColors.BLUE}📍 Step {step_num}{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'-' * 50}{TerminalColors.END}")


def print_action(action: str, action_input: dict):
    print(f"\n{TerminalColors.BLUE}🔧 Action: {TerminalColors.BOLD}{action}{TerminalColors.END}")
    if action_input:
        print(f"   Parameters: {json.dumps(action_input, indent=6, ensure_ascii=False)}")


def print_observation(observation: str):
    print(f"\n{TerminalColors.GREEN}👁️ Observation:{TerminalColors.END}")
    for line in str(observation).split('\n'):
        print(f"   {line}")


def print_thought(thought: str):
    print(f"{TerminalColors.YELLOW}💭 Thought:{TerminalColors.END}")
    for line in thought.split('\n'):
        print(f"   {line}")


def print_final_answer(answer: str):
    print(f"\n{TerminalColors.BOLD}{TerminalColors.GREEN}{'=' * 60}")
    print(f"✅ Final Answer")
    print(f"{'=' * 60}{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}{answer}{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}{'=' * 60}{TerminalColors.END}\n")


def print_error(error: str):
    print(f"\n{TerminalColors.RED}❌ Error: {error}{TerminalColors.END}")


# ============================================================
# Netlist Plan-and-Execute Agent State
# ============================================================

class PlanStep(TypedDict):
    step_num: int
    description: str
    tool: Optional[str]
    status: str  # pending, executing, completed, failed
    result: Optional[str]


class NetlistReActStep(TypedDict):
    step_num: int
    thought: str
    action: Optional[str]
    action_input: Optional[Dict]
    observation: Optional[str]


class NetlistReActState(TypedDict, total=False):
    """State for the Netlist Plan-and-Execute agent"""
    # Problem context
    question: str
    analysis_type: Optional[str]
    ir_dict: Dict[str, Any]
    
    # Conversation
    messages: List[Any]
    
    # Plan-and-Execute state
    plan: List[PlanStep]
    current_plan_step: int
    _step_retry_count: int
    
    # ReAct loop state
    current_step: int
    max_steps: int
    steps: List[NetlistReActStep]
    
    # Tools
    available_tools: List[str]
    
    # Output
    final_answer: Optional[str]
    is_finished: bool
    
    # Token tracking
    total_tokens: int
    input_tokens: int
    output_tokens: int
    llm_calls: int


# ============================================================
# Netlist-specific prompt
# ============================================================

NETLIST_SYSTEM_PROMPT = """You are an expert circuit analyst.

## Problem:
{question}

## Circuit (Already Loaded):
{ir_info}

## Important:
The tools provide INTERMEDIATE analysis results (e.g., transfer function, impedance expressions).
YOU must reason about the tool outputs to derive the FINAL ANSWER based on the question:
- If the question asks for a specific value, substitute component values
- If it's multiple choice, compare your result with each option
- Simplify expressions if needed for comparison
For multiple choice questions:
1. Use tools to compute the expression
2. Compare/simplify the result with each option
3. Output ONLY the option number (1-5) as your final answer

## Output Format:
- Expressions: Python/SymPy format (NOT LaTeX)
- Numeric: plain numbers (e.g., 3.14, 42)
- Multiple choice: just the option number (1, 2, 3, 4, or 5)

End with: FINAL ANSWER: <your answer>"""


PLAN_PROMPT = """Generate a concise problem-solving plan.

Rules:
- Each step: tool call (tool: "name") OR reasoning (tool: null)
- Keep it simple: 2-4 steps is usually enough
- LAST step MUST be: "Summarize and provide final answer" with tool: null
- ⚠️ COMPLEXITY: For circuits with >8 nodes, use this 2-step flow:
2. voltage_gain or transfer_function → now works fast with numerical values
CRITICAL: Output ONLY the JSON array. NO explanations, NO reasoning, NO other text.

```json
[
  {{"step": 1, "description": "...", "tool": "tool_name_or_null"}},
  {{"step": N, "description": "Summarize and provide final answer", "tool": null}}
]
```
"""


EXECUTE_PROMPT = """Execute the current step of the plan.

## Current Plan:
{plan_display}

## Current Step ({step_num}): {step_description}
Tool to use: {tool_name}

## Instructions:
- Focus ONLY on completing this specific step
- Call the appropriate tool with correct parameters
- If no tool needed, provide your analysis

## Output Format:
- Symbolic expressions: Python/SymPy format (NOT LaTeX!)
- Multiple choice: just the option number (1-5)
"""


FINAL_ANSWER_PROMPT = """EXTRACT the final answer from tool results above.

⚠️ STRICT RULES:
1. If a tool returned an equation such as "H(s) = ...", "Vn1(s) = ...",
   or "IV1(s) = ...", copy that equation exactly as the final answer
2. Do NOT re-derive, re-calculate, or simplify - the tool result IS the correct answer
3. Do NOT output 0 unless the tool explicitly returned 0
4. For MCQ: compare tool result with options, output the matching number (1-5)

Output (ONE LINE ONLY - NO REASONING):
FINAL ANSWER: <copy the tool result or option number>"""


# ============================================================
# Netlist ReAct Agent Nodes
# ============================================================

class NetlistReActAgentNodes:
    """Nodes for the Netlist Plan-and-Execute agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = {}
        self.tools_by_name = {}

    def _is_transient_response(self, state: NetlistReActState) -> bool:
        """Return True when this netlist problem asks for an absolute s-domain response."""
        if state.get("analysis_type") == "transient_response":
            return True

        q = (state.get("question") or "").lower()
        if "transfer function" in q or "gain" in q or "ratio" in q:
            return False

        return (
            "nodal equation" in q
            or "node voltage" in q
            or "voltage source current" in q
            or re.search(r"\bvn\s*\d+\s*\(s\)", q) is not None
            or re.search(r"\b(?:iv|il|ieint)\s*\d+\s*\(s\)", q) is not None
            or "in s-domain" in q
        )

    def _transient_tool_for_question(self, question: str) -> tuple[str, str]:
        """Choose the deterministic MNA tool for transient-response questions."""
        q = (question or "").lower()
        asks_for_current = (
            re.search(r"\b(?:iv|il|ieint)\s*\d+\b", q) is not None
            or "voltage source current" in q
            or "branch current" in q
            or re.search(r"\bcurrent\b", q) is not None
        )
        if asks_for_current:
            return "solve_branch_current", "Solve for the requested branch current"
        return "solve_node_voltage", "Solve for the requested node voltage"

    def _normalize_transient_tool_args(
        self,
        action: Optional[str],
        action_input: Optional[Dict],
        state: NetlistReActState,
    ) -> Dict:
        """Normalize LLM arguments for MNA transient tools."""
        args = dict(action_input or {})
        question = state.get("question") or ""

        if action == "solve_branch_current":
            raw_target = (
                args.get("target")
                or args.get("branch")
                or args.get("source")
                or args.get("input_source")
                or args.get("voltage_source")
                or ""
            )
            target = str(raw_target).strip()

            if not target:
                match = re.search(r"\b(iv|il|ieint)\s*(\d+)\b", question, re.IGNORECASE)
                target = f"{match.group(1)}{match.group(2)}" if match else "iv1"

            # The MNA tool expects iv1/il1/ieint1. Convert common source names.
            match = re.fullmatch(r"[Vv]\s*(\d+)", target)
            if match:
                target = f"iv{match.group(1)}"

            match = re.fullmatch(r"(iv|il|ieint)\s*(\d+)", target, re.IGNORECASE)
            if match:
                target = f"{match.group(1).lower()}{match.group(2)}"

            args = {"target": target}

        elif action == "solve_node_voltage":
            raw_node = args.get("node") or args.get("output_node") or args.get("target") or ""
            node = str(raw_node).strip()

            if "," in node:
                node = node.split(",", 1)[0].strip()

            match = re.fullmatch(r"[Vv]n?\s*(\d+)(?:\(s\))?", node)
            if match:
                node = match.group(1)

            if not node:
                match = re.search(r"\bnode\s*(\d+)\b", question, re.IGNORECASE)
                node = match.group(1) if match else "1"

            args = {"node": node}

        return args
    
    def initialize_tools(self, state: NetlistReActState) -> NetlistReActState:
        """Initialize Netlist tools"""
        ir_dict = state["ir_dict"]
        
        # Create Netlist tools
        tools = create_netlist_tools(ir_dict)
        transient_tool_names = {
            "solve_node_voltage",
            "solve_branch_current",
            "symbolic_equiv_check",
        }
        if self._is_transient_response(state):
            tools = [tool for tool in tools if tool.name in transient_tool_names]
        else:
            tools = [tool for tool in tools if tool.name not in transient_tool_names]
        self.tools_by_name = {tool.name: tool for tool in tools}
        
        # Format IR info
        ir_info = self._format_netlist_info(ir_dict)
        
        return {
            **state,
            "available_tools": [tool.name for tool in tools],
            "plan": [],
            "current_plan_step": 0,
            "_step_retry_count": 0,
            "messages": [
                SystemMessage(content=NETLIST_SYSTEM_PROMPT.format(
                    question=state["question"],
                    ir_info=ir_info,
                )),
                HumanMessage(content=f"Solve: {state['question']}")
            ],
        }
    
    def _format_netlist_info(self, ir_dict: Dict[str, Any]) -> str:
        """Format Netlist info"""
        lines = []
        netlist = ir_dict.get("netlist", "")
        
        if netlist:
            netlist_lines = netlist.strip().split("\n")
            lines.append(f"Circuit Components ({len(netlist_lines)} total):")
            for nl in netlist_lines[:15]:
                lines.append(f"  {nl}")
            if len(netlist_lines) > 15:
                lines.append(f"  ... and {len(netlist_lines) - 15} more")
            
            # Extract available nodes list
            nodes = set()
            output_node_hint = None
            for nl in netlist_lines:
                parts = nl.strip().split()
                if len(parts) >= 3:
                    for p in parts[1:4]:
                        if p.isdigit() or p == '0':
                            nodes.add(p)
                    # Identify output node: Cc is usually connected to output
                    if parts[0].lower().startswith('cc'):
                        output_node_hint = parts[1] if parts[1] != '0' else (parts[2] if len(parts) > 2 else None)
            
            if nodes:
                sorted_nodes = sorted(nodes, key=lambda x: int(x) if x.isdigit() else 0)
                lines.append(f"\n📋 Available nodes: {sorted_nodes}")
                if output_node_hint:
                    lines.append(f"💡 Likely output node: {output_node_hint} (connected to Cc)")
            
            # ⭐ Extract all symbolic parameters (gm*, ro*, rpi*, etc.)
            gm_params = sorted(set(re.findall(r'\bgm\d+\b', netlist)))
            ro_params = sorted(set(re.findall(r'\bro\d+\b', netlist)))
            other_params = sorted(set(re.findall(r'\b(rpi|gds|Cgs|Cgd|beta|rd)\d*\b', netlist, re.IGNORECASE)))
            
            if gm_params or ro_params or other_params:
                lines.append(f"\n🔧 Symbolic parameters to substitute (MUST include ALL in batch_calc_transistors):")
                if gm_params:
                    lines.append(f"   gm: {', '.join(gm_params)}")
                if ro_params:
                    lines.append(f"   ro: {', '.join(ro_params)}")
                if other_params:
                    lines.append(f"   other: {', '.join(other_params)}")
                
                # Complexity warning: symbolic analysis may be slow when nodes > 8
                if len(nodes) > 8:
                    lines.append(f"\n⚠️ COMPLEXITY: This circuit has {len(nodes)} nodes (>8).")
                    lines.append("   Use batch_calc_transistors first to substitute ALL gm/ro values:")
                    # Generate example call, containing all actual parameters
                    example_specs = []
                    for gm in gm_params[:5]:  # Only show first 5 as examples
                        num = gm[2:]  # Extract numeric part
                        ro = f"ro{num}" if f"ro{num}" in ro_params else ""
                        example_specs.append(f"{gm}/{ro}:ID:n:λ")
                    if len(gm_params) > 5:
                        example_specs.append("...")
                    lines.append(f'   batch_calc_transistors("{"; ".join(example_specs)}")')
                    lines.append("   ⚠️ CRITICAL: Include ALL parameters listed above, not just some!")
        
        # Display I/O mapping (supports multi-input multi-output)
        io_mappings = ir_dict.get('io_mappings')
        if io_mappings and len(io_mappings) > 0:
            lines.append(f"\n🔌 I/O Mappings ({len(io_mappings)} pair{'s' if len(io_mappings) > 1 else ''}):")
            for mapping in io_mappings:
                input_src = mapping.get('input_source', 'unknown')
                output_nodes = mapping.get('output_nodes')
                # Format output
                if output_nodes is None:
                    output_str = "* (all nodes)"
                elif isinstance(output_nodes, list):
                    output_str = f"V({output_nodes[0]}) - V({output_nodes[1]})" if len(output_nodes) > 1 else f"V({output_nodes[0]})"
                else:
                    output_str = f"V({output_nodes})"
                lines.append(f"   {input_src} → {output_str}")
        else:
            # Backward compatibility: single input/output
            lines.append(f"\nInput Source: {ir_dict.get('input_source', 'V1')}")
            output_node = ir_dict.get('output_node', 'N/A')
            if output_node and output_node != 'N/A':
                lines.append(f"Output Node: {output_node}")
        
        # Display constraints extracted from diagram/question
        constraints = ir_dict.get('constraints')
        if constraints:
            lines.append(f"\n📐 Constraints (from diagram/question):")
            lines.append(f"   {constraints}")
        
        return "\n".join(lines)
    
    def _format_progress(self, state: NetlistReActState) -> str:
        """Format current progress for think node"""
        if not state["steps"]:
            return "No actions taken yet. Starting fresh."
        
        lines = []
        for step in state["steps"][-3:]:  # Show last 3 steps
            lines.append(f"Step {step['step_num']}:")
            if step.get("thought"):
                lines.append(f"  Thought: {step['thought'][:200]}...")
            if step.get("action"):
                lines.append(f"  Action: {step['action']}({step.get('action_input', {})})")
            if step.get("observation"):
                obs = str(step['observation'])[:300]
                lines.append(f"  Result: {obs}...")
        
        return "\n".join(lines) if lines else "No progress yet."
    
    
    def _format_plan_display(self, plan: List[PlanStep], current_idx: int) -> str:
        """Format plan for display"""
        lines = []
        for i, step in enumerate(plan):
            status_icon = "⏳" if step["status"] == "pending" else \
                         "🔄" if step["status"] == "executing" else \
                         "✅" if step["status"] == "completed" else "❌"
            marker = ">>> " if i == current_idx else "    "
            lines.append(f"{marker}{status_icon} Step {step['step_num']}: {step['description']}")
            if step.get("result"):
                result_preview = str(step['result'])[:100]
                lines.append(f"        Result: {result_preview}...")
        return "\n".join(lines)
    
    
    def plan(self, state: NetlistReActState) -> NetlistReActState:
        """Create execution plan"""
        print_separator("PLANNING PHASE", "=", 60)
        print(f"{TerminalColors.MAGENTA}📋 Creating Execution Plan...{TerminalColors.END}")
        
        # Build plan prompt
        plan_message = HumanMessage(content=PLAN_PROMPT)
        
        # Bind tools so LLM knows available tools
        tools = [self.tools_by_name[name] for name in self.tools_by_name]
        llm_with_tools = self.llm.bind_tools(tools)
        
        start_time = time.time()
        response = llm_with_tools.invoke(state["messages"] + [plan_message])
        elapsed = time.time() - start_time
        print(f"{TerminalColors.CYAN}⏱️ Planning Time: {elapsed:.2f}s{TerminalColors.END}")
        
        plan_text = extract_text_content(response.content)
        
        # Extract PLAN section
        if "PLAN:" in plan_text.upper():
            plan_start = plan_text.upper().find("PLAN:")
            plan_text = plan_text[plan_start:]
        
        print(f"\n{TerminalColors.YELLOW}📝 Generated Plan:{TerminalColors.END}")
        print(plan_text[:500] if len(plan_text) > 500 else plan_text)
        
        # Parse plan
        plan = self._parse_plan(plan_text, state)
        
        print(f"\n{TerminalColors.GREEN}✅ Parsed Plan ({len(plan)} steps):{TerminalColors.END}")
        for step in plan:
            tool_info = f" -> Tool: {step['tool']}" if step['tool'] else ""
            print(f"   {step['step_num']}. {step['description']}{tool_info}")
        
        return {
            **state,
            "messages": state["messages"] + [plan_message, response],
            "plan": plan,
            "current_plan_step": 0,
        }
    
    def _parse_plan(self, plan_text: str, state: NetlistReActState) -> List[PlanStep]:
        """Parse plan from LLM response (supports JSON and text formats)"""
        plan = []
        
        # Try JSON format first
        try:
            json_match = re.search(r'\[[\s\S]*\]', plan_text)
            if json_match:
                steps = json.loads(json_match.group())
                for step_data in steps:
                    step_num = step_data.get("step", len(plan) + 1)
                    description = step_data.get("description", "")
                    tool = step_data.get("tool")
                    if tool is None or (isinstance(tool, str) and tool.lower() in ['none', 'null']):
                        tool = None
                    
                    plan.append(PlanStep(
                        step_num=step_num,
                        description=description,
                        tool=tool,
                        status="pending",
                        result=None,
                    ))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        
        # Fallback: parse text format
        if not plan:
            lines = plan_text.split('\n')
            for line in lines:
                line = line.strip()
                match = re.match(r'^(\d+)[.\)]\s*(.+?)(?:\s*->\s*Tool:\s*(\w+|none))?$', line, re.IGNORECASE)
                if match:
                    step_num = int(match.group(1))
                    description = match.group(2).strip()
                    tool = match.group(3)
                    if tool and tool.lower() in ['none', 'null']:
                        tool = None
                    
                    plan.append(PlanStep(
                        step_num=step_num,
                        description=description,
                        tool=tool,
                        status="pending",
                        result=None,
                    ))
        
        question = state.get("question") or ""
        is_transient = self._is_transient_response(state)

        # Default plan if none parsed
        if not plan:
            default_tool = "node_transfer"
            default_description = "Analyze circuit and compute result"
            if is_transient:
                default_tool, default_description = self._transient_tool_for_question(question)
            plan = [
                PlanStep(step_num=1, description=default_description, tool=default_tool, status="pending", result=None),
                PlanStep(step_num=2, description="Summarize and provide final answer", tool=None, status="pending", result=None),
            ]

        # Transient-response questions must use MNA tools, even if the LLM
        # generated a generic transfer-function plan.
        if is_transient:
            transient_tool, transient_description = self._transient_tool_for_question(question)
            has_tool_step = False
            for step in plan:
                if step.get("tool"):
                    step["tool"] = transient_tool
                    step["description"] = transient_description
                    has_tool_step = True
                    break
            if not has_tool_step:
                plan.insert(0, PlanStep(
                    step_num=1,
                    description=transient_description,
                    tool=transient_tool,
                    status="pending",
                    result=None,
                ))
            for idx, step in enumerate(plan, start=1):
                step["step_num"] = idx
        else:
            target_element = self._target_element_for_question(question, state)
            if target_element and "element_transfer" in self.tools_by_name:
                has_tool_step = False
                for step in plan:
                    if step.get("tool"):
                        step["tool"] = "element_transfer"
                        step["description"] = f"Calculate the transfer function to element {target_element}"
                        has_tool_step = True
                        break
                if not has_tool_step:
                    plan.insert(0, PlanStep(
                        step_num=1,
                        description=f"Calculate the transfer function to element {target_element}",
                        tool="element_transfer",
                        status="pending",
                        result=None,
                    ))
                for idx, step in enumerate(plan, start=1):
                    step["step_num"] = idx
        
        return plan

    def _target_element_for_question(self, question: str, state: NetlistReActState) -> Optional[str]:
        """Return the requested output element when the question names one."""
        elements = self._netlist_element_names(state)
        if not elements:
            return None

        question_text = question or ""
        target_patterns = [
            r'\bfrom\s+\w+\s+to\s+([A-Za-z]+\w*)\b',
            r'\bto\s+([A-Za-z]+\w*)\s+in\s+this\s+circuit\b',
            r'\bacross\s+([A-Za-z]+\w*)\b',
        ]
        candidates = []
        for pattern in target_patterns:
            candidates.extend(re.findall(pattern, question_text, flags=re.IGNORECASE))

        element_lookup = {name.lower(): name for name in elements}
        for candidate in candidates:
            match = element_lookup.get(candidate.lower())
            if match:
                return match
        return None

    def _netlist_element_names(self, state: NetlistReActState) -> List[str]:
        """Extract element names from the generated/provided netlist."""
        ir_dict = state.get("ir_dict") or {}
        netlist_text = ir_dict.get("netlist") or ir_dict.get("netlist_code") or state.get("ir_code") or ""
        elements = []
        for line in str(netlist_text).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "*", ";")):
                continue
            name = stripped.split()[0]
            if re.match(r'^[A-Za-z]+\w*$', name):
                elements.append(name)
        return elements

    def _source_for_question(self, question: str, state: NetlistReActState) -> str:
        match = re.search(r'\bfrom\s+([A-Za-z]+\w*)\s+to\b', question or "", flags=re.IGNORECASE)
        if match:
            return match.group(1)
        ir_dict = state.get("ir_dict") or {}
        return ir_dict.get("input_source") or "V1"

    def _normalize_transfer_tool_args(self, action: Optional[str], action_input: Optional[Dict], state: NetlistReActState) -> Dict:
        """Normalize args for deterministic transfer-function tool execution."""
        args = dict(action_input or {})
        if action == "element_transfer":
            target_element = self._target_element_for_question(state.get("question", ""), state)
            if target_element:
                args["output_element"] = target_element
                args.pop("output_node", None)
            if not args.get("input_source"):
                args["input_source"] = self._source_for_question(state.get("question", ""), state)
        return args
    
    def execute(self, state: NetlistReActState) -> NetlistReActState:
        """Execute current step of the plan"""
        plan = state["plan"]
        current_idx = state["current_plan_step"]
        current_step = state["current_step"]
        
        # Check retry count to prevent infinite loop
        max_retries = 2
        retry_count = state.get("_retry_count", 0)
        if retry_count >= max_retries:
            print(f"{TerminalColors.RED}⚠️ Max retries ({max_retries}) reached, moving to next step{TerminalColors.END}")
            # Force move to next step
            if current_idx < len(plan):
                plan[current_idx]["status"] = "skipped"
                plan[current_idx]["result"] = f"Skipped after {max_retries} failed attempts"
            return {
                **state, 
                "plan": plan,
                "current_plan_step": current_idx + 1,
                "_retry_count": 0,
            }
        
        if current_idx >= len(plan):
            return {**state, "is_finished": True}
        
        current_plan_step = plan[current_idx]
        
        print_separator(f"EXECUTE STEP {current_plan_step['step_num']}", "-", 50)
        print(f"{TerminalColors.BLUE}📍 {current_plan_step['description']}{TerminalColors.END}")
        if current_plan_step.get('tool'):
            print(f"   Tool: {current_plan_step['tool']}")
        
        plan[current_idx]["status"] = "executing"
        
        # Determine if it is the last step
        is_final_step = (
            current_plan_step.get('tool') is None and 
            ('final' in current_plan_step['description'].lower() or 
             'summarize' in current_plan_step['description'].lower() or
             'answer' in current_plan_step['description'].lower())
        )
        
        # Use different prompt
        if is_final_step:
            # Collect tool results from previous steps
            tool_results = []
            for step in plan[:current_idx]:
                if step.get("status") == "completed" and step.get("result"):
                    tool_results.append(f"Step {step['step_num']}: {step['result']}")
            
            results_text = "\n".join(tool_results) if tool_results else "No tool results."
            
            # Streamline context: keep only question, tool results, instructions
            final_messages = [
                SystemMessage(content="You are an answer extractor. Copy the tool result as the final answer."),
                HumanMessage(content=f"""Question: {state['question']}

Tool Results:
{results_text}

{FINAL_ANSWER_PROMPT}""")
            ]
            
            start_time = time.time()
            response = self.llm.invoke(final_messages)
            elapsed = time.time() - start_time
            print(f"{TerminalColors.CYAN}⏱️ Execute Time: {elapsed:.2f}s{TerminalColors.END}")
            
            thought = extract_text_content(response.content)
            final_answer = None
            
            if "FINAL ANSWER:" in thought.upper():
                match = re.search(r'FINAL ANSWER[:\s]*(.+?)(?:\n|$)', thought, re.IGNORECASE | re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
            
            if final_answer:
                print_final_answer(final_answer)
            
            plan[current_idx]["status"] = "completed"
            plan[current_idx]["result"] = final_answer or thought
            
            return {
                **state,
                "messages": state["messages"] + [HumanMessage(content="[Final Answer Step]"), response],
                "current_step": state["current_step"] + 1,
                "current_plan_step": current_idx + 1,
                "steps": state["steps"] + [NetlistReActStep(
                    step_num=state["current_step"] + 1,
                    thought=thought,
                    action=None,
                    action_input=None,
                    observation=None,
                )],
                "plan": plan,
                "final_answer": final_answer,
                "is_finished": True,
            }
        else:
            execute_message = HumanMessage(content=EXECUTE_PROMPT.format(
                plan_display=self._format_plan_display(plan, current_idx),
                step_num=current_plan_step['step_num'],
                step_description=current_plan_step['description'],
                tool_name=current_plan_step.get('tool') or 'none (reasoning only)',
            ))
        
        # Bind only the planned tool when a step specifies one. This prevents
        # transient-response questions from accidentally calling node_transfer.
        planned_tool = current_plan_step.get('tool')
        if planned_tool and planned_tool in self.tools_by_name:
            tools = [self.tools_by_name[planned_tool]]
        else:
            tools = [self.tools_by_name[name] for name in state["available_tools"] if name in self.tools_by_name]
        llm_with_tools = self.llm.bind_tools(tools)
        
        start_time = time.time()
        response = llm_with_tools.invoke(state["messages"] + [execute_message])
        elapsed = time.time() - start_time
        print(f"{TerminalColors.CYAN}⏱️ Execute Time: {elapsed:.2f}s{TerminalColors.END}")
        
        # Parse response
        thought = extract_text_content(response.content)
        action = None
        action_input = None
        final_answer = None
        is_finished = False
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            action = tool_call["name"]
            action_input = tool_call.get("args", {})
            if planned_tool and action != planned_tool:
                action = planned_tool
            if self._is_transient_response(state):
                action_input = self._normalize_transient_tool_args(action, action_input, state)
            else:
                action_input = self._normalize_transfer_tool_args(action, action_input, state)
            print(f"{TerminalColors.BLUE}🔧 Tool Call: {action}{TerminalColors.END}")
            if action_input:
                print(f"   Args: {json.dumps(action_input)}")
        
        # Check for final answer
        if "FINAL ANSWER:" in thought.upper():
            match = re.search(r'FINAL ANSWER[:\s]*(.+?)(?:\n|$)', thought, re.IGNORECASE | re.DOTALL)
            if match:
                candidate_answer = match.group(1).strip()
                # Verify if it's a valid final answer (not transitional text)
                if is_valid_final_answer(candidate_answer):
                    final_answer = candidate_answer
                    is_finished = True
                    print_final_answer(final_answer)
        
        if thought:
            print_thought(thought)
        
        # Create step record
        new_step = NetlistReActStep(
            step_num=current_step + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=None,
        )
        
        next_plan_idx = current_idx
        if not action and planned_tool:
            action = planned_tool
            action_input = {}
            if self._is_transient_response(state):
                action_input = self._normalize_transient_tool_args(action, action_input, state)
            else:
                action_input = self._normalize_transfer_tool_args(action, action_input, state)
            new_step["action"] = action
            new_step["action_input"] = action_input
            print(f"{TerminalColors.YELLOW}⚠️ Expected tool '{planned_tool}' but LLM did not emit a tool call; executing planned tool directly.{TerminalColors.END}")
        elif not action:
            plan[current_idx]["status"] = "completed"
            plan[current_idx]["result"] = thought
            next_plan_idx = current_idx + 1
        
        return {
            **state,
            "messages": state["messages"] + [execute_message, response],
            "current_step": current_step + 1,
            "current_plan_step": next_plan_idx,
            "steps": state["steps"] + [new_step],
            "plan": plan,
            "final_answer": final_answer,
            "is_finished": is_finished,
        }
    
    def act(self, state: NetlistReActState) -> NetlistReActState:
        """Execute tool and let LLM judge if step is complete"""
        if not state["steps"]:
            return state
        
        last_step = state["steps"][-1]
        action = last_step["action"]
        action_input = last_step["action_input"] or {}
        
        # ============================================================
        # 1. Execute tool
        # ============================================================
        if not action or action not in self.tools_by_name:
            observation = "No tool called or invalid tool name."
        else:
            tool = self.tools_by_name[action]
            start_time = time.time()
            
            try:
                observation = str(tool.invoke(action_input))
            except Exception as e:
                observation = f"Error: {e}"
            
            elapsed = time.time() - start_time
            print(f"{TerminalColors.CYAN}⏱️ Tool Execution Time: {elapsed:.2f}s{TerminalColors.END}")
        
        print_observation(observation)
        
        # Update step
        updated_steps = list(state["steps"])
        updated_steps[-1] = {**updated_steps[-1], "observation": observation}
        
        plan = state["plan"]
        current_idx = state["current_plan_step"]
        current_plan_step = plan[current_idx] if current_idx < len(plan) else None
        step_retry_count = state.get("_step_retry_count", 0)
        max_step_retries = 3
        
        # ============================================================
        # 2. Simple judgment: if tool doesn't report Error, proceed to next step
        # ============================================================
        # No longer using LLM Judge to avoid misjudgment and extra token consumption
        has_error = observation.strip().startswith("Error:")
        
        # step_retry_count: Number of retries already performed (0 means this is the 1st call)
        # When step_retry_count + 1 >= max_step_retries, this is the last chance
        # For example: when max_step_retries=3, step_retry_count=2 is the 3rd call (last one)
        if has_error and step_retry_count + 1 >= max_step_retries:
            decision = "skip"  # Last chance also failed, skip directly
        elif has_error:
            decision = "retry"  # Still have retry opportunities
        else:
            decision = "done"  # No Error, proceed to next step
        
        # ============================================================
        # 3. Process based on judgment result
        # ============================================================
        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=f"call_{action}_{state['current_step']}",
        )
        
        if decision == "done":
            print(f"{TerminalColors.GREEN}✅ Step DONE - Moving to next step{TerminalColors.END}")
            if current_plan_step:
                plan[current_idx]["status"] = "completed"
                plan[current_idx]["result"] = observation
            
            return {
                **state,
                "steps": updated_steps,
                "messages": state["messages"] + [tool_message],
                "plan": plan,
                "current_plan_step": current_idx + 1,
                "_step_retry_count": 0,
            }
        
        elif decision == "retry" and step_retry_count < max_step_retries:
            print(f"{TerminalColors.YELLOW}🔄 Step RETRY ({step_retry_count + 1}/{max_step_retries}){TerminalColors.END}")
            if current_plan_step:
                plan[current_idx]["status"] = "retrying"
            
            step_desc = current_plan_step["description"] if current_plan_step else "Unknown"
            retry_feedback = HumanMessage(
                content=f"⚠️ Tool returned an error.\n"
                        f"Step goal: {step_desc}\n"
                        f"Error: {observation}\n"
                        f"Please fix the issue:\n"
                        f"- If parameters were wrong, correct them\n"
                        f"- If tool was wrong, try a different one\n"
                        f"Retry {step_retry_count + 1}/{max_step_retries}"
            )
            
            return {
                **state,
                "steps": updated_steps,
                "messages": state["messages"] + [tool_message, retry_feedback],
                "plan": plan,
                "current_plan_step": current_idx,
                "_step_retry_count": step_retry_count + 1,
            }
        
        else:
            print(f"{TerminalColors.YELLOW}⏭️ Step SKIP - Moving to next step{TerminalColors.END}")
            if current_plan_step:
                plan[current_idx]["status"] = "skipped"
                plan[current_idx]["result"] = f"[Skipped after {step_retry_count} retries] {observation[:200]}"
            
            return {
                **state,
                "steps": updated_steps,
                "messages": state["messages"] + [tool_message],
                "plan": plan,
                "current_plan_step": current_idx + 1,
                "_step_retry_count": 0,
            }
    
    def should_continue(self, state: NetlistReActState) -> str:
        """Decide whether to continue or finish"""
        if state["is_finished"]:
            return "finish"
        if state["current_step"] >= state["max_steps"]:
            return "finish"
        
        # Check if all plan steps are done
        plan = state.get("plan", [])
        current_idx = state.get("current_plan_step", 0)
        if current_idx >= len(plan):
            return "finish"
        
        return "continue"


# ============================================================
# Build Netlist Agent Graph
# ============================================================

def create_netlist_react_subgraph(llm, max_steps: int = 10):
    """Build the Netlist Plan-and-Execute agent graph"""
    
    nodes = NetlistReActAgentNodes(llm)
    
    workflow = StateGraph(NetlistReActState)
    
    # Add nodes
    workflow.add_node("initialize", nodes.initialize_tools)
    workflow.add_node("plan", nodes.plan)
    workflow.add_node("execute", nodes.execute)
    workflow.add_node("act", nodes.act)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "plan")
    workflow.add_edge("plan", "execute")
    
    # execute -> act (if tool call) / next_step (if no tool) / finish
    def after_execute(state: NetlistReActState) -> str:
        if state["is_finished"]:
            return "finish"
        if state["steps"] and state["steps"][-1]["action"]:
            return "act"
        plan = state.get("plan", [])
        current_idx = state.get("current_plan_step", 0)
        if current_idx < len(plan):
            return "next_step"
        return "finish"
    
    workflow.add_conditional_edges(
        "execute",
        after_execute,
        {"act": "act", "next_step": "execute", "finish": END}
    )
    
    # act -> execute (continue executing plan) / finish
    workflow.add_conditional_edges(
        "act",
        nodes.should_continue,
        {"continue": "execute", "finish": END}
    )
    
    return workflow.compile()


# ============================================================
# Main Solve Node
# ============================================================

def solve_netlist_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Solve Netlist circuit problem using Plan-and-Execute agent.
    
    Args:
        state: Workflow state with IR and question
        llm: Language model
    
    Returns:
        State update with answer
    """
    start_time = time.time()
    solve_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    llm_calls = 0
    
    ir_dict = state.get("ir")
    
    print_separator("SOLVE NETLIST AGENT START", "=", 60)
    print(f"{TerminalColors.BOLD}📊 Type: {TerminalColors.END}Electronic Circuit / Netlist (Lcapy)")
    print(f"{TerminalColors.BOLD}❓ Question: {TerminalColors.END}{state.get('question', 'N/A')[:100]}...")
    
    if not ir_dict:
        print_error("No IR available")
        return {
            "answer": None,
            "success": False,
            "error": "No IR available for solving",
        }
    
    # Build and run agent
    try:
        agent = create_netlist_react_subgraph(llm, max_steps=10)
        
        # max_steps calculation: assuming plan is at most 5 steps, each step at most 3 retries = 5 x 4 = 20
        initial_state = NetlistReActState(
            question=state.get("question", ""),
            analysis_type=state.get("analysis_type"),
            ir_dict=ir_dict,
            messages=[],
            plan=[],
            current_plan_step=0,
            _step_retry_count=0,
            current_step=0,
            max_steps=20,
            steps=[],
            available_tools=[],
            final_answer=None,
            is_finished=False,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            llm_calls=0,
        )
        
        result = agent.invoke(initial_state)
        
        # Extract answer
        final_answer = result.get("final_answer")
        solve_steps = [
            {
                "thought": s.get("thought", ""),
                "action": f"{s.get('action', 'N/A')}({json.dumps(s.get('action_input', {})) if s.get('action_input') else ''})" if s.get('action') else "N/A",
                "observation": s.get("observation"),
            }
            for s in result.get("steps", [])
        ]
        
        duration = time.time() - start_time
        llm_calls = result.get("llm_calls", len(solve_steps))
        
        print_separator("SOLVE NETLIST AGENT RESULT", "=", 60)
        print(f"{TerminalColors.BOLD}📊 Total Steps: {TerminalColors.END}{len(solve_steps)}")
        print(f"{TerminalColors.BOLD}✅ Success: {TerminalColors.END}{final_answer is not None}")
        print(f"{TerminalColors.BOLD}⏱️ Duration: {TerminalColors.END}{duration:.2f}s")
        
        # Update metrics
        existing_metrics = state.get("metrics", {})
        if "solve" not in existing_metrics:
            existing_metrics["solve"] = {
                "duration_seconds": 0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "llm_calls": 0,
            }
        
        existing_metrics["solve"]["duration_seconds"] += round(duration, 2)
        existing_metrics["solve"]["tokens"]["input_tokens"] += solve_tokens["input_tokens"]
        existing_metrics["solve"]["tokens"]["output_tokens"] += solve_tokens["output_tokens"]
        existing_metrics["solve"]["tokens"]["total_tokens"] += solve_tokens["total_tokens"]
        existing_metrics["solve"]["llm_calls"] += llm_calls
        
        if final_answer:
            return {
                "answer": final_answer,
                "success": True,
                "solve_steps": solve_steps,
                "metrics": existing_metrics,
            }
        else:
            return {
                "answer": None,
                "success": False,
                "error": "No final answer generated",
                "solve_steps": solve_steps,
                "metrics": existing_metrics,
            }
            
    except Exception as e:
        import traceback
        print_error(str(e))
        traceback.print_exc()
        
        duration = time.time() - start_time
        existing_metrics = state.get("metrics", {})
        if "solve" not in existing_metrics:
            existing_metrics["solve"] = {
                "duration_seconds": 0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "llm_calls": 0,
            }
        existing_metrics["solve"]["duration_seconds"] += round(duration, 2)
        
        return {
            "answer": None,
            "success": False,
            "error": str(e),
            "metrics": existing_metrics,
        }
