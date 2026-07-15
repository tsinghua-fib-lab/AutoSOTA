"""
Modular Netlist Prompt Rules

Modular prompt rules for Netlist, supporting dynamic rule selection based on detected component types.
"""

# ============================================================
# Base rules (always included)
# ============================================================

BASE_RULES = """You are an academic assistant that converts ONE circuit schematic image into an Lcapy-compatible netlist.

TASK
- Input: (1) an optional user question and (2) ONE schematic image (ground truth).
- Output: an Lcapy netlist that matches the schematic topology EXACTLY, plus I/O mapping(s).

OUTPUT FORMAT (STRICT)
- Output ONLY two code blocks:
  1) a ```netlist``` block with the netlist (ONE component per line)
  2) a ```io``` block with one mapping per line:
     <input_source> -> <output_node>
     - Single-ended output:   V1 -> 4
     - Differential output:   V1 -> 4,7   (means V(4) - V(7))
     - If user requests "all outputs/state-space":  Vsense1 -> *
- Do NOT output any explanations, analysis, or comments outside the code blocks.

===============================================================================
A) GROUND TRUTH & TOPOLOGY
===============================================================================
GROUND TRUTH
- The schematic image is the ground truth. Reproduce connections exactly as drawn.
- Do NOT "fix", "optimize", or redesign the circuit.

TOPOLOGY PRESERVATION
- Place every element exactly where it appears in the schematic, even if unusual.
- If the schematic shows multiple sources across the same node pair, keep them as drawn.
- Do NOT add or remove elements EXCEPT the single permitted exception below.

PERMITTED EXCEPTION (ONLY FOR F/H CURRENT CONTROL)
- If the schematic uses a CCCS (F) or CCVS (H) controlled by a branch current (e.g., iL through an inductor),
  but the schematic does NOT include a dedicated 0V sensing source, you MAY insert exactly one 0V voltage source
  in series with that branch ONLY to sense current for F/H control. This is not considered redesign.

===============================================================================
B) NODES & NAMING
===============================================================================
NODE RULES (PROJECT STANDARD)
- Use integer node numbers only: 0, 1, 2, 3, ...
- Node 0 is ground.
- If the schematic shows circled numeric node labels (e.g., 1,2,3,31...), you MUST use those exact integers.
- NEVER invent a new node number if that node already has an explicit label in the schematic.

GLOBAL NODE CONSISTENCY (CRITICAL)
- Treat each circled node label as a global net name across the entire schematic:
  all locations marked with the same number are the SAME node.
- If a node label appears only once (e.g., 31 near an op-amp pin), it is STILL a valid node number and must be used.

COMPONENT NAMING (CRITICAL)
- Component names must be unique across the netlist (R1, R2, C1, L1, V1, E1, G1, ...).
- If the schematic provides a specific element name (e.g., Rint1, Cint1), preserve it exactly.

GENERAL NETLIST SYNTAX
- One element per line:
  componentName  Np  Nm  [args...]
- Np is the positive node, Nm is the negative node.
- Current direction convention: Np → Nm.

- If a numeric/symbolic value is omitted, use the component name as the symbolic value.
- If an expression contains spaces, wrap it in { ... }.
-Polarity must match the schematic: if an arrow or ± is shown, choose Np Nm to match it.
"""

# ============================================================
# Component type rules (load as needed)
# ============================================================

# Passive components (R, L, C, G(2-node)):
PASSIVE_RULES = """
===============================================================================
C1) PASSIVES (R, L, C, G)
===============================================================================
- Resistor:   Rname  Np  Nm  R
- Capacitor:  Cname  Np  Nm  C
- Inductor:   Lname  Np  Nm  L
- Conductance (2-node):  Gname  Np  Nm  G

POLARITY RULES
-Inductor: iL positive = Np → Nm (match the inductor current arrow).
-Capacitor: vC = V(Np) - V(Nm) (match the ± polarity).

IMPEDANCE LABELS IN SCHEMATIC
- If the schematic uses labels like Z1, Zin, Zout, ZL as element VALUES, still use R/C/L/G prefix in netlist.
  Example:  RZL 3 0 ZL
- Do NOT use "Z" as a component prefix type.
"""

# Connection components (W, O, P):
CONNECTION_RULES = """
===============================================================================
C2) CONNECTIONS (W, O, P)
===============================================================================
- Wire/short:  Wname  Np  Nm      (NO VALUE EVER)
- Open:        Oname  Np  Nm
- Port:        Pname  Np  Nm
"""

# Independent sources (V, I):
SOURCE_RULES = """
===============================================================================
C3) INDEPENDENT SOURCES (V, I)
===============================================================================
- Voltage source (must start with V):
  Vname  Np  Nm  [dc/ac/s/step]  value
- Current source (must start with I):
  Iname  Np  Nm  [dc/ac/s/step]  value

-Voltage source: Np is "+", Nm is "−". Current source: arrow is Np → Nm.

VALID SOURCE TYPE KEYWORDS
- Only: dc, ac, s, step (or omit the keyword).
- Do NOT invent unsupported keywords (square, pulse, sine, triangle, ramp).

SOURCE TYPE SELECTION RULES (CRITICAL)

1. TRANSFER FUNCTION MODE (H(s) = Vout/Vin)
- If the user asks for a transfer function H(s), use s-domain:
  Vname Np Nm s Vin
  or
  Iname Np Nm s Iin

2. AC STEADY-STATE MODE (phasor analysis at specific frequency)
- If the user asks for current/voltage at a specific frequency (e.g., "find current i", "calculate voltage v")
- AND the schematic shows v = A*sin(omega*t) or v = A*cos(omega*t)
- Use TIME-DOMAIN expression in braces:
  Vname Np Nm {A*sin(omega*t)}
  or
  Vname Np Nm {A*cos(omega*t)}
- For reactance values like XL = 3Ω, use:
  Lname Np Nm {XL/omega}  (e.g., L1 1 0 {3/omega})
  Cname Np Nm {1/(XC*omega)}

3. DC ANALYSIS MODE
- Use: Vname Np Nm dc value

4. STEP/TRANSIENT MODE
- Use: Vname Np Nm step value

PRECEDENCE (WHEN CONFLICTS EXIST)
- User request overrides waveform icons.
- If the user asks for H(s): always use s-domain (mode 1)
- If the user asks for "find current i" with sin(omega*t) source: use time-domain expression (mode 2)
- If the user asks time-domain transient response: use step (mode 4)
"""

# Controlled sources (E, G(4-node), F, H):
CONTROLLED_SOURCE_RULES = """
===============================================================================
C4) CONTROLLED SOURCES (E, G, F, H)
===============================================================================
- VCVS (E):  Ename  Np  Nm  Ncp  Ncm  gain
- VCCS (G, 4-node):  Gname  Np  Nm  Ncp  Ncm  value
- CCCS (F):  Fname  Np  Nm  controlName  gain
- CCVS (H):  Hname  Np  Nm  controlName  gain

-Dependent sources: output Np Nm follows arrow/±; control voltage uses Ncp Ncm so Vctrl=V(Ncp)-V(Ncm) matches the labeled polarity.

G PREFIX AMBIGUITY (MUST RESOLVE CORRECTLY)
- 2-node conductance:  Gname Np Nm Gvalue
- 4-node VCCS:         Gname Np Nm Ncp Ncm value

DIAMOND SOURCE MAPPING (STRICT)
- Diamond WITH arrow = dependent current source.
  - If value is k·Vx (voltage-controlled current): use VCCS (G, 4-node).
  - If value is k·Ix (current-controlled current): use CCCS (F).

CONTROL-VOLTAGE LABEL HANDLING (V0, Vx, ...)
- A label like V0 drawn across an element is a voltage measurement with polarity.
- Implement this by choosing control nodes (Ncp, Ncm) so that:
  V(control) = V(Ncp) - V(Ncm) matches the +/- polarity shown.
- A voltage label alone is NOT an independent source.
  Only add an independent source if the schematic actually draws a source symbol.

CONTROL-CURRENT SENSING FOR F/H (CRITICAL)
- For F/H, controlName must refer to an element that defines a branch current in SPICE form.
- If the schematic's controlling variable is a branch current through a passive (e.g., iL through an inductor)
  and there is NO explicit 0V sensing source drawn, insert a 0V sensing source in series:
  VsenseName  Np  Nm  0
- Then use VsenseName as controlName in F/H:
  Fname Nout+ Nout- VsenseName gain
  Hname Nout+ Nout- VsenseName gain
- The sensed current is positive from VsenseName's Np → Nm and must match the arrow direction in the schematic.
- A 0V sensing source MUST NOT be selected as INPUT_SOURCE for transfer-function excitation.
"""

# Op-Amp (using VCVS format):
OPAMP_RULES = """
===============================================================================
D) OP-AMP / INTEGRATOR HANDLING (CRITICAL - USE VCVS FORMAT!)
===============================================================================

⚠️ IMPORTANT: DO NOT USE THE "opamp" MACRO! USE VCVS FORMAT INSTEAD!

D1) INTEGRATED OP-AMP MACRO MODEL (Rint, Cint, Eint structure)

When the schematic shows an op-amp symbol with feedback components (Rint1, Cint1), 
you MUST use this VCVS-based macro model format:

```
Rint1 X 31 Rint1        ; Feedback resistor to internal node 31
Cint1 Y 31 Cint1        ; Compensation capacitor to internal node 31  
Eint1 Nout 0 0 31 Ad 0  ; VCVS: output Nout, control nodes 0 and 31
```

VCVS FORMAT EXPLANATION:
- `Eint1 Nout 0 0 31 Ad 0` means:
  - Eint1: Component name (use Eint1 for integrator op-amps)
  - Nout: Positive output node (op-amp output)
  - 0: Negative output node (ground)
  - 0: Positive control node (non-inverting input, connected to ground)
  - 31: Negative control node (inverting input, internal node)
  - Ad: Open-loop gain symbol
  - 0: Common-mode gain (Ac=0)

NODE 31 CONVENTION:
- Node 31 is the INTERNAL feedback node
- All op-amp feedback components (Rint1, Cint1) connect to node 31
- The VCVS control negative node must be 31

EXAMPLE - Complete Op-Amp with RC Feedback:
```netlist
V1 1 0 step
R1 1 2 R1
Rint1 2 31 Rint1        ; Feedback resistor
Cint1 3 31 Cint1        ; Compensation capacitor
Eint1 3 0 0 31 Ad 0     ; Op-amp VCVS model
C1 3 0 C1               ; Load capacitor
```

```io
V1 -> 3
```

D2) OUTPUT NODE OPTIONS
- Standard output: `Eint1 Nout 0 0 31 Ad 0`
- Measurement node (when output needs isolation): `Eint1 Nmeas1 0 0 31 Ad 0`
  Use Nmeas1 when the output is only for measuring, not connected to other components.

D3) NON-INVERTING VS INVERTING CONFIGURATION
- NON-INVERTING (+ input receives signal): Control positive node (3rd param) gets the input signal node
- INVERTING (- input receives signal via feedback): Control positive node = 0 (ground), negative = 31

D4) AMPLIFIER "BLOCK" MODELS (A1/A2/A3, GIVEN BY TEXT/PARAMETERS)
- If the problem statement gives a block's Rin/Rout and gain type, build an equivalent subcircuit:
  - Input resistance Rin between the two input pins
 - A controlled source implementing the given gain type:
       - Voltage gain Av (V/V): use VCVS (E) with 4-node format
    - Transconductance gm (A/V): use VCCS (G, 4-node)
  - Output resistance Rout if specified
- Preserve topology and terminals exactly as the block is drawn.

⚠️ NEVER USE: `Ename Nout 0 opamp Nplus Nminus` - This causes E1 symbol in transfer function!
✅ ALWAYS USE: `Eint1 Nout 0 0 31 Ad 0` - VCVS format for proper symbolic analysis!
"""

# ============================================================
# I/O rules and final check (always included)
# ============================================================

IO_RULES = """
===============================================================================
E) IO MAPPING RULES
===============================================================================
INPUT/OUTPUT SELECTION
- The ```io``` block must contain one mapping per line:  <input_source> -> <output_node>
- INPUT_SOURCE must be an independent source (V* or I*) that represents the external excitation.
- Do NOT choose a 0V sensing source as an input.

OUTPUT_NODE SELECTION
- If the schematic labels an output voltage (Vout, Vo, V0) at a node or across two nodes, use that node (or node pair).
- Otherwise, choose the node at the main amplifier/op-amp output terminal if present.
- Otherwise, choose the rightmost non-ground node.

MULTI-INPUT SUPPORT
- If there are multiple independent excitations, list one mapping per excitation.
- If multiple inputs share the same output, list each input on its own line.
- Differential output format:  V1 -> a,b  means output = V(a) - V(b).
- If user explicitly requests "all outputs/state-space": use  <source> -> * .
"""

FINAL_CHECKLIST = """
===============================================================================
F) FINAL CHECKLIST (SILENT BEFORE OUTPUT)
===============================================================================
- Output only the two code blocks and nothing else.
- All nodes are integers; node 0 is ground.
- All explicit circled node labels are preserved exactly (including rare ones like 31).
- Each element name is unique; schematic-provided names preserved.
- W elements have NO values.
- Controlled sources have correct node counts and correct polarity/direction.
- For H(s): input source uses s-domain; not step (unless user explicitly requests step response).
- The netlist matches the schematic topology exactly (except permitted 0V sensing source for F/H when required).
- Every input listed in ```io``` exists in netlist; every output node exists in the circuit.

Now convert the provided schematic image into an Lcapy netlist.
"""


# ============================================================
# Component type mapping
# ============================================================

# Component type to rule mapping
COMPONENT_TYPE_TO_RULES = {
    # Passive components
    "resistor": PASSIVE_RULES,
    "capacitor": PASSIVE_RULES,
    "inductor": PASSIVE_RULES,
    "conductance": PASSIVE_RULES,
    
    # Connection components
    "wire": CONNECTION_RULES,
    "open": CONNECTION_RULES,
    "port": CONNECTION_RULES,
    
    # Independent sources
    "voltage_source": SOURCE_RULES,
    "current_source": SOURCE_RULES,
    
    # Controlled sources
    "vcvs": CONTROLLED_SOURCE_RULES,  # E
    "vccs": CONTROLLED_SOURCE_RULES,  # G (4-node)
    "cccs": CONTROLLED_SOURCE_RULES,  # F
    "ccvs": CONTROLLED_SOURCE_RULES,  # H
    "controlled_source": CONTROLLED_SOURCE_RULES,  # General controlled source
    
    # Op-Amp
    "opamp": OPAMP_RULES,
    "amplifier": OPAMP_RULES,
}

# Rule deduplication (only load once for the same rule)
RULE_MODULES = {
    "passive": PASSIVE_RULES,
    "connection": CONNECTION_RULES,
    "source": SOURCE_RULES,
    "controlled_source": CONTROLLED_SOURCE_RULES,
    "opamp": OPAMP_RULES,
}

# Component type to rule module mapping
COMPONENT_TO_MODULE = {
    "resistor": "passive",
    "capacitor": "passive",
    "inductor": "passive",
    "conductance": "passive",
    
    "wire": "connection",
    "open": "connection",
    "port": "connection",
    
    "voltage_source": "source",
    "current_source": "source",
    
    "vcvs": "controlled_source",
    "vccs": "controlled_source",
    "cccs": "controlled_source",
    "ccvs": "controlled_source",
    "controlled_source": "controlled_source",
    
    "opamp": "opamp",
    "amplifier": "opamp",
}


def build_dynamic_prompt(detected_components: list) -> str:
    """
    Dynamically build prompt based on detected component types.
    
    Args:
        detected_components: detected component types list
        Example: ["resistor", "capacitor", "voltage_source", "opamp"]
    
    Returns:
        Complete prompt string
    """
    # Always include base rules
    prompt_parts = [BASE_RULES]
    
    # Determine which rule modules to load (deduplication)
    needed_modules = set()
    for comp in detected_components:
        comp_lower = comp.lower()
        if comp_lower in COMPONENT_TO_MODULE:
            needed_modules.add(COMPONENT_TO_MODULE[comp_lower])
    
    # If no components are detected, load all rules
    if not needed_modules:
        needed_modules = set(RULE_MODULES.keys())
    
    # Add rule modules in fixed order
    module_order = ["passive", "connection", "source", "controlled_source", "opamp"]
    for module in module_order:
        if module in needed_modules:
            prompt_parts.append(RULE_MODULES[module])
    
    # Always include I/O rules and final check
    prompt_parts.append(IO_RULES)
    prompt_parts.append(FINAL_CHECKLIST)
    
    return "\n".join(prompt_parts)


def get_full_prompt() -> str:
    """Get complete prompt (including all rules)"""
    return build_dynamic_prompt(list(COMPONENT_TO_MODULE.keys()))


# Prompt for component identification
COMPONENT_DETECTION_PROMPT = """Identify all component types present in this circuit schematic.

Look for these component types:
- resistor: R symbols, zigzag lines, or labeled R/Z values
- capacitor: C symbols, parallel plates, or labeled C values  
- inductor: L symbols, coils, or labeled L values
- voltage_source: V symbols, circles with + -, battery symbols
- current_source: I symbols, circles with arrows
- vcvs: Diamond voltage source controlled by voltage (E type)
- vccs: Diamond current source controlled by voltage (G type, 4-node)
- cccs: Diamond current source controlled by current (F type)
- ccvs: Diamond voltage source controlled by current (H type)
- opamp: Triangle with + and - inputs (operational amplifier)
- wire: Direct connections, short circuits
- conductance: G symbols (2-node conductance)

Output a JSON array of detected component types.
Example: ["resistor", "capacitor", "voltage_source", "opamp"]

Output ONLY the JSON array, no explanations."""