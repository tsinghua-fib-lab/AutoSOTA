"""
Netlist-IR: Circuit Netlist Intermediate Representation

For representing electronic circuits in Lcapy-compatible format.

Supported component types (based on Lcapy netlists):
- Passive elements: R (resistor), NR (noiseless resistor), G (conductance), L (inductor), C (capacitor)
- Independent sources: V (voltage), I (current)
- Controlled sources: E (VCVS), VCCS, H (CCVS), F (CCCS)
- Active devices: Ideal op-amps (E prefix with 'opamp' keyword)
- Connections: W (wire), O (open circuit), P (port)

Reference: https://lcapy.readthedocs.io/en/latest/netlists.html
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import re


class ComponentType(Enum):
    """Types of circuit components
    
    Classification corresponding to the modular netlist rules:
    1) PASSIVES: R, C, L, G(2-node)
    2) INDEPENDENT SOURCES: V, I
    3) CONNECTIONS: W, O, P
    4) CONTROLLED SOURCES: E(VCVS), G(4-node VCCS), F(CCCS), H(CCVS)
    5) OP-AMP MACROS: opamp, fdopamp, inamp
    6) TWO-PORT: TP
    7) TRANSFORMER/GYRATOR: TF, GY
    8) SWITCH: SW
    """
    # 1) PASSIVES
    RESISTOR = "R"              # Rname Np Nm R
    NOISELESS_RESISTOR = "NR"   # NRname Np Nm R
    CAPACITOR = "C"             # Cname Np Nm C
    INDUCTOR = "L"              # Lname Np Nm L
    CONDUCTANCE = "G_passive"   # Gname Np Nm G (2-node)
    
    # 2) INDEPENDENT SOURCES
    VOLTAGE_SOURCE = "V"        # Vname Np Nm [dc/ac/s/step] value
    CURRENT_SOURCE = "I"        # Iname Np Nm [dc/ac/s/step] value
    
    # 3) CONNECTIONS
    WIRE = "W"                  # Wname Np Nm (no value)
    OPEN = "O"                  # Oname Np Nm
    PORT = "P"                  # Pname Np Nm
    
    # 4) CONTROLLED SOURCES
    VCVS = "E"                  # Ename Np Nm Ncp Ncm gain
    VCCS = "G_controlled"       # Gname Np Nm Ncp Ncm value (4-node)
    CCCS = "F"                  # Fname Np Nm controlName gain
    CCVS = "H"                  # Hname Np Nm controlName gain
    
    # 5) OP-AMP MACROS
    OPAMP = "OPAMP"             # Ename Np Nm opamp Nip Nim Ad [Ac] [Ro]
    FDOPAMP = "FDOPAMP"         # Ename Np Nm fdopamp Nip Nim Nocm Ad Ac
    INAMP = "INAMP"             # Ename Np Nm inamp Nip Nim Nrp Nrm Ad [Ac] [Rf]
    
    # 6) TWO-PORT & TRANSMISSION LINE
    TWOPORT = "TP"              # TPname Np Nm Nip Nim ...
    
    # 7) TRANSFORMER / GYRATOR
    TRANSFORMER = "TF"          # TFname Nsp Nsm Npp Npm [Ns] [Np]
    GYRATOR = "GY"              # GYname Np Nm Nip Nim R
    
    # 8) SWITCH
    SWITCH = "SW"               # SW Np Nm activation-time
    
    @classmethod
    def from_prefix(cls, prefix: str, num_args: int = 0) -> Optional["ComponentType"]:
        """Get component type from component ID prefix
        
        Args:
            prefix: Component name prefix (e.g., 'R', 'G', 'NR', 'TF')
            num_args: Number of arguments after nodes (helps distinguish G types)
        
        Note: G is ambiguous - can be conductance (2 nodes + value) or VCCS (4 nodes + value)
        This is resolved during parsing based on actual argument count.
        """
        prefix_upper = prefix.upper()
        
        # Handle multi-char prefixes first
        multi_char_map = {
            "NR": cls.NOISELESS_RESISTOR,
            "TF": cls.TRANSFORMER,
            "GY": cls.GYRATOR,
            "TP": cls.TWOPORT,
            "SW": cls.SWITCH,
        }
        
        if len(prefix_upper) >= 2:
            two_char = prefix_upper[:2]
            if two_char in multi_char_map:
                return multi_char_map[two_char]
        
        # Single-char prefixes
        prefix_map = {
            "R": cls.RESISTOR,
            "L": cls.INDUCTOR,
            "C": cls.CAPACITOR,
            "V": cls.VOLTAGE_SOURCE,
            "I": cls.CURRENT_SOURCE,
            "E": cls.VCVS,  # May be VCVS or OPAMP, resolved during parsing
            "G": cls.CONDUCTANCE,  # May be CONDUCTANCE or VCCS, resolved during parsing
            "H": cls.CCVS,
            "F": cls.CCCS,
            "W": cls.WIRE,
            "O": cls.OPEN,
            "P": cls.PORT,
        }
        
        return prefix_map.get(prefix_upper[0] if prefix_upper else "")
    
    def get_netlist_prefix(self) -> str:
        """Get the prefix used in Lcapy netlist"""
        prefix_map = {
            ComponentType.RESISTOR: "R",
            ComponentType.NOISELESS_RESISTOR: "NR",
            ComponentType.CONDUCTANCE: "G",
            ComponentType.INDUCTOR: "L",
            ComponentType.CAPACITOR: "C",
            ComponentType.VOLTAGE_SOURCE: "V",
            ComponentType.CURRENT_SOURCE: "I",
            ComponentType.VCVS: "E",
            ComponentType.VCCS: "G",
            ComponentType.CCVS: "H",
            ComponentType.CCCS: "F",
            ComponentType.OPAMP: "E",
            ComponentType.FDOPAMP: "E",
            ComponentType.INAMP: "E",
            ComponentType.WIRE: "W",
            ComponentType.OPEN: "O",
            ComponentType.PORT: "P",
            ComponentType.TWOPORT: "TP",
            ComponentType.TRANSFORMER: "TF",
            ComponentType.GYRATOR: "GY",
            ComponentType.SWITCH: "SW",
        }
        return prefix_map.get(self, self.value)
    
    def get_type_label(self) -> str:
        """Get the TYPE label corresponding to the modular netlist rules."""
        label_map = {
            # 1) PASSIVES
            ComponentType.RESISTOR: "RESISTOR",
            ComponentType.NOISELESS_RESISTOR: "RESISTOR",
            ComponentType.CAPACITOR: "CAPACITOR",
            ComponentType.INDUCTOR: "INDUCTOR",
            ComponentType.CONDUCTANCE: "CONDUCTANCE",
            # 2) INDEPENDENT SOURCES
            ComponentType.VOLTAGE_SOURCE: "VOLTAGE SOURCE",
            ComponentType.CURRENT_SOURCE: "CURRENT SOURCE",
            # 3) CONNECTIONS
            ComponentType.WIRE: "WIRE",
            ComponentType.OPEN: "OPEN",
            ComponentType.PORT: "PORT",
            # 4) CONTROLLED SOURCES
            ComponentType.VCVS: "VCVS",
            ComponentType.VCCS: "VCCS",
            ComponentType.CCCS: "CCCS",
            ComponentType.CCVS: "CCVS",
            # 5) OP-AMP MACROS
            ComponentType.OPAMP: "OPAMP",
            ComponentType.FDOPAMP: "FDOPAMP",
            ComponentType.INAMP: "INAMP",
            # 6) TWO-PORT
            ComponentType.TWOPORT: "TWO-PORT",
            # 7) TRANSFORMER / GYRATOR
            ComponentType.TRANSFORMER: "TRANSFORMER",
            ComponentType.GYRATOR: "GYRATOR",
            # 8) SWITCH
            ComponentType.SWITCH: "SWITCH",
        }
        return label_map.get(self, self.value)


@dataclass
class Component:
    """A circuit component
    
    Reference: modular netlist rules
    Format: componentName Np Nm [args...]
    """
    id: str                           # Component ID (e.g., R1, C2, V1)
    type: ComponentType               # Component type
    node1: str                        # First node (Np = positive node)
    node2: str                        # Second node (Nm = negative node)
    value: str = ""                   # Value (symbolic or numeric)
    
    # For voltage-controlled sources (4-terminal): E (VCVS), G (VCCS)
    ctrl_node1: Optional[str] = None  # Control node+ (Ncp)
    ctrl_node2: Optional[str] = None  # Control node- (Ncm)
    
    # For current-controlled sources: H (CCVS), F (CCCS)
    ctrl_element: Optional[str] = None  # Control element name
    
    # For op-amps
    in_plus: Optional[str] = None     # Non-inverting input (Nip)
    in_minus: Optional[str] = None    # Inverting input (Nim)
    opamp_Ad: Optional[str] = None    # Differential gain
    opamp_Ac: Optional[str] = None    # Common-mode gain
    opamp_Ro: Optional[str] = None    # Output resistance
    
    # For L/C with initial conditions
    initial_value: Optional[str] = None  # i0 for L, v0 for C
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "node1": self.node1,
            "node2": self.node2,
            "value": self.value,
            "ctrl_node1": self.ctrl_node1,
            "ctrl_node2": self.ctrl_node2,
            "ctrl_element": self.ctrl_element,
            "in_plus": self.in_plus,
            "in_minus": self.in_minus,
            "opamp_Ad": self.opamp_Ad,
            "opamp_Ac": self.opamp_Ac,
            "opamp_Ro": self.opamp_Ro,
            "initial_value": self.initial_value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Component":
        """Create Component from dictionary"""
        type_val = data["type"]
        try:
            comp_type = ComponentType(type_val)
        except ValueError:
            # Fallback for legacy data or string type values
            type_upper = type_val.upper()
            if type_upper == "OPAMP":
                comp_type = ComponentType.OPAMP
            elif type_upper == "G_PASSIVE" or type_upper == "CONDUCTANCE":
                comp_type = ComponentType.CONDUCTANCE
            elif type_upper == "G_CONTROLLED" or type_upper == "VCCS":
                comp_type = ComponentType.VCCS
            elif type_upper == "NR":
                comp_type = ComponentType.NOISELESS_RESISTOR
            else:
                # Try to get from prefix
                comp_type = ComponentType.from_prefix(type_val) or ComponentType.RESISTOR
        
        return cls(
            id=data["id"],
            type=comp_type,
            node1=data["node1"],
            node2=data["node2"],
            value=data.get("value", ""),
            ctrl_node1=data.get("ctrl_node1"),
            ctrl_node2=data.get("ctrl_node2"),
            ctrl_element=data.get("ctrl_element"),
            in_plus=data.get("in_plus"),
            in_minus=data.get("in_minus"),
            opamp_Ad=data.get("opamp_Ad"),
            opamp_Ac=data.get("opamp_Ac"),
            opamp_Ro=data.get("opamp_Ro"),
            initial_value=data.get("initial_value"),
        )
    
    def to_netlist_line(self, with_comment: bool = False) -> str:
        """Convert to Lcapy netlist line
        
        Args:
            with_comment: Whether to add TYPE comment (Default: False)
        """
        line = self._to_netlist_line_core()
        if with_comment:
            comment = self._get_type_comment()
            if comment:
                line = f"{line}  ; {comment}"
        return line
    
    def _get_type_comment(self) -> str:
        """Generate component type comment, corresponding to the modular netlist rules.
        
        Special case: 0V voltage source (used for CCCS/CCVS current sensing) 
        labeled as CURRENT SENSE (0V)
        """
        # Detect 0V sensing source: voltage source + value is 0
        if self.type == ComponentType.VOLTAGE_SOURCE:
            value_str = str(self.value).strip().lower() if self.value else ""
            # Common formats for 0V sensing source: 0, dc 0, {0}
            is_zero = (value_str in ['0', 'dc 0', '{0}', 'dc0'] or
                      value_str.endswith(' 0') or
                      (value_str.split()[-1] == '0' if value_str else False))
            if is_zero:
                return "TYPE: CURRENT SENSE (0V)"
        
        return f"TYPE: {self.type.get_type_label()}"
    
    def _to_netlist_line_core(self) -> str:
        """Core netlist line without comment"""
        # PASSIVES: R, NR, G(2-node), L, C
        if self.type == ComponentType.RESISTOR:
            value = self.value if self.value else self.id
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        elif self.type == ComponentType.NOISELESS_RESISTOR:
            value = self.value if self.value else self.id
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        elif self.type == ComponentType.CONDUCTANCE:
            value = self.value if self.value else self.id
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        elif self.type == ComponentType.INDUCTOR:
            value = self.value if self.value else self.id
            if self.initial_value:
                return f"{self.id} {self.node1} {self.node2} {value} {self.initial_value}"
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        elif self.type == ComponentType.CAPACITOR:
            value = self.value if self.value else self.id
            if self.initial_value:
                return f"{self.id} {self.node1} {self.node2} {value} {self.initial_value}"
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        # CONNECTIONS: W, O, P (no value)
        elif self.type == ComponentType.WIRE:
            return f"{self.id} {self.node1} {self.node2}"
        
        elif self.type == ComponentType.OPEN:
            return f"{self.id} {self.node1} {self.node2}"
        
        elif self.type == ComponentType.PORT:
            return f"{self.id} {self.node1} {self.node2}"
        
        # INDEPENDENT SOURCES: V, I
        elif self.type == ComponentType.VOLTAGE_SOURCE:
            value = self.value if self.value else f"s {self.id}"
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        elif self.type == ComponentType.CURRENT_SOURCE:
            value = self.value if self.value else f"s {self.id}"
            return f"{self.id} {self.node1} {self.node2} {value}"
        
        # CONTROLLED SOURCES: E (VCVS), G (VCCS 4-node), H (CCVS), F (CCCS)
        elif self.type == ComponentType.VCVS:
            gain = self.value if self.value else "1"
            return f"{self.id} {self.node1} {self.node2} {self.ctrl_node1} {self.ctrl_node2} {gain}"
        
        elif self.type == ComponentType.VCCS:
            gm = self.value if self.value else "1"
            return f"{self.id} {self.node1} {self.node2} {self.ctrl_node1} {self.ctrl_node2} {gm}"
        
        elif self.type == ComponentType.CCVS:
            gain = self.value if self.value else "1"
            return f"{self.id} {self.node1} {self.node2} {self.ctrl_element} {gain}"
        
        elif self.type == ComponentType.CCCS:
            gain = self.value if self.value else "1"
            return f"{self.id} {self.node1} {self.node2} {self.ctrl_element} {gain}"
        
        # OP-AMP MACROS
        elif self.type == ComponentType.OPAMP:
            comp_id = self.id if self.id.upper().startswith('E') else f"E{self.id}"
            parts = [comp_id, self.node1, self.node2, "opamp", self.in_plus, self.in_minus]
            if self.opamp_Ad:
                parts.append(self.opamp_Ad)
            if self.opamp_Ac:
                parts.append(f"Ac={self.opamp_Ac}")
            if self.opamp_Ro:
                parts.append(f"Ro={self.opamp_Ro}")
            return " ".join(str(p) for p in parts if p is not None)
        
        # Fallback
        value = self.value if self.value else self.id
        return f"{self.id} {self.node1} {self.node2} {value}"


@dataclass
class NetlistIR:
    """
    Circuit Netlist Intermediate Representation
    
    Core attributes:
    - netlist: Raw netlist text (Lcapy format)
    - components: List of parsed components
    - nodes: Set of all nodes
    
    Reference: modular netlist rules
    """
    
    # Raw netlist (primary representation)
    netlist: str = ""
    
    # Parsed structure
    components: List[Component] = field(default_factory=list)
    nodes: Set[str] = field(default_factory=set)
    
    # Analysis targets
    input_source: Optional[str] = None      # Input source (V1, I1)
    output_node: Optional[str] = None       # Output node
    ground_node: str = "0"                  # Ground node
    
    # Metadata
    name: str = "circuit"
    
    def to_dict(self) -> dict:
        return {
            "type": "netlist",
            "name": self.name,
            "netlist": self.netlist,
            "components": [c.to_dict() for c in self.components],
            "nodes": list(self.nodes),
            "input_source": self.input_source,
            "output_node": self.output_node,
            "ground_node": self.ground_node,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NetlistIR":
        ir = cls(
            name=data.get("name", "circuit"),
            netlist=data.get("netlist", ""),
            input_source=data.get("input_source"),
            output_node=data.get("output_node"),
            ground_node=data.get("ground_node", "0"),
        )
        ir.components = [Component.from_dict(c) for c in data.get("components", [])]
        ir.nodes = set(data.get("nodes", []))
        return ir
    
    @classmethod
    def from_netlist(cls, netlist: str, name: str = "circuit") -> "NetlistIR":
        """Create NetlistIR from netlist text"""
        ir = cls(name=name, netlist=netlist)
        ir._parse_netlist()
        return ir
    
    def _parse_netlist(self):
        """Parse netlist text to extract components and nodes"""
        self.components = []
        self.nodes = set()
        
        for line in self.netlist.strip().split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('*') or line.startswith(';'):
                continue
            
            # Handle inline comments
            if ';' in line:
                line = line.split(';')[0].strip()
                if not line:
                    continue
            
            component = self._parse_line(line)
            if component:
                self.components.append(component)
                self._collect_nodes(component)
        
        # Auto-detect input source
        if not self.input_source:
            for comp in self.components:
                if comp.type == ComponentType.VOLTAGE_SOURCE:
                    self.input_source = comp.id
                    break
    
    def _parse_line(self, line: str) -> Optional[Component]:
        """Parse a single netlist line
        
        Format: componentName Np Nm [args...]
        """
        parts = line.split()
        if len(parts) < 3:  # At minimum: ComponentID Node1 Node2
            return None
        
        comp_id = parts[0]
        node1 = parts[1]
        node2 = parts[2]
        
        # Determine type from prefix
        prefix = self._extract_prefix(comp_id)
        comp_type = ComponentType.from_prefix(prefix)
        
        if not comp_type:
            return None
        
        # Parse based on component type
        
        # PASSIVES: R, NR, G(2-node), L, C
        if comp_type == ComponentType.RESISTOR:
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value=parts[3] if len(parts) > 3 else comp_id,
            )
        
        elif comp_type == ComponentType.NOISELESS_RESISTOR:
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value=parts[3] if len(parts) > 3 else comp_id,
            )
        
        # L: Inductor
        elif comp_type == ComponentType.INDUCTOR:
            initial = parts[4] if len(parts) > 4 else None
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value=parts[3] if len(parts) > 3 else comp_id,
                initial_value=initial,
            )
        
        # C: Capacitor
        elif comp_type == ComponentType.CAPACITOR:
            initial = parts[4] if len(parts) > 4 else None
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value=parts[3] if len(parts) > 3 else comp_id,
                initial_value=initial,
            )
        
        # CONNECTIONS: W, O, P (no value)
        elif comp_type == ComponentType.WIRE:
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value="",
            )
        
        elif comp_type == ComponentType.OPEN:
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value="",
            )
        
        elif comp_type == ComponentType.PORT:
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value="",
            )
        
        # INDEPENDENT SOURCES: V, I
        elif comp_type in [ComponentType.VOLTAGE_SOURCE, ComponentType.CURRENT_SOURCE]:
            value = " ".join(parts[3:]) if len(parts) > 3 else f"s {comp_id}"
            return Component(
                id=comp_id,
                type=comp_type,
                node1=node1,
                node2=node2,
                value=value,
            )
        
        # E: VCVS or Opamp
        # VCVS: Ename Np Nm Nip Nim gain
        # Opamp: Ename Np Nm opamp Nip Nim [Ad] [Ac] [Ro]
        elif comp_type == ComponentType.VCVS:
            if len(parts) >= 4 and parts[3].lower() == 'opamp':
                # Opamp format
                return Component(
                    id=comp_id,
                    type=ComponentType.OPAMP,
                    node1=node1,
                    node2=node2,
                    in_plus=parts[4] if len(parts) > 4 else None,
                    in_minus=parts[5] if len(parts) > 5 else None,
                    opamp_Ad=parts[6] if len(parts) > 6 else None,
                    opamp_Ac=self._extract_named_param(parts[7:], "Ac") if len(parts) > 7 else None,
                    opamp_Ro=self._extract_named_param(parts[7:], "Ro") if len(parts) > 7 else None,
                )
            elif len(parts) >= 5:
                # VCVS format: Ename Np Nm Nip Nim gain
                return Component(
                    id=comp_id,
                    type=comp_type,
                    node1=node1,
                    node2=node2,
                    ctrl_node1=parts[3],
                    ctrl_node2=parts[4],
                    value=parts[5] if len(parts) > 5 else "1",
                )
        
        # G: Conductance (2-node) or VCCS (4-node)
        # Conductance: Gname Np Nm G (4 parts)
        # VCCS: Gname Np Nm Nip Nim gm (6 parts)
        elif comp_type == ComponentType.CONDUCTANCE:
            if len(parts) >= 6:
                # VCCS format: Gname Np Nm Nip Nim gm
                return Component(
                    id=comp_id,
                    type=ComponentType.VCCS,
                    node1=node1,
                    node2=node2,
                    ctrl_node1=parts[3],
                    ctrl_node2=parts[4],
                    value=parts[5] if len(parts) > 5 else "1",
                )
            else:
                # Conductance format: Gname Np Nm G
                return Component(
                    id=comp_id,
                    type=ComponentType.CONDUCTANCE,
                    node1=node1,
                    node2=node2,
                    value=parts[3] if len(parts) > 3 else comp_id,
                )
        
        # H: CCVS
        elif comp_type == ComponentType.CCVS:
            if len(parts) >= 4:
                return Component(
                    id=comp_id,
                    type=comp_type,
                    node1=node1,
                    node2=node2,
                    ctrl_element=parts[3],
                    value=parts[4] if len(parts) > 4 else "1",
                )
        
        # F: CCCS
        elif comp_type == ComponentType.CCCS:
            if len(parts) >= 4:
                return Component(
                    id=comp_id,
                    type=comp_type,
                    node1=node1,
                    node2=node2,
                    ctrl_element=parts[3],
                    value=parts[4] if len(parts) > 4 else "1",
                )
        
        # Fallback
        return Component(
            id=comp_id,
            type=comp_type,
            node1=node1,
            node2=node2,
            value=parts[3] if len(parts) > 3 else "",
        )
    
    def _extract_prefix(self, comp_id: str) -> str:
        """Extract prefix from component ID (e.g., R, NR, TF, SW)"""
        if not comp_id:
            return ""
        
        # Check for multi-char prefixes first
        if len(comp_id) >= 2:
            two_char = comp_id[:2].upper()
            if two_char == "NR":
                return "NR"
        
        return comp_id[0].upper()
    
    def _extract_named_param(self, parts: list, name: str) -> Optional[str]:
        """Extract named parameter like 'Ac=0' or 'Ro=0'"""
        for part in parts:
            if part.startswith(f"{name}="):
                return part.split("=", 1)[1]
        return None
    
    def _collect_nodes(self, component: Component):
        """Collect all nodes from a component"""
        self.nodes.add(component.node1)
        self.nodes.add(component.node2)
        if component.ctrl_node1:
            self.nodes.add(component.ctrl_node1)
        if component.ctrl_node2:
            self.nodes.add(component.ctrl_node2)
        if component.in_plus:
            self.nodes.add(component.in_plus)
        if component.in_minus:
            self.nodes.add(component.in_minus)
    
    def regenerate_netlist(self, with_comments: bool = False) -> str:
        """Regenerate netlist from components"""
        lines = [comp.to_netlist_line(with_comment=with_comments) for comp in self.components]
        return '\n'.join(lines)
    
    def regenerate_netlist_with_types(self) -> str:
        """Generate netlist with TYPE comments."""
        lines = [comp.to_netlist_line(with_comment=True) for comp in self.components]
        return '\n'.join(lines)
    
    def add_type_comments_to_netlist(self) -> str:
        """Add TYPE comments to existing netlist"""
        if not self.components:
            self._parse_netlist()
        return self.regenerate_netlist_with_types()
    
    def get_controlled_sources(self) -> list:
        """Get all controlled sources (VCVS/VCCS/CCVS/CCCS)"""
        sources = []
        for comp in self.components:
            if comp.type in [ComponentType.VCVS, ComponentType.VCCS, ComponentType.CCVS, ComponentType.CCCS]:
                sources.append({
                    "name": comp.id,
                    "type": comp.type.value,
                    "gain": comp.value
                })
        return sources
    
    def get_controlled_source_names(self) -> str:
        """Get controlled source names, comma-separated (for tool calls)"""
        sources = self.get_controlled_sources()
        return ",".join([s["name"] for s in sources])
    
    def get_component(self, comp_id: str) -> Optional[Component]:
        """Get component by ID"""
        for comp in self.components:
            if comp.id == comp_id:
                return comp
        return None
    
    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of a specific type"""
        return [c for c in self.components if c.type == comp_type]
    
    def count_by_type(self) -> Dict[str, int]:
        """Count components by type"""
        counts = {}
        for comp in self.components:
            type_name = comp.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def has_ground(self) -> bool:
        """Check if ground node (0) exists"""
        return self.ground_node in self.nodes or "0" in self.nodes
    
    def has_source(self) -> bool:
        """Check if at least one independent source exists"""
        for comp in self.components:
            if comp.type in [ComponentType.VOLTAGE_SOURCE, ComponentType.CURRENT_SOURCE]:
                return True
        return False
    
    def summary(self) -> str:
        """Get a summary of the IR"""
        counts = self.count_by_type()
        counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
        return (
            f"NetlistIR: {self.name}\n"
            f"  Components: {len(self.components)} ({counts_str})\n"
            f"  Nodes: {len(self.nodes)}\n"
            f"  Input: {self.input_source}\n"
            f"  Output: {self.output_node}"
        )
    
    def __str__(self) -> str:
        return f"NetlistIR: {len(self.components)} components, {len(self.nodes)} nodes"
