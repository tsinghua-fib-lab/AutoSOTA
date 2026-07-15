"""
IR (Intermediate Representation) Module

Intermediate representations:
1. NetlistIR - Circuit Netlist (for electronic circuits, Lcapy-compatible)
"""

from .netlist_ir import NetlistIR, Component

__all__ = [
    # Netlist IR (Lcapy)
    "NetlistIR", "Component",
]








