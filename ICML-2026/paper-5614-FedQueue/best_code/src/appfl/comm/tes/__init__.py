"""
GA4GH Task Execution Service (TES) communication module for APPFL.

This module provides TES-based distributed federated learning communication,
following the same architectural patterns as Globus Compute and Ray communicators.
"""

from .tes_server_communicator import TESServerCommunicator
from .tes_client_communicator import TESClientCommunicator

__all__ = ["TESServerCommunicator", "TESClientCommunicator"]
