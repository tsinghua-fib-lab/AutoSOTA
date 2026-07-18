"""RBCBF: Rollback-Based Control Barrier Function decoding-time safety alignment.

Official open-source implementation of the ICML 2026 paper:
    "RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control"
"""

__version__ = "1.0.0"

from .controllers import JointCBFKLController, NoOpController
from .judges import QwenMarginJudge, StreamingLLMRiskTracker
from .models import BaseLanguageModel
from .runner import GenerationResult, RBCBFRunner
from .scorers import (
    MultiRuleScorer,
    QwenMarginScorer,
)

__all__ = [
    "RBCBFRunner",
    "GenerationResult",
    "JointCBFKLController",
    "NoOpController",
    "QwenMarginJudge",
    "StreamingLLMRiskTracker",
    "BaseLanguageModel",
    "MultiRuleScorer",
    "QwenMarginScorer",
]
