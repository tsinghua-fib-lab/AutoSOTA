from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from quarc.data_handlers.eval_datasets import ReactionInput


@dataclass
class StagePrediction:
    agents: list[int]
    temp_bin: int
    reactant_bins: list[int]
    agent_amount_bins: list[tuple[int, int]]  # (agent_idx, bin)
    score: float

    meta: dict = field(default_factory=dict)  # include individual stage scores

    def __str__(self) -> str:
        """Readable string representation for visualization"""
        lines = [
            f"  Agents: {self.agents}",
            f"  Temperature bin: {self.temp_bin}",
            f"  Reactant bins: {self.reactant_bins}",
            f"  Agent amount bins: {self.agent_amount_bins}",
            f"  Score: {self.score:.4f}",
        ]

        if self.meta:
            meta_items = [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in self.meta.items()
            ]
            lines.append(f"  Meta: {', '.join(meta_items)}")

        return "\n".join(lines)


@dataclass
class PredictionList:
    """Ranked (descending) enumeratedpredictions for a single reaction"""

    doc_id: str
    rxn_class: str
    rxn_smiles: str
    predictions: list[StagePrediction]

    def __str__(self) -> str:
        """Readable string representation for visualization"""
        # Truncate SMILES for readability
        smiles_preview = (
            self.rxn_smiles[:50] + "..." if len(self.rxn_smiles) > 50 else self.rxn_smiles
        )

        lines = [
            f"PredictionList:",
            f"  Doc ID: {self.doc_id}",
            f"  Reaction class: {self.rxn_class}",
            f"  SMILES: {smiles_preview}",
            f"  Predictions ({len(self.predictions)}):",
        ]

        for i, pred in enumerate(self.predictions, 1):
            lines.append(f"    [{i}]")
            pred_lines = str(pred).split("\n")
            for line in pred_lines:
                lines.append(f"    {line}")
            lines.append("")

        return "\n".join(lines[:-1])


class BasePredictor(ABC):
    """Return *ranked* StagePredictions for a single reaction."""

    @abstractmethod
    def predict(self, reaction: ReactionInput, top_k: int = 10) -> PredictionList:
        pass

    def predict_many(
        self, reactions: list[ReactionInput], top_k: int = 10
    ) -> list[PredictionList]:
        return [self.predict(r, top_k) for r in reactions]
