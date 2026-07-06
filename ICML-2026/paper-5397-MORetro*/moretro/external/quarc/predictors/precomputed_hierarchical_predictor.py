import pickle
import numpy as np
from pathlib import Path
from dataclasses import asdict
import gc
from typing import List, Iterator

from quarc.data_handlers.eval_datasets import ReactionInput
from quarc.predictors.base import BasePredictor, PredictionList, StagePrediction
from quarc.predictors.multistage_predictor import HierarchicalPrediction


class PrecomputedHierarchicalPredictor(BasePredictor):
    """
    Fast predictor using precomputed hierarchical predictions.

    This predictor loads cached HierarchicalPrediction objects and performs
    fast enumeration with given weights.
    """

    def __init__(
        self,
        hierarchical_cache_path: Path | str,
        weights: dict[str, float],
        use_geometric: bool = True,
    ):
        """
        Args:
            hierarchical_cache_path: Path to pickle file containing list[HierarchicalPrediction]
            weights: Stage weights for scoring {"agent": 0.25, "temperature": 0.25, ...}
            use_geometric: Whether to use geometric mean for score combination
        """
        self.weights = weights
        self.use_geometric = use_geometric
        self._cache: dict[tuple, HierarchicalPrediction] = (
            {}
        )  # (doc_id, rxn_smiles) -> HierarchicalPrediction
        self._load_cache(Path(hierarchical_cache_path))

    def _load_cache(self, cache_path: Path) -> None:
        """Load hierarchical predictions into lookup cache"""
        # print(f"Loading hierarchical predictions from {cache_path}")
        with open(cache_path, "rb") as f:
            hierarchical_list = pickle.load(f)

        for hier_pred in hierarchical_list:
            key = (hier_pred.doc_id, hier_pred.rxn_smiles)
            self._cache[key] = hier_pred

    def update_weights(self, weights: dict[str, float]):
        """Update weights"""
        self.weights = weights

    def predict(self, reaction: ReactionInput, top_k: int = 10) -> PredictionList:
        """Fast enumeration from cached hierarchical predictions"""
        doc_id = reaction.metadata["doc_id"]
        rxn_smiles = reaction.metadata["rxn_smiles"]
        key = (doc_id, rxn_smiles)

        if key not in self._cache:
            return PredictionList(
                doc_id=doc_id,
                rxn_class=reaction.metadata.get("rxn_class", ""),
                rxn_smiles=rxn_smiles,
                predictions=[],
            )

        hier_pred = self._cache[key]

        enumerated = self._rank_enumerate_combinations(hier_pred, top_k)

        return PredictionList(
            doc_id=hier_pred.doc_id,
            rxn_class=hier_pred.rxn_class,
            rxn_smiles=hier_pred.rxn_smiles,
            predictions=enumerated,
        )

    def _rank_enumerate_combinations(
        self, hierarchical_preds: HierarchicalPrediction, top_k: int
    ) -> list[StagePrediction]:
        """
        Convert HierarchicalPrediction to StagePrediction, with given weights. Same as in EnumeratedPredictor.
        """
        enumerated_predictions = []

        for agent_group in hierarchical_preds.agent_groups:
            agents = agent_group["agent_indices"]
            agent_score = agent_group["agent_score"]  # Stage 1 score

            # Get all predictions for this agent group
            temp_preds = [(pred["bin"], pred["score"]) for pred in agent_group["temperature"]]
            reactant_preds = [
                (pred["bin_indices"], pred["score"]) for pred in agent_group["reactant_amounts"]
            ]
            agent_amount_preds = [
                (pred["amounts"], pred["score"]) for pred in agent_group["agent_amounts"]
            ]

            # Generate all combinations
            for temp_bin, temp_score in temp_preds:
                for reactant_bins, reactant_score in reactant_preds:
                    for agent_amount_items, agent_amount_score in agent_amount_preds:

                        # Calculate weighted combined score
                        combined_score = self._calculate_combined_score(
                            agent_score=agent_score,
                            temp_score=temp_score,
                            reactant_score=reactant_score,
                            agent_amount_score=agent_amount_score,
                            n_reactants=len(reactant_bins),
                            n_agents=len(agent_amount_items),
                        )

                        # Create StagePrediction with individual scores stored in meta
                        stage_pred = StagePrediction(
                            agents=agents,
                            temp_bin=temp_bin,
                            reactant_bins=reactant_bins,
                            agent_amount_bins=agent_amount_items,
                            score=combined_score,
                            meta={
                                "s1_score": agent_score,
                                "s2_score": temp_score,
                                "s3_score": self._normalize_reactant_score(
                                    reactant_score, len(reactant_bins)
                                ),
                                "s4_score": self._normalize_agent_amount_score(
                                    agent_amount_score, len(agent_amount_items)
                                ),
                            },
                        )

                        enumerated_predictions.append(stage_pred)

        # Sort by combined score (descending)
        enumerated_predictions.sort(key=lambda x: x.score, reverse=True)

        return enumerated_predictions[:top_k]

    def _rank_enumerate_combinations_iter(
        self, hierarchical_preds: HierarchicalPrediction
    ) -> list[StagePrediction]:
        """
        Memory-efficient enumeration using generators and streaming
        """
        # Use generator to avoid storing all combinations in memory
        all_predictions = self._generate_combinations_stream(hierarchical_preds)

        # Use heap to keep only top-k without storing all
        import heapq

        # Keep track of top predictions using a min-heap
        # We'll use negative scores since heapq is a min-heap
        top_predictions = []

        for stage_pred in all_predictions:
            if len(top_predictions) < 100:  # Keep more than needed for better ranking
                heapq.heappush(top_predictions, (-stage_pred.score, stage_pred))
            elif stage_pred.score > -top_predictions[0][0]:
                # Replace worst prediction if current is better
                heapq.heapreplace(top_predictions, (-stage_pred.score, stage_pred))

        # Extract and sort final results
        final_predictions = [pred for _, pred in top_predictions]
        final_predictions.sort(key=lambda x: x.score, reverse=True)

        # Force garbage collection
        del top_predictions
        gc.collect()

        return final_predictions

    def _generate_combinations_stream(
        self, hierarchical_preds: HierarchicalPrediction
    ) -> Iterator[StagePrediction]:
        """
        Generator that yields combinations one at a time
        """
        for agent_group in hierarchical_preds.agent_groups:
            agents = agent_group["agent_indices"]
            agent_score = agent_group["agent_score"]

            # Get predictions for this agent group
            temp_preds = [(pred["bin"], pred["score"]) for pred in agent_group["temperature"]]
            reactant_preds = [
                (pred["bin_indices"], pred["score"]) for pred in agent_group["reactant_amounts"]
            ]
            agent_amount_preds = [
                (pred["amounts"], pred["score"]) for pred in agent_group["agent_amounts"]
            ]

            # Generate combinations one at a time
            for temp_bin, temp_score in temp_preds:
                for reactant_bins, reactant_score in reactant_preds:
                    for agent_amount_items, agent_amount_score in agent_amount_preds:
                        # Calculate score
                        combined_score = self._calculate_combined_score(
                            agent_score=agent_score,
                            temp_score=temp_score,
                            reactant_score=reactant_score,
                            agent_amount_score=agent_amount_score,
                            n_reactants=len(reactant_bins),
                            n_agents=len(agent_amount_items),
                        )

                        # Yield one prediction at a time
                        yield StagePrediction(
                            agents=agents,
                            temp_bin=temp_bin,
                            reactant_bins=reactant_bins,
                            agent_amount_bins=agent_amount_items,
                            score=combined_score,
                            meta={
                                "s1_score": agent_score,
                                "s2_score": temp_score,
                                "s3_score": self._normalize_reactant_score(
                                    reactant_score, len(reactant_bins)
                                ),
                                "s4_score": self._normalize_agent_amount_score(
                                    agent_amount_score, len(agent_amount_items)
                                ),
                            },
                        )

    def _calculate_combined_score(
        self,
        agent_score: float,
        temp_score: float,
        reactant_score: float,
        agent_amount_score: float,
        n_reactants: int,
        n_agents: int,
    ) -> float:
        """
        Calculate combined score (same as in EnumeratedPredictor)
        """
        # Stage 1 & 2 scores are already individual scores
        normalized_agent_score = agent_score
        normalized_temp_score = temp_score

        # Stage 3: Normalize reactant amount score using geometric mean
        normalized_reactant_score = reactant_score ** (1 / n_reactants) if n_reactants > 0 else 1.0

        # Stage 4: Normalize agent amount score using geometric mean
        normalized_agent_amount_score = (
            agent_amount_score ** (1 / n_agents) if n_agents > 0 else 1.0
        )

        if self.use_geometric:
            combined_score = (
                normalized_agent_score ** self.weights["agent"]
                * normalized_temp_score ** self.weights["temperature"]
                * normalized_reactant_score ** self.weights["reactant_amount"]
                * normalized_agent_amount_score ** self.weights["agent_amount"]
            ) ** (1 / sum(self.weights.values()))
        else:
            combined_score = (
                self.weights["agent"] * normalized_agent_score
                + self.weights["temperature"] * normalized_temp_score
                + self.weights["reactant_amount"] * normalized_reactant_score
                + self.weights["agent_amount"] * normalized_agent_amount_score
            )

        return combined_score

    def _normalize_reactant_score(self, reactant_score: float, n_reactants: int) -> float:
        """Normalize reactant score using geometric mean"""
        return reactant_score ** (1 / n_reactants) if n_reactants > 0 else 1.0

    def _normalize_agent_amount_score(self, agent_amount_score: float, n_agents: int) -> float:
        """Normalize agent amount score using geometric mean"""
        return agent_amount_score ** (1 / n_agents) if n_agents > 0 else 1.0
