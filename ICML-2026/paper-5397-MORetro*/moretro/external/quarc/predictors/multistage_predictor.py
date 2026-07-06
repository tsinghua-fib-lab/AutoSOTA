from typing import Any
import itertools, numpy as np, torch
from dataclasses import dataclass
from chemprop.data.collate import BatchMolGraph

from quarc.predictors.base import BasePredictor, StagePrediction, PredictionList
from quarc.models_code.search import beam_search_gnn, beam_search
from quarc.data_handlers.eval_datasets import ReactionInput


@dataclass
class HierarchicalPrediction:
    """
    Generates hierarchical predictions (non-enumerated). A cleaner version of intermediate_prediction.py.
    """

    doc_id: str
    rxn_class: str
    rxn_smiles: str
    agent_groups: list[dict[str, Any]]  # List of agent prediction groups

    @classmethod
    def from_models(
        cls,
        reaction_data: dict[str, Any],
        agent_model,
        temperature_model,
        reactant_amount_model,
        agent_amount_model,
        model_types: dict[str, str],
        agent_encoder,
        top_k_agents: int = 5,
        top_k_temp: int = 2,
        top_k_reactant_amount: int = 2,
        top_k_agent_amount: int = 2,
        beam_size: int = 10,
        device: str = "cuda",
    ) -> "HierarchicalPrediction":
        """
        Generate comprehensive predictions for a single reaction including agents and conditions.

        This function takes a reaction and predicts:
        1. Top-k agent sets
        2. For each agent set, top-k temperature predictions
        3. For each agent set, top-k reactant amount predictions
        4. For each agent set, top-k agent amount predictions

        Returns agent prediction with associated subsequent predictions for stage 2-4.

        Args:
            reaction_data: Dictionary containing reaction data
            agent_model: Model to predict agents
            temperature_model: Model to predict temperature
            reactant_amount_model: Model to predict reactant amounts
            agent_amount_model: Model to predict agent amounts
            agent_encoder: Agent encoder object
            top_k_agents: Number of agent combinations to predict
            top_k_temp: Number of temperature bins to predict per agent combination
            top_k_reactant_amount: Number of reactant amount combinations to predict
            top_k_agent_amount: Number of agent amount combinations to predict
            device: Device to run models on
            model_types: Type of models ('gnn' or 'ffn')

        Returns:
            HierarchicalPrediction object containing all predictions in a hierarchical structure
        """
        # Extract metadata
        metadata = reaction_data.metadata

        # Extract model inputs
        FP_reactants = reaction_data.model_inputs["FP_reactants"]
        # rxn_class = reaction_data.model.inputs["rxn_class"]
        mg = reaction_data.model_inputs["mg"]
        FP_inputs = reaction_data.model_inputs["FP_inputs"]

        # rxn_class_tensor = torch.tensor(rxn_class, dtype=torch.float).unsqueeze(0).to(device)
        FP_inputs_tensor = FP_inputs.unsqueeze(0).to(device)
        FP_reactants_tensor = torch.tensor(FP_reactants, dtype=torch.float).unsqueeze(0).to(device)
        bmg = BatchMolGraph([mg])
        bmg.to(device)

        # Stage 1: Predict agents
        beam_results = cls._predict_agents(
            agent_model=agent_model,
            agent_model_type=model_types["agent"],
            agent_encoder=agent_encoder,
            bmg=bmg,
            # rxn_class=rxn_class_tensor,
            FP_inputs=FP_inputs_tensor,
            a_input=torch.zeros(len(agent_encoder), dtype=torch.float).unsqueeze(0).to(device),
            top_k_agents=top_k_agents,
            beam_size = beam_size
        )

        agent_groups = []

        # Process each agent prediction
        for agent_pred_tensor, agent_score in beam_results:
            predicted_agent_indices = agent_pred_tensor.nonzero().squeeze().tolist()
            if isinstance(predicted_agent_indices, int):
                predicted_agent_indices = [predicted_agent_indices]

            a_inputs = agent_pred_tensor.unsqueeze(0).to(device)

            # Stage 2: Predict temperature
            temp_predictions = cls._predict_temperature(
                temperature_model,
                model_types["temperature"],
                a_inputs,
                FP_inputs_tensor,
                bmg,
                top_k_temp,
            )

            # Stage 3: Predict reactant amounts
            reactant_predictions = cls._predict_reactant_amounts(
                reactant_amount_model,
                model_types["reactant_amount"],
                a_inputs,
                FP_inputs_tensor,
                FP_reactants_tensor,
                bmg,
                top_k_reactant_amount,
            )

            # Stage 4: Predict agent amounts
            agent_amount_predictions = cls._predict_agent_amounts(
                agent_amount_model,
                model_types["agent_amount"],
                a_inputs,
                FP_inputs_tensor,
                bmg,
                predicted_agent_indices,
                top_k_agent_amount,
            )

            # Store as agent group
            agent_group = {
                "agent_indices": predicted_agent_indices,
                "agent_score": float(agent_score),
                "temperature": [
                    {"bin": bin_idx, "score": score} for bin_idx, score in temp_predictions
                ],
                "reactant_amounts": [
                    {"bin_indices": bins, "score": score} for bins, score in reactant_predictions
                ],
                "agent_amounts": [
                    {
                        "amounts": [(agent_idx, bin_idx) for agent_idx, bin_idx in amounts],
                        "score": score,
                    }
                    for amounts, score in agent_amount_predictions
                ],
            }

            agent_groups.append(agent_group)

        return cls(
            doc_id=metadata["doc_id"],
            rxn_class=metadata["rxn_class"],
            rxn_smiles=metadata["rxn_smiles"],
            agent_groups=agent_groups,
        )

    @staticmethod
    def _predict_agents(
        agent_model,
        agent_model_type: str,
        agent_encoder,
        bmg,
        # rxn_class,
        FP_inputs,
        a_input,
        top_k_agents,
        beam_size
    ) -> list[tuple[torch.Tensor, float]]:
        """use given model to predict agents, returns beam search results list[(agent_pred_tensor, score)]"""

        if agent_model_type == "gnn":
            beam_results = beam_search_gnn(
                model=agent_model,
                bmg=bmg,
                V_d=None,
                x_d=None,
                num_classes=len(agent_encoder),
                agents_input=a_input,
                max_steps=6,
                beam_size=beam_size,
                eos_id=0,
                return_top_n=top_k_agents,
                verbosity=0,
            )
        elif agent_model_type == "ffn":
            beam_results = beam_search(
                model=agent_model,
                rxn_input=FP_inputs,
                # rxn_class=None,
                num_classes=len(agent_encoder),
                agents_input=a_input,
                max_steps=6,
                beam_size=beam_size,
                eos_id=0,
                return_top_n=top_k_agents,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model type for agent prediction: {agent_model_type}")
        return beam_results

    @staticmethod
    def _predict_temperature(
        temperature_model, model_type, a_inputs, FP_inputs_tensor, bmg, top_k_temp
    ):
        """Predict temperature for given agent inputs"""
        if model_type == "gnn":
            with torch.no_grad():
                temp_preds = temperature_model(a_inputs, bmg)
        elif model_type == "ffn":
            with torch.no_grad():
                temp_preds = temperature_model(FP_inputs_tensor, a_inputs)

        temp_probs = torch.softmax(temp_preds.squeeze(0), dim=-1)
        top_temp_scores, top_temp_preds = temp_probs.topk(top_k_temp, dim=-1)

        return [
            (int(bin_idx), float(score))
            for bin_idx, score in zip(top_temp_preds.cpu(), top_temp_scores.cpu())
        ]

    @staticmethod
    def _predict_reactant_amounts(
        reactant_amount_model,
        model_type,
        a_inputs,
        FP_inputs_tensor,
        FP_reactants_tensor,
        bmg,
        top_k_reactant_amount,
    ):
        """Predict reactant amounts for given agent inputs"""
        if model_type == "gnn":
            with torch.no_grad():
                reactant_preds = reactant_amount_model(a_inputs, FP_reactants_tensor, bmg)
        elif model_type == "ffn":
            with torch.no_grad():
                reactant_preds = reactant_amount_model(
                    FP_inputs_tensor, a_inputs, FP_reactants_tensor
                )

        # Process reactant amounts (same logic as original)
        valid_reactants = (
            FP_reactants_tensor.squeeze(0).nonzero(as_tuple=True)[0].unique().tolist()
        )
        if isinstance(valid_reactants, int):
            valid_reactants = [valid_reactants]

        filtered_reactant_preds = reactant_preds.squeeze(0)[valid_reactants]
        reactant_probs = torch.softmax(filtered_reactant_preds, dim=-1)

        top_reactant_scores, top_reactant_preds = reactant_probs.topk(2, dim=-1)
        combo_lists = [
            [(int(top_reactant_preds[i, j]), float(top_reactant_scores[i, j])) for j in range(2)]
            for i in range(len(valid_reactants))
        ]

        # Get all combinations of reactant predictions
        # combos is a list of ((r1_idx, r1_score), (r2_idx, r2_score), ..., (rn_idx, rn_score))
        combos = list(itertools.product(*combo_lists))
        combo_joint_scores = [np.prod([score for (_, score) in combo]) for combo in combos]
        combo_with_scores = list(zip(combos, combo_joint_scores))

        combo_with_scores_sorted = sorted(combo_with_scores, key=lambda x: x[1], reverse=True)
        top_combos = combo_with_scores_sorted[:top_k_reactant_amount]

        output = []
        for combo, joint_score in top_combos:
            bin_indices = [idx for (idx, _) in combo]
            output.append((bin_indices, joint_score))
        return output

    @staticmethod
    def _predict_agent_amounts(
        agent_amount_model,
        model_type,
        a_inputs,
        FP_inputs_tensor,
        bmg,
        agent_indices,
        top_k_agent_amount,
    ):
        """Predict agent amounts for given agent inputs"""
        if model_type == "gnn":
            with torch.no_grad():
                agent_amount_preds = agent_amount_model(a_inputs, bmg)
        elif model_type == "ffn":
            with torch.no_grad():
                agent_amount_preds = agent_amount_model(FP_inputs_tensor, a_inputs)

        agent_amount_preds = agent_amount_preds.squeeze(0)
        if isinstance(agent_indices, int):
            agent_indices = [agent_indices]

        filtered_agent_amount_preds = agent_amount_preds[agent_indices]
        agent_amount_probs = torch.softmax(filtered_agent_amount_preds, dim=-1)
        top_agent_amount_scores, top_agent_amount_preds = agent_amount_probs.topk(
            2, dim=-1
        )  # n_agents x 2

        agent_combo_lists = [
            [
                (int(top_agent_amount_preds[i, j]), float(top_agent_amount_scores[i, j]))
                for j in range(2)
            ]
            for i in range(len(agent_indices))
        ]

        # Get all combinations
        agent_combos = list(itertools.product(*agent_combo_lists))
        agent_combo_joint_scores = [
            np.prod([score for (_, score) in combo]) for combo in agent_combos
        ]

        agent_combo_with_scores = list(zip(agent_combos, agent_combo_joint_scores))
        agent_combo_with_scores_sorted = sorted(
            agent_combo_with_scores, key=lambda x: x[1], reverse=True
        )
        top_agent_combos = agent_combo_with_scores_sorted[:top_k_agent_amount]

        # Convert to (agent_idx, bin) format
        results = []
        for combo, joint_score in top_agent_combos:
            amount_items = []
            for i, (amount_bin, _) in enumerate(combo):
                agent_idx = agent_indices[i]
                amount_items.append((agent_idx, amount_bin))
            results.append((amount_items, joint_score))

        return results


class EnumeratedPredictor(BasePredictor):
    """
    Model-based predictor that generates enumerated, ranked predictions.
    This wraps HierarchicalPrediction and flattens the results following
    the score calculation logic from overall_evaluate.py.
    """

    def __init__(
        self,
        agent_model,
        temperature_model,
        reactant_amount_model,
        agent_amount_model,
        model_types: dict[str, str],
        agent_encoder,
        top_k_agents: int = 5,
        top_k_temp: int = 2,
        top_k_reactant_amount: int = 2,
        top_k_agent_amount: int = 2,
        device: str = "cuda",
        weights: dict[str, float] = None,
        use_geometric: bool = True,
    ):
        """Initialize with models and top-k parameters"""
        self.agent_model = agent_model
        self.temperature_model = temperature_model
        self.reactant_amount_model = reactant_amount_model
        self.agent_amount_model = agent_amount_model
        self.model_types = model_types
        self.agent_encoder = agent_encoder

        self.top_k_agents = top_k_agents
        self.top_k_temp = top_k_temp
        self.top_k_reactant_amount = top_k_reactant_amount
        self.top_k_agent_amount = top_k_agent_amount
        self.device = device
        self.use_geometric = use_geometric

        if weights is None:
            weights = {
                "agent": 0.25,
                "temperature": 0.25,
                "reactant_amount": 0.25,
                "agent_amount": 0.25,
            }
        self.weights = weights

        self.agent_model.eval()
        self.temperature_model.eval()
        self.reactant_amount_model.eval()
        self.agent_amount_model.eval()

    def predict(self, reaction: ReactionInput, top_k, beam_size) -> PredictionList:
        """
        Generate enumerated predictions by first creating hierarchical view,
        then flattening and ranking all combinations.
        """
        hierarchical_view = HierarchicalPrediction.from_models(
            reaction_data=reaction,
            agent_model=self.agent_model,
            temperature_model=self.temperature_model,
            reactant_amount_model=self.reactant_amount_model,
            agent_amount_model=self.agent_amount_model,
            model_types=self.model_types,
            agent_encoder=self.agent_encoder,
            top_k_agents=self.top_k_agents,
            top_k_temp=self.top_k_temp,
            top_k_reactant_amount=self.top_k_reactant_amount,
            top_k_agent_amount=self.top_k_agent_amount,
            beam_size=beam_size,
            device=self.device,
        )

        enumerated_predictions = self._rank_enumerate_combinations(hierarchical_view, top_k)

        return PredictionList(
            doc_id=hierarchical_view.doc_id,
            rxn_class=hierarchical_view.rxn_class,
            rxn_smiles=hierarchical_view.rxn_smiles,
            predictions=enumerated_predictions,
        )

    def _rank_enumerate_combinations(
        self, hierarchical_preds: HierarchicalPrediction, top_k: int
    ) -> list[StagePrediction]:
        """
        Convert hierarchical view to flat enumerated predictions.
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

                        combined_score = self._calculate_combined_score(
                            agent_score=agent_score,
                            temp_score=temp_score,
                            reactant_score=reactant_score,
                            agent_amount_score=agent_amount_score,
                            n_reactants=len(reactant_bins),
                            n_agents=len(agent_amount_items),
                        )

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

        enumerated_predictions.sort(key=lambda x: x.score, reverse=True)

        return enumerated_predictions[:top_k]

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
        Calculate combined score following overall_evaluate.py logic:
        1. Normalize multi-item scores using geometric mean
        2. Combine using weighted arithmetic or geometric mean
        """

        # Stage 1 & 2 scores are already individual scores
        normalized_agent_score = agent_score
        normalized_temp_score = temp_score

        # Stage 3: Normalize reactant amount score using geometric mean
        # (Note: In hierarchical view, this is already the joint score from product of individual reactant scores)
        normalized_reactant_score = reactant_score ** (1 / n_reactants) if n_reactants > 0 else 1.0

        # Stage 4: Normalize agent amount score using geometric mean
        # (Note: In hierarchical view, this is already the joint score from product of individual agent scores)
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
