import json
import torch
from pathlib import Path
from typing import Any
import argparse
import pickle
from time import time

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

from quarc.models_code.modules.agent_encoder import AgentEncoder
from quarc.models_code.modules.agent_standardizer import AgentStandardizer
from quarc.data_handlers.datapoints import AgentRecord, ReactionDatum
from quarc.data_handlers.eval_datasets import EvaluationDatasetFactory
from quarc.data_handlers.binning import BinningConfig
from quarc.utils.smiles_utils import parse_rxn_smiles
from quarc.predictors.base import PredictionList
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.multistage_predictor import EnumeratedPredictor
from quarc.predictors.multistage_predictor_batch import EnumeratedBatchPredictor


def prepare_reaction_data(rxn_smiles: str) -> ReactionDatum:
    """Convert input data to ReactionDatum objects"""

    reactants, agents, products = parse_rxn_smiles(rxn_smiles)

    return ReactionDatum(
        rxn_smiles=rxn_smiles,
        reactants=[AgentRecord(smiles=r, amount=None) for r in reactants],
        agents=[AgentRecord(smiles=a, amount=None) for a in agents],
        products=[AgentRecord(smiles=p, amount=None) for p in products],
        rxn_class=None,
        document_id=None,
        date=None,
        temperature=None,
    )


class QuarcPredictor:
    """QUARC predictor wrapper. Used exisiting src/quarc/predictors logic."""

    def __init__(self, config_path: Path, device: str):
        self.config_path = config_path
        self.device = device

        self._initialize_components()
        self._load_models()
    
    def _initialize_components(self):
        json_path = self.config_path.parent / "data" / "processed"
        self.agent_encoder = AgentEncoder(
            class_path=str(json_path / "agent_encoder/agent_encoder_list.json")
        )
        self.agent_standardizer = AgentStandardizer(
            conv_rules=str(json_path / "agent_encoder/agent_rules_v1.json"),
            other_dict=str(json_path / "agent_encoder/agent_other_dict.json"),
        )
        self.featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
        self.binning_config = BinningConfig.default()  # TODO: allow to be set by user

    def _load_models(self):
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")

        models, model_types, weights = load_models_from_yaml(config_file, self.device)

        self.predictor = EnumeratedPredictor(
            agent_model=models["agent"],
            temperature_model=models["temperature"],
            reactant_amount_model=models["reactant_amount"],
            agent_amount_model=models["agent_amount"],
            model_types=model_types,
            agent_encoder=self.agent_encoder,
            device=self.device,
            weights=weights["use_top_5"],
            use_geometric=weights["use_geometric"],
        )

        self.batch_predictor = EnumeratedBatchPredictor(
            agent_model=models["agent"],
            temperature_model=models["temperature"],
            reactant_amount_model=models["reactant_amount"],
            agent_amount_model=models["agent_amount"],
            model_types=model_types,
            agent_encoder=self.agent_encoder,
            device=self.device,
            weights=weights["use_top_5"],
            use_geometric=weights["use_geometric"],
        )

    def _format_prediction_results(
        self, predictions: PredictionList, top_k: int = 10
    ) -> dict[str, Any]:
        """Format prediction results into structured output"""

        temp_labels = self.binning_config.get_bin_labels("temperature")
        reactant_labels = self.binning_config.get_bin_labels("reactant")
        agent_labels = self.binning_config.get_bin_labels("agent")

        results = []

        reactants_smiles, _, _ = parse_rxn_smiles(predictions.rxn_smiles)

        for i, pred in enumerate(predictions.predictions[:top_k]):
            agent_smiles = self.agent_encoder.decode(pred.agents)
            temp_label = temp_labels[pred.temp_bin]
            reactant_labels_list = [
                reactant_labels[bin_idx] for bin_idx in pred.reactant_bins
            ]

            agent_amounts = []
            for agent_idx, bin_idx in pred.agent_amount_bins:
                agent_smi = self.agent_encoder.decode([agent_idx])[0]
                amount_label = agent_labels[bin_idx]
                agent_amounts.append({"agent": agent_smi, "amount_range": amount_label})

            reactant_amounts = []
            for reactant_smi, reactant_label in zip(
                reactants_smiles, reactant_labels_list
            ):
                reactant_amounts.append(
                    {"reactant": reactant_smi, "amount_range": reactant_label}
                )

            prediction = {
                "rank": i + 1,
                "reagents": agent_smiles,
                "temperature": temp_label,
                "reactant_amounts": reactant_amounts,
                "agent_amounts": agent_amounts,
                "score": pred.score,
                # "raw_scores": pred.meta if hasattr(pred, "meta") else {},
            }
            results.append(prediction)

        return results

    def predict(
        self, input_smiles: list[str], top_k: int, beam_size: int = 10
    ) -> list[list[dict[str, Any]]]:
        # prepare smiles to ReactionInput
        reactions = [prepare_reaction_data(s) for s in input_smiles]
        dataset = EvaluationDatasetFactory.for_inference(
            data=reactions,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=self.featurizer,
        )

        # Run predictions
        if self.device == "cpu":
            all_results = []
            for reaction in dataset:
                predictions = self.predictor.predict(
                    reaction, top_k=top_k, beam_size=beam_size
                )
                result = self._format_prediction_results(
                    predictions, top_k=top_k
                )
                all_results.append(result)
        elif self.device == "cuda":
            all_results = self.batch_predictor.predict_many(dataset, top_k=top_k, beam_size=beam_size)
            all_results = [self._format_prediction_results(r, top_k=top_k) for r in all_results]

        return all_results
