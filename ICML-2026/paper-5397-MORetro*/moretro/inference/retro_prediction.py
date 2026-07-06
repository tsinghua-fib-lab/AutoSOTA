import json
import logging
from pathlib import Path
from typing import Any

import gin
import torch
from rdkit import Chem

from moretro.external.quarc.quarc_predictor import QuarcPredictor
from moretro.external.template_models import PDVN, TemplRel
from moretro.external.tf_models import Graph2EditsPolicy
from moretro.inference.calculate_costs import COST_MAPPING, calculate_costs
from moretro.utils.base_paths import MODELS_DIR
from moretro.utils.typing_hints import Predictions

logger = logging.getLogger(__name__)
file_path = Path(__file__).parent


@gin.configurable()
class OneStepModel:
    """
    A one-step model class for retro prediction.
    This class incorporates different one step models
    """

    # TODO specify device for the models
    def __init__(
        self,
        model_type: str,
        cost_functions: list[str],
        device: str,
    ):
        self.model_type = model_type
        if self.model_type == "template":
            self.checkpoint_path = MODELS_DIR / "template/model_retro.pt"
            self.template_path = MODELS_DIR / "template/idx2template_retro.json"
        elif self.model_type == "pdvn":
            self.checkpoint_path = MODELS_DIR / "pdvn/rollout_model.ckpt"
            self.template_path = MODELS_DIR / "pdvn/template_rules_1.dat"
        elif self.model_type == "g2e":
            self.checkpoint_path = MODELS_DIR / "g2e/graph2edits.pth"
            self.template_path = None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            # * Add new models here

        self.condition_model = ConditionModel(gin.REQUIRED)  # type: ignore
        self.device = device
        logger.info(f"Loading Single-Step Model from {self.checkpoint_path}")

        self.cost_functions = []
        for cost_name in cost_functions:
            if cost_name in COST_MAPPING:
                self.cost_functions.append(COST_MAPPING[cost_name])
            else:
                logger.error(f"Unknown cost function: {cost_name}")
                raise ValueError("Please ensure that all cost functions are defined")

        if self.template_path:
            logger.info(f"Using template path: {self.template_path}")
        else:
            logger.info("No template path provided, expected for non-template models.")

        if model_type == "template" and self.template_path:
            with open(self.template_path, encoding="utf-8") as f:
                template_dict = json.load(f)
            self.templates = {}
            for k, v in template_dict.items():
                self.templates[int(k)] = v
            retro_checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            pretrain_args = retro_checkpoint["args"]
            self.model = TemplRel(pretrain_args)
            self.model.to(self.device)
            state_dict = retro_checkpoint["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        elif model_type == "pdvn" and self.template_path:
            self.model = PDVN(
                trained_model=self.checkpoint_path,
                template_path=self.template_path,
                device=self.device,
                fp_dim=2048,
                realistic_filter=True,
            )
            self.templates = {}  # PDVN does not need templates passed in prediction
        elif model_type == "g2e":
            self.model = Graph2EditsPolicy(
                model_checkpoint=self.checkpoint_path,
                vocab_checkpoint=self.checkpoint_path.parent / "vocab",
                device=self.device,
            )
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location="cpu")
            )
            self.templates = {}  # G2E does not need templates passed in prediction
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            # * Add new models here
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def predict(self, target: str | list[str], top_n: int = 50) -> Predictions:
        """
        Predict the retro reactions for a given molecule or list of molecules up to top_n reactions.

        Parameters
        ----------
        target : str | list[str]
            The SMILES representation of the target molecule or a list of SMILES strings.
        top_n : int
            The number of top predictions to return.

        Returns
        -------
        Predictions
            A list of lists of dictionaries containing the predicted retro reactions.
            Each prediction dict contains: ["rxn_smiles", "reactants", "template", "score", "costs", "reagents", "temperature"]
        """
        # Get predictions from the underlying model
        if isinstance(target, list) and len(target) == 1:
            target = target[0]
        predictions = self.model.predict(target, top_n, self.templates)
        updated_predictions = self._add_cost_and_condition(predictions)
        return updated_predictions

    def _add_cost_and_condition(self, predictions: Predictions) -> Predictions:
        # Add cost calculations and missing fields to each prediction
        updated_predictions: Predictions = []
        for mol_predictions in predictions:
            rxn_smiles = [pred["rxn_smiles"] for pred in mol_predictions]
            if not rxn_smiles:
                updated_predictions.append(mol_predictions)
                continue
            recording_indices = []
            for rxn in rxn_smiles:
                rxn_smiles_parts = rxn.split(">>")
                lengths = []
                for n in rxn_smiles_parts:
                    mol = Chem.MolFromSmiles(n)
                    lengths.append(len(mol.GetAtoms()) if mol else 0)  # type: ignore
                if all([length == 1 for length in lengths]):
                    recording_indices.append(False)
                else:
                    recording_indices.append(True)
            # delete predictions with only single atom molecules
            rxn_smiles = [
                rxn
                for rxn, record in zip(rxn_smiles, recording_indices, strict=True)
                if record
            ]
            checked_mol_predictions = [
                pred
                for pred, record in zip(mol_predictions, recording_indices, strict=True)
                if record
            ]
            conditions = self.condition_model.predict(rxn_smiles)
            expanded_mol_predictions = []
            for pred, topk_cond in zip(
                checked_mol_predictions, conditions, strict=True
            ):
                for cond in topk_cond:
                    # Create a copy of the prediction for each condition
                    pred_copy = pred.copy()
                    pred_copy["temperature"] = cond["temperature"]
                    pred_copy["reagents"] = cond["reagents"]
                    if "agent_amounts" in cond:
                        pred_copy["agent_amounts"] = cond["agent_amounts"]
                    expanded_mol_predictions.append(pred_copy)
            updated_predictions.append(expanded_mol_predictions)

        updated_predictions = calculate_costs(updated_predictions, self.cost_functions)
        return updated_predictions


@gin.configurable()
class ConditionModel:
    """
    Prediction of reaction conditions given the reaction string
    """

    def __init__(
        self,
        model_type: str,
        config_path: str,
        device: str,
        top_k: int,
        beam_size: int,
    ):
        self.model_type = model_type
        self.config_path = file_path.parent / config_path
        self.device = device
        self.top_k = top_k
        self.beam_size = beam_size
        if self.model_type == "quarc":
            self.model = QuarcPredictor(
                config_path=self.config_path, device=self.device
            )
        elif self.model_type == "rct":
            raise NotImplementedError("R-CT model not implemented yet.")
        else:
            raise ValueError(f"Unsupported condition model type: {self.model_type}")

    def predict(self, rxn_smiles: list[str]) -> list[list[dict[str, Any]]]:
        """
        Predict the reaction conditions for a given reaction SMILES.
        This method should be implemented by subclasses.
        """
        results = self.model.predict(rxn_smiles, self.top_k, self.beam_size)
        results = self._clean_up_prediction(results)
        return results

    def _clean_up_prediction(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[list[dict[str, Any]]]:
        """
        Clean up the predictions by removing duplicate entries.
        """
        cleaned_predictions = []
        for mol_preds in predictions:
            seen = set()
            unique_preds = []
            for pred in mol_preds:
                # NOTE: This ignores the reagent amount for now.
                pred_tuple = (pred["temperature"], tuple(pred["reagents"]))
                if pred_tuple not in seen:
                    seen.add(pred_tuple)
                    unique_preds.append(pred)
            cleaned_predictions.append(unique_preds)
        return cleaned_predictions
