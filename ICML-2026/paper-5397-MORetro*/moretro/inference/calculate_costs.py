# this file introduces the cost functions to calculate costs of reactions
# * All costs are scaled between 0 and 1 -> 0 = best, 1 = worst
import json
import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from rdkit import Chem

from moretro.external.logp_prediction import LogPPredictor
from moretro.utils.base_paths import MODELS_DIR, PACKAGE_DIR
from moretro.utils.typing_hints import (
    BatchedCostFunction,
    CostFunctions,
    IndividualCostFunction,
    Predictions,
)

logger = logging.getLogger(__name__)

# Global variables to hold loaded models/data for cost functions
TOXICITY_DATA: dict[str, float] | None = None
LOGP_PREDICTOR: LogPPredictor | None = None


def cost_loader(cost_name: str, device: str) -> None:
    """
    Loads the necessary models for the cost functions based on the provided cost name.
    This function is called at module load time to ensure that models are loaded only once.
    """
    global TOXICITY_DATA, LOGP_PREDICTOR

    if cost_name == "scaleup_cost":
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logp_model_path = MODELS_DIR / "objectives/model_logp.ckpt"
        if not logp_model_path.exists():
            logger.error(
                "LogP checkpoint %s missing. Scale-up cost cannot be calculated anymore.",
                logp_model_path,
            )
        try:
            LOGP_PREDICTOR = LogPPredictor(
                checkpoint_path=logp_model_path, device=device
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load LogP predictor (%s).", exc)
    elif cost_name == "toxicity_cost":
        # Toxicity data is loaded at module level for efficiency
        data_path = PACKAGE_DIR / "external/tox_score/agents_with_scores.json"
        try:
            with open(data_path, "rb") as f:
                TOXICITY_DATA = json.load(f)
        except FileNotFoundError:
            logger.error(
                "Toxicity data not found; toxicity_cost will not be calculated correctly."
            )
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse toxicity data (%s).", exc)
    elif cost_name in {
        "sustainability_cost",
        "convergence_cost",
        "retro_star_cost",
        "policy_cost",
    }:
        # No specific model to load for these costs, but could be added here if needed
        pass
    else:
        raise ValueError(f"Unknown cost function: {cost_name}")


def batch_fn(func: Callable) -> Callable:
    """
    Decorator to mark a cost function as batched.
    """
    func.batched = True  # type: ignore
    return func


def calculate_costs(
    predictions: Predictions, cost_functions: CostFunctions
) -> Predictions:
    """
    Calculate multi-dimensional costs for batched reactions and add them to predictions.

    Automatically detects whether cost functions expect individual predictions
    or batched predictions based on their type annotations.

    Parameters:
    -----------
    predictions : Predictions
        List of lists of prediction dictionaries, where each inner list represents
        reactions for a specific molecule. Each prediction dict contains at minimum:
        - "rxn_smiles": str - Full reaction SMILES string
        - "reactants": list[str] - List of reactant SMILES strings
        - "template": str or list[str] - Reaction template in SMARTS format
        - "score": float - Model prediction score
    cost_functions : CostFunctions
        List of cost functions that take either individual predictions
        (dict[str, Any] -> float) or batched predictions (list[dict[str, Any]] -> list[float])

    Returns:
    --------
    Predictions
        Updated predictions with costs added as both:
        - "costs": dict mapping function names to cost values
        - "cost_vector": list of costs in same order as cost_functions

    """
    # Process each molecule's reactions
    for mol_predictions in predictions:
        mol_costs = []

        # Initialize cost arrays for each reaction
        num_reactions = len(mol_predictions)
        for _ in range(num_reactions):
            mol_costs.append([])

        # Process each cost function
        for cost_fn in cost_functions:
            if hasattr(cost_fn, "batched"):
                batched_fn = cast(BatchedCostFunction, cost_fn)
                batch_costs = batched_fn(mol_predictions)
                # Add costs to each reaction
                for rxn_idx, cost in enumerate(batch_costs):
                    mol_costs[rxn_idx].append(cost)
            else:
                # Individual function - calculate cost for each reaction
                individual_fn = cast(IndividualCostFunction, cost_fn)
                for rxn_idx, pred in enumerate(mol_predictions):
                    cost = individual_fn(pred)
                    mol_costs[rxn_idx].append(cost)

        # Assign cost vectors to each prediction dictionary
        for rxn_idx, pred in enumerate(mol_predictions):
            pred["costs"] = mol_costs[rxn_idx]

    return predictions


def atom_economy_cost(prediction: dict[str, Any]) -> float:
    """
    Atom economy cost based on the number of atoms in reactants vs products.
    Lower atom economy = higher cost.
    """
    rxn_smiles = prediction["rxn_smiles"]
    reactants_smiles, products_smiles = rxn_smiles.split(">>")

    # Calculate total heavy atoms in reactants
    reactant_atoms = 0
    for reactant in reactants_smiles.split("."):
        mol = Chem.MolFromSmiles(reactant.strip())
        if mol:
            reactant_atoms += mol.GetNumHeavyAtoms()

    # Calculate total heavy atoms in products
    product_atoms = 0
    for product in products_smiles.split("."):
        mol = Chem.MolFromSmiles(product.strip())
        if mol:
            product_atoms += mol.GetNumHeavyAtoms()

    atom_economy = product_atoms / reactant_atoms
    return max(1 - atom_economy, 0.0)


def temperature_cost(prediction: dict[str, Any]) -> float:
    """
    Cost based on the reaction temperature.
    Best ambient, lower for cryo and high
    """
    temp = prediction["temperature"]
    if isinstance(temp, str):
        temp = temp.replace(")", "]")
        temp = eval(temp.replace("inf", "1000"))  # no inf in eval
        temp = np.mean(temp)
    if 15 <= temp <= 25:
        return 0.0
    elif 10 <= temp < 15 or 25 < temp <= 40:
        return 0.25
    elif -20 <= temp < 10:
        return 0.6
    elif temp < -20:
        return 1.0
    elif 40 < temp <= 120:
        return 0.4
    # temperature > 120
    return 0.8


def combined_sustainability(prediction: dict[str, Any]) -> float:
    """
    Combine temperature and atom economy costs.
    """
    temp_cost = temperature_cost(prediction)
    atom_cost = atom_economy_cost(prediction)
    combined = (temp_cost + atom_cost) / 2
    combined = min(combined, 1.0)  # Cap at 1.0
    return combined


def toxicity_cost(prediction: dict[str, Any]) -> float:
    """
    Cost based on the toxicity of the agents used in the reaction.
    """
    agents: list[str] = prediction["reagents"]
    score = 0.0
    if not agents:
        return score  # No reagents, no cost
    if not TOXICITY_DATA:
        logger.warning("Toxicity data unavailable; returning zero toxicity cost.")
        return score
    for agent in agents:
        if agent in TOXICITY_DATA:
            score = max(TOXICITY_DATA[agent], score)  # Keep the worst score
        else:
            logger.warning(
                f"Agent '{agent}' not found in toxicity data. Assigning default toxicity cost of 0.5. Please update toxicity data if this agent is expected."
            )
            score = max(0.5, score)  # Assign default cost for missing agent
    return score


@batch_fn
def scaleup_cost(predictions: list[dict[str, Any]]) -> list[float]:
    """
    Cost based on likelihood of ease of separation via LLE
    """
    # * Enhancement: improve this calculation - perhaps also including reagent (solvent)
    if not predictions:
        return []

    scores = []
    molecules = []
    reactants_idx_tracker = []

    # Get product (same for all reactions) and add once
    product_smiles = predictions[0]["rxn_smiles"].split(">>")[1].strip()
    molecules.append(product_smiles)

    # Collect reactants for each reaction and track indices
    for pred in predictions:
        reactants_smiles = pred["rxn_smiles"].split(">>")[0]
        reactants = [r.strip() for r in reactants_smiles.split(".")]

        start_idx = len(molecules)  # Starting index for this reaction's reactants
        molecules.extend(reactants)
        end_idx = len(molecules)  # Ending index for this reaction's reactants

        # Store the indices for this reaction's reactants
        reactants_idx_tracker.append(
            {"reactants_start": start_idx, "reactants_end": end_idx}
        )

    if LOGP_PREDICTOR is None:
        logger.warning("Using baseline LogP scoring for scaleup_cost.")
        logp_predictions = [0.0] * len(molecules)
    else:
        logp_predictions = LOGP_PREDICTOR.predict_batch(molecules)
    prod_logp = logp_predictions[0]  # Product logP is always first

    # Calculate scores for each reaction
    for reaction_indices in reactants_idx_tracker:
        reagent_logps = logp_predictions[
            reaction_indices["reactants_start"] : reaction_indices["reactants_end"]
        ]

        # Calculate absolute differences for all reactants from product
        differences = [abs(r_logp - prod_logp) for r_logp in reagent_logps]

        # Guard against empty reactants
        if not differences:
            scores.append(0.5)  # Default medium cost
            continue

        # Use average difference with threshold-based scoring
        avg_diff = sum(differences) / len(differences)

        # Convert average difference to cost score (higher diff = better separation = lower cost)
        if avg_diff >= 3.0:
            score = 0.0  # Excellent separation
        elif avg_diff >= 2.5:
            score = 0.2  # Very good separation
        elif avg_diff >= 2.0:
            score = 0.4  # Good separation
        elif avg_diff >= 1.0:
            score = 0.6  # Fair separation
        elif avg_diff >= 0.5:
            score = 0.8  # Poor separation
        else:  # avg_diff < 0.5
            score = 1.0  # Very poor separation

        scores.append(score)

    return scores


def log_score(prediction: dict[str, Any]) -> float:
    """
    Logarithmic cost based on the model prediction score.
    Higher score = lower cost.
    """
    score = prediction["score"]
    score = -np.log(np.clip(score, 1e-3, 1.0))  # Natural log
    score /= 10  # Scale down
    score = min(score, 1.0)  # Cap at 1.0
    return score


def even_split(prediction: dict[str, Any]) -> float:
    """
    The more even the reactants are in size, the better the reaction.
    If it is just a single molecule, the cost is 1.0
    """
    reactants_smiles = prediction["rxn_smiles"].split(">>")[0]
    reactants = [r.strip() for r in reactants_smiles.split(".")]
    if len(reactants) < 2:
        return 1.0  # Single reactant, high cost

    sizes = []
    for reactant in reactants:
        mol = Chem.MolFromSmiles(reactant)
        if mol:
            sizes.append(mol.GetNumHeavyAtoms())
        else:
            sizes.append(0)

    if not sizes or sum(sizes) == 0:
        return 1.0  # No valid sizes, high cost

    ratio_split = [size / sum(sizes) for size in sizes]
    ratio_split = np.array(ratio_split)
    # calculate max difference between ratios and ideal even split
    ideal_split = 1 / len(reactants)
    diffs = np.abs(ratio_split - ideal_split)
    cost = np.mean(diffs) * len(reactants)  # scale by number
    cost = float(min(cost, 1.0))  # Cap at 1.0

    return cost


# Cost function mapping for easy configuration
COST_MAPPING = {
    "sustainability_cost": combined_sustainability,
    "scaleup_cost": scaleup_cost,
    "toxicity_cost": toxicity_cost,
    "convergence_cost": even_split,
    "retro_star_cost": log_score,
    "policy_cost": log_score,
}
