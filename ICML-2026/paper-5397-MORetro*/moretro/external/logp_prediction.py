"""
LogP Prediction using ChemProp v2

This module provides functionality to load a ChemProp v2 model checkpoint
and make LogP predictions using the MultiHotAtomFeaturizer.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from lightning import pytorch as pl

# Suppress specific DataLoader warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Consider increasing the value of.*num_workers.*")
warnings.filterwarnings("ignore", ".*is an instance of `nn.Module`.*")
from chemprop import data, featurizers
from chemprop.featurizers import MultiHotAtomFeaturizer
from chemprop.models import MPNN


class LogPPredictor:
    """
    A class for loading ChemProp v2 models and making LogP predictions.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        """
        Initialize the LogP predictor with a model checkpoint.

        Args:
            checkpoint_path: Path to the ChemProp v2 model checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        # Use MultiHotAtomFeaturizer.v1() as the atom featurizer in SimpleMoleculeMolGraphFeaturizer
        atom_featurizer = MultiHotAtomFeaturizer.v1()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=atom_featurizer
        )
        self.device = device

        self._load_model()

    def _load_model(self):
        """Load the model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        try:
            self.model = MPNN.load_from_file(
                self.checkpoint_path, map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    def predict_single(self, smiles: str) -> float:
        """
        Predict LogP for a single SMILES string.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Predicted LogP value
        """
        result = self.predict_batch([smiles])[0]
        return result

    def predict_batch(self, smiles_list: list[str]) -> list[float]:
        """
        Predict LogP for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of predicted LogP values
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check checkpoint path.")

        try:
            # Create molecule datapoints from SMILES (following ChemProp v2 API)
            test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles_list]

            # Create dataset with the datapoints and the v1 featurizer
            test_dataset = data.MoleculeDataset(test_data, featurizer=self.featurizer)

            # Create data loader using ChemProp's build_dataloader
            # Set num_workers=0 for prediction to avoid multiprocessing overhead and warnings
            test_loader = data.build_dataloader(
                test_dataset, shuffle=False, num_workers=0, batch_size=256
            )

            accelerator = "gpu" if self.device == "cuda" else "cpu"

            # Set up trainer for prediction
            with torch.inference_mode():
                trainer = pl.Trainer(
                    logger=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    accelerator=accelerator,
                    devices="auto",
                )

                # Make predictions
                test_preds = trainer.predict(self.model, test_loader)

            # Concatenate predictions and flatten
            predictions = np.concatenate(test_preds, axis=0).flatten()
            return predictions.tolist()

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_from_file(
        self, input_file: str | Path, smiles_column: str = "smiles"
    ) -> list[float]:
        """
        Predict LogP values from a CSV file containing SMILES.

        Args:
            input_file: Path to CSV file containing SMILES
            smiles_column: Name of the column containing SMILES strings

        Returns:
            List of predicted LogP values
        """
        import pandas as pd

        try:
            df = pd.read_csv(input_file)

            if smiles_column not in df.columns:
                raise ValueError(f"Column '{smiles_column}' not found in {input_file}")

            smiles_list = df[smiles_column].tolist()
            return self.predict_batch(smiles_list)

        except Exception as e:
            raise RuntimeError(f"Failed to predict from file: {e}")


def main():
    """
    Example usage of the LogP predictor.
    """
    # Example usage - you'll need to provide the actual checkpoint path
    from pathlib import Path

    model_path = Path(__file__).parent.parent
    checkpoint_path = f"{model_path}/models/model_logp.ckpt"

    # Example SMILES for testing
    test_smiles = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "c1ccccc1O",  # Phenol
        "CCCCCCc1ccccc1",  # Heptylbenzene
        "CCCCCCCCCCCC(=O)OCC",
    ]

    # Initialize predictor
    predictor = LogPPredictor(checkpoint_path)

    # Make predictions
    predictions = predictor.predict_batch(test_smiles)

    print("SMILES -> Predicted LogP")
    print("-" * 30)
    for smiles, logp in zip(test_smiles, predictions):
        print(f"{smiles} -> {logp:.3f}")

    # Example of single prediction
    single_pred = predictor.predict_single("CCO")
    print(f"\nSingle prediction for ethanol (CCO): {single_pred:.3f}")


if __name__ == "__main__":
    main()
