# ignore import errors

import joblib
import numpy as np
import os
import pickle
import time
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, SpacialScore
from pathlib import Path


class MolPrice:
    def __init__(self, weights_path, FP_rad=3, FP_len=4096, debug=False):
        """
        Standalone numpy implementation of Fingerprints model.

        Args:
            weights_path: Path to saved weights (.pickle or .pkl)
            FP_rad: Morgan fingerprint radius
            FP_len: Morgan fingerprint length
            debug: Enable debug output
        """
        self.FP_rad = FP_rad
        self.FP_len = FP_len
        self._restored = False
        self.debug = debug

        # Model weights storage
        self.nn_weights = []
        self.nn_biases = []
        self.final_weight = None
        self.final_bias = None

        # Initialize fingerprint generator
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.FP_rad, fpSize=self.FP_len, countSimulation=False
        )
        # Initialize feature extractor
        self.feature_gen = MolFeatureExtractor(weights_path.parent)

        if weights_path:
            self.restore(weights_path)

    def restore(self, weights_path):
        """
        Load model weights from pickle file.

        Args:
            weights_path: Path to weights file (.pickle or .pkl)
        """
        with open(weights_path, "rb") as f:
            weights_dict = pickle.load(f)

        # Extract neural network layers and pre-transpose for efficiency
        layer_indices = [0, 3, 6, 9]  # Based on your model structure
        for idx in layer_indices:
            weight_key = f"neural_network.{idx}.weight"
            bias_key = f"neural_network.{idx}.bias"
            if weight_key in weights_dict and bias_key in weights_dict:
                # Pre-transpose weights to avoid doing it every forward pass
                self.nn_weights.append(weights_dict[weight_key].T)
                self.nn_biases.append(weights_dict[bias_key])

        # Final linear layer (also pre-transpose)
        if "linear.weight" in weights_dict and "linear.bias" in weights_dict:
            self.final_weight = weights_dict["linear.weight"].T
            self.final_bias = weights_dict["linear.bias"]

        self._restored = True
        return self

    def mol_to_fp(self, mol):
        """
        Convert RDKit molecule to Morgan fingerprint using new generator.

        Args:
            mol: RDKit molecule object

        Returns:
            numpy array fingerprint
        """
        if mol is None:
            return np.zeros(self.FP_len, dtype=np.float32)

        # Use the new fingerprint generator to get numpy array directly
        fp = self.fp_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)

    def smi_to_fp(self, smi):
        """
        Convert SMILES string to Morgan fingerprint.

        Args:
            smi: SMILES string

        Returns:
            numpy array fingerprint
        """
        if not smi:
            return np.zeros(self.FP_len, dtype=np.float32)

        mol = Chem.MolFromSmiles(smi)
        fp = self.mol_to_fp(mol)
        features = self.feature_gen.encode(smi)
        features = self.feature_gen.standardise_features(features)
        features = features.reshape(-1, 1)  # Ensure 2D shape
        features = features.squeeze()
        fp = np.concatenate((fp, features), axis=0)
        return fp.astype(np.float32)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input numpy array (fingerprint or batch of fingerprints)

        Returns:
            output: Final price
            z: Latent representation
        """
        if not self._restored:
            raise ValueError("Must restore model weights first!")

        start_time = time.time()
        z = x

        # Neural network layers
        for i, (weight, bias) in enumerate(zip(self.nn_weights, self.nn_biases)):
            z = z @ weight + bias  # Using @ operator instead of np.dot
            if i < len(self.nn_weights) - 1:  # Apply ReLU except for last layer
                z = self.relu(z)

        # Store intermediate representation
        intermediate = z.copy()

        # Final linear layer
        output = z @ self.final_weight + self.final_bias

        forward_time = time.time() - start_time
        if self.debug:
            print(f"Forward pass time: {forward_time:.4f} seconds")

        return output, intermediate

    def predict(
        self, smi, return_intermediate=False
    ) -> float | tuple[float, np.ndarray]:
        """
        Get prediction directly from SMILES string.

        Args:
            smi: SMILES string
            return_intermediate: Whether to return intermediate representation

        Returns:
            price of molecule (and intermediate if requested)
        """
        fp = self.smi_to_fp(smi)
        if np.sum(fp) == 0:
            print("Warning: Could not generate fingerprint for SMILES")
            return 0.0 if not return_intermediate else (0.0, np.zeros(10))

        # Add batch dimension if single molecule
        if fp.ndim == 1:
            fp = fp.reshape(1, -1)

        output, intermediate = self.forward(fp)

        # Remove batch dimension if single molecule
        if output.shape[0] == 1:
            output = output.squeeze(0)
            intermediate = intermediate.squeeze(0)

        if return_intermediate:
            return output, intermediate
        return output[0]


class MolFeatureExtractor:
    def __init__(self, scaler_path: Path):
        self.scaler_path = scaler_path

    def encode(self, smi: str) -> np.ndarray:
        feat = MolFeatureExtractor._calculate_2D_feat(smi)  # type: ignore
        return np.expand_dims(feat, axis=0)

    @staticmethod
    def _calculate_2D_feat(smi):
        mol = Chem.MolFromSmiles(smi)
        sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        sps = SpacialScore.SPS(mol)
        stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        heterocyc = rdMolDescriptors.CalcNumHeterocycles(mol)
        no_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        no_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_macro, n_multi = MolFeatureExtractor.numMacroAndMulticycle(
            mol, mol.GetNumAtoms()
        )
        return np.array(
            [
                sp3,
                sps,
                stereo,
                rot_bonds,
                tpsa,
                heterocyc,
                no_spiro,
                no_bridgehead,
                n_macro,
                n_multi,
            ]
        )

    @staticmethod
    def numMacroAndMulticycle(mol, nAtoms):
        ri = mol.GetRingInfo()  # type: ignore
        nMacrocycles = 0
        multi_ring_atoms = {i: 0 for i in range(nAtoms)}
        for ring_atoms in ri.AtomRings():
            if len(ring_atoms) > 6:
                nMacrocycles += 1
            for atom in ring_atoms:
                multi_ring_atoms[atom] += 1
        nMultiRingAtoms = sum([v - 1 for k, v in multi_ring_atoms.items() if v > 1])
        return nMacrocycles, nMultiRingAtoms

    def standardise_features(self, features: np.ndarray) -> np.ndarray:
        if os.path.exists(self.scaler_path / f"std.bin"):
            try:
                scalar = joblib.load(self.scaler_path / f"std.bin")
                return scalar.transform(features)
            except Exception as e:
                raise (e)
        else:
            raise FileNotFoundError("StandardScaler not found")


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent
    predictor = MolPrice(weights_path=path / "models/MP_Morgan_hybrid.pkl")
    smi = "CCO"
    price = predictor.predict(smi)
    print(f"Predicted price for {smi}: {price}")
