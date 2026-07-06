import joblib
import json
import torch
from pathlib import Path
from rdkit import Chem
import Graph2Edits.utils

from Graph2Edits.models.graph2edits import Graph2Edits
from Graph2Edits.models.beam_search import BeamSearch

# from Graph2Edits.models.value_fn import DMPNNValue
from Graph2Edits.utils.rxn_graphs import Vocab


class BaseAgent(object):
    def __init__(self, device: str):
        self.device = device
        self.nn_models = None

    def load_state_dict(self, state_dict):
        for key, model in self.nn_models.items():
            model.to(device="cpu")
            model.load_state_dict(state_dict[key])
            model.to(device=self.device)
            model.eval()

    def get_state_dict(self):
        state_dict = {}
        for key, model in self.nn_models.items():
            state_dict[key] = model.state_dict()
        return state_dict


class Graph2EditsPolicy(BaseAgent):
    def __init__(self, model_checkpoint: Path, vocab_checkpoint: Path, device: str):
        super().__init__(device=device)
        self.device = device
        self.updates = 0

        self.bond_vocab = Vocab(joblib.load(vocab_checkpoint / "bond_vocab.txt"))
        self.atom_vocab = Vocab(joblib.load(vocab_checkpoint / "atom_lg_vocab.txt"))
        with open(model_checkpoint.parent / "config.json", "r") as f:
            config = json.load(f)

        self.graph2edits = Graph2Edits(
            config,
            atom_vocab=self.atom_vocab,
            bond_vocab=self.bond_vocab,
            device=device,
        )

        # Load bream search
        self.beam_model = BeamSearch(
            model=self.graph2edits, step_beam_size=5, beam_size=10, use_rxn_class=False
        )

        # # Prepare value function
        # checkpoint = torch.load(
        #     args.graph2edits_model, map_location=torch.device("cpu"), weights_only=False
        # )
        # config = checkpoint["saveables"]
        # self.value_fn = DMPNNValue(**config, device=args.device)
        # self.value_fn.load_state_dict(checkpoint["state"])
        # self.value_fn.to(args.device)
        # self.value_fn.eval()
        # self.value_optim = torch.optim.Adam(self.value_fn.parameters(), lr=args.lr)

        self.nn_models = {"policy": self.graph2edits}  # , "value_fn": self.value_fn}

    def predict(
        self, products: str | list[str], topk: int = 5, templates=None
    ) -> list[list[dict]]:
        if isinstance(products, str):
            products = [products]
        self.beam_model.beam_size = topk

        all_predictions = []
        for p_smi in products:
            mol = Chem.MolFromSmiles(p_smi)
            for i, atom in enumerate(mol.GetAtoms(), start=1):
                atom.SetAtomMapNum(i)
            labeled_smi = Chem.MolToSmiles(mol)
            top_k_results = self.beam_model.run_search(
                prod_smi=labeled_smi, max_steps=8, rxn_class=False
            )

            product_preds = []
            for res in top_k_results:
                if not res["final_smi"].startswith("final_smi_unmapped"):
                    edit_seq_with_params = []
                    for action, atom_or_atoms in zip(res["edits"], res["edits_atom"]):
                        edit_seq_with_params.append((action, atom_or_atoms))
                    edit_seq_with_params.append(("Terminate", None))
                    final_smiles = res["final_smi"]
                    final_mol = Chem.MolFromSmiles(final_smiles)
                    [a.SetAtomMapNum(0) for a in final_mol.GetAtoms()]
                    final_smiles = Chem.MolToSmiles(final_mol)
                    product_preds.append(
                        {
                            "rxn_smiles": f"{final_smiles}>>{p_smi}",
                            "score": res["prob"],
                            "template": edit_seq_with_params,
                            "reactants": final_smiles.split("."),
                        }
                    )
            all_predictions.append(product_preds[:topk])

        return all_predictions


if __name__ == "__main__":
    g2p_checkpoint = Path(__file__).parent.parent / "models/graph2edits.pth"
    g2e_checkpoint = Path(__file__).parent / "Graph2Edits/checkpoint"
    vocab_checkpoint = Path(__file__).parent / "Graph2Edits/vocab"
    model = Graph2EditsPolicy(g2e_checkpoint, vocab_checkpoint, device="cpu")
    model.load_state_dict(torch.load(g2p_checkpoint, map_location="cpu"))
    test_smiles = "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S"
    predictions = model.predict(test_smiles, topk=3)
    for smi, preds in zip(test_smiles, predictions):
        print(f"Predictions for {smi}:")
        for pred in preds:
            print(pred)
