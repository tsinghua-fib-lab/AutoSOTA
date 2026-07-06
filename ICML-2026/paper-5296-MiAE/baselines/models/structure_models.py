import os
import urllib.request

import joblib
import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from sequence_models.pdb_utils import process_coords as mif_process_coords
from sequence_models.pretrained import (
    load_model_and_alphabet as mif_load_model_and_alphabet,
)
from tqdm import tqdm

from .proteinmpnn_utils import (
    ProteinMPNN,
    parse_cif_pdb_biounits,
    parse_PDB,
    tied_featurize,
)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def proteinmpnn_process_fn(pdb_path, ca_only=False, return_chain_id=False):
    atom_array = PDBFile.read(pdb_path).get_structure(model=1)
    chain_id = atom_array.chain_id[0]
    pdb_dict = parse_PDB(
        str(pdb_path),
        input_chain_list=[chain_id],
        ca_only=ca_only,
        parse_fn=parse_cif_pdb_biounits,
    )
    if return_chain_id:
        return pdb_dict[0], chain_id
    return pdb_dict[0]


class WrappedProteinMPNN:
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    MODEL_CKPT_BASEURL = "https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/"
    CKPT_CACHE_DIR = os.path.join(SCRIPT_PATH, "ProteinMPNN")
    CA_ONLY = False

    def __init__(
        self,
        checkpoint_path: str = "vanilla_model_weights/v_48_020.pt",
        device: torch.device | str = "cpu",
    ):
        self.device = device
        # init model and load model weights
        local_checkpoint_path = self._download_model_checkpoint(checkpoint_path)
        self._load_protein_mpnn_model(local_checkpoint_path)

    def _download_model_checkpoint(self, checkpoint_path):
        """Download ProteinMPNN checkpoint from GitHub if not locally cached."""
        ckpt_url = self.MODEL_CKPT_BASEURL + checkpoint_path
        cached_checkpoint_path = os.path.join(self.CKPT_CACHE_DIR, checkpoint_path)
        os.makedirs(os.path.dirname(cached_checkpoint_path), exist_ok=True)
        if not os.path.isfile(cached_checkpoint_path):
            urllib.request.urlretrieve(ckpt_url, cached_checkpoint_path)
        return cached_checkpoint_path

    def _load_protein_mpnn_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = ProteinMPNN(
            ca_only=self.CA_ONLY,
            num_letters=21,
            node_features=self.HIDDEN_DIM,
            edge_features=self.HIDDEN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_encoder_layers=self.NUM_LAYERS,
            num_decoder_layers=self.NUM_LAYERS,
            augment_eps=0.0,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device)

    @torch.no_grad()
    def encode_structure(self, pdb_path_list, labels, batch_size=32, num_workers=32):
        parse_fn = parse_cif_pdb_biounits
        X = []
        y = []
        for i in tqdm(
            range(0, len(pdb_path_list), batch_size), desc="Encoding structures"
        ):
            pdb_path_batch = pdb_path_list[i : min(i + batch_size, len(pdb_path_list))]
            labels_batch = labels[i : min(i + batch_size, len(pdb_path_list))]
            pdb_dict_list = joblib.Parallel(n_jobs=num_workers)(
                joblib.delayed(proteinmpnn_process_fn)(str(pdb_path), self.CA_ONLY)
                for pdb_path in pdb_path_batch
            )

            # this function comes from ProteinMPNN:
            (
                h_in,
                _,
                mask,
                _,
                _,
                chain_encoding_all,
                _,
                _,
                _,
                _,
                _,
                _,
                residue_idx,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = tied_featurize(
                pdb_dict_list,
                self.device,
                None,
                None,
                None,
                None,
                None,
                None,
                ca_only=self.CA_ONLY,
            )

            out, h_E = self.model.encode(h_in, mask, residue_idx, chain_encoding_all)
            # out: [B, L, hidden_dim]
            mask = mask.bool()
            for j in range(len(out)):
                X_cur = out[j][mask[j]].mean(0, keepdim=True).cpu()
                X.append(X_cur)
                y.append(labels_batch[j])

        X = torch.concat(X)
        y = torch.tensor(y)
        return X, y


def mif_process_fn(pdb_path):
    pdb_dict, chain_id = proteinmpnn_process_fn(pdb_path, False, return_chain_id=True)
    coords = {
        "N": np.array(pdb_dict[f"coords_chain_{chain_id}"][f"N_chain_{chain_id}"]),
        "CA": np.array(pdb_dict[f"coords_chain_{chain_id}"][f"CA_chain_{chain_id}"]),
        "C": np.array(pdb_dict[f"coords_chain_{chain_id}"][f"C_chain_{chain_id}"]),
    }
    dist, omega, theta, phi = mif_process_coords(coords)
    return [
        pdb_dict[f"seq_chain_{chain_id}"],
        torch.tensor(dist, dtype=torch.float),
        torch.tensor(omega, dtype=torch.float),
        torch.tensor(theta, dtype=torch.float),
        torch.tensor(phi, dtype=torch.float),
    ]


class WrappedMIF:
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.model, self.mif_collater = mif_load_model_and_alphabet("mif")
        self.model.eval()
        self.model = self.model.to(device)

    @torch.no_grad()
    def encode_structure(self, pdb_path_list, labels, batch_size=32, num_workers=32):
        X = []
        y = []
        for i in tqdm(
            range(0, len(pdb_path_list), batch_size), desc="Encoding structures"
        ):
            pdb_path_batch = pdb_path_list[i : min(i + batch_size, len(pdb_path_list))]
            labels_batch = labels[i : min(i + batch_size, len(pdb_path_list))]
            batch = joblib.Parallel(n_jobs=num_workers)(
                joblib.delayed(mif_process_fn)(str(pdb_path))
                for pdb_path in pdb_path_batch
            )

            src, nodes, edges, connections, edge_mask = self.mif_collater(batch)
            src = src.to(self.device)
            nodes = nodes.to(self.device)
            edges = edges.to(self.device)
            connections = connections.to(self.device)
            edge_mask = edge_mask.to(self.device)

            out = self.model(src, nodes, edges, connections, edge_mask)

            for j in range(len(out)):
                X_cur = out[j][: len(batch[j][0])].mean(0, keepdim=True).cpu()
                X.append(X_cur)
                y.append(labels_batch[j])

        X = torch.concat(X)
        y = torch.tensor(y)

        return X, y


baseline_models = {
    "proteinmpnn": WrappedProteinMPNN,
    "mif": WrappedMIF,
}
