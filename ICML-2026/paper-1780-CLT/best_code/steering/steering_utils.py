import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import os
import re
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_transcoder'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'circuit_utils'))
from steering.full_replacement_models import FullCLTReplacementModel, FullPLTReplacementModel, FullCLTDirectReplacementModel
from training.clt_module import CLTLightningModule
from training_transcoder.plt_module import PLTLightningModule
from circuit_utils import compute_attribution, rank_nodes

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT/PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# K: Number of nodes in circuit
# ──────────────────────────────────────────────────────────────────────────────

def ablate_latents(model, seq, circuit, freeze_attention = None):
    """
    Ablate latents in the given circuit for the input sequences using the provided model.
    Args:
        model: The full replacement model (CLT or PLT).
        seqs: Wild-type DMS sequence
        circuit: Dictionary mapping layer indices to lists of latent indices to ablate.
    Returns:
        seqs: A list of output sequences after ablating latents length k, where k is circuit size.
    """
    L = model.num_layers
    seqs_K = []
    embs_K = []
    # should be a total of k model forward passes
    for i in range(L + 1):
        try:
            latents_to_ablate = circuit[i]
        except KeyError:
            continue
        for latent in latents_to_ablate:
            ablate_nodes = {i: [latent]}
            batch_seqs = [seq]
            # forward pass through the model and get the return embedding
            with torch.no_grad():
                if freeze_attention is not None:
                    emb_1TH, _, _, _, mask_1T = model(batch_seqs, ablate_nodes=ablate_nodes, freeze_attention=freeze_attention)
                else:
                    emb_1TH, _, _, _, mask_1T = model(batch_seqs, ablate_nodes=ablate_nodes)
                embs_K.append(emb_1TH.squeeze(0))
                # convert to seq
                seq_out, _ = model.get_sequences(emb_1TH, mask_1T)
                seqs_K = seqs_K + seq_out
    return seqs_K, embs_K

def ablate_latents_steering(model, seq, circuit, freeze_attention = None):
    """
    Ablate latents in the given circuit for the input sequences using the provided model.
    Args:
        model: The full replacement model (CLT or PLT).
        seqs: Wild-type DMS sequence
        circuit: Dictionary mapping layer indices to lists of latent indices to ablate.
    Returns:
        seqs: A list of output sequences after ablating latents length k, where k is circuit size.
    """
    L = model.num_layers
    seqs_K = []
    embs_K = []
    # should be a total of k model forward passes
    for i in range(L + 1):
        try:
            latents_to_ablate = circuit[i]
        except KeyError:
            continue
        for latent in latents_to_ablate:
            ablate_nodes = {i: [latent]}
            # set ablate_nodes to the circuit minus i: [latent]
            # ablate_nodes = circuit.copy()
            # ablate_nodes[i].remove(latent)
            # forward pass through the model and get the return embedding
            with torch.no_grad():
                if freeze_attention is not None:
                    emb_1TH, _, _, _, mask_1T = model.forward_steered(seq, 1, circuit, before=True, ablate_nodes=ablate_nodes, freeze_attention=freeze_attention)
                else:
                    emb_1TH, _, _, _, mask_1T = model.forward_steered(seq, 1, circuit, before=True, ablate_nodes=ablate_nodes)
                embs_K.append(emb_1TH.squeeze(0))
                # convert to seq
                seq_out, _ = model.get_sequences(emb_1TH, mask_1T)
                seqs_K = seqs_K + seq_out
    return seqs_K, embs_K


def keep_only_latents_via_ablation(model, seq, circuit, D, freeze_attention=None):
    """
    For each (layer l, latent i) in circuit:
      - ablate (zero) all latents in layer l except i
      - run a forward pass and decode sequence

    Uses only `ablate_nodes` (no active_nodes needed).

    Returns:
        seqs_K: list[str]
        embs_K: list[Tensor]
    """

    L = model.num_layers
    seqs_K = []
    embs_K = []
    batch_seqs = [seq]

    # Precompute full index list once
    all_latents = list(range(D))

    for l in range(L + 1):
        if l not in circuit:
            continue

        for keep_latent in circuit[l]:
            # ablate everything except keep_latent
            ablate_list = all_latents.copy()
            ablate_list.pop(keep_latent)  # removes the element at position keep_latent (which equals its value)

            ablate_nodes = {l: ablate_list}

            with torch.no_grad():
                if freeze_attention is not None:
                    emb_1TH, _, _, _, mask_1T = model(
                        batch_seqs,
                        ablate_nodes=ablate_nodes,
                        freeze_attention=freeze_attention,
                    )
                else:
                    emb_1TH, _, _, _, mask_1T = model(
                        batch_seqs,
                        ablate_nodes=ablate_nodes,
                    )

            embs_K.append(emb_1TH.squeeze(0))
            seq_out, _ = model.get_sequences(emb_1TH, mask_1T)
            seqs_K.extend(seq_out)

    return seqs_K, embs_K

def score_seqs(scoring_model, embs):
    """
    Score the given embeddings using the provided scoring model.
    Args:
        scoring_model: The model used to score the embeddings (CNN)
        embs: A list of embeddings to be scored (K, T, H)
    """
    # we can just pass in all the embs as 1 batch since K isn't much
    embs_tensor = torch.stack(embs, dim=0)  # (K, T, H)
    with torch.no_grad():
        scores = scoring_model(embs_tensor)  # (K, 1)
    return scores.squeeze(-1)  # (K,)

def get_wt_from_dms(dms_csv):
    """
    Returns wild-type sequence as a string given the DMS CSV
    dms_csv should be a path to one of the SINGLE MUTATIONS CSVS
    """
    dms_df = pd.read_csv(dms_csv)
    first_mutant = dms_df['mutant'].iloc[0]
    first_sequence = dms_df['mutated_sequence'].iloc[0]
    index = int(first_mutant[1:-1]) - 1
    return first_sequence[:index] + first_mutant[0] + first_sequence[index + 1:]
    

# def steer_latent_circuit(model, seq, circuit, alpha):
#     """
#     Steer the latents in the given circuit for the input sequences using the provided model.
#     This method uses the circuit, adds it with strength alpha, then decodes.
#     Args:
#         model: The full replacement model (CLT or PLT).
#         seqs: Wild-type DMS sequence
#         circuit: Dictionary mapping layer indices to lists of latent indices to steer.
#         alpha: Steering strength
#     """
#     model.eval()
#     tokens_1T = model.tokenize([seq])  # (1, T)
#     with torch.no_grad():
#         x_curr_BTH = model.esm.embed_tokens(tokens_1T)
#         # 1. Transpose: (B, T, H) -> (T, B, H)
#         x_curr_TBH = x_curr_BTH.transpose(0, 1)
#         for l in range(model.num_layers):
#             x_curr_TBH = model.esm.layers[l](x_curr_TBH)
#             # If this layer has latents to steer, do so
#             if l in circuit:
#                 for latent_idx in circuit[l]:
#                     # Steer the latent by adding alpha
#                     x_curr_TBH[:, 0, latent_idx] += alpha


def rank_nodes_by_attribution(discoverer, probe, seq, circuit, sequential=True, freeze_attention=True):
    """
    Rank nodes in a circuit by gradient-based attribution scores.
    Much faster and more reliable than single-latent ablation.

    Args:
        discoverer: CircuitDiscovererCLT or CircuitDiscovererPLT
        probe: CNN probe model
        seq: Wildtype sequence string
        circuit: Dict mapping layer indices to lists of node indices
        sequential: Whether to use sequential mode
        freeze_attention: Whether to freeze attention

    Returns:
        ranked_nodes: List of (layer, node_idx, score) sorted by importance (descending)
    """

    # 1. Compute attribution for the sequence
    attr = compute_attribution(discoverer, probe, [seq],
                               sequential=sequential,
                               freeze_attention=freeze_attention)

    # 2. Rank all nodes
    ranking = rank_nodes(attr)

    # 3. Filter to only nodes in the circuit
    circuit_nodes = set()
    for l, nodes in circuit.items():
        for n in nodes:
            circuit_nodes.add((int(l), n))

    filtered_ranking = [(l, n, s) for l, n, s in ranking if (l, n) in circuit_nodes]

    return filtered_ranking


def get_mutant_string(wt_seq, gen_seq):
    """
    Compares wildtype and generated sequences to return a mutation string.
    Format: WTPOSMUT (e.g. "A12C")
    Multiples joined by colon: "A12C:T45G"
    """
    if len(wt_seq) != len(gen_seq):
        return "LengthMismatch"
    
    mutations = []
    for i, (wt_char, gen_char) in enumerate(zip(wt_seq, gen_seq)):
        if wt_char != gen_char:
            # i + 1 for 1-based indexing standard in biology
            mutations.append(f"{wt_char}{i+1}{gen_char}")
            
    if not mutations:
        return "WT" # Or empty string "" if preferred
        
    return ":".join(mutations)


def get_probe_input(inference, seq):
    """
    Mirror of 'precompute_embeddings' logic for a single sequence.
    Returns the MLP outputs for the final layer, stripped of CLS/EOS.
    """
    # 1. Tokenize
    tokens = inference.tokenize([seq])
    B, T_full = tokens.shape
    
    # 2. Collect activations (Raw ESM forward pass via collector)
    # The collector returns: x_stack, x_ln_stack, x_mlp_stack_flat, ...
    _, _, x_mlp_stack_flat, _, _ = inference.collector.collect(tokens)
    
    # 3. Reshape flattened stack
    # precompute_embeddings: x_mlp_stack = x_mlp_stack_flat.view(B, T_full, -1, D)
    if hasattr(inference.model, "module"):
        D = inference.model.module.embed_dim
    else:
        D = inference.model.embed_dim
    x_mlp_stack = x_mlp_stack_flat.view(B, T_full, -1, D)
    
    # 4. Select Target Layer (-1 means final layer)
    # precompute_embeddings: x_layer = x_mlp_stack[:, :, target_layer, :]
    x_layer = x_mlp_stack[:, :, -1, :]
    
    # 5. Strip CLS/EOS
    # precompute_embeddings: x_clean = x_layer[:, 1:-1, :]
    x_clean = x_layer[:, 1:-1, :]
    
    return x_clean


def get_scoring_model_path(circuit_json):
    """
    Derive scoring model path from circuit JSON path.
    Circuit: .../function_circuit/functions/{model_type}/{mode}/{dms_name}/{fold}.json
    Probe:   .../function_circuit/probe/{mode}/{dms_name}/{fold}_cnn.pt
    """
    if "/functions_35M/" in circuit_json:
        probe_path = re.sub(r'/functions_35M/[^/]+/', '/probe_35M/', circuit_json)
    elif "/functions/" in circuit_json:
        probe_path = re.sub(r'/functions/[^/]+/', '/probe/', circuit_json)
    else:
        probe_path = circuit_json.replace("/functions/", "/probe/")
    
    probe_path = probe_path.replace('.json', '_cnn.pt')
    return probe_path


def run_circuit_attribution(circuit_json, config_map, config_name, wildtype, supp, esm_weights, output_json=None):
    """Run circuit attribution via subprocess."""
    ckpt, model_type, freeze_attention, _ = config_map[config_name]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "find_steering_circuit_attribution.py")
    scoring_model_path = get_scoring_model_path(circuit_json)

    cmd = [
        sys.executable, script_path,
        "--json_path", circuit_json,
        "--ckpt", ckpt,
        "--model_type", model_type,
        "--scoring_model", scoring_model_path,
        "--esm_weights", esm_weights,
        "--wt", wildtype,
        "--supp", str(supp),
    ]

    if freeze_attention:
        cmd.append("--freeze_attention")
        
    steering_circuit = circuit_json.replace('.json', f'_steering_{model_type}_attr.json')
    
    if output_json:
        cmd.extend(["--output_file", output_json])
        steering_circuit = output_json

    print(f"   -> Running attribution for {model_type}...")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) # Suppress verbose output

    if result.returncode != 0:
        print(f"WARNING: Attribution failed. Error: {result.stderr.decode('utf-8')}")
        return None

    return steering_circuit


def get_full_model(model_type, ckpt, device, esm_weights_path):
    """Load the appropriate full replacement model."""
    if model_type == "clt":
        pl_module = CLTLightningModule.load_from_checkpoint(ckpt, map_location=device, esm2_weight=esm_weights_path, weights_only=False)
        pl_module.eval().to(device)
        return FullCLTReplacementModel(pl_module, device)
    elif model_type == "plt":
        pl_module = PLTLightningModule.load_from_checkpoint(ckpt, map_location=device, esm2_weight=esm_weights_path, weights_only=False)
        pl_module.eval().to(device)
        return FullPLTReplacementModel(pl_module, device)
    elif model_type == "clt_direct":
        pl_module = CLTLightningModule.load_from_checkpoint(ckpt, map_location=device, esm2_weight=esm_weights_path, weights_only=False)
        pl_module.eval().to(device)
        return FullCLTDirectReplacementModel(pl_module, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    

def generate_random_mutant_sequence(wt, mutated_seq, seed=None, min_position=None, max_position=None):
    """
    Generate a random mutant sequence with the same number of mutations as mutated_seq.
    Shared logic for random baseline computation.
    
    Args:
        wt: Wildtype sequence string
        mutated_seq: Mutated sequence string
        seed: Random seed
    
    Returns:
        rand_seq: Random sequence string (or None if length mismatch)
        rand_mutant_str: Mutation string (e.g., "M1A:R5H") or "WT" if no mutations
    """
    alphabet = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    if len(wt) != len(mutated_seq): return None, None
    wt_list = list(wt)
    seq_list = list(mutated_seq)
    num_mutations = sum(1 for a, b in zip(wt_list, seq_list) if a != b)
    if num_mutations == 0: return mutated_seq, "WT"
    rng = np.random.RandomState(seed)
    rand_seq_list = list(wt)
    rand_mutant_parts = []
    if min_position is not None and max_position is not None:
        lo = min_position - 1
        hi = max_position - 1
        allowed_indices = np.arange(lo, hi + 1)
    else:
        allowed_indices = np.arange(len(wt))
    if num_mutations > len(allowed_indices):
        return None, None
    rand_indices = rng.choice(allowed_indices, num_mutations, replace=False)
    rand_indices.sort()
    for idx in rand_indices:
        choices = [aa for aa in alphabet if aa != wt_list[idx]]
        rand_aa = rng.choice(choices)
        rand_seq_list[idx] = rand_aa
        rand_mutant_parts.append(f"{wt_list[idx]}{idx+1}{rand_aa}")
    rand_seq = "".join(rand_seq_list)
    rand_mutant_str = ":".join(rand_mutant_parts)
    return rand_seq, rand_mutant_str

def infer_wildtype(row):
    """Get wildtype sequence by reverting mutations in DMS."""
    mutant_str = row['mutant']
    seq = list(row['mutated_sequence'])
    parts = mutant_str.split(':') if ':' in mutant_str else [mutant_str]
    for m in parts:
        if len(m) < 2:
            continue
        wt_aa = m[0]
        idx_str = "".join([c for c in m if c.isdigit()])
        if not idx_str:
            continue
        idx = int(idx_str) - 1
        if 0 <= idx < len(seq):
            seq[idx] = wt_aa
    return "".join(seq)
