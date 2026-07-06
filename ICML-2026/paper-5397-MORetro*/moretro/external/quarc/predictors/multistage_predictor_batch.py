from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
from chemprop.data.collate import BatchMolGraph

from quarc.data_handlers.eval_datasets import ReactionInput
from quarc.predictors.base import BasePredictor, PredictionList, StagePrediction
from quarc.predictors.multistage_predictor import HierarchicalPrediction


@dataclass
class HierarchicalBatchPrediction:
    """
    Batched hierarchical predictions. This is a thin wrapper that runs the
    full multi-stage pipeline for each reaction in a batch and aggregates
    the per-reaction HierarchicalPrediction results.

    This leverages the proven single-reaction implementation to ensure
    correctness while providing a convenient batched interface.
    """

    items: List[HierarchicalPrediction]

    @classmethod
    def from_models(
        cls,
        reactions: List[ReactionInput],
        agent_model: Any,
        temperature_model: Any,
        reactant_amount_model: Any,
        agent_amount_model: Any,
        model_types: dict[str, str],
        agent_encoder: Any,
        top_k_agents: int = 5,
        top_k_temp: int = 2,
        top_k_reactant_amount: int = 2,
        top_k_agent_amount: int = 2,
        beam_size: int = 10,
        device: str = "cuda",
    ) -> "HierarchicalBatchPrediction":
        """
        Run the full multi-stage prediction pipeline for a batch of reactions.

        Args:
            reactions: List of ReactionInput items to predict.
            agent_model: Stage-1 agent prediction model.
            temperature_model: Stage-2 temperature prediction model.
            reactant_amount_model: Stage-3 reactant amount prediction model.
            agent_amount_model: Stage-4 agent amount prediction model.
            model_types: Dict mapping stage name to model type ("gnn" or "ffn").
            agent_encoder: Agent encoder instance.
            top_k_agents: Number of agent sets to consider per reaction.
            top_k_temp: Number of temperature bins per agent set.
            top_k_reactant_amount: Number of reactant amount combinations per agent set.
            top_k_agent_amount: Number of agent amount combinations per agent set.
            device: Compute device.

        Returns:
            HierarchicalBatchPrediction containing one HierarchicalPrediction per input reaction.
        """

        # Put models in eval mode
        agent_model.eval()
        temperature_model.eval()
        reactant_amount_model.eval()
        agent_amount_model.eval()

        B = len(reactions)
        num_classes = len(agent_encoder)

        # Collect per-sample inputs
        metas = [r.metadata for r in reactions]
        mgs = [r.model_inputs["mg"] for r in reactions]
        FP_inputs_list = [r.model_inputs["FP_inputs"] for r in reactions]
        FP_reactants_list = [r.model_inputs["FP_reactants"] for r in reactions]

        # Stack FFN inputs
        FP_reactants_batch = torch.tensor(np.stack(FP_reactants_list, axis=0), dtype=torch.float).to(device)

        # end prep

        # Stage 1 (per-sample): agent beam search
        if model_types["agent"] == "ffn":
            beam_results_per_sample = _batched_beam_search_ffn(
                model=agent_model,
                FP_inputs_list=FP_inputs_list,
                num_classes=num_classes,
                max_steps=6,
                beam_size=10,
                return_top_n=top_k_agents,
                eos_id=0,
                device=device,
            )
        elif model_types["agent"] == "gnn":
            beam_results_per_sample = _batched_beam_search_gnn(
                model=agent_model,
                mg_list=mgs,
                num_classes=num_classes,
                max_steps=6,
                beam_size=5,
                return_top_n=top_k_agents,
                eos_id=0,
                device=device,
            )
        else:
            raise ValueError(f"Unknown model type for agent prediction: {model_types['agent']}")

        # end stage 1

        # Flatten all (sample, beam) pairs for batched stages 2-4
        pair_map: List[Tuple[int, int]] = []  # (sample_idx, beam_idx)
        a_inputs_pairs: List[torch.Tensor] = []
        FP_inputs_pairs: List[torch.Tensor] = []
        FP_reactants_pairs: List[torch.Tensor] = []
        mg_pairs: List[Any] = []
        agent_indices_per_pair: List[List[int]] = []
        agent_scores_per_pair: List[float] = []

        for i in range(B):
            beams = beam_results_per_sample[i]
            for j, (agent_onehot, agent_score) in enumerate(beams):
                agent_vec = agent_onehot.squeeze(0) if agent_onehot.dim() == 2 else agent_onehot
                agent_vec = agent_vec.to(device, dtype=torch.float)

                # record mapping and inputs
                pair_map.append((i, j))
                a_inputs_pairs.append(agent_vec)
                FP_inputs_pairs.append(FP_inputs_list[i])
                FP_reactants_pairs.append(FP_reactants_batch[i])
                mg_pairs.append(mgs[i])
                agent_scores_per_pair.append(float(agent_score))

                nz = agent_vec.nonzero().view(-1).tolist()
                indices_list: List[int] = [int(x) for x in (nz if isinstance(nz, list) else [nz])]
                agent_indices_per_pair.append(indices_list)

        # end flatten

        if len(pair_map) == 0:
            # No beams found; return empty predictions
            return cls(items=[
                HierarchicalPrediction(
                    doc_id=meta.get("doc_id", ""),
                    rxn_class=meta.get("rxn_class", ""),
                    rxn_smiles=meta.get("rxn_smiles", ""),
                    agent_groups=[],
                )
                for meta in metas
            ])

        N = len(pair_map)

        # Build batched tensors for all pairs
        a_inputs_tensor = torch.stack(a_inputs_pairs, dim=0).to(device)
        FP_inputs_tensor = torch.stack(FP_inputs_pairs, dim=0).to(device)
        FP_reactants_tensor = torch.stack(FP_reactants_pairs, dim=0).to(device)

        # Build batched graph input for GNN types by repeating per pair
        if model_types["temperature"] == "gnn" or model_types["reactant_amount"] == "gnn" or model_types["agent_amount"] == "gnn":
            bmg_pairs = BatchMolGraph(mg_pairs)
            bmg_pairs.to(device)
        else:
            bmg_pairs = None

        # end build bmg

        # Stage 2 (batched): temperature predictions for all pairs
        if model_types["temperature"] == "gnn":
            with torch.no_grad():
                temp_logits = temperature_model(a_inputs_tensor, bmg_pairs)
        elif model_types["temperature"] == "ffn":
            with torch.no_grad():
                temp_logits = temperature_model(FP_inputs_tensor, a_inputs_tensor)
        else:
            raise ValueError(f"Unknown model type for temperature: {model_types['temperature']}")
        # end stage 2

        temp_probs = torch.softmax(temp_logits, dim=-1)
        top_temp_scores, top_temp_bins = temp_probs.topk(top_k_temp, dim=-1)

        # Stage 3 (batched): reactant amounts for all pairs
        if model_types["reactant_amount"] == "gnn":
            with torch.no_grad():
                reactant_logits = reactant_amount_model(a_inputs_tensor, FP_reactants_tensor, bmg_pairs)
        elif model_types["reactant_amount"] == "ffn":
            with torch.no_grad():
                reactant_logits = reactant_amount_model(FP_inputs_tensor, a_inputs_tensor, FP_reactants_tensor)
        else:
            raise ValueError(f"Unknown model type for reactant_amount: {model_types['reactant_amount']}")
        # end stage 3

        # Stage 4 (batched): agent amounts for all pairs
        if model_types["agent_amount"] == "gnn":
            with torch.no_grad():
                agent_amount_logits = agent_amount_model(a_inputs_tensor, bmg_pairs)
        elif model_types["agent_amount"] == "ffn":
            with torch.no_grad():
                agent_amount_logits = agent_amount_model(FP_inputs_tensor, a_inputs_tensor)
        else:
            raise ValueError(f"Unknown model type for agent_amount: {model_types['agent_amount']}")
        # end stage 4

        # Assemble per-sample agent groups by consuming per-pair outputs
        agent_groups_per_sample: List[List[dict]] = [[] for _ in range(B)]

        for g in range(N):
            sample_idx, beam_idx = pair_map[g]
            agent_indices = agent_indices_per_pair[g]
            agent_score = agent_scores_per_pair[g]

            # Temperature top-k for this pair
            t_bins = top_temp_bins[g]
            t_scores = top_temp_scores[g]
            # keep as tensors; convert to ints/floats only when building dicts
            t_preds = [{"bin": int(b), "score": float(s)} for b, s in zip(t_bins, t_scores)]

            # Reactant amounts: determine valid reactants by nonzero fingerprint rows
            fp_r = FP_reactants_tensor[g]  # [MAX_R, fp_dim]
            valid_mask = fp_r.abs().sum(dim=1) > 0
            valid_indices_t = torch.nonzero(valid_mask, as_tuple=True)[0]
            valid_indices = valid_indices_t.tolist()

            reactant_logits_g = reactant_logits[g]  # [MAX_R, n_bins]
            reactant_logits_valid = reactant_logits_g[valid_mask]
            reactant_probs = torch.softmax(reactant_logits_valid, dim=-1)

            # take top-2 per valid reactant and select top-K combinations without full enumeration
            r_scores, r_bins = reactant_probs.topk(2, dim=-1)  # [R, 2]
            if r_scores.numel() == 0:
                reactant_amount_preds = []
            else:
                # Build top-K product across reactants using log-space merging
                score_lists = [r_scores[i] for i in range(r_scores.size(0))]
                combo_choices, combo_scores = _topk_cartesian_product(score_lists, top_k_reactant_amount, device)
                reactant_amount_preds = []
                for choices, sc in zip(combo_choices, combo_scores):
                    # Map local choice (0/1) to actual bin index per reactant
                    bin_indices = [int(r_bins[i, choices[i]].item()) for i in range(r_bins.size(0))]
                    reactant_amount_preds.append({
                        "bin_indices": bin_indices,
                        "score": float(sc),
                    })

            # Agent amounts: filter per selected agents, top-2 each, enumerate
            aa_logits = agent_amount_logits[g]  # [num_classes, n_bins]
            if not agent_indices:
                agent_amount_preds = []
            else:
                aa_sel = aa_logits[agent_indices]
                aa_probs = torch.softmax(aa_sel, dim=-1)
                aa_scores, aa_bins = aa_probs.topk(2, dim=-1)  # [A, 2]
                score_lists = [aa_scores[i] for i in range(aa_scores.size(0))]
                combo_choices, combo_scores = _topk_cartesian_product(score_lists, top_k_agent_amount, device)
                agent_amount_preds = []
                for choices, sc in zip(combo_choices, combo_scores):
                    amounts = [(int(agent_indices[i]), int(aa_bins[i, choices[i]].item())) for i in range(aa_bins.size(0))]
                    agent_amount_preds.append({
                        "amounts": amounts,
                        "score": float(sc),
                    })

            agent_groups_per_sample[sample_idx].append(
                {
                    "agent_indices": agent_indices,
                    "agent_score": float(agent_score),
                    "temperature": t_preds,
                    "reactant_amounts": reactant_amount_preds,
                    "agent_amounts": agent_amount_preds,
                }
            )

        # Build HierarchicalPrediction per sample
        results: List[HierarchicalPrediction] = []
        for i in range(B):
            meta = metas[i]
            results.append(
                HierarchicalPrediction(
                    doc_id=meta.get("doc_id", ""),
                    rxn_class=meta.get("rxn_class", ""),
                    rxn_smiles=meta.get("rxn_smiles", ""),
                    agent_groups=agent_groups_per_sample[i],
                )
            )
        return cls(items=results)


def _topk_cartesian_product(
    score_lists: List[torch.Tensor],
    top_k: int,
    device: str | torch.device,
) -> Tuple[List[List[int]], List[float]]:
    """
    Compute top-K combinations over a list of 1D score tensors by maximizing the product
    of selected entries (equivalently maximizing the sum of log-scores) without enumerating
    the full Cartesian product.

    Args:
        score_lists: List of tensors [m_i], values assumed in [0, 1].
        top_k: Number of combinations to return.
        device: Torch device for computations.

    Returns:
        (choices, scores):
            - choices: list of index lists, one index per input tensor.
            - scores: list of product scores (floats).
    """
    if not score_lists:
        return [], []

    # Convert to log-space for numerical stability
    logs = [torch.log(torch.clamp(s.to(device), min=1e-12)) for s in score_lists]

    # Initialize with empty path
    sums = torch.zeros(1, device=device)
    paths = torch.empty((1, 0), dtype=torch.long, device=device)

    for v in logs:
        # Broadcast-add current choices
        new_sums = sums.unsqueeze(1) + v.unsqueeze(0)  # [K_prev, m_i]
        K_prev, m_i = new_sums.shape
        flat = new_sums.reshape(-1)
        K_next = min(top_k, flat.numel())
        top_vals, top_idx = torch.topk(flat, k=K_next, dim=0)
        prev_idx = torch.div(top_idx, m_i, rounding_mode='floor')
        choice_idx = top_idx % m_i
        # Build new paths by appending choice indices
        if paths.numel() == 0:
            new_paths = choice_idx.unsqueeze(1)
        else:
            new_paths = torch.cat([paths[prev_idx], choice_idx.unsqueeze(1)], dim=1)
        sums = top_vals
        paths = new_paths

    scores = torch.exp(sums).tolist()
    choices = paths.tolist()
    return choices, scores


def _dedup_and_top_k(
    results: List[Tuple[torch.Tensor, float, frozenset[int]]],
    k: int,
) -> List[Tuple[torch.Tensor, float, frozenset[int]]]:
    score_map: dict[frozenset[int], float] = {}
    tensor_map: dict[frozenset[int], torch.Tensor] = {}
    
    for agent_vec, score, agent_set in results:
        if agent_set in score_map:
            score_map[agent_set] += float(score)
        else:
            score_map[agent_set] = float(score)
            tensor_map[agent_set] = agent_vec
            
    items = [
        (tensor_map[agent_set], val, agent_set)
        for agent_set, val in score_map.items()
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def _batched_beam_search_ffn(
    model: Any,
    FP_inputs_list: List[torch.Tensor],
    num_classes: int,
    max_steps: int = 6,
    beam_size: int = 10,
    return_top_n: int = 10,
    eos_id: int = 0,
    device: str = "cuda",
) -> List[List[Tuple[torch.Tensor, float]]]:
    B = len(FP_inputs_list)
    FP_inputs_list = [fp.to(device) for fp in FP_inputs_list]

    # Initialize per-sample beams
    active_beams: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [
        [(torch.zeros(num_classes, dtype=torch.float, device=device), 1.0, frozenset())] for _ in range(B)
    ]
    completed: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [[] for _ in range(B)]

    for _ in range(max_steps):
        # Flatten current beams across all samples
        batch_FP: List[torch.Tensor] = []
        batch_agents: List[torch.Tensor] = []
        owner: List[int] = []  # which sample this beam belongs to

        for i in range(B):
            for agent_vec, _score, _agent_set in active_beams[i]:
                batch_FP.append(FP_inputs_list[i])
                batch_agents.append(agent_vec)
                owner.append(i)

        if not batch_agents:
            break

        FP_batch = torch.stack(batch_FP, dim=0)
        A_batch = torch.stack(batch_agents, dim=0)

        with torch.no_grad():
            logits = model(FP_batch, A_batch)
            masked = torch.where(A_batch == 1, torch.tensor(-1e6, device=logits.device, dtype=logits.dtype), logits)
            probs = torch.softmax(masked, dim=-1)

        top_scores, top_idx = torch.topk(probs, k=beam_size, dim=-1)

        # Build next beams per sample
        new_active: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [[] for _ in range(B)]
        # Rebuild an iterator in the same order as stacking
        flat_iter = []
        for i in range(B):
            for item in active_beams[i]:
                flat_iter.append((i, item))

        for g, (i, (curr_agents, curr_score, curr_set)) in enumerate(flat_iter):
            k_scores = top_scores[g].tolist()
            k_idx = top_idx[g].tolist()
            curr_agents_2d = curr_agents.unsqueeze(0) if curr_agents.dim() == 1 else curr_agents

            for agent_j, p in zip(k_idx, k_scores):
                new_agents = curr_agents_2d.clone()
                new_score = float(curr_score) * float(p)
                if agent_j == eos_id:
                    completed[i].append((new_agents.squeeze(0), new_score, curr_set))
                else:
                    new_agents[0][agent_j] = 1
                    new_set = curr_set.union([agent_j])
                    new_active[i].append((new_agents.squeeze(0), new_score, new_set))

        # Trim beams per sample and remove duplicates
        for i in range(B):
            active_beams[i] = _dedup_and_top_k(new_active[i], beam_size)

        # Early exit if no active beams
        if all(len(ab) == 0 for ab in active_beams):
            break

    # Return top results per sample
    outputs: List[List[Tuple[torch.Tensor, float]]] = []
    for i in range(B):
        outs = _dedup_and_top_k(completed[i], return_top_n)
        outputs.append([(vec, score) for vec, score, _set in outs])
    return outputs


def _batched_beam_search_gnn(
    model: Any,
    mg_list: List[Any],
    num_classes: int,
    max_steps: int = 6,
    beam_size: int = 10,
    return_top_n: int = 10,
    eos_id: int = 0,
    device: str = "cuda",
) -> List[List[Tuple[torch.Tensor, float]]]:
    B = len(mg_list)
    active_beams: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [
        [(torch.zeros(num_classes, dtype=torch.float, device=device), 1.0, frozenset())] for _ in range(B)
    ]
    completed: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [[] for _ in range(B)]

    for _ in range(max_steps):
        batch_agents: List[torch.Tensor] = []
        batch_mgs: List[Any] = []

        for i in range(B):
            for agent_vec, _score, _set in active_beams[i]:
                batch_agents.append(agent_vec)
                batch_mgs.append(mg_list[i])

        if not batch_agents:
            break

        A_batch = torch.stack(batch_agents, dim=0).to(device)
        bmg_batch = BatchMolGraph(batch_mgs)
        bmg_batch.to(device)

        with torch.no_grad():
            logits = model(A_batch, bmg_batch, None, None)
            masked = torch.where(A_batch == 1, torch.tensor(-1e6, device=logits.device, dtype=logits.dtype), logits)
            probs = torch.softmax(masked, dim=-1)

        top_scores, top_idx = torch.topk(probs, k=beam_size, dim=-1)

        new_active: List[List[Tuple[torch.Tensor, float, frozenset[int]]]] = [[] for _ in range(B)]
        flat_iter = []
        for i in range(B):
            for item in active_beams[i]:
                flat_iter.append((i, item))

        for g, (i, (curr_agents, curr_score, curr_set)) in enumerate(flat_iter):
            k_scores = top_scores[g].tolist()
            k_idx = top_idx[g].tolist()
            curr_agents_2d = curr_agents.unsqueeze(0) if curr_agents.dim() == 1 else curr_agents

            for agent_j, p in zip(k_idx, k_scores):
                new_agents = curr_agents_2d.clone()
                new_score = float(curr_score) * float(p)
                if agent_j == eos_id:
                    completed[i].append((new_agents.squeeze(0), new_score, curr_set))
                else:
                    new_agents[0][agent_j] = 1
                    new_set = curr_set.union([agent_j])
                    new_active[i].append((new_agents.squeeze(0), new_score, new_set))

        for i in range(B):
            active_beams[i] = _dedup_and_top_k(new_active[i], beam_size)

        if all(len(ab) == 0 for ab in active_beams):
            break

    outputs: List[List[Tuple[torch.Tensor, float]]] = []
    for i in range(B):
        outs = _dedup_and_top_k(completed[i], return_top_n)
        outputs.append([(vec, score) for vec, score, _set in outs])
    return outputs


class EnumeratedBatchPredictor(BasePredictor):
    """
    Batched enumerated predictor. Runs the full multi-stage pipeline in batch and
    returns ranked `PredictionList` per reaction, matching the scoring logic used
    in the single-sample EnumeratedPredictor.
    """

    def __init__(
        self,
        agent_model: Any,
        temperature_model: Any,
        reactant_amount_model: Any,
        agent_amount_model: Any,
        model_types: dict[str, str],
        agent_encoder: Any,
        top_k_agents: int = 5,
        top_k_temp: int = 2,
        top_k_reactant_amount: int = 2,
        top_k_agent_amount: int = 2,
        device: str = "cuda",
        weights: dict[str, float] | None = None,
        use_geometric: bool = True,
    ) -> None:
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

    def predict_many(self, reactions: List[ReactionInput], top_k: int, beam_size: int) -> List[PredictionList]:
        batch_view = HierarchicalBatchPrediction.from_models(
            reactions=reactions,
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

        outputs: List[PredictionList] = []
        for hp in batch_view.items:
            preds = self._rank_enumerate_combinations(hp, top_k)
            outputs.append(
                PredictionList(
                    doc_id=hp.doc_id,
                    rxn_class=hp.rxn_class,
                    rxn_smiles=hp.rxn_smiles,
                    predictions=preds,
                )
            )
        return outputs

    def predict(self, reaction: ReactionInput, top_k: int = 2) -> PredictionList:
        return self.predict_many([reaction], top_k=top_k)[0]

    def _rank_enumerate_combinations(
        self, hierarchical_preds: HierarchicalPrediction, top_k: int
    ) -> List[StagePrediction]:
        enumerated_predictions: List[StagePrediction] = []

        for agent_group in hierarchical_preds.agent_groups:
            agents = agent_group["agent_indices"]
            agent_score = agent_group["agent_score"]

            temp_preds = [(pred["bin"], pred["score"]) for pred in agent_group["temperature"]]
            reactant_preds = [
                (pred["bin_indices"], pred["score"]) for pred in agent_group["reactant_amounts"]
            ]
            agent_amount_preds = [
                (pred["amounts"], pred["score"]) for pred in agent_group["agent_amounts"]
            ]

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

                        enumerated_predictions.append(
                            StagePrediction(
                                agents=agents,
                                temp_bin=int(temp_bin),
                                reactant_bins=[int(x) for x in reactant_bins],
                                agent_amount_bins=[(int(a), int(b)) for a, b in agent_amount_items],
                                score=float(combined_score),
                                meta={
                                    "s1_score": float(agent_score),
                                    "s2_score": float(temp_score),
                                    "s3_score": self._normalize_reactant_score(
                                        float(reactant_score), len(reactant_bins)
                                    ),
                                    "s4_score": self._normalize_agent_amount_score(
                                        float(agent_amount_score), len(agent_amount_items)
                                    ),
                                },
                            )
                        )

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
        normalized_agent_score = agent_score
        normalized_temp_score = temp_score
        normalized_reactant_score = reactant_score ** (1 / n_reactants) if n_reactants > 0 else 1.0
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
        return float(combined_score)

    def _normalize_reactant_score(self, reactant_score: float, n_reactants: int) -> float:
        return reactant_score ** (1 / n_reactants) if n_reactants > 0 else 1.0

    def _normalize_agent_amount_score(self, agent_amount_score: float, n_agents: int) -> float:
        return agent_amount_score ** (1 / n_agents) if n_agents > 0 else 1.0
