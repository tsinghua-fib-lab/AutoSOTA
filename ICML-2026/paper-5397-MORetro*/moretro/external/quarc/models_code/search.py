import numpy as np
import torch
from chemprop.data import BatchMolGraph
from typing import List, Tuple, Optional, Union


def remove_duplicates_results(
    results: List[Tuple[torch.Tensor, float]],
) -> List[Tuple[torch.Tensor, float]]:
    """Remove duplicate agent combinations and sum their scores.

    Args:
        results: List of (agent_onehot, score) tuples where agent_onehot is a binary tensor
            indicating which agents are selected.

    Returns:
        List of unique (agent_onehot, score) tuples with summed scores for duplicates.
    """
    score_dict = {}
    for agent, score in results:
        agent_key = tuple(agent.view(-1).tolist())
        score_dict[agent_key] = score_dict.get(agent_key, 0) + score

    return [(torch.tensor(agent), score) for agent, score in score_dict.items()]


def top_n_results(
    results: List[Tuple[torch.Tensor, float]], n: int = 10, remove_duplicates: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """Get top n results sorted by score.

    Args:
        results: List of (agent_onehot, score) tuples.
        n: Number of top results to return.
        remove_duplicates: Whether to remove duplicate agent combinations.

    Returns:
        Top n results sorted by score in descending order.
    """
    if remove_duplicates:
        results = remove_duplicates_results(results)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results[:n]


def get_min_score(results: List[Tuple[torch.Tensor, float]]) -> float:
    """Get minimum score from results.

    Args:
        results: List of (agent_onehot, score) tuples.

    Returns:
        Minimum score value.
    """
    scores = np.array([score for _, score in results], dtype="float32")
    return np.min(scores)


def get_max_score(results: List[Tuple[torch.Tensor, float]]) -> float:
    """Get maximum score from results.

    Args:
        results: List of (agent_onehot, score) tuples.

    Returns:
        Maximum score value.
    """
    scores = np.array([score for _, score in results], dtype="float32")
    return np.max(scores)


def apply_geometric_average(
    results: List[Tuple[torch.Tensor, float]],
) -> List[Tuple[torch.Tensor, float]]:
    """Apply geometric average normalization to scores based on number of selected agents.

    Normalizes scores by taking the nth root where n is the number of selected agents + 1.
    This helps account for the fact that longer sequences naturally have lower probabilities.

    Args:
        results: List of (agent_onehot, score) tuples.

    Returns:
        Results with geometrically averaged scores.
    """
    normalized_results = []
    for agent_onehot, score in results:
        num_agents = torch.sum(agent_onehot).item() + 1
        normalized_score = score ** (1.0 / num_agents)
        normalized_results.append((agent_onehot, normalized_score))

    return normalized_results


def beam_search(
    model: torch.nn.Module,
    rxn_input: torch.Tensor,
    # rxn_class: torch.Tensor,
    num_classes: int,
    agents_input: Optional[torch.Tensor] = None,
    max_steps: int = 6,
    beam_size: int = 10,
    eos_id: int = 0,
    return_top_n: int = 10,
    verbosity: int = 0,
) -> List[Tuple[torch.Tensor, float]]:
    """Beam search decoding for FFN-based agent prediction models.

    Args:
        model: Trained FFN model for agent prediction.
        rxn_input: Reaction fingerprint input tensor, shape (4096,) or (1, 4096).
        rxn_class: Reaction class one-hot encoding, shape (num_rxn_classes,) or (1, num_rxn_classes).
        num_classes: Total number of agent classes.
        agents_input: Optional initial agent selection tensor, shape (num_classes,) or (1, num_classes).
            If None, starts with empty agent set.
        max_steps: Maximum number of beam search steps (search depth).
        beam_size: Number of candidates to keep at each step (beam width).
        eos_id: End-of-sequence token ID (typically 0).
        return_top_n: Number of top results to return. Use -1 to return all results.
        verbosity: Logging verbosity level (0=silent, 1=basic, 2=detailed).

    Returns:
        List of (agent_onehot, score) tuples sorted by score in descending order.
        agent_onehot is a binary tensor of shape (num_classes,) indicating selected agents.

    Example:
        >>> results = beam_search(model, rxn_fp, rxn_class, num_agents=1500)
        >>> best_agents, best_score = results[0]
        >>> selected_indices = torch.nonzero(best_agents).squeeze()
    """
    device = next(model.parameters()).device

    if len(rxn_input.shape) == 1:
        rxn_input = rxn_input.unsqueeze(0).to(device)
        # rxn_class = rxn_class.unsqueeze(0).to(device)
        if agents_input is not None:
            agents_input = agents_input.unsqueeze(0)

    completed_sequences = []  # [(agent_onehot, score)]

    if agents_input is None:
        active_beams = [(torch.zeros(num_classes, device=device), 1.0)]
    else:
        active_beams = [(agents_input.to(device), 1.0)]

    for step in range(max_steps):
        if not active_beams:
            break

        if verbosity > 0:
            print(f"Step {step}: {len(active_beams)} active beams")

        new_beams = []

        for current_agents, current_score in active_beams:
            if verbosity > 1:
                selected = torch.nonzero(current_agents.squeeze()).tolist()
                print(f"  Expanding beam with agents {selected}, score: {current_score:.3f}")

            if len(current_agents.shape) == 1:
                current_agents = current_agents.unsqueeze(0)

            with torch.no_grad():
                current_agents = current_agents.to(device)
                # mask already selected agents with very negative logits
                logits = model(rxn_input, current_agents)
                masked_logits = torch.where(current_agents == 1, -1e6, logits)
                probabilities = torch.softmax(masked_logits, dim=-1)

            # get top-k predictions
            top_scores, top_indices = torch.topk(probabilities[0], k=beam_size)

            if verbosity > 1:
                print(f"    Top predictions: {top_indices.tolist()}")
                print(f"    Top scores: {[f'{s:.3f}' for s in top_scores.tolist()]}")

            for agent_idx, prob_score in zip(top_indices, top_scores.tolist()):
                new_agents = current_agents.clone()
                new_score = current_score * prob_score

                if agent_idx == eos_id:
                    if verbosity > 1:
                        print(f"    EOS reached, adding to completed sequences")
                    completed_sequences.append((new_agents.cpu(), new_score))
                else:
                    # add new agent and continue
                    new_agents[0][agent_idx] = 1
                    new_beams.append((new_agents, new_score))

        # keep only top beam_size candidates for next iteration
        active_beams = top_n_results(new_beams, n=beam_size, remove_duplicates=True)

        # early termination: if we have enough good completed sequences
        if len(completed_sequences) > beam_size:
            top_completed = top_n_results(completed_sequences, n=beam_size, remove_duplicates=True)
            min_completed_score = get_min_score(top_completed)
            max_active_score = get_max_score(active_beams) if active_beams else 0

            if max_active_score < min_completed_score:
                if verbosity > 0:
                    print(f"Early termination at step {step}")
                break

    if return_top_n == -1:
        return top_n_results(
            completed_sequences, n=len(completed_sequences), remove_duplicates=True
        )
    else:
        return top_n_results(completed_sequences, n=return_top_n, remove_duplicates=True)


def beam_search_gnn(
    model: torch.nn.Module,
    bmg: BatchMolGraph,
    V_d: torch.Tensor,
    x_d: torch.Tensor,
    num_classes: int,
    agents_input: Optional[torch.Tensor] = None,
    max_steps: int = 6,
    beam_size: int = 10,
    eos_id: int = 0,
    return_top_n: int = 10,
    verbosity: int = 0,
) -> List[Tuple[torch.Tensor, float]]:
    """Beam search decoding for GNN-based agent prediction models.

    Args:
        model: Trained GNN model for agent prediction.
        batch_mol_graph: Molecular graph representation from ChemProp.
        V_d: Atom feature matrix.
        x_d: Additional molecular features.
        num_classes: Total number of agent classes.
        agents_input: Optional initial agent selection tensor, shape (num_classes,) or (1, num_classes).
            If None, starts with empty agent set.
        max_steps: Maximum number of beam search steps (search depth).
        beam_size: Number of candidates to keep at each step (beam width).
        eos_id: End-of-sequence token ID (typically 0).
        return_top_n: Number of top results to return. Use -1 to return all results.
        verbosity: Logging verbosity level (0=silent, 1=basic, 2=detailed).

    Returns:
        List of (agent_onehot, score) tuples sorted by score in descending order.
        agent_onehot is a binary tensor of shape (num_classes,) indicating selected agents.
    """
    device = next(model.parameters()).device

    if agents_input is not None and len(agents_input.shape) == 1:
        agents_input = agents_input.unsqueeze(0)

    completed_sequences = []  # [(agent_onehot, score)]

    if agents_input is None:
        active_beams = [(torch.zeros(num_classes, device=device), 1.0)]
    else:
        active_beams = [(agents_input.to(device), 1.0)]

    for step in range(max_steps):
        if not active_beams:
            break

        if verbosity > 0:
            print(f"Step {step}: {len(active_beams)} active beams")

        new_beams = []

        for current_agents, current_score in active_beams:
            if verbosity > 1:
                selected = torch.nonzero(current_agents.squeeze()).tolist()
                print(f"  Expanding beam with agents {selected}, score: {current_score:.3f}")

            if len(current_agents.shape) == 1:
                current_agents = current_agents.unsqueeze(0)

            with torch.no_grad():
                current_agents = current_agents.to(device)
                # mask already selected agents with very negative logits
                logits = model(current_agents, bmg, V_d, x_d)
                masked_logits = torch.where(current_agents == 1, -1e6, logits)
                probabilities = torch.softmax(masked_logits, dim=-1)

            # get top-k predictions
            top_scores, top_indices = torch.topk(probabilities[0], k=beam_size)

            if verbosity > 0:
                print(f"    Top predictions: {top_indices.tolist()}")
                print(f"    Top scores: {[f'{s:.3f}' for s in top_scores.tolist()]}")

            for agent_idx, prob_score in zip(top_indices, top_scores.tolist()):
                new_agents = current_agents.clone()
                new_score = current_score * prob_score

                if agent_idx == eos_id:
                    if verbosity > 1:
                        print(f"    EOS reached, adding to completed sequences")
                    completed_sequences.append((new_agents.cpu(), new_score))
                else:
                    new_agents[0][agent_idx] = 1
                    new_beams.append((new_agents, new_score))

        # keep only top beam_size candidates for next iteration
        active_beams = top_n_results(new_beams, n=beam_size, remove_duplicates=True)

        # early termination: if we have enough good completed sequences
        if len(completed_sequences) > beam_size:
            top_completed = top_n_results(completed_sequences, n=beam_size, remove_duplicates=True)
            min_completed_score = get_min_score(top_completed)
            max_active_score = get_max_score(active_beams) if active_beams else 0

            if max_active_score < min_completed_score:
                if verbosity > 0:
                    print(f"Early termination at step {step}")
                break

    if return_top_n == -1:
        return top_n_results(
            completed_sequences, n=len(completed_sequences), remove_duplicates=True
        )
    else:
        return top_n_results(completed_sequences, n=return_top_n, remove_duplicates=True)
