#!/usr/bin/env python3
"""
Description: Implementation of evaluation metrics for multimodal embeddings.
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def embedding_distance(
    embedding1: torch.Tensor, embedding2: torch.Tensor, distance: str = "l2"
) -> torch.Tensor:
    """
    Compute the distance between two embeddings.

    Parameters
    ----------
    embedding1 : torch.Tensor
        The first embedding.
    embedding2 : torch.Tensor
        The second embedding.
    distance : str, optional
        The distance metric to use. Can be 'l2', 'cosine', or 'geodesic'.

    Returns
    -------
    torch.Tensor
        The distance between the two embeddings."
    """

    # Compute the distance based on the specified metric
    if distance == "l2":
        return torch.norm(embedding1 - embedding2, p=2)
    elif distance == "cosine":
        return 1 - embedding1 @ embedding2
    elif distance == "geodesic":
        # Compute the cosine similarity
        cosine_sim = embedding1 @ embedding2
        # Ensure numerical stability: clip values to the valid range for accos.
        cosine_sim = torch.clamp(cosine_sim, -1, 1)
        # The geodesic distance is the arc cosine of the cosine similarity.
        return torch.acos(cosine_sim)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")


def compute_mean_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean embedding of a set of embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        The embeddings to compute the mean of.

    Returns
    -------
    torch.Tensor
    """
    return embeddings.mean(axis=0)


def compute_residuals_and_total(
    similarity_matrix: torch.Tensor,
    predicted: torch.Tensor,
    distance: str = "l2",
) -> torch.Tensor:
    """
    Computes the residuals and total for the R-squared-like metric
    in vector space. The residuals are the sum of the distances between
    the predicted embeddings and the true embeddings, while the total
    is the sum of the distances between the mean embedding and the true
    embeddings.

    Parameters
    ----------
    similarity_matrix : torch.Tensor
        The similarity matrix between the true embeddings.
    predicted : torch.Tensor
        The predicted embeddings.
    distance : str, optional
        The distance metric to use. Can be 'l2', 'cosine', or 'geodesic'.
        Default is 'l2'.
    Returns
    -------
    torch.Tensor
        The residuals and total for the R-squared-like metric.
    """
    num_samples = similarity_matrix.shape[0]

    # Get the predicted indices by taking the argmax of the similarity matrix
    predicted_indices = similarity_matrix.argmax(dim=1)
    true_indices = np.arange(num_samples)

    # Get the true and predicted embeddings
    predicted_embeddings = predicted[predicted_indices]
    true_embeddings = predicted[true_indices]

    # Compute the mean embedding
    mean_embedding = compute_mean_embedding(predicted)

    # Compute distances in vector space
    residuals = torch.stack(
        [
            embedding_distance(true, pred, distance=distance)
            for true, pred in zip(true_embeddings, predicted_embeddings)
        ]
    )
    total = torch.stack(
        [
            embedding_distance(true, mean_embedding, distance=distance)
            for true in true_embeddings
        ]
    )

    return residuals, total


def compute_r_squared(
    similarity_matrix: torch.Tensor,
    output0: torch.Tensor,
    output1: torch.Tensor,
    distance: str = "l2",
) -> torch.Tensor:
    """
    Compute the R-squared-like metric for a pair of embeddings.

    Parameters
    ----------
    similarity_matrix : torch.Tensor
        The similarity matrix between the true embeddings.
    output0 : torch.Tensor
        The first set of embeddings.
    output1 : torch.Tensor
        The second set of embeddings.
    distance : str, optional
        The distance metric to use. Can be 'l2', 'cosine', or 'geodesic'.
        Default is 'l2'.
    """

    # Get the residuals and total for the R-squared-like metric
    residuals0, total0 = compute_residuals_and_total(
        similarity_matrix, output1, distance=distance
    )
    residuals1, total1 = compute_residuals_and_total(
        similarity_matrix.T, output0, distance=distance
    )

    # Sum both rows and columns
    residuals = residuals0 + residuals1
    total = total0 + total1

    # Compute R-squared-like metric
    r_squared = 1 - (residuals.sum() / total.sum())

    return r_squared


def compute_r_squared_router(
    similarity_matrix: torch.Tensor,
    output0: torch.Tensor,
    output1: torch.Tensor,
    distance: str = "l2",
    cls_only: bool = True,
) -> torch.Tensor:
    """
    Routes the computation of the R-squared-like metric based on the input
    embeddings. If `cls_only` is True, the metric is computed only for the
    CLS token. Otherwise, the metric is computed for each modality.

    Parameters
    ----------
    similarity_matrix : torch.Tensor
        The similarity matrix between the true embeddings.
    output0 : torch.Tensor
        The first set of embeddings.
    output1 : torch.Tensor
        The second set of embeddings.
    distance : str, optional
        The distance metric to use. Can be 'l2', 'cosine', or 'geodesic'.
        Default is 'l2'.
    cls_only : bool, optional
        Whether to compute the metric only for the CLS token. Default is True.

    Returns
    -------
    torch.Tensor
        The R-squared-like metric.
    """

    # Compute the R-squared-like metric for CLS token or each modality
    if cls_only:
        r_squared_metric = compute_r_squared(
            similarity_matrix, output0, output1, distance=distance
        )
        return r_squared_metric
    else:
        r_squared_modality = []
        for i in range(output0.shape[1]):
            r_squared_modality.append(
                compute_r_squared(
                    similarity_matrix[i],
                    output0[:, i, :],
                    output1[:, i, :],
                    distance=distance,
                )
            )
        return r_squared_modality


def get_correct_count_data_structure(only_cls: bool) -> Union[int, list]:
    """
    Get correct count data structure based on whether we are computing
    the metric only for the CLS token or for each modality.

    Parameters
    ----------
    only_cls : bool
        Whether to compute the metric only for the CLS token.

    Returns
    -------
    int or list
        The correct count data structure.
    """

    if only_cls:
        return 0
    else:
        return []


def update_r_squared(
    r_squared_sum: Union[float, list],
    r_squared: Union[float, list],
    only_cls: bool,
) -> Union[float, list]:
    """
    Updates the R-squared-like metric based on whether we are computing
    the metric only for the CLS token or for each modality.

    Parameters
    ----------
    r_squared_sum : Union[float, list]
        The sum of the R-squared-like metric.
    r_squared : Union[float, list]
        The R-squared-like metric to update.
    only_cls : bool
        Whether to compute the metric only for the CLS token.

    Returns
    -------
    Union[float, list]
        The updated R-squared-like metric.
    """

    if only_cls:
        return r_squared_sum + r_squared
    else:
        if len(r_squared_sum) == 0:
            return r_squared
        else:
            return [
                r_squared_sum[i] + r_squared[i]
                for i in range(len(r_squared_sum))
            ]


def update_correct_count(
    correct_sum: Union[int, list],
    correct: Union[int, list],
    only_cls: bool,
) -> Union[int, list]:
    """
    Update the correct count based on whether we are computing the metric
    only for the CLS token or for each modality.

    Parameters
    ----------
    correct_sum : Union[int, list]
        The sum of the correct count.
    correct : Union[int, list]
        The correct count to update.
    only_cls : bool
        Whether to compute the metric only for the CLS token.
    topk : int, optional
        The number of top predictions to consider for accuracy. Default is 1.

    Returns
    -------
    Union[int, list]
        The updated correct count.
    """
    if only_cls:
        return correct_sum + correct
    else:
        if len(correct_sum) == 0:
            return correct
        else:
            return [
                correct_sum[i] + correct[i] for i in range(len(correct_sum))
            ]


def count_correct_predictions_router(
    output0: torch.Tensor,
    output1: torch.Tensor,
    only_cls: bool = True,
    normalize: bool = True,
    top_k: int = 1,
) -> Union[int, list]:
    """
    Route the computation of the correct predictions based on whether we are
    computing the metric only for the CLS token or for each modality.

    Parameters
    ----------
    output0 : torch.Tensor
        The first set of embeddings.
    output1 : torch.Tensor
        The second set of embeddings.
    only_cls : bool, optional
        Whether to compute the metric only for the CLS token. Default is True.
    normalize : bool, optional
        Whether to normalize the embeddings. Default is True.
    top_k : int, optional
        The 'k' in top-k accuracy. Default is 1 (identical to top-1).

    Returns
    -------
    Union[int, list]
        The correct predictions (in top-k sense).
    """
    if only_cls:
        return clip_count_correct_predictions(
            output0, output1, normalize=normalize, top_k=top_k
        )
    else:
        return clip_count_correct_predictions_multimodal(
            output0, output1, normalize=normalize, top_k=top_k
        )


def clip_accuray_router(
    correct: Union[int, list],
    samples: int,
    print_metrics: bool = True,
    topk: int = 1,
) -> Union[float, list]:
    """
    Route the computation of the accuracy based on whether we are computing
    the metric only for the CLS token or for each modality.

    Parameters
    ----------
    correct : Union[int, list]
        The correct predictions.
    samples : int
        The number of samples.
    print_metrics : bool, optional
        Whether to print the metrics. Default is True.
    topk : int, optional
        The number of top predictions to consider for accuracy. Default is 1.
        Only for CLS token setting, otherwise top 1 is used.

    Returns
    -------
    Union[float, list]
        The accuracy.
    """
    if isinstance(correct, list):
        accuracy = clip_accuracy_multimodal(correct, samples)
        if print_metrics:
            for i, acc in enumerate(accuracy):
                if i == 0:
                    print(f"CLS Accuracy: {acc:.2f}%")
                else:
                    print(f"Token {i+1} Accuracy: {acc:.2f}%")
    else:
        accuracy = clip_accuray(correct, samples, topk)
        if print_metrics:
            print(f"CLS Accuracy: {accuracy:.2f}%")

    return accuracy


def clip_accuray(correct: int, samples: int, topk: int) -> float:
    """
    Compute the accuracy of the a token based on the number of correct
    predictions an the number of samples.

    Parameters
    ----------
    correct : int
        The number of correct predictions.
    samples : int
        The number of samples.
    topk : int
        Top-k accuracy, where k is the number of top predictions to consider.

    Returns
    -------
    float
        The accuracy in percentage.
    """
    return 100 * correct / samples


def clip_accuracy_multimodal(correct_modalities: list, samples: int) -> list:
    """
    Aggregate accuracies for each modality.

    Parameters
    ----------
    correct_modalities : list
        The number of correct predictions for each modality.
    samples : int
        The number of samples.

    Returns
    -------
    list
        The accuracies for each modality.
    """
    accuracies = []
    for correct in correct_modalities:
        accuracy = clip_accuray(correct, samples)
        accuracies.append(accuracy)
    return accuracies


def clip_count_correct_predictions_multimodal(
    output0: torch.Tensor,
    output1: torch.Tensor,
    normalize: bool = True,
    top_k: int = 1,
) -> list:
    """
    Count correct predictions for each modality in a top-k sense.

    Parameters
    ----------
    output0 : torch.Tensor
        The first set of embeddings.
    output1 : torch.Tensor
        The second set of embeddings.
    normalize : bool, optional
        Whether to normalize the embeddings. Default is True.
    top_k : int, optional
        The 'k' in top-k accuracy. Default is 1.

    Returns
    -------
    list
        A tuple of (correct_counts_per_modality, total_samples,
        cosine_similarities_per_modality).
        'correct_counts_per_modality' is a list of integers (one per modality).
        'total_samples' is the total number of “matches” (2N).
        'cosine_similarities_per_modality' is a list of the raw cosine-sim
        matrices (one per modality).
    """
    number_modalities = output0.shape[1]
    correct_modalities = []
    samples_modalities = 0
    cosine_sim_modalities = []

    for i in range(number_modalities):
        correct, samples, cosine_sim = clip_count_correct_predictions(
            output0[:, i, :],
            output1[:, i, :],
            normalize=normalize,
            top_k=top_k,
        )
        correct_modalities.append(correct)
        cosine_sim_modalities.append(cosine_sim)
        if i == 0:
            samples_modalities += samples

    return correct_modalities, samples_modalities, cosine_sim_modalities


def clip_count_correct_predictions(
    output0: torch.Tensor,
    output1: torch.Tensor,
    normalize: bool = True,
    top_k: int = 1,
) -> tuple:
    """
    Count correct predictions for a pair of embeddings in a top-k sense.

    Parameters
    ----------
    output0 : torch.Tensor
        The first set of embeddings.
    output1 : torch.Tensor
        The second set of embeddings.
    normalize : bool, optional
        Whether to normalize the embeddings. Default is True.
    top_k : int, optional
        The 'k' in top-k accuracy. Default is 1.

    Returns
    -------
    tuple
        A 3‐tuple of (correct_count, total_samples, cosine_similarity_matrix).
        - correct_count counts how many times (across both directions) the
          true index appears among the top-k highest cosine similarities.
        - total_samples is 2 * N, where N = number of embeddings.
        - cosine_similarity_matrix is the raw [N, N] matrix before taking
          top-k.
    """

    # Get number of samples
    n = output0.shape[0]

    if normalize:
        output0 = F.normalize(output0, dim=1)
        output1 = F.normalize(output1, dim=1)

    # Compute cosine similarity matrix once: shape [n, n]
    cosine_sim = output0 @ output1.T

    # Ground-truth labels (diagonal from 0 to n−1)
    labels = torch.arange(n, device=output0.device)

    # For “each column j”: find the top_k indices (over rows) that have
    # highest sim[:, j]
    # shape: [top_k, n]
    topk_vals0, topk_inds0 = cosine_sim.topk(top_k, dim=0)

    # For “each row i”: find the top_k indices (over columns) that have
    # highest sim[i, :]
    # shape: [n, top_k]
    topk_vals1, topk_inds1 = cosine_sim.topk(top_k, dim=1)

    # Count correct matches in column-direction:  true index j must appear in
    # top-inds0[:, j]
    # topk_inds0[:, j] is a length‐top_k tensor of row‐indices; label[j] = j.
    # So for each j, check if j is in topk_inds0[:, j]
    correct0 = (topk_inds0 == labels.unsqueeze(0)).any(dim=0).sum().item()

    # Count correct matches in row-direction: true index i must appear in top-
    # inds1[i, :]
    correct1 = (topk_inds1 == labels.unsqueeze(1)).any(dim=1).sum().item()

    # Total correct predictions (across both directions)
    correct = correct0 + correct1

    # Total possible matches is still 2N
    samples = 2 * n

    return correct, samples, cosine_sim


def r_squared_reduction(
    r_squared_sum: Union[float, list],
    num_batches: int,
    only_cls: bool = True,
    print_results: bool = False,
) -> Union[float, list]:
    """
    Reduce the R-squared-like metric based on whether we are computing
    the metric only for the CLS token or for each modality.
    Average the metric over the number of batches.

    Parameters
    ----------
    r_squared_sum : Union[float, list]
        The sum of the R-squared-like metric.
    num_batches : int
        The number of batches.
    only_cls : bool, optional
        Whether to compute the metric only for the CLS token. Default is True.
    print_results : bool, optional
        Whether to print the results. Default is False.

    Returns
    -------
    Union[float, list]
        The R-squared-like metric
    """
    if only_cls:
        r_squared_metric = r_squared_sum / num_batches
        if print_results:
            print(f"R-squared-like Metric for CLS: {r_squared_metric:.4f}")
        if isinstance(r_squared_metric, torch.Tensor):
            r_squared_metric = r_squared_metric.cpu().numpy()
        if not isinstance(r_squared_metric, np.ndarray):
            r_squared_metric = np.array([r_squared_metric])
        return r_squared_metric
    else:

        r_squared_modality = [
            r_squared.cpu().numpy() / num_batches
            for r_squared in r_squared_sum
        ]
        if print_results:
            for i, r_squared in enumerate(r_squared_modality):
                if i == 0:
                    print(f"R-squared-like Metric for CLS: {r_squared:.4f}")
                else:
                    print(
                        f"R-squared-like Metric for Modality {i}: "
                        f"{r_squared:.4f}"
                    )
        return r_squared_modality
