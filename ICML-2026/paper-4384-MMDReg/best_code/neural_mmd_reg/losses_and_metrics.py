import jax.numpy as jnp
import optax


def get_overlap_accuracy(pred_Xo_logits, pred_Yo_logits, true_Xos, true_Yos):
    """Compute overlap classification accuracy over batched logits and masks.

    Args:
        pred_Xo_logits: Batch of predicted logits for source point cloud X.
        pred_Yo_logits: Batch of predicted logits for target point cloud Y.
        true_Xos: Batch of ground truth binary labels for source point cloud X.
        true_Yos: Batch of ground truth binary labels for target point cloud Y.

    Returns:
        The overlap classification accuracy as a percentage.
    """
    accuracy = 0.0
    accuracy = accuracy + jnp.sum((pred_Xo_logits >= 0.0) == true_Xos)
    accuracy = accuracy + jnp.sum((pred_Yo_logits >= 0.0) == true_Yos)
    accuracy = accuracy / (true_Xos.size + true_Yos.size)
    accuracy = accuracy * 100.0
    return accuracy


def get_overlap_loss(pred_overlap_logits, true_overlap_masks):
    """Compute the mean cross-entropy loss over batched logits and masks.

    Args:
        pred_overlap_logits: Batch of predicted logits.
        true_overlap_masks: Batch of ground truth binary labels.

    Returns:
        The mean sigmoid binary cross-entropy loss.
    """
    K = optax.losses.sigmoid_binary_cross_entropy(
        pred_overlap_logits, true_overlap_masks
    )
    overlap_loss = jnp.mean(K)
    return overlap_loss


def get_rotation_errors(pred_rotation_matrices, true_rotation_matrices):
    """Compute angular errors in degrees over batched rotation matrices.

    Args:
        pred_rotation_matrices: Batch of predicted rotation matrices.
        true_rotation_matrices: Batch of ground truth rotation matrices.

    Returns:
        A batch of angular errors in degrees.
    """
    K = jnp.sum(pred_rotation_matrices * true_rotation_matrices, axis=(1, 2))
    errors = jnp.clip(0.5 * (K - 1.0), -1.0, 1.0)
    errors = jnp.arccos(errors) * 180.0 / jnp.pi
    return errors


def get_rotation_loss(pred_rotation_matrices, true_rotation_matrices):
    """Compute the mean angular distance over batched rotation matrices.

    Args:
        pred_rotation_matrices: Batch of predicted rotation matrices.
        true_rotation_matrices: Batch of ground truth rotation matrices.

    Returns:
        The mean angular distance in radians.
    """
    eps = 1e-6
    K = jnp.sum(pred_rotation_matrices * true_rotation_matrices, axis=(1, 2))
    K = jnp.clip(0.5 * (K - 1.0), -1.0 + eps, 1.0 - eps)
    rotation_loss = jnp.mean(jnp.arccos(K))
    return rotation_loss


def get_translation_errors(pred_translation_vectors, true_translation_vectors):
    """Compute L2 distances over batched translation vectors.

    Args:
        pred_translation_vectors: Batch of predicted translation vectors.
        true_translation_vectors: Batch of ground truth translation vectors.

    Returns:
        A batch of L2 distances.
    """
    K = pred_translation_vectors - true_translation_vectors
    errors = jnp.linalg.norm(K, axis=1)
    return errors


def get_translation_loss(pred_translation_vectors, true_translation_vectors):
    """Compute the mean squared L2 distance over batched translation vectors.

    Args:
        pred_translation_vectors: Batch of predicted translation vectors.
        true_translation_vectors: Batch of ground truth translation vectors.

    Returns:
        The mean squared L2 distance.
    """
    K = pred_translation_vectors - true_translation_vectors
    translation_loss = jnp.mean(jnp.sum(K * K, axis=1))
    return translation_loss
