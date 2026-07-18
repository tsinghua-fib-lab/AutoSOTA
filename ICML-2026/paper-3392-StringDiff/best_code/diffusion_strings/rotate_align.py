import torch


def kabsch_align(
    moving: torch.Tensor,
    reference: torch.Tensor,
    return_transform: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align batched 3D point clouds with the Kabsch algorithm.

    Args:
        moving: Tensor of shape (T, N, 3).
        reference: Tensor of shape (T, N, 3), paired with ``moving``.
        return_transform: If ``True``, also return ``(rotation, translation)``.

    Returns:
        If ``return_transform`` is ``False``:
            aligned point clouds of shape (T, N, 3).
        If ``return_transform`` is ``True``:
            tuple ``(aligned, rotation, translation)`` where
            ``rotation`` has shape (T, 3, 3) and ``translation`` has shape (T, 1, 3),
            with ``aligned = moving @ rotation + translation``.
    """
    if moving.ndim != 3 or reference.ndim != 3:
        raise ValueError("moving and reference must have shape (T, N, 3).")
    if moving.shape != reference.shape:
        raise ValueError("moving and reference must have the same shape.")
    if moving.shape[-1] != 3:
        raise ValueError("Last dimension must be 3.")
    if moving.shape[1] < 1:
        raise ValueError("N must be >= 1.")

    moving_centroid = moving.mean(dim=1, keepdim=True)
    reference_centroid = reference.mean(dim=1, keepdim=True)

    moving_centered = moving - moving_centroid
    reference_centered = reference - reference_centroid

    covariance = moving_centered.transpose(-1, -2) @ reference_centered
    u, _, vh = torch.linalg.svd(covariance)

    # Enforce proper rotations (det=+1), avoiding reflections.
    det_uv = torch.det(u @ vh)
    correction = torch.ones((moving.shape[0], 3), dtype=moving.dtype, device=moving.device)
    correction[:, -1] = torch.where(det_uv < 0, -1.0, 1.0).to(dtype=moving.dtype)
    d = torch.diag_embed(correction)

    rotation = u @ d @ vh
    translation = reference_centroid - (moving_centroid @ rotation)
    aligned = moving @ rotation + translation

    if return_transform:
        return aligned, rotation, translation
    return aligned


def align_translations_and_rotations_to_index(
    translations: torch.Tensor,
    rotations: torch.Tensor,
    index: int,
    return_transform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align a trajectory of rigid transforms to a reference frame index.

    Args:
        translations: Tensor of shape (T, N, 3).
        rotations: Tensor of shape (T, N, 3, 3).
        index: Reference frame index in ``[0, T-1]``.
        return_transform: If ``True``, also return Kabsch ``(rotation, translation)``
            used for each frame alignment.

    Returns:
        If ``return_transform`` is ``False``:
            tuple ``(aligned_translations, aligned_rotations)``.
        If ``return_transform`` is ``True``:
            tuple ``(aligned_translations, aligned_rotations, frame_rotation, frame_translation)``.

        ``aligned_translations`` has shape (T, N, 3), ``aligned_rotations`` has shape
        (T, N, 3, 3), ``frame_rotation`` has shape (T, 3, 3), and
        ``frame_translation`` has shape (T, 1, 3).
    """
    if translations.ndim != 3 or translations.shape[-1] != 3:
        raise ValueError("translations must have shape (T, N, 3).")
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (T, N, 3, 3).")
    if translations.shape[:2] != rotations.shape[:2]:
        raise ValueError("translations and rotations must have matching (T, N) dimensions.")

    t_size = translations.shape[0]
    if not (-t_size <= index < t_size):
        raise IndexError("index out of range for translations first dimension.")

    ref = translations[index].unsqueeze(0).expand_as(translations)
    aligned_translations, frame_rotation, frame_translation = kabsch_align(
        translations,
        ref,
        return_transform=True,
    )
    frame_rotation_t = frame_rotation.transpose(-1, -2).unsqueeze(1)
    frame_rotation_u = frame_rotation.unsqueeze(1)
    aligned_rotations = frame_rotation_t @ rotations @ frame_rotation_u

    if return_transform:
        return aligned_translations, aligned_rotations, frame_rotation, frame_translation
    return aligned_translations, aligned_rotations
