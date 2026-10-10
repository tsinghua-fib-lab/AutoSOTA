from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .artifacts import (
    initialize_grad_cam_output_dir,
    save_grad_cam_sample,
    write_grad_cam_metadata,
)
from .performance import unwrap_model


def compute_grad_cam_pp_batch(trainer, batch) -> list[dict]:
    architecture = unwrap_model(trainer.get_architecture())
    target_layer = architecture.get_grad_cam_target_layer()
    device = trainer.device
    images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
    labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    try:
        architecture.eval()
        architecture.zero_grad(set_to_none=True)
        logits = architecture(images)
        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = logits.argmax(dim=1)

        if not activations:
            raise RuntimeError("Grad-CAM target layer did not produce activations")

        batch_activations = activations[-1]
        results = []
        for sample_index in range(images.shape[0]):
            architecture.zero_grad(set_to_none=True)
            gradients.clear()
            target_class = int(predictions[sample_index].item())
            score = logits[sample_index, target_class]
            score.backward(retain_graph=sample_index < images.shape[0] - 1)
            if not gradients:
                raise RuntimeError("Grad-CAM target layer did not produce gradients")

            heatmap = _build_grad_cam_pp_heatmap(
                activations=batch_activations[sample_index],
                gradients=gradients[-1][sample_index],
                output_shape=tuple(int(axis) for axis in images.shape[2:]),
            )
            results.append(
                {
                    "sample_id": str(batch["sample_id"][sample_index]),
                    "case_id": str(batch["case_id"][sample_index]),
                    "roi_name": str(batch["roi_name"][sample_index]),
                    "label": int(labels[sample_index].item()),
                    "prediction": int(predictions[sample_index].item()),
                    "target_class": target_class,
                    "probabilities": probabilities[sample_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "input_shape": tuple(int(axis) for axis in images.shape[2:]),
                    "heatmap": heatmap.detach().cpu().to(torch.float32).numpy(),
                }
            )

        return results
    finally:
        forward_handle.remove()
        architecture.zero_grad(set_to_none=True)


def export_validation_grad_cam(
    trainer,
    *,
    output_dir: Path,
    checkpoint_kind: str,
    log_fn=None,
    log_path: Path | None = None,
) -> Path:
    initialize_grad_cam_output_dir(output_dir)
    run_metadata = {
        "trainer_name": trainer.__class__.__name__,
        "architecture_name": trainer.architecture_name,
        "dataset_id": trainer.dataset_id,
        "dataset_name": trainer.dataset_dir.name,
        "fold": int(trainer.fold),
        "checkpoint_kind": checkpoint_kind,
        "target_policy": "predicted_class",
        "normalization": "min_max_per_sample_[0,1]",
        "space": "validation_input_after_preprocessing",
    }
    metadata_predictions: dict[str, dict] = {}
    for batch in trainer.get_val_dataloader():
        batch_results = compute_grad_cam_pp_batch(trainer, batch)
        for result in batch_results:
            metadata_predictions[str(result["sample_id"])] = save_grad_cam_sample(
                result,
                output_dir,
            )
            metadata_path = write_grad_cam_metadata(
                output_dir,
                run_metadata=run_metadata,
                predictions=metadata_predictions,
            )
            if log_fn is not None and log_path is not None:
                log_fn(
                    "Grad-CAM saved for "
                    f"{result['sample_id']} "
                    f"(target_class={result['target_class']}, "
                    f"prediction={result['prediction']})",
                    log_path,
                )
    return (
        metadata_path
        if metadata_predictions
        else write_grad_cam_metadata(
            output_dir,
            run_metadata=run_metadata,
            predictions=metadata_predictions,
        )
    )


def _build_grad_cam_pp_heatmap(
    *,
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    activations = activations.float()
    gradients = gradients.float()
    grad_sq = gradients.pow(2)
    grad_cube = grad_sq * gradients
    spatial_dims = (1, 2, 3)
    activation_sum = activations.sum(dim=spatial_dims, keepdim=True)
    denominator = 2.0 * grad_sq + activation_sum * grad_cube
    denominator = torch.where(
        denominator.abs() > 0,
        denominator,
        torch.full_like(denominator, 1e-8),
    )
    alpha = grad_sq / (denominator + 1e-8)
    weights = (alpha * torch.relu(gradients)).sum(dim=spatial_dims)
    cam = torch.relu((weights[:, None, None, None] * activations).sum(dim=0))
    cam = F.interpolate(
        cam[None, None],
        size=output_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    cam_min = cam.min()
    cam_max = cam.max()
    if float((cam_max - cam_min).detach()) <= 1e-8:
        return torch.zeros_like(cam)
    return (cam - cam_min) / (cam_max - cam_min)
