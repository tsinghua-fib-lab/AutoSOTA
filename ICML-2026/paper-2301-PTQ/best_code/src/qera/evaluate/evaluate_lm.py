import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model,
    eval_dataloader: DataLoader,
    num_samples: int = None,
    progress_bar: bool = False,
    input_device: str = None,
    description: str = "Evaluating perplexity",
):
    if num_samples is not None:
        if not num_samples >= eval_dataloader.batch_size:
            txt = f"num_samples {num_samples} must be greater than batch_size {eval_dataloader.batch_size}"
            raise ValueError(txt)
        if not num_samples <= eval_dataloader.batch_size * len(eval_dataloader):
            txt = (
                f"num_samples {num_samples} must be less than or equal to "
                f"batch_size * len(eval_dataloader) = "
                f"{eval_dataloader.batch_size} * {len(eval_dataloader)} = {eval_dataloader.batch_size * len(eval_dataloader)}"
            )
            raise ValueError(txt)

    losses = []
    model.eval()

    if num_samples:
        num_batches = num_samples // eval_dataloader.batch_size
    else:
        num_batches = len(eval_dataloader)

    progress_bar = tqdm(
        eval_dataloader,
        desc=description,
        total=num_batches,
        disable=not progress_bar,
    )

    batch_size = eval_dataloader.batch_size
    seq_len = next(iter(eval_dataloader))["input_ids"].shape[1]
    evaluated_samples = 0
    for i, batch in enumerate(eval_dataloader):
        if num_samples and i >= num_batches:
            break

        assert (
            batch["input_ids"].shape[1] == seq_len
        ), f"sequence length is not a constant current seq_len = {batch['input_ids'].shape[1]} != {seq_len}"
        with torch.no_grad():
            if input_device is None:
                input_device = next(iter(model.state_dict().values())).device
            batch = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
        loss = outputs.loss.item() * batch_size * seq_len
        losses.append(loss)
        evaluated_samples += batch_size

        progress_bar.update(1)

    logger.debug(f"evaluated_samples = {evaluated_samples}")

    reduced_loss = sum(losses) / (seq_len * evaluated_samples)
    try:
        perplexity = math.exp(reduced_loss)
    except OverflowError:
        perplexity = float("inf")

    results = {
        "loss": reduced_loss,
        "perplexity": perplexity,
        "num_samples": evaluated_samples,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    return results
