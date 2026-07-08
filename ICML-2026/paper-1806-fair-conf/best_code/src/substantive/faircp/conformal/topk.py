from torch import Tensor
from substantive.faircp.conformity.utils import accuracy


def top_k(
    k: int,
    logits_calib: Tensor,
    targets_calib: Tensor,
    logits_test: Tensor,
    targets_test: Tensor,
) -> float:
    # cvg_topk_val = accuracy(logits_val, targets_val, topk=(k,))[0].item() / 100.0
    # print(f"Empirical coverage of top {k} prediction sets on the validation set: {cvg_topk_val: .4f}")
    cvg_topk_calib = accuracy(logits_calib, targets_calib, topk=(k,))[0].item() / 100.0
    print(
        f"Empirical coverage of top-{k} prediction sets on the calibration set: {cvg_topk_calib: .4f}"
    )
    cvg_topk_test = accuracy(logits_test, targets_test, topk=(k,))[0].item() / 100.0
    print(
        f"Empirical coverage of top-{k} prediction sets on the test set: {cvg_topk_test: .4f}"
    )

    return cvg_topk_calib
