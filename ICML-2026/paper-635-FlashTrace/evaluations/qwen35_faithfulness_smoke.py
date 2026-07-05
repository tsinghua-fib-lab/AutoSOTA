"""Faithfulness smoke evaluation for FlashTrace on Qwen3.5 (hybrid) models.

Each sample is a fixed ``(context, answer)`` pair where the answer depends on
specific context tokens. The faithfulness metric is the Spearman rank
correlation between an attribution method's per-token scores and the
token-level leave-one-out (LOO) importance ground truth -- i.e. how well the
attribution recovers the tokens whose removal actually hurts the fixed answer.

Correlation is bounded in ``[-1, 1]`` and normalises away each model's raw
sensitivity to deletion, so scores are directly comparable across models. We
compare:

* FlashTrace IFR attribution
* a coarse segment-level perturbation baseline

Used both as an importable library (unit-tested on a tiny synthetic model) and
as a script for the real Qwen3.5-9B vs Qwen3-8B comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from flashtrace.tracer import FlashTrace


# --------------------------------------------------------------------------
# Faithfulness primitives (token-grid consistent)
# --------------------------------------------------------------------------

def _answer_logprob(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    """Total log-probability of the answer tokens ``input_ids[prompt_len:]``."""

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    target = input_ids[0, prompt_len:]
    pred = logp[0, prompt_len - 1 : -1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(pred.sum().item())


def _split_groups(indices: Sequence[int], k: int) -> List[List[int]]:
    k = max(1, min(int(k), len(indices)))
    base, rem = divmod(len(indices), k)
    groups, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        groups.append(list(indices[start : start + size]))
        start += size
    return groups


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson correlation of the ranks)."""

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def leave_one_out_importance(
    model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Per-token ground-truth importance: answer log-prob drop when each prompt
    token is individually masked. Higher == the token matters more."""

    prompt_len = int(prompt_ids.shape[1])
    base = _answer_logprob(model, torch.cat([prompt_ids, answer_ids], dim=1), prompt_len)

    importance = torch.zeros(prompt_len, dtype=torch.float32)
    for j in range(prompt_len):
        perturbed = prompt_ids.clone()
        perturbed[0, j] = pad_id
        masked = _answer_logprob(
            model, torch.cat([perturbed, answer_ids], dim=1), prompt_len
        )
        importance[j] = base - masked
    return importance


def segment_perturbation_attribution(
    model,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    pad_id: int,
    *,
    n_segments: int = 20,
) -> torch.Tensor:
    """Coarse perturbation baseline: ablate prompt segments, attribute the drop.

    Each segment is masked in turn; the resulting drop in answer log-prob is
    spread uniformly across the segment's tokens. This mirrors the project's
    sentence-level feature-ablation baseline.
    """

    prompt_len = int(prompt_ids.shape[1])
    base = _answer_logprob(model, torch.cat([prompt_ids, gen_ids], dim=1), prompt_len)

    scores = torch.zeros(prompt_len, dtype=torch.float32)
    for segment in _split_groups(list(range(prompt_len)), min(n_segments, prompt_len)):
        perturbed = prompt_ids.clone()
        for idx in segment:
            perturbed[0, idx] = pad_id
        ablated = _answer_logprob(model, torch.cat([perturbed, gen_ids], dim=1), prompt_len)
        drop = base - ablated
        for idx in segment:
            scores[idx] = drop / max(1, len(segment))
    return scores


# --------------------------------------------------------------------------
# Per-sample evaluation
# --------------------------------------------------------------------------

@dataclass
class SampleResult:
    prompt: str
    n_prompt_tokens: int
    n_answer_tokens: int
    flashtrace_corr: float
    perturbation_corr: float


def _scores_vec(scores: Sequence[float], n: int) -> np.ndarray:
    vec = np.zeros(n, dtype=np.float64)
    m = min(n, len(scores))
    vec[:m] = np.asarray([float(s) for s in scores[:m]], dtype=np.float64)
    return vec


def evaluate_sample(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    *,
    n_segments: int = 8,
) -> SampleResult:
    """Evaluate FlashTrace vs the perturbation baseline on one fixed pair.

    ``answer`` is scored as a fixed continuation of ``prompt``. Faithfulness is
    the Spearman correlation between each method's per-token attribution and the
    token-level leave-one-out importance ground truth -- comparable across
    models because correlation is bounded and sensitivity-normalised.
    """

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    answer_len = int(answer_ids.shape[1])

    loo = leave_one_out_importance(model, prompt_ids, answer_ids, pad_id).numpy()

    tracer = FlashTrace(model, tokenizer)
    ft_result = tracer.trace(
        prompt=prompt,
        target=answer,
        output_span=(0, max(0, answer_len - 1)),
        method="flashtrace",
    )
    ft_vec = _scores_vec(ft_result.scores, prompt_len)

    pert_vec = segment_perturbation_attribution(
        model, prompt_ids, answer_ids, pad_id, n_segments=n_segments
    ).numpy()

    return SampleResult(
        prompt=prompt,
        n_prompt_tokens=prompt_len,
        n_answer_tokens=answer_len,
        flashtrace_corr=_spearman(ft_vec, loo),
        perturbation_corr=_spearman(pert_vec, loo),
    )


# Fixed (context, answer) pairs. Each answer depends on specific context tokens,
# so guided deletion of those tokens collapses the answer log-probability.
DEFAULT_SAMPLES: List[tuple] = [
    (
        "Background notes. The annual report was filed on time. "
        "The secret access code for the east vault is 7492. "
        "Lunch will be served at noon. The weather has been mild. "
        "Question: What is the secret access code for the east vault? Answer:",
        " 7492",
    ),
    (
        "Several people attended the meeting. The budget review went smoothly. "
        "Dr. Helena Ramirez was appointed as the new chief scientist. "
        "Coffee was available in the lobby. The minutes were approved. "
        "Question: Who was appointed as the new chief scientist? Answer:",
        " Dr. Helena Ramirez",
    ),
    (
        "Trip log. The hotel had a nice view. "
        "The package must be delivered to warehouse number 38. "
        "Traffic was light in the morning. The flight landed early. "
        "Question: Which warehouse must the package be delivered to? Answer:",
        " warehouse number 38",
    ),
    (
        "Company memo. The cafeteria menu changed this week. "
        "The new product launch is scheduled for September 14th. "
        "Parking permits were renewed. The printer on floor 3 was fixed. "
        "Question: When is the new product launch scheduled? Answer:",
        " September 14th",
    ),
    (
        "Lab journal. The samples were stored correctly. "
        "The experiment used a temperature of 320 kelvin. "
        "The notebook was updated daily. The equipment was calibrated. "
        "Question: What temperature did the experiment use? Answer:",
        " 320 kelvin",
    ),
    (
        "Reading notes. The novel was published long ago. "
        "The protagonist of the story is a cartographer named Edwin. "
        "The cover art was striking. The chapters were short. "
        "Question: What is the profession of the protagonist? Answer:",
        " cartographer",
    ),
]


def evaluate_model(model_name: str, samples: Sequence[tuple], **kwargs) -> List[SampleResult]:
    """Load ``model_name`` and evaluate every ``(context, answer)`` sample."""

    from flashtrace.model_io import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_name, dtype="bfloat16")
    results = [evaluate_sample(model, tokenizer, p, a, **kwargs) for p, a in samples]
    del model
    torch.cuda.empty_cache()
    return results


def _summary(results: Sequence[SampleResult]) -> dict:
    return {
        "flashtrace_corr": float(np.mean([r.flashtrace_corr for r in results])),
        "perturbation_corr": float(np.mean([r.perturbation_corr for r in results])),
    }


def _print_per_sample(label: str, results: Sequence[SampleResult]) -> None:
    print(f"  {label}:")
    for i, r in enumerate(results):
        print(
            f"    sample {i}: FlashTrace={r.flashtrace_corr:+.4f}  "
            f"Perturbation={r.perturbation_corr:+.4f}  "
            f"(P={r.n_prompt_tokens}, A={r.n_answer_tokens})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--qwen3", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    print(f"== Evaluating Qwen3.5 ({args.qwen35}) ==")
    q35 = evaluate_model(args.qwen35, DEFAULT_SAMPLES)
    _print_per_sample("Qwen3.5-9B", q35)
    print(f"== Evaluating Qwen3  ({args.qwen3}) ==")
    q3 = evaluate_model(args.qwen3, DEFAULT_SAMPLES)
    _print_per_sample("Qwen3-8B", q3)

    s35, s3 = _summary(q35), _summary(q3)
    print("\n--- Mean faithfulness (Spearman corr. with LOO; higher == more faithful) ---")
    for label, summ in [("Qwen3.5-9B", s35), ("Qwen3-8B", s3)]:
        print(
            f"  {label:12s}  FlashTrace={summ['flashtrace_corr']:+.4f}  "
            f"Perturbation={summ['perturbation_corr']:+.4f}"
        )

    ft35 = s35["flashtrace_corr"]
    pert35 = s35["perturbation_corr"]
    ft3 = s3["flashtrace_corr"]
    beats_baseline = ft35 >= pert35 - 1e-6
    close_to_qwen3 = abs(ft35 - ft3) <= 0.15
    print(f"\nFlashTrace(Qwen3.5) beats perturbation baseline: {beats_baseline} "
          f"({ft35:+.4f} vs {pert35:+.4f})")
    print(f"FlashTrace(Qwen3.5) close to FlashTrace(Qwen3-8B): {close_to_qwen3} "
          f"({ft35:+.4f} vs {ft3:+.4f})")
    if not (beats_baseline and close_to_qwen3):
        raise SystemExit("Qwen3.5 faithfulness smoke check FAILED")
    print("\nQwen3.5 faithfulness smoke check PASSED")


if __name__ == "__main__":
    main()
