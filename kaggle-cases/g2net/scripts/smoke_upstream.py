"""Synthetic-input numerical checks only; these are NOT score reproductions."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch
from upstream_models import ROOT, make_eventhorizon, make_team3


def check(name, constructor, batch_size, device):
    started = time.time()
    torch.manual_seed(2021)
    model = constructor().to(device)
    model.train()
    waves = torch.randn(batch_size, 3, 4096, device=device)
    labels = (torch.arange(batch_size, device=device) % 2).float().reshape(-1, 1)
    before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    logits = model(waves)
    if logits.shape != labels.shape or not torch.isfinite(logits).all():
        raise ValueError("Invalid output shape or non-finite output")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Missing or non-finite gradients")
    optimizer.step()
    after = next(p for p in model.parameters() if p.requires_grad).detach()
    if torch.equal(before, after):
        raise ValueError("Optimizer did not update the checked parameter")
    model.eval()
    with torch.no_grad():
        prediction = model(waves).sigmoid()
    if not torch.isfinite(prediction).all():
        raise ValueError("Invalid inference output")
    result = {"model": name, "passed": True, "input": list(waves.shape),
              "output": list(logits.shape), "loss_on_random_inputs": float(loss),
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": round(time.time() - started, 2),
              "pretrained_weights": False, "competition_auc": None}
    del model, optimizer, waves, before, after, logits, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    destination = ROOT / "experiments/numerical_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    # Unit spectrum is a fixture for software checks, never used for training.
    psd_path = destination / "unit_spectrum_fixture.pt"
    torch.save(torch.ones(3, 8192), psd_path)
    tests = [
        ("EventHorizon/Seq03aug3", lambda: make_eventhorizon("Seq03aug3", False)[0]),
        ("EventHorizon/Nspec22aug1", lambda: make_eventhorizon("Nspec22aug1", False)[0]),
        ("GOGOGO/V2StochasticDepth_c16", lambda: make_team3(psd_path, 16, 1.0)),
    ]
    results = []
    for name, constructor in tests:
        try:
            result = check(name, constructor, args.batch_size, args.device)
        except Exception as exc:
            result = {"model": name, "passed": False, "error": repr(exc), "traceback": traceback.format_exc()}
        results.append(result)
        print(json.dumps(result), flush=True)
    output = {"test_kind": "random-input forward/backward/optimizer/inference checks",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(args.device), "results": results}
    (ROOT / "reports/numerical_smoke.json").write_text(json.dumps(output, indent=2))
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
