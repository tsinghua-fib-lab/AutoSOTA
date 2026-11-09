from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ramrt_nested_cv import (
    BASE_PATH,
    ID_MAP_PATH,
    ROOT,
    SEEDS,
    TARGET_CELLS,
    bootstrap_interval,
    gate_lambdas,
    load_data,
    predict_corrections,
    quality,
    raw_reliability,
    read_submission,
)


REPORT_PATH = ROOT / "experiments" / "ramrt_nested_cv_results.json"
OOF_PATH = ROOT / "experiments" / "ramrt_oof_predictions.npz"
FULL_PATH = ROOT / "experiments" / "ramrt_b_abstention_full_corrections.npz"
OUTPUT_PATH = ROOT / "final_submission" / "ramrt_b_abstention_candidate.zip"
AUDIT_PATH = ROOT / "final_submission" / "ramrt_b_abstention_candidate_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    oof = np.load(OOF_PATH)
    compounds = oof["compounds"].astype(str)
    base = oof["base"]
    truth = oof["truth"]
    corrections = oof["corrections"]
    nested_lambdas = oof["lambdas"].copy()
    nested_lambdas[1] = 0.0
    base_error = np.sqrt(np.mean((base - truth) ** 2, axis=-1))
    prediction = base + nested_lambdas[..., None] * corrections.mean(axis=0)
    delta = base_error - np.sqrt(np.mean((prediction - truth) ** 2, axis=-1))
    b_delta = delta[0]
    per_compound = delta.mean(axis=0)
    positive_sum = np.maximum(per_compound, 0.0).sum()
    max_gain_share = float(np.maximum(per_compound, 0.0).max() / positive_sum)
    seed_b_deltas = {}
    for position, seed in enumerate(SEEDS):
        seed_prediction = base[0] + nested_lambdas[0, :, None] * corrections[position, 0]
        seed_delta = base_error[0] - np.sqrt(
            np.mean((seed_prediction - truth[0]) ** 2, axis=-1)
        )
        seed_b_deltas[str(seed)] = float(seed_delta.mean())
    b_ci = bootstrap_interval(b_delta, 117)
    combined_ci = bootstrap_interval(per_compound, 217)
    validation = {
        "selection": "target-level B activation from aggregate cross-fitted evidence; Myeloid abstains",
        "b_delta": float(b_delta.mean()),
        "b_wins": int((b_delta > 0).sum()),
        "b_ci95": b_ci,
        "combined_delta": float(per_compound.mean()),
        "combined_ci95_by_compound": combined_ci,
        "seed_b_deltas": seed_b_deltas,
        "max_single_compound_positive_gain_share": max_gain_share,
    }
    validation["accept"] = bool(
        b_delta.mean() > 0.0
        and (b_delta > 0).sum() >= 10
        and b_ci[0] > 0.0
        and combined_ci[0] > 0.0
        and all(value > 0.0 for value in seed_b_deltas.values())
        and max_gain_share <= 0.5
    )
    if not validation["accept"]:
        raise RuntimeError(f"Frozen validation gate failed: {validation}")

    data = load_data()
    fold_reliability = {name: [] for name in ("agreement", "distance", "strength")}
    compound_position = {drug: position for position, drug in enumerate(compounds)}
    for fold in report["folds"]:
        valid = np.asarray(fold["valid"])
        positions = np.asarray([compound_position[drug] for drug in valid])
        fold_corrections = corrections[:, :, positions]
        reliability = raw_reliability(data, np.asarray(fold["train"]), valid, fold_corrections)
        for name, values in reliability.items():
            fold_reliability[name].append(values)
    references = {
        name: np.concatenate(values, axis=1).reshape(-1)
        for name, values in fold_reliability.items()
    }

    params = dict(report["final_refit_gate"])
    params["rho"] = 0.0
    all_drugs = np.asarray(data.drugs)
    full_corrections = predict_corrections(data, compounds, all_drugs, 2900000)
    full_reliability = raw_reliability(data, compounds, all_drugs, full_corrections)
    q = quality(full_reliability, str(params["family"]), references)
    lambdas = gate_lambdas(q, params)
    lambdas[1] = 0.0
    np.savez_compressed(
        FULL_PATH,
        drugs=all_drugs,
        corrections=full_corrections,
        lambdas=lambdas,
        quality=q,
        seeds=np.asarray(SEEDS),
    )

    base_submission = read_submission(BASE_PATH)
    id_map = pd.read_csv(ID_MAP_PATH)
    if len(base_submission) != len(id_map):
        raise ValueError("Submission and id_map row counts differ")
    if not np.array_equal(base_submission.iloc[:, 0].to_numpy(), id_map["id"].to_numpy()):
        raise ValueError("Submission IDs do not align with id_map")
    drug_position = {drug: position for position, drug in enumerate(all_drugs)}
    values = base_submission.iloc[:, 1:].to_numpy(dtype=np.float64)
    original = values.copy()
    modified = []
    for row in range(len(id_map)):
        if id_map.at[row, "cell_type"] != "B cells":
            continue
        drug = id_map.at[row, "sm_name"]
        position = drug_position[drug]
        correction = full_corrections[:, 0, position].mean(axis=0)
        values[row] += lambdas[0, position] * correction
        modified.append(row)
    if not np.isfinite(values).all():
        raise ValueError("Candidate contains non-finite values")
    candidate = base_submission.copy()
    candidate.iloc[:, 1:] = values
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.csv", candidate.to_csv(index=False).encode("utf-8"))
    with zipfile.ZipFile(OUTPUT_PATH) as archive:
        bad_member = archive.testzip()
    reread = read_submission(OUTPUT_PATH)
    reread_values = reread.iloc[:, 1:].to_numpy(dtype=np.float64)
    audit = {
        "method": "RAMRT reliability gate on B cells with target-level Myeloid abstention",
        "base": BASE_PATH.name,
        "gate": params,
        "validation": validation,
        "modified_rows": len(modified),
        "mean_b_lambda": float(lambdas[0].mean()),
        "min_b_lambda": float(lambdas[0].min()),
        "max_b_lambda": float(lambdas[0].max()),
        "shape": list(reread.shape),
        "unique_ids": int(reread.iloc[:, 0].nunique()),
        "finite": bool(np.isfinite(reread_values).all()),
        "zip_bad_member": bad_member,
        "non_b_max_abs_difference": float(
            np.max(
                np.abs(
                    (reread_values - original)[id_map["cell_type"].ne("B cells").to_numpy()]
                )
            )
        ),
        "delta_rms": float(np.sqrt(np.mean((reread_values - original) ** 2))),
        "delta_max_abs": float(np.max(np.abs(reread_values - original))),
        "sha256": sha256(OUTPUT_PATH),
        "output": str(OUTPUT_PATH),
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
