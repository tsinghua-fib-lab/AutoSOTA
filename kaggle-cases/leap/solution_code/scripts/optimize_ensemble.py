"""Optimize champion weights using calibration data and report a separate audit split.

No competition test data or test labels enter this program. The audit split is
never used to fit weights, choose regularization, or select postprocessing.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("temperature", 0, 60), ("vapor", 60, 120), ("liquid", 120, 180),
          ("ice", 180, 240), ("wind_u", 240, 300), ("wind_v", 300, 360), ("scalars", 360, 368)]


def score(y, p, weights):
    return float(r2_score(y*weights, p*weights, multioutput="uniform_average", force_finite=True))


def physics(predictions, humidity, weights):
    result = predictions.copy()
    result[:, 60:240] = np.maximum(result[:, 60:240], -humidity/1200)
    result[:, weights == 0] = 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/champion-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimized-ensemble")
    parser.add_argument("--source-ratio",type=float)
    parser.add_argument("--relative-radius",type=float)
    args = parser.parse_args()
    models = json.loads((ROOT / "configs/champion_models.json").read_text())
    missing=[m["id"] for m in models if not (args.root/m["id"]/"metrics.json").exists()]
    if missing:raise SystemExit(f"Original champion members missing: {missing}")
    arrays = [np.load(args.root/m["id"]/"predictions.npy", mmap_mode="r") for m in models]
    reference = args.root/models[0]["id"]
    labels = np.load(reference/"labels.npy", mmap_mode="r")
    humidity = np.load(reference/"humidity.npy", mmap_mode="r")
    reports=[json.loads((args.root/m["id"]/"metrics.json").read_text()) for m in models]
    if len(labels)!=200000 or any(r["rows"]!=200000 or r["records"]!=reports[0]["records"] for r in reports):
        raise SystemExit("Requires identical original 200k holdout and record order for all members")
    original = np.asarray([m["weight"] for m in models], np.float64)
    family=np.asarray([m['id'].startswith('lr_') for m in models],np.float64)
    if args.source_ratio is not None:
        if not 0<args.source_ratio<1:raise ValueError('Source ratio must lie in (0,1)')
        original*=np.where(family>0,args.source_ratio/.35,(1-args.source_ratio)/.65)
    np.testing.assert_allclose(original.sum(), 1)
    with (ROOT / "data/greysnow/leap-sample-submission-stds-ds/stds_new.p").open("rb") as f:
        weights=np.asarray(pickle.load(f), np.float64)
    baseline = np.zeros(labels.shape, np.float64)
    for w,p in zip(original,arrays):baseline+=w*p
    result = np.zeros_like(baseline)
    fit = slice(0,75000);select = slice(75000,100000);audit = slice(100000,200000)
    group_weights = {}
    selected_regularization = {}
    for name,start,end in GROUPS:
        columns = np.arange(start,end)
        y = labels[fit, columns]
        sd = y.std(axis=0)
        active = (weights[columns]!=0)&(sd>0)
        columns = columns[active];sd=sd[active]
        residuals = np.stack([(p[fit,columns]-labels[fit,columns])/sd for p in arrays], axis=-1)
        flat = residuals.reshape(-1,len(models))
        with threadpool_limits(limits=8):gram=flat.T@flat/len(flat)
        penalty_scale=max(float(np.trace(gram)/len(models)),1e-12)
        best_loss = np.inf;best_w = original.copy();best_reg = None
        for regularization in [0.0,0.001,0.01,0.1,1.0,10.0]:
            penalty=regularization*penalty_scale
            def objective(w):
                return float(w@gram@w+penalty*np.sum((w-original)**2))
            def gradient(w):
                return 2*gram@w+2*penalty*(w-original)
            bounds=[(0,1)]*len(models) if args.relative_radius is None else [(max(0,w*(1-args.relative_radius)),min(1,w*(1+args.relative_radius))) for w in original]
            constraints=[{"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones_like(w)}]
            if args.source_ratio is not None:constraints.append({"type":"eq","fun":lambda w:w@family-args.source_ratio,"jac":lambda w:family})
            fitted=minimize(objective,original,jac=gradient,method="SLSQP",bounds=bounds,
                            constraints=constraints,
                            options={"ftol":1e-12,"maxiter":1000})
            if not fitted.success:raise RuntimeError(f"Weight optimizer failed: {fitted.message}")
            selection_prediction=sum(w*p[select,columns] for w,p in zip(fitted.x,arrays))
            loss=float(np.mean(((selection_prediction-labels[select,columns])/sd)**2))
            if loss<best_loss:best_loss=loss;best_w=fitted.x.copy();best_reg=regularization
        group_weights[name]=best_w.tolist();selected_regularization[name]=best_reg
        for w,p in zip(best_w,arrays):result[:,start:end]+=w*p[:,start:end]
        print(f"Group {name}: selected regularization={best_reg}, calibration normalized MSE={best_loss:.8f}",flush=True)
    candidates={"original":baseline,"original_physics":physics(baseline,humidity,weights),
                "group_weights":result,"group_weights_physics":physics(result,humidity,weights)}
    selection_scores={name:score(labels[select],p[select],weights) for name,p in candidates.items()}
    # Freeze all choices before the independent audit is scored.
    selected=max(selection_scores,key=selection_scores.get)
    args.output.mkdir(parents=True,exist_ok=True)
    frozen={"selected":selected,"models":[m["id"] for m in models],"original_weights":original.tolist(),
            "group_weights":group_weights,"regularization":selected_regularization,
            "selection_scores":selection_scores,"fit_rows":[0,75000],"selection_rows":[75000,100000],
            "audit_rows":[100000,200000],"test_labels_used":False,"source_ratio":args.source_ratio,"relative_radius":args.relative_radius}
    (args.output/"frozen_config.json").write_text(json.dumps(frozen,indent=2))
    audit_scores={name:score(labels[audit],p[audit],weights) for name,p in candidates.items()}
    report={**frozen,"audit_scores":audit_scores,"audit_gain":audit_scores[selected]-audit_scores["original"],
            "baseline_full_holdout":score(labels,baseline,weights),
            "selected_full_holdout":score(labels,candidates[selected],weights),
            "score_type":"local_validation_not_Kaggle_LB",
            "note":"Same original pretrained members; configuration selected on first 100k only. Audit is the other 100k of the champion's training-era holdout, not the secret competition test set."}
    np.save(args.output/"baseline_predictions.npy",baseline)
    np.save(args.output/"selected_predictions.npy",candidates[selected])
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
