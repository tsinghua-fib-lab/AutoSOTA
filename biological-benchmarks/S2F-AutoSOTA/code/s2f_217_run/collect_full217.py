import csv
import glob
import math
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TAG = (ROOT / "full217_run/RUN_TAG.txt").read_text().strip()
BASE = ROOT / "S3F/results/ResidueTypePrediction/DMSAssay/FusionNetwork"
DATA = ROOT / "S3F/dataset/DMS_ProteinGym_substitutions"
OUT = ROOT / "full217_run/collected"
PRED_OUT = OUT / "raw_predictions"

workdirs = sorted(glob.glob(str(BASE / (TAG + "_b*") / "20*")))
assert len(workdirs) == 8, (len(workdirs), workdirs)
meta = list(csv.DictReader(open(DATA / "DMS_substitutions.csv")))
expected = {row["DMS_id"]: row for row in meta}
assert len(expected) == 217

score_rows = {}
pred_src = {}
for directory in workdirs:
    result = Path(directory) / "results.csv"
    assert result.is_file(), result
    for row in csv.DictReader(open(result)):
        aid = row["DMS_id"]
        assert aid not in score_rows, ("duplicate assay", aid)
        score_rows[aid] = row
    for path in Path(directory).glob("*.csv"):
        if path.name != "results.csv":
            assert path.name not in pred_src, ("duplicate prediction", path.name)
            pred_src[path.name] = path

assert set(score_rows) == set(expected), (set(expected) - set(score_rows), set(score_rows) - set(expected))
assert len(pred_src) == 217, len(pred_src)
PRED_OUT.mkdir(parents=True, exist_ok=True)
for aid, row in expected.items():
    name = row["DMS_filename"]
    assert name in pred_src, (aid, name)
    truth = list(csv.DictReader(open(DATA / name)))
    with open(pred_src[name]) as stream:
        reader = csv.DictReader(stream)
        pred_fields = reader.fieldnames
        predicted = list(reader)
    true_mut = [x["mutant"] for x in truth]
    pred_mut = [x["mutant"] for x in predicted]
    assert len(true_mut) == len(set(true_mut)), ("duplicate input mutation", aid)
    assert len(pred_mut) == len(set(pred_mut)), ("duplicate prediction mutation", aid)
    assert set(pred_mut) == set(true_mut), (aid, len(pred_mut), len(true_mut))
    assert all(math.isfinite(float(x["model_score"])) for x in predicted), ("nonfinite score", aid)
    assert math.isfinite(float(score_rows[aid]["spearmanr"])), ("nonfinite assay score", aid)
    by_mutant = {row["mutant"]: row for row in predicted}
    with open(PRED_OUT / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(by_mutant[mutant] for mutant in true_mut)

with open(OUT / "assay_scores.csv", "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(next(iter(score_rows.values())).keys()))
    writer.writeheader()
    writer.writerows(score_rows[aid] for aid in sorted(score_rows))

scores = pd.read_csv(OUT / "assay_scores.csv")
metadata = pd.read_csv(DATA / "DMS_substitutions.csv")
df = scores[["DMS_id", "spearmanr"]].merge(
    metadata[["DMS_id", "UniProt_ID", "coarse_selection_type"]],
    on="DMS_id", validate="one_to_one")
assert len(df) == 217 and df["spearmanr"].notna().all()
protein = df.groupby(["coarse_selection_type", "UniProt_ID"], as_index=False)["spearmanr"].mean()
category = protein.groupby("coarse_selection_type")["spearmanr"].mean()
official_style = category.mean()
df.to_csv(OUT / "assay_scores_with_metadata.csv", index=False)
category.rename("spearmanr").to_csv(OUT / "category_scores.csv", header=True)
(OUT / "OFFICIAL_STYLE_SCORE.txt").write_text("%.8f\n" % official_style)
print("PASS: 217 assays and 217 mutation files match input rows", flush=True)
print("simple assay mean:", round(df["spearmanr"].mean(), 6), flush=True)
print("category means:\n", category.to_string(), flush=True)
print("official-style aggregate:", round(official_style, 6), flush=True)
