import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

featurizer = ElementProperty.from_preset("magpie", impute_nan=True)

df = pd.read_csv("./data/bandgap.csv")
all_feats = []
nan_count = 0
for formula in df["material formula"]:
    try:
        comp = Composition(formula)
        feats = featurizer.featurize(comp)
        all_feats.append(feats)
        if any(np.isnan(f) for f in feats):
            nan_count += 1
    except Exception as e:
        print("Error for {}: {}".format(formula, e))
        all_feats.append([np.nan]*132)

all_feats = np.array(all_feats)
print("Total featurized:", len(all_feats))
print("NaN rows (any):", nan_count)
nan_cols = np.where(np.isnan(all_feats).any(axis=0))[0]
print("Total columns with NaN:", len(nan_cols))
if len(nan_cols) > 0:
    print("NaN count per feature column (first 10):")
    for col in nan_cols[:10]:
        print("  Col {}: {} NaN".format(col, np.isnan(all_feats[:, col]).sum()))

valid = all_feats[~np.isnan(all_feats).any(axis=1)]
print("Valid rows:", len(valid))
print("Feature mean range: [{:.2f}, {:.2f}]".format(np.nanmin(all_feats), np.nanmax(all_feats)))
feat_stds = np.nanstd(all_feats, axis=0)
print("Feature std range: [{:.4f}, {:.2f}]".format(np.nanmin(feat_stds), np.nanmax(feat_stds)))
