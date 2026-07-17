import pandas as pd
import numpy as np
from pandas.api.types import is_integer_dtype
import pytest
from data.jigsaw import build_jigsaw_frame

@pytest.mark.parametrize("missing", ["comment_text", "target", "toxicity_annotator_count"])
def test_build_jigsaw_frame_missing_required_columns_raises(missing):
    df = pd.DataFrame({
        "comment_text": ["a"],
        "target": [0.0],
        "toxicity_annotator_count": [1],
    }).drop(columns=[missing])

    # Expect ValueError for missing required columns
    with pytest.raises(ValueError):
        build_jigsaw_frame(df)


def test_annotation_counts():
    df = pd.DataFrame({
        "comment_text": ["a", "b", "c", "d", "e"],
        "target": [0.0, 0.5, 1.0, 0.49, 0.51],
        "toxicity_annotator_count": [1, 2, 3, 100, 100],
    })

    jf = build_jigsaw_frame(df).reset_index(drop=True)

    # Check that n_yes and n_no are integers and correctly computed
    assert(is_integer_dtype(jf["n_yes"]))
    assert(is_integer_dtype(jf["n_no"]))
    exp_n_yes = np.floor(df["target"] * df["toxicity_annotator_count"] + 0.5).astype("int32")
    pd.testing.assert_series_equal(jf["n_yes"], exp_n_yes, check_names=False)
    pd.testing.assert_series_equal(
        (jf["n_yes"] + jf["n_no"]).astype("int32"),
        df["toxicity_annotator_count"].astype("int32"),
        check_names=False
    )

@pytest.mark.parametrize("annotators", [2, 10, 200])
@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("tie_positive", [True, False])
def test_n_yes_n_no_tie_rule(annotators, eps, tie_positive):
    df = pd.DataFrame({
        "comment_text": ["a"],
        "target": [0.5],
        "toxicity_annotator_count": [annotators],
    })

    jf = build_jigsaw_frame(df, eps=eps, tie_positive=tie_positive).reset_index(drop=True)

    # Check y_star and delta values for tie cases
    assert np.equal(jf.loc[0, "y_star"], 1 if tie_positive else 0)
    assert np.isclose(jf.loc[0, "delta_signed"], 0.0)
    assert np.isclose(jf.loc[0, "abs_delta"], 0.0)
    
@pytest.mark.parametrize("eps", [0, -1e-9])
def test_eps_guard(eps):
    df = pd.DataFrame({
        "comment_text": ["a"],
        "target": [0.5],
        "toxicity_annotator_count": [2],
    })

    # Expect ValueError for non-positive epsilon
    with pytest.raises(ValueError):
        build_jigsaw_frame(df, eps=eps).reset_index(drop=True)

def test_na_leakage():
    df = pd.DataFrame({
        "comment_text": ["keep", "drop_target_none", "drop_count_nan", "drop_target_pdNA"],
        "target": [0.5, None, 0.5, pd.NA],
        "toxicity_annotator_count": [2, 2, np.nan, 2]
    })

    jf = build_jigsaw_frame(df).reset_index(drop=True)

    # Only valid row remains
    assert len(jf) == 1
    assert jf.loc[0, "comment_text"] == "keep"

    # Ensure no NaN values in the resulting DataFrame
    assert not jf.isna().any().any()

    # Additional sanity checks
    assert int(jf.loc[0, "n_yes"] + jf.loc[0, "n_no"]) == 2
    assert np.isclose(jf.loc[0, "delta_signed"], 0.0)
    assert jf.loc[0, "y_star"] == 1