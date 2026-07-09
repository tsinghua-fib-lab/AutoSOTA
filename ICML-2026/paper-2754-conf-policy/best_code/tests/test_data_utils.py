import numpy as np
import pandas as pd

from cpc_llm.data.combine_and_split import append_df_len_to_fp
from cpc_llm.data.synthetic_dataset_formatter import abs_subtract_replace_infs


class TestAppendDfLenToFp:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = append_df_len_to_fp("/path/to/file.jsonl", df)
        assert result == "/path/to/file_3.jsonl"

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = append_df_len_to_fp("data.csv", df)
        assert result == "data_0.csv"

    def test_nested_path(self):
        df = pd.DataFrame({"x": range(100)})
        result = append_df_len_to_fp("/a/b/c/output.jsonl", df)
        assert result == "/a/b/c/output_100.jsonl"


class TestAbsSubtractReplaceInfs:
    def test_normal_values(self):
        assert abs_subtract_replace_infs(3.0, 1.0) == 2.0
        assert abs_subtract_replace_infs(1.0, 3.0) == 2.0

    def test_inf_replaced_with_zero(self):
        # inf is replaced with 0, so abs(0 - 1) = 1
        assert abs_subtract_replace_infs(np.inf, 1.0) == 1.0
        assert abs_subtract_replace_infs(1.0, np.inf) == 1.0

    def test_both_inf(self):
        # Both replaced with 0, so abs(0 - 0) = 0
        assert abs_subtract_replace_infs(np.inf, np.inf) == 0.0

    def test_nan_replaced_with_zero(self):
        assert abs_subtract_replace_infs(np.nan, 2.0) == 2.0

    def test_both_zero(self):
        assert abs_subtract_replace_infs(0.0, 0.0) == 0.0
