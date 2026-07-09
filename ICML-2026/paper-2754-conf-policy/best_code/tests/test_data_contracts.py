from cpc_llm.data_contracts import (
    CON_LIK_PREFIX,
    LIK_PREFIX,
    con_lik_col,
    lik_col,
)


class TestLikColHelpers:
    def test_lik_col(self):
        assert lik_col(0) == "lik_r0"
        assert lik_col(5) == "lik_r5"

    def test_con_lik_col(self):
        assert con_lik_col(0) == "con_lik_r0"
        assert con_lik_col(3) == "con_lik_r3"

    def test_lik_col_uses_prefix(self):
        assert lik_col(0).startswith(LIK_PREFIX)

    def test_con_lik_col_uses_prefix(self):
        assert con_lik_col(0).startswith(CON_LIK_PREFIX)

    def test_generated_cols_are_sequential(self):
        cols = [lik_col(i) for i in range(5)]
        assert cols == ["lik_r0", "lik_r1", "lik_r2", "lik_r3", "lik_r4"]
