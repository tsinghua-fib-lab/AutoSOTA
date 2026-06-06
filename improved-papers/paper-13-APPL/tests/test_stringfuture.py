import time

import appl
from appl import CallFuture
from appl import StringFuture as S


def _check_str(s: S, target: str):
    assert type(s) is S
    assert str(s) == target


def test_concat():
    _check_str(S("ab") + S("cd"), "abcd")
    _check_str(S("AP") + "PL", "APPL")
    _check_str("AP" + S("PL"), "APPL")
    s = ""
    s += S("AP")
    _check_str(s, "AP")
    s += "PL"
    _check_str(s, "APPL")
    s += S("E")
    _check_str(s, "APPLE")


def test_cmp():
    assert S("APPL") == "APPL"
    assert "APPL" == S("APPL")
    assert S("APPL") == S("APPL")

    assert S("APPL") != "APPLE"
    assert "APPL" != S("APPLE")
    assert S("APPL") != S("APPLE")

    assert "APPL" >= S("AAA")
    assert S("APPL") >= S("AAA")
    assert S("APPL") >= "AAA"
    assert S("APPL") > "AAA"
    assert S("APPL") <= "ZZZ"
    assert S("APPL") < "ZZZ"


def test_contains():
    assert "AP" in S("APPL")
    assert "AP" not in S("AAA")


def test_getitem():
    assert S("APPL")[1] == "P"
    assert S("APPL")[1:] == "PPL"
    assert S("A.P.P.L").split(".") == ["A", "P", "P", "L"]


def test_format():
    assert f"{S('A'):3}" == "A  "
    assert appl.format(1.234, ".2f") == "1.23"
    assert isinstance(appl.format(1.234, ".2f"), S)


def test_as_key():
    d = {}
    d[S("APPL")] = 1
    assert d[S("APPL")] == 1
    assert d["APPL"] == 1


def test_future():
    def run(t):
        time.sleep(t)
        return "a"

    t0 = time.time()
    n = 3
    s = ""
    for i in range(n):
        s += S(CallFuture(run, t=0.2))
    assert time.time() - t0 < 0.15
    assert str(s) == "a" * n
    assert time.time() - t0 < 0.4
