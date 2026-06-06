import asyncio
import multiprocessing as mp
import queue
import re
import warnings
from datetime import datetime
from math import isclose
from typing import Any, Callable

from pebble import ProcessPool

# Useful for `eval` despite not appearing in the code
from sympy import (
    E,
    FiniteSet,
    I,
    Intersection,
    Interval,
    Matrix,
    N,
    Union,
    pi,
    simplify,
    sqrt,
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.exceptions import SymPyDeprecationWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)

from .data import RespSampleBase
from .olympiadbench import OlympiadMathJudger
from .parallel import seq_consume_preset_queue_w_each_timeout
from .grader import math_equal
from .qwen25_utils import qwen25_math_extract

STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS
# Answer prefixes
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


# %% ../nbs/04_eval.ipynb 0
def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None


class EvaluatorBase:
    """Base class for evaluators.

    Parameters
    ----------
    strict_extract: bool, default: False
        Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.
    """

    def __init__(
        self,
        strict_extract: bool = False,
        use_qwen25_math_extract: bool = False,
    ):
        self.strict_extract = strict_extract
        self.use_qwen25_math_extract = use_qwen25_math_extract

    def eval(self, sample: RespSampleBase) -> bool:
        """Evaluate a sample based on comprehensive information."""
        ans = sample.ans if sample.ans is not None else self.extract_ans(sample.resp)
        correct = self.eq(sample.ref_ans, ans)
        return correct

    def eq(self, ref: str, ans: str) -> bool:
        """Check if reference answer and prediction answer are **literally** equal."""
        return ref == ans

    def extract_ans(self, resp_str: str) -> str:
        """Extract answer segment from complete `resp`."""
        if self.use_qwen25_math_extract:
            ans = qwen25_math_extract(resp_str)
            if ans is not None:
                return ans
        ans = self.extract_explicit_ans(resp_str)
        if ans is not None:
            return ans
        elif not self.strict_extract:
            # Speculate with the last latex formula
            matches = re.findall(
                r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
            )
            if len(matches) > 0:
                return matches[-1]
            # Speculate with the last number
            matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
            if len(matches) > 0:
                return matches[-1]
        return ""  # Empty str if no answer is found

    def extract_explicit_ans(self, resp_str: str) -> str:
        resp_str = self.clean_trailing(resp_str)
        # might be answer only
        if "herefore" in resp_str:
            resp_str = resp_str.split("herefore")[-1].strip()
        if GSM8K_ANS_PREFIX in resp_str:
            resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
        if PRM800K_ANS_PRRFIX in resp_str:
            resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()

        if "oxed{" in resp_str:
            resp = extract_boxed(resp_str)
        else:
            resp = resp_str

            # should be answer only
            if "is the ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                    -1
                ].strip()
            elif "is our ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                    -1
                ].strip()
            elif "answer is" in resp:
                resp = resp.split("answer is")[-1].strip()
            elif "answer:" in resp:
                resp = resp.split("answer:")[-1].strip()
            elif "answer :" in resp:
                resp = resp.split("answer :")[-1].strip()
            elif "statement" in resp:
                bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
                if bool_resp is not None:
                    return str(bool_resp)
            else:
                return None

            if resp.startswith("$") and resp.endswith("$"):
                resp = resp[1:-1]

        return resp

    def clean(self, ans: str) -> str:
        """Clean the extracted answer."""

        ans = ans.strip()
        ans = self.clean_preceding(ans)
        ans = self.clean_trailing(ans)

        return ans

    def clean_preceding(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with preceding punctuation marks removed.
        """Removes preceding punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[0] in NO_PRECEDING_PUNCS:
            s = s[1:].strip()

        return s

    def clean_trailing(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with trailing punctuation marks removed.
        """Removes trailing punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[-1] in NO_TRAILING_STRS:
            s = s[:-1].strip()
        return s


DEF_TIMEOUT = 5


class EvaluatorBatchBase(EvaluatorBase):
    """Base class for batch evaluators, providing additional method for batch evaluation.

    Parameters
    ----------
    strict_extract: bool, default: False
        Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.
    timeout : int, default: DEF_TIMEOUT:=5
        The timeout for each evaluation in seconds.
    """

    def __init__(self, strict_extract: bool = False, timeout: int = DEF_TIMEOUT, use_qwen25_math_extract: bool = False):
        EvaluatorBase.__init__(self, use_qwen25_math_extract=use_qwen25_math_extract)
        self.strict_extract = strict_extract
        self.use_qwen25_math_extract = use_qwen25_math_extract
        self.timeout = timeout


    def batch_eval(
        self, samples: list[RespSampleBase], n_procs: int = 2, use_tqdm: bool = True
    ) -> tuple[list[str], list[bool]]:
        """Evaluate a batch of `samples` based on comprehensive information in place."""

        n_samples = len(samples)
        with ProcessPool(max_workers=min(n_procs, n_samples), max_tasks=1024) as pool:
            # NOTE: multi-processing does not support modifications in place
            resps = [sample.resp for sample in samples]
            iterator = pool.map(self.extract_ans, resps, timeout=self.timeout).result()
            answers = []
            pbar = tqdm(total=n_samples, desc="Extracting") if use_tqdm else None
            corrects = []
            while True:
                try:
                    result = next(iterator)
                    answers.append(result)
                except StopIteration:
                    break
                except Exception:
                    answers.append("")
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

            for sample, ans in zip(samples, answers):
                sample.ans = str(ans)

            iterator = pool.map(self.eval, samples, timeout=self.timeout).result()
            pbar = tqdm(total=n_samples, desc="Evaluating") if use_tqdm else None
            corrects = []
            while True:
                try:
                    result = next(iterator)
                    corrects.append(result)
                except StopIteration:
                    break
                except Exception:
                    corrects.append(False)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

            for sample, correct in zip(samples, corrects):
                sample.correct = bool(correct)

        return answers, corrects


def latex2sympy_fix(s: str):
    sp_symbol = parse_latex(s)

    if "," in s:
        first_term = None
        try:
            first_term = parse_latex(s.split(",")[0])
        except Exception:
            pass
        if sp_symbol == first_term:
            raise LaTeXParsingError(f"{s} != {first_term}")

    return sp_symbol


def latex2sympy_interval(s: str):
    """Parse LaTeX expression like (-\\infty,0] as SymPy Interval object."""
    s = s.replace(" ", "")

    if "\\cup" in s:
        exps = s.split("\\cup")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Union(*intervals)

    if "\\cap" in s:
        exps = s.split("\\cap")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Intersection(*intervals)

    if s.startswith("\\{") and s.endswith("\\}"):
        return FiniteSet(simplify(latex2sympy_fix(s[2:-2])))
    elif s.startswith("{") and s.endswith("}"):
        return FiniteSet(simplify(latex2sympy_fix(s[1:-1])))

    if s.startswith("("):
        left_open = True
        s = s[1:]
    elif s.startswith("\\("):
        left_open = True
        s = s[2:]
    elif s.startswith("["):
        left_open = False
        s = s[1:]
    elif s.startswith("\\["):
        left_open = False
        s = s[2:]
    else:
        raise ValueError(f"Invalid interval: {s}")

    if s.endswith(")"):
        right_open = True
        s = s[:-1]
    elif s.endswith("\\)"):
        right_open = True
        s = s[:-2]
    elif s.endswith("]"):
        right_open = False
        s = s[:-1]
    elif s.endswith("\\]"):
        right_open = False
        s = s[:-2]
    else:
        raise ValueError(f"Invalid interval: {s}")

    left, right = s.split(",")
    left = simplify(latex2sympy_fix(left))
    right = simplify(latex2sympy_fix(right))
    if left.is_comparable and right.is_comparable and left >= right:
        raise ValueError(f"Invalid interval: {left}, {right}")
    interval = Interval(left, right, left_open, right_open)

    return interval


PAREN_MAP = {
    r"\(": r"\)",
    r"\[": r"\]",
    r"\{": r"\}",
    "(": ")",
    "[": "]",
    "{": "}",
}

DATETIME_FMTS = [
    # Date formats
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    # Date and time formats
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M",
    # Time formats only
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",  # 24-hour and 12-hour formats
]

BASIC_FN_NAMES = (
    "sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|coth|sech|csch|log|ln|exp"
).split("|")

UNITS = [
    "hour",
    "minute",
    "min",
    "sec",
    "second",
    "day",
    "week",
    "month",
    "year",
    "meter",
    "mile",
    "kg",
    "mg",
    "g",
    "t",
    "ton",
    "nm",
    "pm",
    "um",
    "μm",
    "m",
    "cm",
    "mm",
    "dm",
    "km",
    "kilometer",
    "inch",
    "feet",
    "piece",
    "bit",
    "hz",
    "Hz",
    "m/s",
    "km/s",
    "m/(min^2)",
    "billion",
    "eV",
    "V",
    "C",
    "s",
    r"a\.?m\.?",
    r"(?<!\\)p\.?m\.?",  # 1\pm\sqrt{5}
]


DEF_REL_TOL = 1e-9  # Following `is_close`
# Highest precision: `sys.float_info.epsilon == 2.220446049250313e-16`
DEF_ABS_TOL = 1e-8  # Following OlympiadBench
# https://github.com/OpenBMB/OlympiadBench/blob/1289b12c1067dfe01210f7153bd9ffaaddd42ed5/inference/code/math_judger.py#L33
DEF_PERCENT_REL_TOL = 1e-3


def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False


def is_querying4set(query):
    return "ind the" in query or ("all" in query and "separate" in query)


NDAYS_PER_WEEK = 7
WEEKDAY_ABBRS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_FULLS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def norm_str2weekday(s: str) -> str | None:
    """Converts a string representation of a weekday to its normalized form. Returns `None` if the input is not a valid weekday"""
    s = str(s).lower().strip()
    if " " in s:  # not a word
        return None

    for i_day in range(NDAYS_PER_WEEK):
        if s.startswith(WEEKDAY_ABBRS[i_day]):
            return WEEKDAY_FULLS[i_day].capitalize()
    return None


def parse(parser: Callable, s_to_parse: str, parse_errs: list[Exception]) -> Any | None:
    try:
        return parser(s_to_parse)
    except Exception as e:
        parse_errs.append(e)
    return None


def norm_deg(s: str) -> str:
    """Normalize expressions including degrees, except independent <num>\\circ"""
    s = s.replace("rad", "")
    s = re.sub(r"^(\d+) ?\^?\\?circ$", r"\1", s)
    s = re.sub(r"(\d+) ?\^?\\?circ", r"{\1*\\frac{\\pi}{180}}", s)

    return s


def is_set(s: str):
    return (
        re.search(r"[^a-z]or(x|[^a-z])", s) is not None
        or (s.startswith("{") and s.endswith("}"))
        or (s.startswith("\\{") and s.endswith("\\}"))
    )


def fix_sqrt(
    s: str,
) -> str:
    """Fixes the formatting of square root expressions in a given string."""
    _s = re.sub(r"\\?sqrt[\(\{\[](\w+)[\)\}\]]", r"\\sqrt{\1}", s)
    _s = re.sub(r"\\?sqrt\s*(\d+)", r"\\sqrt{\1}", _s)
    return _s


def fix_fracs(s: str) -> str:
    """Fixes the formatting of fractions in a given string."""
    substrs = s.split("\\frac")
    _s = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            _s += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                _s += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return s
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        _s += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}" + b + post_substr
                    else:
                        _s += "{" + a + "}" + b
    return _s


def fix_a_slash_b(s: str) -> str:
    """
    Fixes the formatting of fractions in a given string using regular expressions.
    """
    # Define a regular expression to match fractions. Here we match two parts: the numerator (a) and the denominator (b).
    # The numerator and denominator can be numbers (\d+) or expressions containing sqrt (sqrt\(.*?\)).
    fraction_pattern = r"(\b\d+|sqrt\(.*?\))\/(\d+|sqrt\(.*?\)\b)"

    # Use `re.sub` to replace the matched fractions with properly formatted fractions.
    result = re.sub(
        fraction_pattern, lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s
    )

    return result


STR2NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX environment from a string.

    Parameters
    ----------
    s : str
        The input string.
    env : str
        The LaTeX environment name to remove.

    Returns
    -------
    str
        The string with the specified LaTeX environment removed.
    """
    s = s.replace(f"\\begin{{{env}}}", "")
    s = s.replace(f"\\end{{{env}}}", "")
    return s


LATEX_CMDS = [
    "\\textbf",
    "\\textit",
    "\\textsl",
    "\\texttt",
    "\\textsc",
    "\\textsf",
    "\\textrm",
    "\\mathrm",
    "\\mathbf",
    "\\mathit",
    "\\mathsf",
    "\\mathtt",
    "\\mathbb",
    "\\mathcal",
    "\\mathscr",
    "\\mathfrak",
    "\\bm",
    "\\em",
    "\\emph",
    "\\underline",
    "\\overline",
    "\\tiny",
    "\\scriptsize",
    "\\footnotesize",
    "\\small",
    "\\normalsize",
    "\\large",
    "\\Large",
    "\\LARGE",
    "\\huge",
    "\\Huge",
    "\\newline",
    "\\par",
    "\\noindent",
    "\\indent",
    "\\footnote",
    "\\cite",
    "\\ref",
    "\\label",
    "\\textsuperscript",
    "\\textsubscript",
    "\\text",
    "\mbox",
    "\\renewcommand{\\arraystretch}",
]

LATEX_FMT_ENVS = [
    # Align
    "align",
    "align*",
    "center",
    "flushleft",
    "flushright",
]
LATEX_LIST_ENVS = [
    "itemize",
    "enumerate",
    "description",
]


SIMPLE_RM_STRS = [
    "\n",
    "\t",
    "approximately",
    "'",
    '"',
    "\\$",
    "$",
    "￥",
    "£",
    "€",
    "{,}",
    "\\!",
    "\\,",
    "\\:",
    "\\;",
    "\\quad",
    "\\qquad",
    "\\space",
    "\\thinspace",
    "\\medspace",
    "\\thickspace",
    "~,",
    "\\ ",
    # Note the order
    "\\\\%",
    "\\%",
    "%",
    "\\left",
    "\\right",
    "^{\\circ}",
    "^\\circ",
]

SIMPLE_REPLACE_MAP = {
    "∪": "\\cup",
    "π": "\\pi",
    "∞": "\\infty",
    "∈": "\\in",
    "∩": "\\cap",
    "−": "-",
    "\\item": ",",
    "and": ",",
    ";": ",",
    "infinity": "\\infty",
    "+\\infty": "\\infty",
    "tfrac": "frac",
    "dfrac": "frac",
    "\\approx": "=",
    "\\times": "*",
    "\\cdot": "*",
    "{.": "{0.",  # "{0." equivalent to "{."
    " .": " 0.",  # " 0." equivalent to " ."
    ":": "/",  # Ratio like 3:2
}


class EvaluatorMath(EvaluatorBase):
    """Evaluator for math problems, capable of extracting answer segment from complex resp and processing various mathematical objects
    (e.g. fractions, symbolic expressions, matrices, vectors) and special text (e.g. bool values).

    Parameters
    ----------
    strict_extract: bool, default: False
        Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.
    use_orig_eq_for_olympiadbench : bool, default: True
        Whether to use the original implementation of `eq` for OlympiadBench.
        For OlympiadBench, by default, we use the official implementation of `eq` by He et al. (2024),
        which utilizing the numerical error range information provided with query,
        but keep the `extract_nas` of ours,
        because the official implementation fails to extract a non-negligible part of answers, especially for base model ICL.
        You could set `use_orig_eq_for_olympiadbench` to `False` to use our implementation of `eq`
        for better consistency across benchmarks in our evaluation setting.
    include_percentage : bool, default: True
        Whether to include percentage comparisons.
    rel_tol : float, default: DEF_REL_TOL
        The relative tolerance for numerical comparisons.
    abs_tol : float, default: DEF_ABS_TOL
        The absolute tolerance for numerical comparisons. Necessary for precision issues.
    percent_rel_tol : float, default: DEF_PERCENT_REL_TOL
        The relative tolerance for percentage comparisons. Relative for different surface forms (e.g. 99% v.s. 0.99).
    ascii_only : bool, default: True
        Only allowing ASCII characters
    """

    def __init__(
        self,
        strict_extract: bool = False,
        use_orig_eq_for_olympiadbench: bool = True,
        include_percentage: bool = True,
        rel_tol: float = DEF_REL_TOL,
        abs_tol: float = DEF_ABS_TOL,
        percent_rel_tol: float = DEF_PERCENT_REL_TOL,
        ascii_only: bool = True,
        use_qwen25_math_extract: bool = False,
    ):
        EvaluatorBase.__init__(self, use_qwen25_math_extract=use_qwen25_math_extract)
        self.strict_extract = strict_extract
        self.use_qwen25_math_extract = use_qwen25_math_extract
        if use_orig_eq_for_olympiadbench:
            self.olympiad_math_judger = OlympiadMathJudger()
        else:
            self.olympiad_math_judger = None

        self.include_percentage = include_percentage
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.percent_rel_tol = percent_rel_tol
        self.ascii_only = ascii_only

    def eval(self, sample: RespSampleBase) -> tuple[str, bool]:
        if sample.ans is None:
            ans = self.extract_ans(sample.resp)
        else:
            ans = sample.ans

        # if getattr(sample, "finish_reason", None) in [
        #     "length",
        #     "abort",
        # ]:
        #     correct = False
        if getattr(sample, "finish_reason", None) in [
            "abort",
        ]:
            correct = False
        elif self.olympiad_math_judger is None or "olympiad" not in sample.dataset:
            correct = self.eq(sample.ref_ans, ans, is_querying4set(sample.query))
        else:  # OlympiadBench Correctness Judger
            abs_tol = getattr(sample, "abs_tol", None)
            if abs_tol is None:
                correct = self.olympiad_math_judger.judge(sample.ref_ans, ans)
            else:
                abs_tol = (
                    float(sample.abs_tol)
                    if isinstance(sample.abs_tol, str)
                    else sample.abs_tol
                )
                correct = self.olympiad_math_judger.judge(sample.ref_ans, ans, abs_tol)

        return correct

    def extract_ans(self, resp_str: str) -> str:
        raw_ans = EvaluatorBase(use_qwen25_math_extract=self.use_qwen25_math_extract).extract_ans(resp_str)
        math_ans = self.norm_ans_str(raw_ans)
        return math_ans

    def eq(
        self,
        ref: float | str,  # The reference answer value.
        pred: bool | float | str,  # The predicted answer value.
        compare_sets: bool = False,  # Whether to compare sets of values.
    ) -> bool:  # True if the values are mathematically equal, False otherwise.
        """
        Check if two values are mathematically equal.
        Return `False` by default.
        Notes:
        - The function checks for three types of equality: literal equality, numerical equality, and symbolic equality.
        - If the reference value is a list of two elements, the second element is treated as the numerical reference value.
        - The function normalizes the input strings before performing comparisons.
        - If compare_sets is True, the function compares sets of values instead of individual values.
        - If timeout is True, the function uses a timeout for symbolic comparisons.
        """
        if self.use_qwen25_math_extract:
            return math_equal(pred, ref)
        if isinstance(ref, list) and len(ref) == 2:
            ref, ref_num = ref
        else:
            ref_num = None

        if ref is None:
            return None

        if pred is None:
            return False

        # datetime
        pred_datetime = self.norm_str2date_time(str(pred))
        ref_datetime = self.norm_str2date_time(str(ref))
        if (
            pred_datetime is not None
            and ref_datetime is not None
            and pred_datetime == ref_datetime
        ):
            return True  # Stricter than ratio

        # 0. Normalize
        pred_str = self.norm_ans_str(pred)
        ref_str = self.norm_ans_str(ref)

        if len(pred_str) == 0:
            return False

        # NOTE: some non-ASCII characters are also allowed for control, they should be removed by the above normalization
        if self.ascii_only and has_non_ascii(pred_str):
            return False

        # 1. literally equal
        lower_pred = pred_str.lower()
        lower_ref = ref_str.lower()
        if lower_pred == lower_ref:
            return True
        if compare_sets:
            preds = self.extract_set(pred_str)
            refs = self.extract_set(ref_str)

            if len(preds) != len(refs):
                return False
            for pred in preds:
                exist = False
                for ref in refs:
                    exist = self.eq(
                        pred,
                        ref,
                        compare_sets=False,
                    )
                    if exist:
                        break
                if not exist:
                    return False
                refs.remove(ref)
            return True

        pred_parse_errs = []
        ref_parse_errs = []

        # 2. Numerically equal
        # no `norm_float_str` for possible mistakes like "123,456"(123 and 456) -> "123456"
        pred_num = parse(float, pred_str, pred_parse_errs)
        if ref_num is None:
            ref_num = parse(float, ref_str, ref_parse_errs)
        num_eq = self.is_num_eq(ref_num, pred_num)
        if num_eq is not None:
            return num_eq

        # 3. Symbolically equal (w/ SymPy and antlr4)
        # Return `True` if the two expressions can be interpreted as equal in **any** unified form.
        # NOTE: possible ambiguity 1,234 -> (1,234) / 1234 ?

        # 3.1 Python object
        # NOTE: parse_expr("1,234") == (1, 234)
        pred_obj = parse(parse_expr, pred_str, pred_parse_errs)
        ref_obj = parse(parse_expr, ref_str, ref_parse_errs)
        # print(pred_obj, ref_obj, symbol_equal(pred_obj, ref_obj))  # debug
        if pred_obj is not None and ref_obj is not None and pred_obj == ref_obj:
            return True

        # 3.2 SymPy object
        # ImportError: LaTeX parsing requires the antlr4 Python package, provided by pip (antlr4-python3-runtime) or conda (antlr-python-runtime), version 4.11
        pred_spobj = parse(latex2sympy_interval, pred_str, pred_parse_errs)
        ref_spobj = parse(latex2sympy_interval, ref_str, ref_parse_errs)
        # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True

        pred_spobj = parse(self.latex2matrix, pred_str, pred_parse_errs)
        ref_spobj = parse(self.latex2matrix, ref_str, ref_parse_errs)
        # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True

        # WARNING: parse_latex("a,b") -> a but parse_latex("1,234") -> 1234, `latex2sympy_fix` fixed the former by raising a `LaTeXParsingError``
        pred_spobj = parse(latex2sympy_fix, pred_str, pred_parse_errs)
        ref_spobj = parse(latex2sympy_fix, ref_str, ref_parse_errs)
        # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True

        if (
            pred_spobj is not None
            and ref_obj is not None
            and self.is_sym_eq(pred_spobj, ref_obj)
        ):
            return True

        if (
            pred_obj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_obj, ref_spobj)
        ):
            return True

        n_checks = 5
        expr_parse_errs = {}
        if len(pred_parse_errs) == n_checks:
            expr_parse_errs["pred"] = pred_parse_errs
        if len(ref_parse_errs) == n_checks:
            expr_parse_errs["ref"] = ref_parse_errs

        # print(expr_parse_errs)
        if len(expr_parse_errs) > 0:
            raise ValueError(expr_parse_errs)
        else:
            return False

    def could_be_percent(self, v) -> bool:
        """Check if a value could be a percentage."""
        return 0 < v < 1 or 1 < v < 100

    def is_num_eq(self, ref_num, pred_num) -> bool | None:
        """Compare two numbers with specified feautures:
        - relative tolerance
        - flexible percentage surface forms
        """
        if ref_num is None or pred_num is None:
            return None
        if isclose(ref_num, pred_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return True

        if self.include_percentage and self.could_be_percent(pred_num):
            percent_ref_nums = [
                num
                for num in [ref_num / 100, ref_num * 100]
                if self.could_be_percent(num)
            ]
            for item in percent_ref_nums:
                # "For the values to be considered close, the difference between them must be smaller than at least one of the tolerances."
                if isclose(
                    item, pred_num, rel_tol=self.percent_rel_tol, abs_tol=self.abs_tol
                ):
                    return True
        return None

    def norm_ans_str(self, ans: str) -> str:
        """Normalize answer string for **all kinds** of answers."""
        ans = str(ans)
        ans = ans.replace("\n", "")  # no answer must need \n
        ans = ans.strip()

        # remove impropriate trailing punctuations
        ans = self.clean(ans)

        # cornor cases

        # bool
        ans_bool = norm_str2bool(ans)
        if ans_bool is not None:
            return str(ans_bool)

        # weekdays
        ans_weekday = norm_str2weekday(ans)
        if ans_weekday is not None:
            return ans_weekday

        # math normalize
        ans = self.norm_math_str(ans)

        return ans

    def latex2matrix(self, latex_mat_str: str):
        """This function convert latex matrix into sympy matrix (always 2)"""
        if not isinstance(latex_mat_str, str):
            raise ValueError(f"{latex_mat_str} is not a `str`!")
        latex_mat_str = latex_mat_str.replace(" ", "")

        pattern = r"(?:\[|\()?\\begin{[a-zA-Z]?(?:matrix|array)}(?:\[lcr\])*?(.*)\\end{[a-zA-Z]?(?:matrix|array)}(?:\]|\))?"
        data = re.search(pattern, latex_mat_str)
        python_matrix = []
        if data is not None:
            data = data[1]
            # \+ not followed by frac or sqrt
            rows = re.split(r"\\+(?!frac|sqrt)", data)
            for row in rows:
                elements_list = row.split("&")
                python_matrix.append(elements_list)
        else:
            if "," in latex_mat_str:
                if is_set(latex_mat_str):
                    # print("set")
                    python_matrix = [self.extract_set(latex_mat_str)]
                else:
                    python_matrix = [self.remove_out_paren(latex_mat_str).split(",")]
            else:
                raise LaTeXParsingError(
                    f"{latex_mat_str} can not be parsed in a `Matrix`!"
                )

        # print(data)
        # print(python_matrix)
        sympy_matrix = []
        for row in python_matrix:
            # print(row)
            sympy_row = [latex2sympy_fix(element) for element in row]
            sympy_matrix.append(sympy_row)

        matrix = Matrix(sympy_matrix)

        # print(s)
        # unify one row/col into vector
        if len(matrix.shape) == 2 and matrix.shape[1] == 1:
            matrix = matrix.T
        return matrix

    def remove_latex_cmd(self, s: str, cmd: str) -> str:
        try:
            cmd_idx = s.index(cmd)
        except ValueError:
            return s

        pfx = s[:cmd_idx].strip()
        sfx = s[cmd_idx + len(cmd) :].strip()

        if len(sfx) > 0 and sfx[0] == "{":  # Common command
            sfx = self.remove_first_paren_pair(sfx, "{")
        elif len(pfx) > 0 and pfx[-1] == "{":  # Declaration command
            left_idx_in_sfx = sfx.find("}")
            if left_idx_in_sfx != -1:
                pfx = pfx[:-1]
                sfx = sfx[:left_idx_in_sfx] + sfx[left_idx_in_sfx + 1 :]
        else:  # Indepedent command
            pass

        return pfx + sfx

    def is_sym_eq(self, a: Any, b: Any) -> bool | None:
        """Compare two objects symbolically."""
        if a is None or b is None:
            return None

        try:
            if a == b:
                return True
        except Exception:
            pass

        try:
            diff = simplify(a - b)
            # For non-symmetric operations like subtraction between sets
            diff_rev = simplify(b - a)

            if hasattr(diff, "__iter__") and hasattr(
                diff_rev, "__iter__"
            ):  # If diff is iterable (e.g. Matrix)
                if all(element == 0 for element in diff) and all(
                    element == 0 for element in diff_rev
                ):
                    return True
            else:
                if (
                    not diff and not diff_rev
                ):  # use `not` for non-zero values like `sympy.EmptySet`
                    return True
        except Exception:
            pass

        try:
            v_a, v_b = (N(eval(str(v))) for v in [a, b])
            num_eq = self.is_num_eq(v_a, v_b)
            if num_eq:
                return True
        except Exception:
            pass

        return None

    def norm_str2date_time(self, string: str):
        """Normalize date or time string to a standard and precise format."""

        for fmt in DATETIME_FMTS:
            try:
                dt = datetime.strptime(string, fmt)
                has_time, has_date = ":" in string, "/" in string or "-" in string
                if has_date and has_time:
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                elif has_date:
                    return dt.strftime("%Y-%m-%d")
                elif has_time:
                    return dt.strftime("%H:%M:%S")
                else:
                    pass
            except ValueError:
                continue
        return None

    def index_first_paren_pair(self, s: str, l: str) -> tuple[int, int]:
        r = PAREN_MAP[l]
        try:
            i_l = s.index(l)
        except ValueError:
            return -1, -1
        len_paren = len(l)

        depth = 0
        i_r = -1
        for i_c in range(i_l, len(s)):
            if s[i_c : i_c + len_paren] == l:
                depth -= 1
            elif s[i_c : i_c + len_paren] == r:
                depth += 1
            if depth == 0:
                i_r = i_c
                break

        return i_l, i_r

    def remove_first_paren_pair(
        self,
        s: str,
        l: str,  # Left parenthesis
    ) -> str:
        i_l, i_r = self.index_first_paren_pair(s, l)
        if i_l != -1 and i_r != -1:
            len_paren = len(l)
            s = s[:i_l] + s[i_l + len_paren : i_r] + s[i_r + len_paren :]

        return s

    def remove_out_paren(self, s: str) -> str:
        """Remove until there are no parentheses outside."""
        done = False
        while not done:
            done = True
            for left, _ in PAREN_MAP.items():
                len_paren = len(left)
                i_l, i_r = self.index_first_paren_pair(s, left)
                if i_l == 0 and i_r == len(s) - len_paren:
                    s = s[len_paren:-len_paren]
                    done = False
        return s

    def extract_set(self, norm_s: str) -> list[str]:
        clean_s = self.remove_out_paren(norm_s)
        ele_strs = clean_s.replace("or", ",").split(",")
        ele_strs = [s.strip() for s in ele_strs]

        # ele_strs.sort()
        # return ele_strs

        merged_strs = []
        for i in range(len(ele_strs)):
            s_i = ele_strs[i]
            existing = False
            for j in range(i):
                s_j = ele_strs[j]
                if self.eq(s_i, s_j):
                    existing = True
                    break
            if not existing:
                merged_strs.append(s_i)

        merged_strs.sort()

        return merged_strs

    def norm_basic_fn(self, s: str) -> str:
        """Avoid potential LaTex errors caused by removing spaces:
        - \\{fn}[a-z] : followed by some letter without middle spaces
        - \\{fn}^{pow}{expr}

        Returns
        -------
        str
            Normalized format of basic function expression: \\{fn}^{{pow}}{{expr}}
        """
        # \2 matches \d+ without {} around, if there has been {}, there is no need to normalize
        # Existing nude power, i.e. ^<pow_d+>
        s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})\^(\d+)", r"\\\1^{\2}", s)
        # No power
        s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})(?!\^)", r"\\\1^{1}", s)
        return s

    def norm_pm(self, s: str) -> str:
        """Replaces the LaTeX symbols '$1\\pm$2' or '$1\\mp$2' with '$1-$2,$1+$2'."""

        def replace_pm(match):
            # Extracts the first and second parts of the match.
            first_part, second_part = match.groups()
            # Creates the replacement string as specified.
            return f"{first_part}-{second_part},{first_part}+{second_part}"

        _s = self.remove_out_paren(s)
        # Define the pattern that matches '$1\\pm$2' or '$1\\mp$2'.
        # We use non-greedy matching (.*?) to capture the parts before and after \pm or \mp.
        # The pattern is corrected to include the '$' signs and to capture the expressions correctly.
        pattern = r"([\w\.\\{}\+\-\*\^]+?)(?:\\pm|\\mp)([\w\.\\{}\+\-\*\^]+)"

        if re.search(pattern, _s):
            # Use re.sub to replace all occurrences of the pattern in the input string.
            return re.sub(pattern, replace_pm, _s)
        else:
            return s

    def norm_math_str(self, string: str):
        # delay logics for multi-choice to after extraction from model output
        # lower_str = string.lower()
        # for choice in ALL_CHOICES:
        #     choice_lower = choice.lower()
        #     if lower_str == choice_lower or lower_str == f"({choice_lower})":
        #         return choice

        # Replacement-based normalization

        string = str(string).strip()
        string = self.clean(string)

        # Simple removals
        for rm_str in SIMPLE_RM_STRS:
            string = string.replace(rm_str, "")

        # Simple replacements
        for k, v in SIMPLE_REPLACE_MAP.items():
            string = string.replace(k, v)
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")

        # Remove spaces after all space-related operations
        string = string.replace(" ", "")

        for latex_cmd in LATEX_CMDS:
            string = self.remove_latex_cmd(string, latex_cmd)

        for env in LATEX_FMT_ENVS + LATEX_LIST_ENVS:
            string = rm_latex_env(string, env)

        # Normalize local expressions
        string = norm_deg(string)  # Normalize degrees
        string = re.sub(
            rf"(?<!\\)(pi\b|{'|'.join(BASIC_FN_NAMES)})", r"\\\1", string
        )  # Fix backslashes
        string = self.norm_basic_fn(string)  # Normalize basic functions

        # Normalize matrix and array
        string = re.sub(r"{[a-z]?matrix}", r"{array}", string)
        string = re.sub(r"\\begin{array}{[lcr]*}", r"\\begin{array}{}", string)
        # NOTE: the substituion str should alse obey the regex syntax, like r"\\begin{array}"
        if "\\begin{array}" not in string:
            string = string.replace("\\\\", "")

        # i, j
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = re.sub(r"(\d+)\.0+$", r"\1", string)

        # remove units
        for unit in UNITS:
            string = re.sub(f"([-\d\.\*\^{{}}]+){unit}e?s?$", "\\1", string)

        # Check if empty before splitting
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # Splitting-based normalization

        # Process complex expressions without parentheses
        s_is_set = is_set(string)
        if s_is_set:
            raw_strings = self.extract_set(string)
        else:
            raw_strings = [string]

        strings = []
        for string in raw_strings:
            string = fix_sqrt(string)

            if string.startswith("frac"):
                string = "\\" + string
            # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
            string = fix_fracs(string)

            # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
            string = fix_a_slash_b(string)

            string = re.sub(r"^[a-z]\\in", "", string)

            if "," not in string:
                string = self.remove_out_paren(string)

            if "\\begin{array}" not in string:
                # to consider: get rid of chain of equalities like "a = b = c = d"
                if len(string.split("=")) > 2:
                    string = string.split("=")[-1]

                # to consider: get rid of e.g. "k = " or "q = " at beginning
                if len(string.split("=")) == 2:
                    first_part = string.split("=")[0].strip()
                    if (
                        re.match(
                            r"^([a-z]|[A-Z]{2}|\\?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sin|cos|sec|csc|tan|cot|sinh|cosh|sech|csch|tanh|coth|log|ln|exp))\^?{?-?('|\\prime|\d)*}?(\(-?([\d\.]+|[a-z])?\))?$",
                            first_part,
                        )
                        is not None
                    ):
                        string = string.split("=")[1]

                # to consider: get rid of equalities but not equations
                if len(string.split("=")) == 2:
                    if len(re.findall(r"[a-zA-Z]", string.split("=")[0].strip())) == 0:
                        string = string.split("=")[1]
            # replace \pm with +,-
            # string = re.sub(r"(.*?)\\pm(.+?)", r"\1-\2,\1+\2", string)
            string = self.norm_pm(string)  # might add comma ","

            string = re.sub(r"^0+([1-9])", r"\1", string)

            strings.append(string)
        string = ",".join(strings)

        if "," not in string:
            string = self.remove_out_paren(string)

        if STR2NUM.get(string):
            string = str(STR2NUM[string])

        # add space
        string = re.sub(r"\\mid([a-z])", r"\\mid \1", string)
        string = self.clean(string)

        # If there are multiple same inequality signs and no commas
        for ineq in ["<", ">"]:
            if len(re.findall(f"{ineq}=?", string)) > 1 and not any(
                delim in string.lower() for delim in [",", "and", "or"]
            ):
                string = string.replace(ineq, ",")

        return string


class EvaluatorMathBatch(EvaluatorMath, EvaluatorBatchBase):
    """Batch evaluator for math problems, capable of extracting answer segment from complex resp and processing various mathematical objects
    (e.g. fractions, symbolic expressions, matrices, vectors) and special text (e.g. bool values).

    Parameters
    ----------
    strict_extract: bool, default: False
        Whether to extract answers strictly. If `False`, speculate the answer from the last number if needed.
    timeout : int, default: DEF_TIMEOUT:=5
    use_orig_eq_for_olympiadbench : bool, default: True
        Whether to use the original implementation of `eq` for OlympiadBench.
        For OlympiadBench, by default, we use the official implementation of `eq` by He et al. (2024),
        which utilizing the numerical error range information provided with query,
        but keep the `extract_nas` of ours,
        because the official implementation fails to extract a non-negligible part of answers, especially for base model ICL.
        You could set `use_orig_eq_for_olympiadbench` to `False` to use our implementation of `eq`
        for better consistency across benchmarks in our evaluation setting.
    include_percentage : bool, default: True
        Whether to include percentage comparisons.
    rel_tol : float, default: DEF_REL_TOL
        The relative tolerance for numerical comparisons.
    abs_tol : float, default: DEF_ABS_TOL
        The absolute tolerance for numerical comparisons. Necessary for precision issues.
    percent_rel_tol : float, default: DEF_PERCENT_REL_TOL
        The absolute tolerance for percentage comparisons.
    ascii_only : bool, default: True
        Only allowing ASCII characters
    """

    def __init__(
        self,
        strict_extract: bool = False,
        use_orig_eq_for_olympiadbench: bool = True,
        include_percentage: bool = True,
        rel_tol: float = DEF_REL_TOL,
        abs_tol: float = DEF_ABS_TOL,
        percent_rel_tol: float = DEF_PERCENT_REL_TOL,
        ascii_only: bool = True,
        timeout: int = DEF_TIMEOUT,
        use_qwen25_math_extract: bool = False,
    ):
        
        EvaluatorMath.__init__(
            self,
            use_orig_eq_for_olympiadbench=use_orig_eq_for_olympiadbench,
            include_percentage=include_percentage,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            percent_rel_tol=percent_rel_tol,
            ascii_only=ascii_only,
            use_qwen25_math_extract=use_qwen25_math_extract,
        )
        EvaluatorBatchBase.__init__(self, timeout=timeout, use_qwen25_math_extract=use_qwen25_math_extract)
        self.strict_extract = strict_extract
        self.use_qwen25_math_extract = use_qwen25_math_extract