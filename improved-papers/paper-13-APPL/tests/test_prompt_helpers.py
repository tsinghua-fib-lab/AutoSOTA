from typing import Dict, Optional

from appl import BracketedDefinition as Def
from appl import define, define_bracketed, ppl, records
from appl.compositor import *


def test_compositor():
    @ppl
    def func():
        with NumberedList(indent=INDENT, sep="???\n"):
            with LetterList(indent=INDENT, sep="!!!\n"):
                with DashList(indent=INDENT, sep="|||\n"):
                    "a"
                "b"
            "c"
        return records()

    assert str(func()) == f"{INDENT * 3}- a!!!\n{INDENT * 2}A. b???\n{INDENT}1. c"


def test_iter():
    @ppl
    def func():
        for i in iter(range(3), compositor=RomanList()):
            f"item {i}"
        return records()

    assert str(func()) == f"I. item 0\nII. item 1\nIII. item 2"


def test_logged():
    @ppl
    def func():
        with Logged(prolog="start", epilog="end"):
            "a"
            "b"
        return records()

    assert str(func()) == f"start\na\nb\nend"


def test_tagged():
    @ppl
    def func(attrs: Optional[Dict] = None):
        with Tagged("tag1", attrs=attrs, indent_inside=4):
            "a"
            "b"
        return records()

    @ppl
    def inline():
        with InlineTagged("tag2"):
            "hello"
            "world"
        return records()

    content = "    a\n    b\n"
    assert str(func()) == f"<tag1>\n{content}</tag1>"
    assert str(func({"x": "1"})) == f'<tag1 x="1">\n{content}</tag1>'
    assert str(inline()) == f"<tag2>helloworld</tag2>"


def test_definition():
    # recommended way to define
    class ADef(Def):
        name = "A"
        fstr = "({})"  # overwrite the format string

    # A shorter way to define
    BDef = define_bracketed("B")
    DDef = define("D")

    @ppl
    def func():
        ADef(desc="This is A.")
        BDef(details="This is B.")
        (c := Def("C", "This is C."))  # inline definition
        DDef(desc="This is D.", sep=":: ")
        f"{ADef}{BDef}{c}{DDef}"
        return records()

    target = [
        "A: This is A.",
        "B: \nThis is B.",
        "C: This is C.",
        "D:: This is D.",
        "(A)[B][C]D",
    ]
    assert str(func()) == "\n".join(target)


def test_compositor_decorator():
    @ppl(compositor=NumberedList())
    def func():
        f"first line"
        f"second line"
        with IndentedList():
            "third line"
            "fourth line"
        return records()

    assert (
        str(func())
        == f"1. first line\n2. second line\n{INDENT}third line\n{INDENT}fourth line"
    )
