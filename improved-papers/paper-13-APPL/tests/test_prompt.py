import pytest

from appl import Generation, convo, define, gen, need_ctx, ppl, records
from appl.compositor import *


def test_return():
    @ppl
    def func():
        "Hello World"
        return "answer"

    assert func() == "answer"


def test_prompt():
    @ppl
    def func(_ctx):
        "Hello World"
        return records()

    assert str(func()) == "Hello World"


def test_fstring():
    @ppl
    def f1():
        f"a is {1!r}"
        return records()

    assert str(f1()) == f"a is {1!r}"

    @ppl
    def f2():
        f"a is {3.1415:.2f}"
        return records()

    assert str(f2()) == f"a is {3.1415:.2f}"


def test_prompts_change():
    @ppl
    def func():
        "Hello"
        ret1 = records()  # the reference
        ret2 = records().copy()  # the copy of the current prompt
        "World"
        ret3 = records()
        return ret1, ret2, ret3

    ret1, ret2, ret3 = func()
    assert str(ret1) == "Hello\nWorld"
    assert str(ret2) == "Hello"
    assert str(ret3) == "Hello\nWorld"


def test_return_prompt():
    @ppl(default_return="prompt")
    def f1():
        "Hello World"

    assert str(f1()) == "Hello World"

    @ppl(default_return="prompt")
    def f2():
        "Hello World"
        return "answer"

    # The return is unchanged.
    assert str(f2()) == "answer"


def test_record():
    @ppl
    def f2():
        "Hello"
        "World"
        return records()

    @ppl
    def func():
        with NumberedList():
            "first line"
            "second line"
            f2()  # add the prompts from f2, following the current format.
        return records()

    assert str(func()) == f"1. first line\n2. second line\n3. Hello\n4. World"


def test_inner_func():
    @ppl
    def func():
        "Hello"

        def func2():  # the inner function use the same context from the outer function.
            "World"

        func2()
        return records()

    assert str(func()) == "Hello\nWorld"


def test_tripple_quote():
    @ppl
    def func1():
        """This is a docstring"""
        """
        begin
            1. first
            2. second
        """
        return records()

    @ppl
    def func2():
        """This is a docstring"""
        # Note that dedent will remove the leading spaces.
        """begin
            1. first
            2. second
        """
        return records()

    @ppl
    def func3():
        """This is a docstring"""
        # Not recommended
        """
    begin
        1. first
        2. second
        """  # note the leading spaces here
        return records()

    @ppl
    def func4():
        """This is a docstring"""
        # Not recommended, but still works.
        """begin
1. first
2. second
"""
        return records()

    assert str(func1()) == "begin\n    1. first\n    2. second"
    assert str(func2()) == "begin\n1. first\n2. second"
    assert str(func3()) == "begin\n    1. first\n    2. second\n    "
    assert str(func4()) == "begin\n1. first\n2. second"


def test_tripple_quote_fstring():
    @ppl
    def func1():
        x = "end"
        f"""
        begin
        {x}
        """
        return records()

    @ppl
    def func2():
        x = f"""
        begin
            Hello
            
            World
        """
        f"""
        {x}
        end
        """
        return records()

    @ppl
    def func3():
        f"""
        1+1={1 + \
                1
        }
        """
        # The recovered code from libcst will become:
        #         f"""
        #         1+1={1 + \
        #         1
        # }
        #         """
        return records()

    assert str(func1()) == "begin\nend"
    assert str(func2()) == "begin\n    Hello\n\n    World\nend"
    assert str(func3()) == "1+1=2"


def test_include_docstring():
    @ppl(docstring_as="system")
    def func1():
        """This is a docstring"""
        "Hello"
        return records()

    assert func1().as_convo().as_list() == [
        {"role": "system", "content": "This is a docstring"},
        {"role": "user", "content": "Hello"},
    ]

    @ppl(docstring_as="user")
    def func2():
        """This is a docstring"""
        "Hello"
        return records()

    assert str(func2()) == "This is a docstring\nHello"


def test_include_multiline_docstring():
    @ppl(docstring_as="user")
    def func():
        """This is a
        multiline docstring"""

        "Hello"
        return records()

    assert str(func()) == "This is a\nmultiline docstring\nHello"

    @ppl(docstring_as="user")
    def func2():
        """
        This is a
            multiline docstring
        """
        return records()

    assert str(func2()) == "This is a\n    multiline docstring"


def test_default_no_docstring():
    @ppl()
    def func():
        """This is a docstring"""
        "Hello"
        return records()

    @ppl()
    def func2():
        """Same string"""  # the first is docstring
        # the second string is not a docstring anymore, should be included
        """Same string"""
        return records()

    assert str(func()) == "Hello"
    assert str(func2()) == "Same string"


def test_copy_ctx():
    @ppl(ctx="copy")
    def addon():
        "World"
        return str(convo())

    @ppl
    def func2():
        "Hello"
        first = addon()
        second = addon()
        return first, second, records()

    first, second, origin = func2()
    assert first == "Hello\nWorld"
    assert second == "Hello\nWorld"
    assert str(origin) == "Hello"


def test_resume_ctx():
    @ppl(ctx="resume")
    def resume_ctx():
        "Hello"
        return convo()

    target = []
    for i in range(3):
        res = resume_ctx()
        target += ["Hello"]
        assert str(res) == "\n".join(target)


def test_class_resume_ctx():
    class A:
        @ppl(ctx="resume")
        def append(self, msg: str):
            msg
            return convo()

        @classmethod
        @ppl(ctx="resume")
        def append_cls(cls, msg: str):
            msg
            return convo()

    a = A()
    b = A()
    target_a = []
    target_b = []
    target_cls = []
    for i in range(3):
        res = a.append("Hello")
        target_a += ["Hello"]
        assert str(res) == "\n".join(target_a)
        res = b.append("World")
        target_b += ["World"]
        assert str(res) == "\n".join(target_b)
        res = A.append_cls("Class")
        target_cls += ["Class"]
        assert str(res) == "\n".join(target_cls)


def test_class_func():
    class ComplexPrompt:
        def __init__(self, condition: str):
            self._condition = condition

        @ppl(ctx="same")
        def sub1(self):
            if self._condition:
                "sub1, condition is true"
            else:
                "sub1, condition is false"

        @ppl(ctx="same")
        def sub2(self):
            if self._condition:
                "sub2, condition is true"
            else:
                "sub2, condition is false"

        @ppl
        def func(self):
            self.sub1()
            self.sub2()
            return records()

    prompt1 = ComplexPrompt(False).func()
    prompt2 = ComplexPrompt(True).func()
    assert str(prompt1) == "sub1, condition is false\nsub2, condition is false"
    assert str(prompt2) == "sub1, condition is true\nsub2, condition is true"


def test_generation_message():
    @ppl
    def func():
        "Hello World"
        gen1 = gen(lazy_eval=True)
        "Hi"
        gen2 = gen(lazy_eval=True)
        return gen1, gen2

    gen1, gen2 = func()
    assert str(gen1._args.messages) == "Hello World"
    assert str(gen2._args.messages) == "Hello World\nHi"


def test_generation_message2():
    def fakegen():
        return "24"

    @ppl
    def func():
        f"Q: 1 + 2 = ?"
        f"A: 3"
        f"Q: 15 + 9 = ?"
        f"A: {fakegen()}"
        return convo()

    assert str(func()) == "Q: 1 + 2 = ?\nA: 3\nQ: 15 + 9 = ?\nA: 24"
