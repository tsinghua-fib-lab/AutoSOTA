from inspect import signature

import pytest

from appl import as_func, need_ctx, partial, ppl, records

GLOBAL_V = 123


def test_docstring():
    @ppl
    def func():
        "This is a docstring"

    assert func.__doc__ == "This is a docstring"


def test_signature():
    @ppl
    def add(x: int, y: int) -> int:
        return x + y

    def add2(x: int, y: int, *, config=None) -> int:
        # config (langsmith_extra) is added by langsmith @traceable
        return x + y

    new_sig = signature(add)
    assert new_sig == signature(add.__wrapped__) or new_sig == signature(add2)


def test_functionality():
    @ppl
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 3


def test_decorator():
    def extra(func):
        @need_ctx
        def inner(*args, **kwargs):
            return func(*args, **kwargs) + 1

        return inner

    @extra
    @ppl
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 4


def test_classmethod():
    class A:
        @classmethod
        @ppl(ctx="same")
        def addon(cls):
            "addon"

        @classmethod
        @ppl
        def func(cls):
            "start"
            cls.addon()
            return records()

    assert str(A.func()) == "start\naddon"


def test_super_in_class():
    class A:
        def func(self):
            return 1

    class B(A):
        @ppl
        def func(self):
            return super(B, self).func()

    # NOTE: super() is not supported in ppl decorator
    # TODO: fix by filling the class name and self in the function during compiling.
    assert B().func() == 1


def test_nested_ppl_error():
    with pytest.raises(SyntaxError) as excinfo:

        @ppl
        def func():
            "Hello"

            @ppl  # nested ppl decorator is not supported yet.
            def func2():
                "World"

            func2()
            return records()

    assert "Nested ppl decorator is not" in str(excinfo.value)


def test_function_inside_ppl():
    @ppl
    def f1():
        "Hello"

        def f2():
            "World"

        f2()
        return records()

    assert str(f1()) == "Hello\nWorld"

    @ppl
    def f3():
        "Hello"

        def f4():
            "World"

        return records()

    assert str(f3()) == "Hello"


def test_closure():
    a = 1

    @ppl
    def func():
        return a

    assert func() == a
    a = 2
    assert func() == a
    f = as_func(func)
    assert f() == a

    b = 1

    def f2():
        @ppl
        def f3():
            return b

        return f3()

    assert f2() == b


def test_wrapper():
    a = 2

    def wrapper():
        a = 1

        @ppl
        def func(b):
            f"{a}+{b}={a+b}"
            return records()

        return as_func(func)  # use as_func to inject the locals of wrapper (a=1)

    # if not use as_func, the caller's locals will be used, which is from test_wrapper's locals (a=2)
    func = wrapper()
    assert str(func(2)) == "1+2=3"


def test_global():
    @ppl
    def func():
        return GLOBAL_V

    assert func() == GLOBAL_V
