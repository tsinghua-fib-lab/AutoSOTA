from __future__ import annotations

import inspect
import sys
import time
import traceback
from threading import Lock
from typing import Any, Callable, Literal, Optional

from .compile import appl_compile
from .context import PromptContext
from .modifiers import Compositor


class PromptFunc:
    """A wrapper for an APPL function, can be called as a normal function.

    The function contains a prompt context, which could be same as or
    copied from its caller function, or created from scratch, or resumed
    from the last run.
    """

    def __init__(
        self,
        func: Callable,
        ctx_method: str = "new",
        compositor: Optional[Compositor] = None,
        default_return: Optional[Literal["prompt"]] = None,
        docstring_as: Optional[str] = None,
        new_ctx_func: Callable = PromptContext,
        # default_sep: Optional[str] = None,
        # ? set the default printer behavior for the prompt function?
    ):
        """Initialize the PromptFunc.

        Args:
            func (Callable): the function being wrapped
            ctx_method (str):
                the method to deal with the child context, available methods includes:

                - (default) "new" or "new_ctx": create a brand new context.
                - "copy" or "copy_ctx":
                    copy from the parent's context, the change will not
                    affect the parent's context.
                - "same" or "same_ctx":
                    use the same context as the parent's, the change will
                    affect the parent's context.
                - "resume" or "resume_ctx":
                    resume its own context from the last run.
                    For the first run, it will copy the parent's context.

            compositor (Compositor, optional):
                the default compositor to be used. Defaults to None.
            default_return (str, optional):
                The default return value, "prompt" means return the prompt within
                the function. Defaults to None.
            docstring_as (str, optional):
                Include the triple-quoted docstring as a message in the prompt.
                Options include "user" and "system". Defaults to None.
            new_ctx_func (Callable, optional):
                the function to create a new context. Defaults to PromptContext.
        """
        self._func = appl_compile(func)
        self._signature = inspect.signature(func)
        self._doc = func.__doc__
        self._name = func.__name__
        self._qualname = func.__qualname__
        self._default_ctx_method = self._process_ctx_method(ctx_method)
        self._default_compositor = compositor
        if default_return is not None and default_return != "prompt":
            raise NotImplementedError("Only support default_return='prompt' now.")
        self._default_return = default_return
        self._docstring_as = docstring_as
        self._new_ctx_func = new_ctx_func
        self._persist_ctx: Optional[PromptContext] = None
        self._reset_context_func: Optional[Callable[[], None]] = None
        # self._default_sep = default_sep

    @property
    def compiled_func(self):
        """The compiled function."""
        return self._func

    def _process_ctx_method(self, ctx_method: str) -> str:
        available_methods = ["new", "copy", "same", "resume"]
        alias = [m + "_ctx" for m in available_methods]
        res = ctx_method
        if res in alias:
            res = available_methods[alias.index(ctx_method)]
        if res not in available_methods:
            raise ValueError(f"Unknown ctx_method: {ctx_method}")
        return res

    def _run(
        self,
        parent_ctx: PromptContext,
        child_ctx: PromptContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run the prompt function with desired context, deal with the exception."""
        # if parent_ctx.is_outmost:
        #     exc = None
        #     try:
        #         results = self._func(_ctx=child_ctx, *args, **kwargs)
        #     except Exception:
        #         # if parent_ctx.is_outmost:  # only print the trace in outmost appl function
        #         traceback.print_exc()
        #         print()  # empty line
        #         exc = sys.exc_info()

        #     if exc is not None:  # the outmost appl function
        #         _, e_instance, e_traceback = exc
        #         # already printed above, clean up the traceback
        #         if e_instance and e_traceback:
        #             e_traceback.tb_next = None
        #             raise e_instance.with_traceback(e_traceback)
        # else:  # run normally, capture the exception at the outmost appl function
        #     results = self._func(_ctx=child_ctx, *args, **kwargs)
        results = self._func(_ctx=child_ctx, *args, **kwargs)
        return results

    def _call(
        self, *args: Any, _ctx: Optional[PromptContext] = None, **kwargs: Any
    ) -> Any:
        """Call the prompt function."""
        parent_ctx = _ctx or self._new_ctx_func()
        is_class_method = kwargs.pop("_is_class_method", False)
        ctx_method = kwargs.pop("ctx_method", self._default_ctx_method)
        if ctx_method == "new":
            child_ctx = self._new_ctx_func()
        elif ctx_method == "copy":
            child_ctx = parent_ctx.copy()
        elif ctx_method == "same":
            child_ctx = parent_ctx.inherit()
        elif ctx_method == "resume":
            # NOTE: the resume method is not thread-safe
            # For standalone functions or class methods, they should de defined within the thread
            #   So that the function or `cls` is thread-local
            # For object methods, the object need to be created within the thread
            #   So that `self` is thread-local
            if is_class_method:
                self_or_cls = args[0]  # is it guaranteed? need double check
                var_name = f"{self._name}_appl_ctx_"

                def reset_context():
                    setattr(self_or_cls, var_name, None)

                # try to retrieve the context from the class
                if (ctx := getattr(self_or_cls, var_name, None)) is None:
                    # copy the parent context if not exist
                    child_ctx = parent_ctx.copy()
                    setattr(self_or_cls, var_name, child_ctx)
                else:
                    # resume from the last run, but with clean locals
                    child_ctx = ctx.inherit()
            else:

                def reset_context():
                    self._persist_ctx = None

                if self._persist_ctx is None:
                    self._persist_ctx = parent_ctx

                child_ctx = self._persist_ctx.inherit()

            self._reset_context_func = reset_context
        else:
            raise ValueError(f"Unknown ctx_method: {ctx_method}")
        child_ctx.is_outmost = False
        child_ctx._func_name = self._name
        child_ctx._func_docstring = self._doc
        child_ctx._docstring_quote_count = self._func._docstring_quote_count
        child_ctx._docstring_as = self._docstring_as  # set in the context

        compositor: Optional[Compositor] = kwargs.pop(
            "compositor", self._default_compositor
        )
        # push the compositor
        if compositor is not None:
            child_ctx.push_printer(compositor.push_args)

        # run the function
        results = self._run(parent_ctx, child_ctx, *args, **kwargs)
        if results is None and self._default_return == "prompt":
            results = child_ctx.records  # the default return result

        # pop the compositor
        if compositor is not None:
            child_ctx.pop_printer()

        if ctx_method == "same":
            parent_ctx.add_records(child_ctx.records, write_to_prompt=False)

        return results

    __call__ = _call

    def __repr__(self):
        res = f"[PromptFunc]{self._name}{self._signature}"
        if self._doc is not None:
            res += f": {self._doc}"
        return res
